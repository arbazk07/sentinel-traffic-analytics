"""
analytics.py
------------
The actual "purposeful" layer on top of raw detection+tracking: turns a
per-frame list of tracked boxes into real metrics over TIME.

WHY THIS NEEDS TRACKING (NOT JUST DETECTION):
Plain per-frame detection has no memory — every frame, "a person" is
detected fresh, with no link to the person detected in the previous
frame. If you tried to count "how many people visited" by counting
detections, a single person standing still for 10 seconds at 30fps would
get counted as 300 different "people." Tracking IDs are what make it
possible to ask real questions like "how many DISTINCT people were here"
and "how long did each one stay" — this file is where that distinction
actually pays off.

THREE METRICS TRACKED PER OBJECT ID:
  1. Dwell time — how long (in seconds) this ID has been continuously
     visible, from first detection to most recent detection.
  2. Trajectory — a short, capped history of this ID's center-point
     positions, used to draw a fading motion trail.
  3. Zone crossings — if a zone line is configured, whether this ID has
     crossed it, and in which direction (in vs out).

UNIQUE VISITOR COUNTING:
We count every DISTINCT track_id we've ever seen, not detections-per-
frame. A track_id that disappears (object left frame or was occluded)
and never reappears is simply done contributing — but it still counts
toward the total unique visitor count, since it really was here once.
"""

import time
from collections import deque, defaultdict


class TrackAnalytics:
    def __init__(self, trajectory_length=30, stale_after_seconds=3.0):
        """
        trajectory_length: how many recent center-points to remember per
            track, for drawing motion trails. Longer = longer visible
            trails, but more drawing cost per frame.
        stale_after_seconds: if a track ID hasn't been seen for this long,
            we consider it gone (left frame, fully occluded, etc.) and stop
            counting it as "currently present" — but its historical stats
            (total dwell time, unique-visitor count) are preserved.
        """
        self.trajectory_length = trajectory_length
        self.stale_after_seconds = stale_after_seconds

        # Per-track-id bookkeeping. Using plain dicts (not a class) keeps
        # this easy to serialize to JSON for the frontend stats endpoint.
        self.first_seen = {}       # track_id -> timestamp of first detection
        self.last_seen = {}        # track_id -> timestamp of most recent detection
        self.class_names = {}      # track_id -> class name (e.g. "person")
        self.trajectories = defaultdict(lambda: deque(maxlen=self.trajectory_length))

        # Zone crossing state, set up via configure_zone(). None means no
        # zone is configured and crossing detection is simply skipped.
        self.zone_line = None      # ((x1, y1), (x2, y2))
        self.zone_side_history = {}  # track_id -> last known side of the line ("left"/"right")
        self.zone_entries = 0
        self.zone_exits = 0

    def configure_zone(self, point_a, point_b):
        """
        Defines a line segment. Crossing it triggers an entry/exit count,
        based on which side of the line a track's center point was on
        the previous frame vs. this frame.
        """
        self.zone_line = (point_a, point_b)

    def _side_of_line(self, point):
        """
        Returns "left" or "right" of the configured zone line, using the
        sign of the 2D cross product — a standard way to determine which
        side of a directed line a point falls on without needing
        trigonometry.
        """
        (x1, y1), (x2, y2) = self.zone_line
        px, py = point
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        return "left" if cross > 0 else "right"

    def update(self, detections):
        """
        Call this once per frame with the current frame's detections
        (the same list shape detector.py's detect_and_track() returns).
        Updates all bookkeeping and returns nothing — read results back
        via get_snapshot().
        """
        now = time.time()

        for det in detections:
            track_id = det["track_id"]
            if track_id is None:
                continue  # tracker hasn't assigned this detection an ID yet — nothing to accumulate

            if track_id not in self.first_seen:
                self.first_seen[track_id] = now
            self.last_seen[track_id] = now
            self.class_names[track_id] = det["class_name"]

            x1, y1, x2, y2 = det["box"]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.trajectories[track_id].append(center)

            if self.zone_line is not None:
                current_side = self._side_of_line(center)
                previous_side = self.zone_side_history.get(track_id)
                if previous_side is not None and previous_side != current_side:
                    # Crossed the line this frame. Direction convention:
                    # left-to-right counts as an "entry," right-to-left as
                    # an "exit" — arbitrary but consistent, and documented
                    # here so the frontend can label it sensibly.
                    if previous_side == "left" and current_side == "right":
                        self.zone_entries += 1
                    else:
                        self.zone_exits += 1
                self.zone_side_history[track_id] = current_side

    def get_active_track_ids(self):
        """
        Returns track IDs considered "currently present" — seen recently
        enough to not be stale. Used for both the live "currently in
        frame" count and for deciding which trajectories to still draw.
        """
        now = time.time()
        return [
            tid for tid, last in self.last_seen.items()
            if (now - last) <= self.stale_after_seconds
        ]

    def get_dwell_time(self, track_id):
        """Seconds between first and most recent detection for this ID."""
        if track_id not in self.first_seen:
            return 0.0
        return self.last_seen[track_id] - self.first_seen[track_id]

    def get_trajectory(self, track_id):
        """Recent center-point history for drawing a motion trail."""
        return list(self.trajectories.get(track_id, []))

    def get_snapshot(self):
        """
        Returns a complete, JSON-serializable snapshot of current
        analytics state — this is what the /api/stats endpoint sends to
        the frontend's live stats panel.
        """
        active_ids = self.get_active_track_ids()

        dwell_leaderboard = sorted(
            [
                {
                    "track_id": tid,
                    "class_name": self.class_names.get(tid, "unknown"),
                    "dwell_seconds": round(self.get_dwell_time(tid), 1),
                    "is_active": tid in active_ids
                }
                for tid in self.first_seen
            ],
            key=lambda entry: entry["dwell_seconds"],
            reverse=True
        )[:10]  # top 10 longest-dwelling, most relevant for a live leaderboard

        return {
            "unique_visitors_total": len(self.first_seen),
            "currently_in_frame": len(active_ids),
            "zone_entries": self.zone_entries,
            "zone_exits": self.zone_exits,
            "zone_configured": self.zone_line is not None,
            "dwell_leaderboard": dwell_leaderboard
        }
