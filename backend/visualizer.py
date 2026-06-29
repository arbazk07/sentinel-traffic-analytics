"""
visualizer.py
-------------
Draws bounding boxes, class labels, and tracking IDs onto a video frame.

Kept separate from detector.py on purpose — detector.py answers "what did
we find," this file answers "how do we draw it." This separation means
we could swap in a completely different visual style (different colors,
a minimap, a counter overlay) without touching any detection logic.
"""

import cv2
import hashlib

# A small fixed palette so boxes are visually distinct without being
# garish — colors are BGR (OpenCV's color order, not RGB).
COLOR_PALETTE = [
    (66, 135, 245),   # blue
    (52, 199, 89),    # green
    (255, 149, 0),    # orange
    (175, 82, 222),   # purple
    (255, 59, 48),    # red
    (0, 199, 190),    # teal
]


def color_for_track_id(track_id):
    """
    Picks a consistent color for a given tracking ID, so the SAME
    tracked object keeps the SAME box color across frames — this is a
    small but important visual cue that reinforces "this is the same
    object being tracked," not a fresh detection each frame.

    We use a hash of the ID rather than just `track_id % len(palette)`
    purely so consecutive IDs (1, 2, 3...) don't end up with visually
    similar adjacent colors in the palette — a cosmetic nicety, not a
    functional requirement.
    """
    if track_id is None:
        return (128, 128, 128)  # gray for not-yet-tracked detections
    digest = int(hashlib.md5(str(track_id).encode()).hexdigest(), 16)
    return COLOR_PALETTE[digest % len(COLOR_PALETTE)]


def _rects_overlap(a, b):
    """True if two (x1, y1, x2, y2) rectangles intersect at all."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def draw_detections(frame, detections, reserved_top_margin=45, dwell_times=None):
    """
    Draws all detections onto the given frame IN PLACE.

    reserved_top_margin: the HUD (drawn separately by draw_hud) occupies
    the top-left corner of the frame. Any detection label that would
    land inside that reserved area is drawn BELOW its box's top edge
    instead, so it never gets visually buried under the HUD bar.

    dwell_times: optional dict of {track_id: seconds}. When provided,
    each box's label includes how long that ID has been tracked — this
    is what turns a generic "person #3" label into something that
    visibly demonstrates WHY tracking matters (a number that only makes
    sense if the system remembers this object across many frames).

    LABEL COLLISION HANDLING: when objects stand close together (e.g.
    several people shoulder-to-shoulder), their labels would naturally
    land at the same height and overlap into illegible text. We track
    every label rectangle already placed THIS frame, and if a new one
    would collide, push it downward in small steps until it clears —
    same idea as how mapping libraries avoid overlapping pin labels.
    """
    placed_label_rects = []

    # Draw boxes first in one pass, labels in a second pass — this keeps
    # label placement decisions based on a stable set of boxes rather
    # than interleaving, and means a later box's outline is never drawn
    # on top of an earlier box's label.
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = color_for_track_id(det["track_id"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = color_for_track_id(det["track_id"])

        if det["track_id"] is not None:
            label = f"{det['class_name']} #{det['track_id']}"
            if dwell_times and det["track_id"] in dwell_times:
                label += f" - {dwell_times[det['track_id']]:.1f}s"
        else:
            label = f"{det['class_name']} ({det['confidence']:.2f})"

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        label_y_top = y1 - text_h - 8
        if label_y_top < reserved_top_margin:
            label_bg_top = y1
            label_bg_bottom = y1 + text_h + 8
        else:
            label_bg_top = label_y_top
            label_bg_bottom = y1

        candidate_rect = (x1, label_bg_top, x1 + text_w + 6, label_bg_bottom)

        # Nudge downward in small steps until this label clears every
        # label already placed this frame, or we give up after a
        # reasonable number of attempts (dense clusters of objects will
        # always have SOME visual crowding — we just avoid the worst,
        # fully-illegible overlaps).
        attempts = 0
        step = text_h + 10
        while any(_rects_overlap(candidate_rect, placed) for placed in placed_label_rects) and attempts < 6:
            label_bg_top += step
            label_bg_bottom += step
            candidate_rect = (x1, label_bg_top, x1 + text_w + 6, label_bg_bottom)
            attempts += 1

        placed_label_rects.append(candidate_rect)

        # Text sits near the bottom of whichever label box we ended up
        # with, regardless of whether it was nudged down or not.
        text_y = label_bg_bottom - 5

        cv2.rectangle(frame, (x1, label_bg_top), (x1 + text_w + 6, label_bg_bottom), color, -1)
        cv2.putText(frame, label, (x1 + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def draw_trajectories(frame, analytics):
    """
    Draws each currently-active track's recent motion trail as a fading
    line — this is a direct VISUAL proof that tracking is working, since
    a trail can only exist if the system has correctly linked many
    frames' detections together under one consistent ID. A pure
    per-frame detector could never produce this; there'd be no "history"
    to draw a line through.

    The trail fades from transparent (oldest point) to fully opaque
    (most recent point), using OpenCV's weighted blending per segment.
    """
    active_ids = analytics.get_active_track_ids()

    for track_id in active_ids:
        trajectory = analytics.get_trajectory(track_id)
        if len(trajectory) < 2:
            continue

        color = color_for_track_id(track_id)
        num_points = len(trajectory)

        for i in range(1, num_points):
            # Older segments are drawn more faded (lower alpha) by
            # blending toward the existing frame content rather than
            # drawing at full opacity — this is what produces the visual
            # "fading tail" effect rather than a single flat-color line.
            alpha = i / num_points
            overlay = frame.copy()
            cv2.line(overlay, trajectory[i - 1], trajectory[i], color, 2)
            cv2.addWeighted(overlay, alpha * 0.6, frame, 1 - (alpha * 0.6), 0, frame)

    return frame


def draw_zone_line(frame, zone_line, entries, exits):
    """
    Draws the configured analytics zone line plus a live entry/exit
    counter readout near it, so the line's purpose is visually obvious
    rather than an unexplained line across the frame.
    """
    if zone_line is None:
        return frame

    (x1, y1), (x2, y2) = zone_line
    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

    label = f"IN: {entries}   OUT: {exits}"
    label_x = max(10, min(x1, x2) - 20)
    label_y = max(20, min(y1, y2) - 10)
    cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def draw_hud(frame, fps, object_count):
    """
    Draws a small heads-up display in the corner: current FPS and how
    many objects are currently being tracked. Useful both for genuinely
    monitoring performance and for making a demo video look more
    intentional/informative rather than just "boxes on a video."
    """
    hud_text = f"FPS: {fps:.1f}  |  Objects: {object_count}"
    cv2.rectangle(frame, (10, 10), (260, 40), (30, 30, 30), -1)
    cv2.putText(frame, hud_text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame
