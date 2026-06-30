# Sentinel - Real-Time Foot Traffic & Dwell Analytics

A live object detection and tracking system that goes beyond drawing
boxes: it turns tracked identities into real foot-traffic analytics —
unique visitor counts, per-object dwell time, motion trails, and
directional entry/exit counting across a configurable line — streamed to
a live web dashboard.

## Why this exists

Per-frame object detection alone can't answer the questions that actually
matter for spaces, queues, or storefronts: *how many distinct people
visited*, *how long did each one stay*, *how many crossed into this
area*. Naively counting detections overcounts massively — a single
person standing still for 10 seconds at 30fps would register as 300
"people" if you just summed boxes per frame. **Tracking IDs are what
make these real questions answerable**, and this project exists to
demonstrate exactly that gap: every metric on the dashboard depends on
identity persisting across frames, not just detection within one.

## What it does

- Detects and tracks people, vehicles, and other objects in real time
  from a webcam or an uploaded video file.
- Counts **unique visitors** (deduplicated by tracked identity, not
  raw per-frame detections).
- Tracks **dwell time** per object — how long each tracked identity has
  remained visible — shown live in a sorted leaderboard.
- Draws **fading motion trails** showing each object's recent path.
- Lets you draw a **counting line** directly on the video; tracked
  objects crossing it are counted directionally (in / out), using the
  side of the line each track's center-point falls on, frame to frame.
- Streams everything — annotated video plus live stats — to a real-time
  web dashboard.

## Tech stack

| Layer        | Tech                                          |
|--------------|----------------------------------------------|
| Detection    | YOLO11 (nano variant), via `ultralytics`       |
| Tracking     | ByteTrack (built into `ultralytics`)            |
| Analytics    | Custom dwell-time, unique-counting, and line-crossing engine |
| Streaming    | Flask, MJPEG video stream + polled JSON stats API |
| Frontend     | HTML, CSS, vanilla JavaScript                    |

## Project structure

```
sentinel-traffic-analytics/
├── backend/
│   ├── detector.py        # YOLO loading + per-frame detect_and_track()
│   ├── analytics.py         # dwell time, unique counting, zone-line crossing logic
│   ├── visualizer.py          # draws boxes, labels, motion trails, zone line, HUD
│   ├── app.py                   # Flask server: video streaming + stats API
│   ├── yolo11n.pt                 # pretrained YOLO11 nano weights
│   └── requirements.txt
└── frontend/
    ├── index.html              # dashboard layout
    ├── style.css                 # console-style dashboard theme
    └── script.js                   # video controls, zone-line drawing, live stats polling
```

## How it works

1. **Detection + tracking** (`detector.py`): each frame is passed to a
   pretrained YOLO11 model via `.track()` (not `.predict()`), which
   additionally runs ByteTrack to assign a persistent ID to each object
   across frames — this identity is the foundation everything else
   depends on.
2. **Analytics** (`analytics.py`): for every tracked ID, records first-seen
   and last-seen timestamps (dwell time), a capped history of recent
   center-points (for motion trails), and — if a counting line is
   configured — which side of that line the object's center currently
   falls on, using a 2D cross-product test. A side change between
   consecutive frames is logged as a directional crossing.
3. **Visualization** (`visualizer.py`): draws bounding boxes, labels
   (including live dwell time), fading trajectory trails, the counting
   line with a running in/out tally, and an FPS/object-count HUD. Labels
   that would otherwise overlap when objects stand close together are
   automatically nudged apart to stay legible.
4. **Streaming** (`app.py`): runs detection in a background thread,
   serves the annotated video as an MJPEG stream (`/video_feed`) and
   exposes live analytics as JSON (`/api/stats`) for the dashboard to
   poll. Supports both webcam and uploaded video files as input.

## Running it locally

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5003` in your browser. Choose a webcam or upload
a video file, then optionally click **Draw counting line** and click two
points on the video to define a crossing line.

## Notes

- The YOLO11 nano weights are bundled directly in `backend/` for
  convenience; `ultralytics` will also auto-download them if missing.
- Dwell time and unique-visitor counts depend entirely on tracking
  identity persistence — if the tracker loses and re-acquires an object
  (heavy occlusion, object leaving and re-entering frame), it will be
  counted as a new visitor. This is an inherent limitation of any
  appearance/motion-based tracker, not specific to this implementation.
- The crossing-line direction convention (left-to-right = entry,
  right-to-left = exit) is arbitrary but consistent — useful to know if
  adapting this for a real doorway, where the "correct" in/out mapping
  depends on which side faces the entrance.
