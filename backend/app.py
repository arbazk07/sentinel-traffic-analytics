"""
app.py
------
Flask backend for the live analytics dashboard. Streams annotated video
(detection boxes + fading trajectory trails + zone line) to the browser
via MJPEG, and exposes a JSON endpoint for the live stats panel.

ARCHITECTURE:
A single background thread continuously reads frames from the video
source (webcam or uploaded file), runs detection+tracking+analytics on
each one, and stores the latest ANNOTATED frame plus the latest stats
snapshot in shared state. Two separate Flask routes then just read from
that shared state:
  - /video_feed streams the annotated frames as MJPEG
  - /api/stats returns the latest analytics snapshot as JSON
This way the expensive work (YOLO inference) happens exactly once per
frame, regardless of how many browser tabs are watching the stream or
how often the stats panel polls for updates.
"""

import os
import cv2
import time
import threading
from flask import Flask, Response, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from detector import ObjectDetector
from analytics import TrackAnalytics
from visualizer import draw_detections, draw_trajectories, draw_zone_line, draw_hud

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "../data/uploads"

# Frames wider than this get downscaled before running detection. YOLO's
# inference time scales with input pixel count, and high-resolution
# uploaded videos (e.g. 1080p+ phone recordings) were the actual cause
# of very slow processing (sub-1 FPS) — capping width here is the fix.
# 640 matches YOLO11n's typical internal input size, so this loses
# little real detection accuracy while substantially cutting per-frame
# inference time.
MAX_PROCESSING_WIDTH = 640
os.makedirs(UPLOAD_DIR, exist_ok=True)

detector = ObjectDetector(model_name="yolo11n.pt", confidence_threshold=0.4)


class VideoStream:
    """
    Owns the background processing thread and all shared state between
    it and the Flask routes. One instance per "active session" — created
    fresh whenever the user starts a new source (webcam or a newly
    uploaded file), and cleanly stopped before starting another.
    """

    def __init__(self, source):
        self.source = source
        self.cap = None
        self.analytics = TrackAnalytics()
        self.latest_frame_jpeg = None
        self.latest_stats = {}
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.prev_frame_time = time.time()
        self.fps = 0.0

    def configure_zone(self, point_a, point_b):
        self.analytics.configure_zone(point_a, point_b)

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()

    def _process_loop(self):
        """
        Runs continuously in a background thread: read a frame, detect,
        track, update analytics, draw, encode to JPEG, store. Looping
        back to the start of the file when a video file ends (rather
        than stopping) keeps a demo recording playing indefinitely
        instead of freezing on the last frame.
        """
        while self.running:
            success, frame = self.cap.read()

            if not success:
                if isinstance(self.source, str):
                    # Reached the end of a video file — loop back to the
                    # start so a demo recording plays continuously rather
                    # than freezing, which matters for an unattended dashboard.
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break  # webcam disconnected — nothing sensible to loop

            # Downscale before running detection if the frame is larger
            # than MAX_PROCESSING_WIDTH. This is the main fix for slow
            # uploaded-video performance: YOLO's inference time scales
            # with input pixel count, and a phone-shot video can easily
            # be 1080p or higher — running full-resolution inference on
            # every frame is the actual bottleneck, not the model itself.
            # YOLO11n internally resizes to ~640px on its longest side
            # regardless, so downscaling beforehand loses little real
            # detection accuracy while cutting inference time substantially.
            original_height, original_width = frame.shape[:2]
            if original_width > MAX_PROCESSING_WIDTH:
                scale = MAX_PROCESSING_WIDTH / original_width
                frame = cv2.resize(
                    frame,
                    (MAX_PROCESSING_WIDTH, int(original_height * scale)),
                    interpolation=cv2.INTER_AREA
                )

            detections = detector.detect_and_track(frame)
            self.analytics.update(detections)

            dwell_times = {
                tid: self.analytics.get_dwell_time(tid)
                for tid in self.analytics.get_active_track_ids()
            }

            frame = draw_trajectories(frame, self.analytics)
            frame = draw_detections(frame, detections, dwell_times=dwell_times)
            frame = draw_zone_line(
                frame, self.analytics.zone_line,
                self.analytics.zone_entries, self.analytics.zone_exits
            )

            now = time.time()
            self.fps = 1.0 / (now - self.prev_frame_time) if now != self.prev_frame_time else 0.0
            self.prev_frame_time = now
            frame = draw_hud(frame, self.fps, len(detections))

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            with self.lock:
                self.latest_frame_jpeg = jpeg.tobytes()
                self.latest_stats = self.analytics.get_snapshot()
                self.latest_stats["fps"] = round(self.fps, 1)

    def get_frame_bytes(self):
        with self.lock:
            return self.latest_frame_jpeg

    def get_stats(self):
        with self.lock:
            return dict(self.latest_stats)


# Global active stream — this app supports one active session at a time,
# which matches the actual use case (one dashboard, one camera/video at
# a time), and avoids the complexity of multi-session video pipelines.
active_stream = None
stream_lock = threading.Lock()


def _stop_active_stream():
    global active_stream
    with stream_lock:
        if active_stream is not None:
            active_stream.stop()
            active_stream = None


@app.route("/")
def serve_frontend():
    return send_from_directory("../frontend", "index.html")


@app.route("/<path:filename>")
def serve_static_files(filename):
    return send_from_directory("../frontend", filename)


@app.route("/api/start", methods=["POST"])
def start_stream():
    """
    POST /api/start
    Expects: {"source": "webcam"} or {"source": "uploaded", "filename": "..."}
    Starts (or restarts) the analytics session on the chosen source.
    """
    global active_stream

    data = request.get_json(silent=True) or {}
    source_type = data.get("source", "webcam")

    if source_type == "webcam":
        video_source = 0
    elif source_type == "uploaded":
        filename = data.get("filename")
        if not filename:
            return jsonify({"error": "filename is required for uploaded source"}), 400
        video_source = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(video_source):
            return jsonify({"error": "Uploaded file not found"}), 404
    else:
        return jsonify({"error": "source must be 'webcam' or 'uploaded'"}), 400

    _stop_active_stream()

    try:
        with stream_lock:
            active_stream = VideoStream(video_source)
            active_stream.start()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "started", "source": source_type})


@app.route("/api/stop", methods=["POST"])
def stop_stream():
    _stop_active_stream()
    return jsonify({"status": "stopped"})


@app.route("/api/upload", methods=["POST"])
def upload_video():
    """
    POST /api/upload (multipart/form-data, field name "video")
    Saves an uploaded video file for later use as a source.
    """
    if "video" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    return jsonify({"status": "uploaded", "filename": filename})


@app.route("/api/configure_zone", methods=["POST"])
def configure_zone():
    """
    POST /api/configure_zone
    Expects: {"point_a": [x, y], "point_b": [x, y]}
    Sets the entry/exit counting line on the CURRENTLY active stream.
    """
    if active_stream is None:
        return jsonify({"error": "No active stream — start one first"}), 400

    data = request.get_json(silent=True) or {}
    point_a = data.get("point_a")
    point_b = data.get("point_b")

    if not point_a or not point_b:
        return jsonify({"error": "point_a and point_b are required"}), 400

    active_stream.configure_zone(tuple(point_a), tuple(point_b))
    return jsonify({"status": "zone configured"})


def _mjpeg_generator():
    """
    Yields frames in the multipart/x-mixed-replace format browsers
    expect for MJPEG streaming. If no stream is active yet, yields
    nothing — the <img> tag will simply show as broken until a stream
    starts, which the frontend handles by hiding/showing the feed based
    on session state rather than relying on this to look correct alone.
    """
    while True:
        if active_stream is None:
            time.sleep(0.2)
            continue

        frame_bytes = active_stream.get_frame_bytes()
        if frame_bytes is None:
            time.sleep(0.05)
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.03)  # cap streaming rate; the actual processing rate is independent of this


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def stats():
    if active_stream is None:
        return jsonify({"active": False})
    snapshot = active_stream.get_stats()
    snapshot["active"] = True
    return jsonify(snapshot)


if __name__ == "__main__":
    app.run(debug=False, port=5003, threaded=True)
