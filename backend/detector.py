"""
detector.py
-----------
Wraps the YOLO model: loading it once, and running detection+tracking on
a single frame at a time, returning clean structured results.

WHY YOLO?
YOLO ("You Only Look Once") is a real-time object detection architecture
that processes an entire image in a single pass through the network,
predicting all bounding boxes and class labels simultaneously. This is
what makes it fast enough for live video — older detection approaches
(like R-CNN) scan an image in multiple stages/passes and are much slower.

WE USE A PRETRAINED MODEL — NO TRAINING NEEDED.
Unlike Task 3 (where we trained our own model from scratch), here we use
a model Ultralytics has ALREADY trained on the COCO dataset — 80 common
object classes (person, car, dog, bicycle, etc.) using millions of
labeled images. Training an object detector from scratch is a massive
undertaking; using a strong pretrained model is the standard, practical
approach for almost all real-world object detection projects.

WHY .track() INSTEAD OF .predict()?
.predict() would re-detect every object fresh on every frame, with NO
memory of what was detected in the previous frame — so a person walking
across the screen would technically be a "new" detection every frame,
with no consistent identity. .track() additionally runs a tracking
algorithm (we use ByteTrack, the modern default) that matches each new
frame's detections to objects seen in previous frames, assigning a
STABLE ID that persists as long as the object keeps being detected.
This stable ID is what lets us say "Person #3" consistently rather than
relabeling everyone every frame.

NOTE ON SORT/DEEPSORT:
The original task brief mentions SORT or Deep SORT specifically. ByteTrack
(used here via Ultralytics' built-in .track()) is a more modern tracker
that builds on the same core ideas (Kalman filter motion prediction +
IOU-based matching between frames) but with improved handling of
low-confidence detections. It's a reasonable, well-regarded substitute —
worth knowing the relationship if asked about it.
"""

from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_name="yolo11n.pt", confidence_threshold=0.4):
        """
        model_name: which YOLO weights to use. "yolo11n.pt" is the
        "nano" variant — smallest and fastest, ideal for real-time CPU
        inference. Larger variants (yolo11s/m/l/x) trade speed for
        accuracy if you have a GPU and want better detection quality.

        confidence_threshold: detections below this confidence score are
        discarded. 0.4 is a reasonable default — low enough to catch most
        real objects, high enough to filter out a lot of noisy false
        positives.
        """
        print(f"Loading YOLO model: {model_name}...")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        print(f"Model loaded. Tracking {len(self.model.names)} object classes.")

    def detect_and_track(self, frame):
        """
        Runs detection + tracking on a single video frame.

        Returns a list of dicts, one per detected object:
            {
                "track_id": int or None,   # None if tracker hasn't assigned an ID yet
                "class_name": str,         # e.g. "person", "car"
                "confidence": float,
                "box": (x1, y1, x2, y2)    # bounding box corners, in pixel coordinates
            }

        persist=True tells the tracker to remember object identities
        BETWEEN calls to this method — without this, calling .track()
        repeatedly would behave like .predict(), losing all tracking
        memory every single frame.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence_threshold,
            verbose=False
        )

        detections = []
        result = results[0]

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])

                track_id = int(box.id[0]) if box.id is not None else None

                detections.append({
                    "track_id": track_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": (int(x1), int(y1), int(x2), int(y2))
                })

        return detections
