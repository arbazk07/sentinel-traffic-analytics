"""
run.py
------
Main entry point. Opens a video source (webcam or file), runs detection +
tracking on every frame, draws the results, and displays them live —
optionally also saving the annotated video to a file.

USAGE:
    python run.py --source webcam
    python run.py --source path/to/video.mp4
    python run.py --source webcam --save output.mp4
    python run.py --source path/to/video.mp4 --save output.mp4 --no-display

Press 'q' at any time while the video window is focused to quit early.
"""

import argparse
import time
import cv2
from detector import ObjectDetector
from visualizer import draw_detections, draw_hud


def open_video_source(source_arg):
    """
    Translates our --source argument into something cv2.VideoCapture
    understands. "webcam" maps to camera index 0 (the default camera);
    anything else is treated as a file path.
    """
    if source_arg.lower() == "webcam":
        cap = cv2.VideoCapture(0)
        source_description = "webcam"
    else:
        cap = cv2.VideoCapture(source_arg)
        source_description = f"file: {source_arg}"

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source ({source_description}). "
                            f"Check that the webcam is connected, or the file path is correct.")

    return cap, source_description


def main():
    parser = argparse.ArgumentParser(description="Real-time object detection and tracking.")
    parser.add_argument("--source", type=str, default="webcam",
                         help="'webcam' for a live camera feed, or a path to a video file.")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                         help="Which YOLO model weights to use.")
    parser.add_argument("--confidence", type=float, default=0.4,
                         help="Minimum confidence score to keep a detection.")
    parser.add_argument("--save", type=str, default=None,
                         help="If set, saves the annotated video to this file path.")
    parser.add_argument("--no-display", action="store_true",
                         help="Don't open a live preview window (useful when only --save matters).")
    args = parser.parse_args()

    detector = ObjectDetector(model_name=args.model, confidence_threshold=args.confidence)
    cap, source_description = open_video_source(args.source)
    print(f"Video source opened: {source_description}")

    writer = None
    if args.save:
        fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_source, (frame_width, frame_height))
        print(f"Saving annotated output to: {args.save}")

    prev_frame_time = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("No more frames to read (end of video or camera disconnected).")
                break

            detections = detector.detect_and_track(frame)
            frame = draw_detections(frame, detections)

            current_time = time.time()
            fps = 1.0 / (current_time - prev_frame_time) if current_time != prev_frame_time else 0.0
            prev_frame_time = current_time

            frame = draw_hud(frame, fps, len(detections))

            if writer is not None:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("Object Detection & Tracking — press 'q' to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quit requested by user.")
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("Resources released. Done.")


if __name__ == "__main__":
    main()
