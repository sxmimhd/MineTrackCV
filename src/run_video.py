
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def main():
    parser = argparse.ArgumentParser(
        description="MineTrackCV: real-time Minecraft mob tracking."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Video file path or webcam index, e.g. 0.",
    )
    args = parser.parse_args()

    with (ROOT / "configs" / "config.yaml").open(
        "r", encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)

    weights = Path(config["model"]["weights"])
    if not weights.is_absolute():
        weights = ROOT / weights
    if not weights.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {weights}")

    output_dir = ROOT / config["video"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    source_name = Path(args.source).stem if isinstance(source, str) else "webcam"
    output_video = output_dir / f"{source_name}_tracked.mp4"
    output_csv = output_dir / f"{source_name}_metrics.csv"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = config["video"]["output_fps_fallback"]

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    model = YOLO(str(weights))
    tracker = config["tracking"]["tracker"]
    device = config["model"]["device"]
    confidence = config["model"]["confidence"]
    image_size = config["model"]["image_size"]

    frame_index = 0
    total_detections = 0
    start_time = time.perf_counter()

    csv_file = (
        output_csv.open("w", newline="", encoding="utf-8")
        if config["monitoring"]["save_csv"]
        else None
    )
    csv_writer = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            inference_start = time.perf_counter()
            results = model.track(
                frame,
                persist=True,
                tracker=tracker,
                conf=confidence,
                imgsz=image_size,
                device=device,
                verbose=False,
                iou=0.85,
                max_det=500,
            )
            inference_ms = (time.perf_counter() - inference_start) * 1000

            result = results[0]
            boxes = result.boxes
            detection_count = len(boxes)
            total_detections += detection_count

            track_ids = []
            if boxes is not None and boxes.id is not None:
                track_ids = [
                    int(track_id)
                    for track_id in boxes.id.cpu().tolist()
                ]

            elapsed = time.perf_counter() - start_time
            pipeline_fps = (frame_index + 1) / elapsed if elapsed else 0.0

            annotated = result.plot()
            cv2.putText(
                annotated,
                f"FPS: {pipeline_fps:.1f} | Detections: {detection_count}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated,
                f"Inference: {inference_ms:.1f} ms",
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("Could not create output video.")

                if csv_file is not None:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([
                        "frame",
                        "inference_ms",
                        "pipeline_fps",
                        "detections",
                        "track_ids",
                    ])

            writer.write(annotated)

            if csv_writer is not None:
                csv_writer.writerow([
                    frame_index,
                    round(inference_ms, 3),
                    round(pipeline_fps, 3),
                    detection_count,
                    ";".join(map(str, track_ids)),
                ])

            frame_index += 1

            if frame_index % 30 == 0:
                print(
                    f"Frames: {frame_index} | "
                    f"FPS: {pipeline_fps:.1f} | "
                    f"Detections: {detection_count}"
                )

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()

    elapsed = time.perf_counter() - start_time
    print("\nMineTrackCV finished.")
    print(f"Frames processed: {frame_index}")
    print(f"Total detections: {total_detections}")
    print(f"Average pipeline FPS: {frame_index / elapsed:.2f}" if elapsed else "Average pipeline FPS: 0")
    print(f"Video: {output_video}")
    if config["monitoring"]["save_csv"]:
        print(f"Metrics: {output_csv}")


if __name__ == "__main__":
    main()
