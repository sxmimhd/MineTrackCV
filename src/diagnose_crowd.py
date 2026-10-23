
from pathlib import Path
from collections import Counter

import cv2
from ultralytics import YOLO

VIDEO = r"C:\Users\samym\Downloads\ALL MOBS 500v500 BATTLE in Minecraft Mob Battle - Alpha Wise (1080p).mp4"

WEIGHTS = (
    r"C:\Users\samym\Desktop\Projects\SceneForgeCV"
    r"\runs\detection\yolov8n_minecraft_mobs\weights\best.pt"
)

OUTPUT = Path("runs/diagnostics")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Grab a crowded frame around 2:28.
cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_MSEC, 148_000)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError(f"Could not read video: {VIDEO}")

model = YOLO(WEIGHTS)

tests = [
    {
        "name": "high_recall",
        "conf": 0.05,
        "imgsz": 1280,
        "iou": 0.70,
    },
    {
        "name": "crowded_nms",
        "conf": 0.05,
        "imgsz": 1280,
        "iou": 0.85,
    },
    {
        "name": "higher_resolution",
        "conf": 0.05,
        "imgsz": 1536,
        "iou": 0.85,
    },
]

for test in tests:
    result = model.predict(
        source=frame,
        conf=test["conf"],
        imgsz=test["imgsz"],
        iou=test["iou"],
        max_det=500,
        device=0,
        verbose=False,
    )[0]

    boxes = result.boxes
    count = len(boxes)

    class_counts = Counter(
        model.names[int(class_id)]
        for class_id in boxes.cls.tolist()
    )

    print(f"\n--- {test['name']} ---")
    print(f"Image size: {test['imgsz']}")
    print(f"Confidence: {test['conf']}")
    print(f"Detections: {count}")
    print(f"By class: {dict(class_counts)}")

    annotated = result.plot()
    output_path = OUTPUT / f"{test['name']}.jpg"
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved: {output_path.resolve()}")
