
import cv2
from pathlib import Path

VIDEO = Path(
    r"C:\Users\samym\Downloads\ALL MOBS 500v500 BATTLE in Minecraft Mob Battle - Alpha Wise (1080p).mp4"
)
OUTPUT = Path("runs/crowd_test.mp4")

START_SECONDS = 130  # 2:10
DURATION_SECONDS = 30

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0 or width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("Could not read video metadata.")

start_frame = int(START_SECONDS * fps)
end_frame = min(
    start_frame + int(DURATION_SECONDS * fps),
    total_frames,
)

if start_frame >= total_frames:
    cap.release()
    raise RuntimeError("Start time is beyond the end of the video.")

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

writer = cv2.VideoWriter(
    str(OUTPUT),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")

written = 0

try:
    for _ in range(end_frame - start_frame):
        ok, frame = cap.read()
        if not ok:
            break

        writer.write(frame)
        written += 1
finally:
    cap.release()
    writer.release()

if written == 0:
    raise RuntimeError("No frames were written.")

print(f"Created: {OUTPUT.resolve()}")
print(f"Frames: {written}")
print(f"Duration: {written / fps:.2f} seconds")
print(f"Resolution: {width}x{height}")
