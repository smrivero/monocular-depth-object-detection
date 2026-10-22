import cv2
import numpy as np
from ultralytics import YOLO

INPUT = "longer-input.mov"
OUTPUT = "yolo26-detect-depth.mp4"

detect_model = YOLO("yolo26n.pt")
depth_model = YOLO("yolo26n-depth.pt")

cap = cv2.VideoCapture(INPUT)

if not cap.isOpened():
    raise RuntimeError(f"No pude abrir {INPUT}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

print("===== YOLO26 DETECTION + DEPTH =====")
print("Resolucion:", width, "x", height)
print("FPS original:", fps)
print("Frames:", frames)
print()

frame_count = 0

while True:
    ok, frame = cap.read()

    if not ok:
        break

    det = detect_model(
        frame,
        imgsz=640,
        conf=0.35,
        device=0,
        verbose=False
    )[0]

    dep = depth_model(
        frame,
        imgsz=768,
        device=0,
        verbose=False
    )[0]

    depth = dep.depth.data.cpu().numpy()

    for box in det.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = det.names[cls]

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            continue

        bw = x2 - x1
        bh = y2 - y1

        # Usamos el 40% central del objeto
        rx1 = int(x1 + bw * 0.30)
        rx2 = int(x2 - bw * 0.30)
        ry1 = int(y1 + bh * 0.30)
        ry2 = int(y2 - bh * 0.30)

        roi = depth[ry1:ry2, rx1:rx2]

        valid = roi[
            np.isfinite(roi) &
            (roi > 0)
        ]

        if valid.size == 0:
            continue

        distance = float(np.median(valid))

        label = f"{name} {conf:.2f} | {distance:.1f} m"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        ty1 = max(0, y1 - th - 12)

        cv2.rectangle(
            frame,
            (x1, ty1),
            (x1 + tw + 8, y1),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    writer.write(frame)

    frame_count += 1

    if frame_count % 50 == 0:
        print(f"{frame_count}/{frames}")

cap.release()
writer.release()

print()
print("Frames procesados:", frame_count)
print("Output:", OUTPUT)
