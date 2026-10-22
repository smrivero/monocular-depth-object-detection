import cv2
import time
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colorize_depth

INPUT = "input.mp4"
OUTPUT = "yolo26-depth-video.mp4"

MODEL = "yolo26n-depth.pt"
IMGSZ = 768

model = YOLO(MODEL)

cap = cv2.VideoCapture(INPUT)

if not cap.isOpened():
    raise RuntimeError(f"No pude abrir {INPUT}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("===== INPUT =====")
print("Resolucion:", width, "x", height)
print("FPS:", fps)
print("Frames:", frames)

# Output:
# original | depth
out_width = width * 2

writer = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (out_width, height)
)

frame_count = 0
total_inference = 0.0
total_processing = 0.0

print("\nProcesando...")

while True:
    ok, frame = cap.read()

    if not ok:
        break

    t0 = time.perf_counter()

    t_inf = time.perf_counter()

    result = model(
        frame,
        imgsz=IMGSZ,
        device=0,
        verbose=False
    )[0]

    total_inference += time.perf_counter() - t_inf

    depth = result.depth.data.cpu().numpy()

    # Rango fijo para todos los frames
    depth_color = colorize_depth(
        depth,
        vmin=0.0,
        vmax=20.0,
        cmap="inferno",
        mode="metric"
    )

    # Por seguridad, igualamos tamaño
    depth_color = cv2.resize(
        depth_color,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    # Labels
    cv2.putText(
        frame,
        "RGB",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        depth_color,
        "YOLO26 DEPTH  0-20 m",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    combined = np.hstack((frame, depth_color))

    writer.write(combined)

    total_processing += time.perf_counter() - t0

    frame_count += 1

    if frame_count % 50 == 0:
        print(f"{frame_count}/{frames}")

cap.release()
writer.release()

print("\n===== RESULTADOS =====")
print("Frames:", frame_count)

if frame_count:
    inf_ms = total_inference / frame_count * 1000
    total_ms = total_processing / frame_count * 1000

    print(f"Inference:       {inf_ms:.2f} ms/frame")
    print(f"Processing total:{total_ms:.2f} ms/frame")
    print(f"FPS end-to-end:  {1000 / total_ms:.2f}")

print("\nOutput:", OUTPUT)
