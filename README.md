# Monocular Depth and Object Detection on NVIDIA Jetson

Video-based object detection and monocular depth estimation using YOLO26 on NVIDIA Jetson.

## Overview

This project combines object detection with neural depth estimation to produce annotated video output.

Unlike a stereo depth camera pipeline, the implementation estimates depth from ordinary RGB video frames using a learned model.

The current Python scripts do not consume RealSense hardware depth measurements.

## Processing Pipeline

1. Decode RGB video.
2. Execute YOLO26 object detection.
3. Execute a YOLO26 depth model.
4. Extract depth estimates.
5. Associate detection boxes with depth regions.
6. Visualize depth maps and annotated objects.
7. Encode output video.

## Components

| File | Purpose |
|---|---|
| `video_depth.py` | RGB video and estimated depth visualization |
| `detect_depth_video.py` | Object detection with depth estimates |

## Models

The scripts reference:

    yolo26n.pt
    yolo26n-depth.pt

The depth model requires a compatible Ultralytics implementation.

Model checkpoints are not included in Git.

## Requirements

- NVIDIA Jetson Orin Nano Super 8 GB
- CUDA-enabled PyTorch
- Compatible Ultralytics installation
- OpenCV
- NumPy
- Input RGB video

## Execution

For side-by-side RGB and depth visualization, provide `input.mp4` and run:

    python3 video_depth.py

For detections with estimated depth values, provide `longer-input.mov` and run:

    python3 detect_depth_video.py

Input videos and model checkpoints are excluded from the repository.

## Depth Estimation

The detection pipeline samples the central region of each detected bounding box and computes the median of valid depth predictions.

This reduces the effect of some outlying pixels compared with using a single point.

Estimated depth must not be treated as a validated physical distance measurement without independent calibration and testing.

## Limitations

- Depth is inferred from RGB images.
- No hardware depth stream is consumed.
- Estimated depth accuracy has not been independently validated.
- Performance depends on scene geometry and model generalization.
- No verified live camera FPS or depth accuracy result is claimed.

## Project Scope

Monocular depth estimation, object detection, GPU inference, video processing and spatial visualization.
