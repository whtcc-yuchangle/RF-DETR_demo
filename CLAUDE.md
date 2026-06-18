# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Environment

Always use the `PyTorch` conda environment:

```bash
/d/miniconda3/envs/PyTorch/python <script>.py
```

## Project Overview

A minimal demo using **RF-DETR** (Roboflow's real-time detection transformer) with **supervision** for annotation and **OpenCV** for video I/O. Covers object detection, instance segmentation, and keypoint detection.

### Scripts

- `02th_detect_image.py` — Static image detection. Run first to verify the model works on `dog.jpg`. Uses `supervision.plot_image()` to display results inline.
- `01th_detect_video.py` — Real-time video detection with on-screen display + output video. Press **Q** to quit, **Space** to pause/resume.
- `04th_segment_image.py` — Static image instance segmentation. Uses `sv.MaskAnnotator` to overlay segmentation masks.
- `03th_segment_video.py` — Real-time video instance segmentation with on-screen display + output video.
- `05th_keypoint_image.py` — Static image keypoint detection using COCO 17-point skeleton. Uses `RFDETRKeypointPreview`, `sv.EdgeAnnotator`, and `sv.VertexAnnotator` to draw keypoints and skeleton edges.

### Key Libraries

- `rfdetr` — Provides `RFDETRMedium` (detection), `RFDETRSegment` (segmentation), `RFDETRKeypointPreview` (keypoint), and `COCO_CLASSES` (80 classes)
- `supervision` (sv) — `BoxAnnotator`, `LabelAnnotator` for detections; `MaskAnnotator` for segmentation; `EdgeAnnotator`, `VertexAnnotator` for keypoints
- `cv2` — Video read/write, `imshow` display, BGR↔RGB conversion

### Model Input/Output Pattern

```python
model = RFDETRMedium()
detections = model.predict(image, threshold=0.5)  # image: path | np.ndarray | PIL.Image | torch.Tensor
# detections.class_id  -> class indices
# detections.metadata["source_image"] -> original image (when include_source_image=True)
```

**Important:** `model.predict()` accepts RGB-order images. OpenCV reads BGR, so convert with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before inference, but annotate the original BGR frame for correct display/saving.

### Key Differences for RF-DETR vs Other Detection Models

- After `model.predict()` for detection/segmentation, source image is in `detections.metadata["source_image"]` (not `detections.data["source_image"]` as in older supervision).
- For keypoint detection (`RFDETRKeypointPreview`), `model.predict()` returns `KeyPoints` object. Source image is in `key_points.data["source_image"]` (a list, one per detection). The image may need deduplication — multiple detections on the same image share the same backing image.
- Edge annotation requires explicit edge definitions. COCO 17-keypoint skeleton edges are defined in the script as `COCO_EDGES`.
- COCO class mapping: `COCO_CLASSES[class_id]` returns the English class name string.
