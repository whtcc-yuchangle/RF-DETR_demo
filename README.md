# RF-DETR Object Detection Demo

基于 [RF-DETR](https://github.com/roboflow/rf-detr) 的目标检测、实例分割与关键点检测演示项目，使用 COCO 预训练模型对图片和视频进行实时推理。

## 环境

- Python 环境：`PyTorch` (conda)
- 依赖：`rfdetr`, `supervision`, `opencv-python`, `torch`

## 运行

```bash
# 目标检测 — 图片
python 02th_detect_image.py

# 目标检测 — 视频（实时显示 + 保存结果）
python 01th_detect_video.py

# 实例分割 — 图片
python 04th_segment_image.py

# 实例分割 — 视频（实时显示 + 保存结果）
python 03th_segment_video.py

# 关键点检测 — 图片
python 05th_keypoint_image.py
```

## 文件

| 文件 | 说明 |
|------|------|
| `02th_detect_image.py` | 目标检测 — 图片，弹窗显示标注结果 |
| `01th_detect_video.py` | 目标检测 — 视频，逐帧推理并输出 `video_output.mp4` |
| `04th_segment_image.py` | 实例分割 — 图片，弹窗显示 mask 标注结果 |
| `03th_segment_video.py` | 实例分割 — 视频，逐帧推理并输出 `video_output.mp4` |
| `05th_keypoint_image.py` | 关键点检测 — 图片，使用 COCO 17 点骨架，弹窗显示关键点及连线 |
| `dog.jpg` | 目标检测测试图片 |
| `zidane.jpg` | 关键点检测测试图片 |
| `video.mp4` | 测试视频 |

## 视频快捷键

- **Q** — 退出
- **空格** — 暂停 / 继续
