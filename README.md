# RF-DETR Object Detection Demo

基于 [RF-DETR](https://github.com/roboflow/rf-detr) 的目标检测演示项目，使用 COCO 预训练模型对图片和视频进行实时目标检测。

## 环境

- Python 环境：`PyTorch` (conda)
- 依赖：`rfdetr`, `supervision`, `opencv-python`, `torch`

## 运行

```bash
# 图片检测
python 02th_detect_image.py

# 视频检测（实时显示 + 保存结果）
python 01th_detect_video.py
```

## 文件

| 文件 | 说明 |
|------|------|
| `02th_detect_image.py` | 对 `dog.jpg` 做目标检测，弹窗显示标注结果 |
| `01th_detect_video.py` | 对 `video.mp4` 逐帧检测，实时显示并输出 `video_output.mp4` |
| `dog.jpg` | 测试图片 |
| `video.mp4` | 测试视频 |

## 视频检测快捷键

- **Q** — 退出
- **空格** — 暂停 / 继续
