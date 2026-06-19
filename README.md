# RF-DETR Object Detection Demo

基于 [RF-DETR](https://github.com/roboflow/rf-detr) 的目标检测、实例分割与关键点检测演示项目，使用 COCO 预训练模型对图片和视频进行实时推理。

## 环境

- Python 环境：`PyTorch` (conda)
- 依赖：`rfdetr`, `supervision`, `opencv-python`, `torch`

```bash
# 激活环境
conda activate PyTorch

# 确保 opencv-python 为非 headless 版本（否则 GUI 不可用）
pip uninstall opencv-python-headless -y 2>nul
pip install opencv-python --force-reinstall --no-deps
```

## 项目结构

```
RF-DETR_demo/
├── 02th_detect_image.py      # 目标检测 — 图片（COCO 80 类）
├── 01th_detect_video.py      # 目标检测 — 视频（COCO 80 类）
├── 04th_segment_image.py     # 实例分割 — 图片
├── 03th_segment_video.py     # 实例分割 — 视频
├── 05th_keypoint_image.py    # 关键点检测 — 图片（COCO 17 点骨架）
├── dog.jpg                   # 目标检测测试图片
├── zidane.jpg                # 关键点检测测试图片
├── video.mp4                 # 测试视频
└── train/
    └── helmet/               # 安全帽检测自定义训练项目
        ├── README.md         # 训练项目完整文档
        ├── train.py          # 训练脚本
        ├── test.py           # 视频推理测试脚本
        ├── datasets/         # 数据集（230 训练 / 20 验证）
        ├── output/           # 训练输出（模型权重 + 日志）
        ├── helmet.mp4        # 测试视频
        ├── test_helmet.jpg   # 正样本测试图片（戴安全帽）
        └── test_not.jpg      # 负样本测试图片（未戴安全帽）
```

## 快速开始 — COCO 预训练模型推理

### 目标检测

```bash
# 图片
python 02th_detect_image.py

# 视频（实时显示 + 保存结果）
python 01th_detect_video.py
```

### 实例分割

```bash
# 图片
python 04th_segment_image.py

# 视频（实时显示 + 保存结果）
python 03th_segment_video.py
```

### 关键点检测

```bash
# 图片（人体 17 点骨架）
python 05th_keypoint_image.py
```

## 自定义训练 — 安全帽检测

项目支持基于 RF-DETR 的自定义数据训练。`train/helmet/` 是一个完整的安全帽佩戴检测训练示例。

### 数据集

- 2 分类：`helmet`（戴安全帽）、`not`（未戴安全帽）
- 训练集 230 张，验证集 20 张
- YOLO 格式标注

### 训练

```bash
cd train/helmet
python train.py
```

### 测试

```bash
cd train/helmet
python test.py
```

### 训练结果

| 指标 | 最优值 |
|---|---|
| helmet AP | 75.8% |
| not AP | 62.8% |
| mAP@50-95 | 60.5% |

> 完整文档请参阅 [train/helmet/README.md](train/helmet/README.md)

## 文件说明

| 文件 | 说明 |
|------|------|
| `02th_detect_image.py` | 目标检测 — 图片，弹窗显示标注结果 |
| `01th_detect_video.py` | 目标检测 — 视频，逐帧推理并输出 `video_output.mp4` |
| `04th_segment_image.py` | 实例分割 — 图片，弹窗显示 mask 标注结果 |
| `03th_segment_video.py` | 实例分割 — 视频，逐帧推理并输出 `video_output.mp4` |
| `05th_keypoint_image.py` | 关键点检测 — 图片，使用 COCO 17 点骨架 |
| `dog.jpg` | 目标检测测试图片 |
| `zidane.jpg` | 关键点检测测试图片 |
| `video.mp4` | 测试视频 |
| `train/helmet/` | 安全帽检测自定义训练项目（数据集 + 训练 + 测试） |

## 视频快捷键

- **Q** — 退出
- **空格** — 暂停 / 继续

## 模型输入/输出规范

```python
model = RFDETRMedium()
detections = model.predict(image, threshold=0.5)
# image 支持: path | np.ndarray | PIL.Image | torch.Tensor
# detections.class_id  -> 类别索引
# detections.confidence -> 置信度
# detections.xyxy       -> 检测框坐标
```

- **色彩空间**: `model.predict()` 接受 RGB 输入；OpenCV 读取的是 BGR，需 `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` 转换
- **检查点加载**: 自定义训练后使用 `RFDETRMedium(pretrain_weights="path/to/checkpoint.pth", num_classes=N)` 加载微调权重

## 常见问题

| 问题 | 解决方法 |
|------|----------|
| `cv2.error: The function is not implemented` | 卸载 `opencv-python-headless`，保留 `opencv-python` |
| 训练时 DataLoader 崩溃 | 使用 `if __name__ == "__main__":` 保护 + `num_workers=0` |
| 模型加载 class 数量警告 | 传入 `num_classes` 参数匹配训练时的类别数 |

## 参考

- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
- [Supervision 文档](https://supervision.roboflow.com)
