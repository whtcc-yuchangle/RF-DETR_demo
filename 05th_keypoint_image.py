import cv2
import supervision as sv
from rfdetr import RFDETRKeypointPreview

# COCO 17 关键点骨架连接（边）
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # 鼻子 ↔ 眼睛/耳朵
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 肩膀 ↔ 手臂
    (5, 11), (6, 12), (11, 12),           # 肩膀 ↔ 髋部
    (11, 13), (13, 15), (12, 14), (14, 16),   # 髋部 ↔ 腿
    (0, 5), (0, 6),                        # 鼻子 ↔ 肩膀
]

# 初始化模型
model = RFDETRKeypointPreview()

# 关键点检测
key_points = model.predict("zidane.jpg", threshold=0.5)

# 获取原始图片（KeyPoints 将 source_image 存在 data 中，每个检测一张）
source_images = key_points.data.get("source_image", [])
if source_images:
    # 去重：同一张图的多个检测共享同一张背景图
    image = source_images[0]
else:
    image = cv2.imread("zidane.jpg")

# RGB → BGR（supervision 标注器工作在 RGB 下，但 matplotlib/show 需要 BGR）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] == 3 else image

# 初始化标注器
edge_annotator = sv.EdgeAnnotator(color=sv.Color.GREEN, thickness=2, edges=COCO_EDGES)
vertex_annotator = sv.VertexAnnotator(color=sv.Color.RED, radius=4)

# 标注关键点及骨架连线
annotated_image = edge_annotator.annotate(image_rgb, key_points)
annotated_image = vertex_annotator.annotate(annotated_image, key_points)

# 显示结果
sv.plot_image(annotated_image)
