import cv2
import supervision as sv
from rfdetr import RFDETRMedium

# -------- 配置 --------
INPUT_VIDEO = "helmet.mp4"
OUTPUT_VIDEO = "helmet_output.mp4"
CONFIDENCE_THRESHOLD = 0.5
MODEL_PATH = "output\checkpoint_best_total.pth"
# 与训练时的 data.yaml 保持一致
CLASS_NAMES = ["helmet", "not"]
# --------------------

# 初始化模型（num_classes=2 避免加载时的警告）
model = RFDETRMedium(pretrain_weights=MODEL_PATH, num_classes=2)

# 初始化标注器
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 打开输入视频
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"视频信息: {width}x{height}, {fps:.2f} FPS, {total_frames} 帧")

# 创建输出视频写入器
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# 创建实时显示窗口
cv2.namedWindow("RF-DETR Detection", cv2.WINDOW_NORMAL)

frame_idx = 0
paused = False
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB（rfdetr 需要 RGB 输入）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 目标检测
        detections = model.predict(frame_rgb, threshold=CONFIDENCE_THRESHOLD)

        # 生成标签（自定义类别名 + 置信度）
        labels = [
            f"{CLASS_NAMES[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # 在原始 BGR 帧上标注（OpenCV 写入需要 BGR）
        annotated_frame = box_annotator.annotate(frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

        # 写入输出视频
        out.write(annotated_frame)

        # 实时显示
        cv2.imshow("RF-DETR Detection", annotated_frame)

        # 进度显示
        frame_idx += 1
        if frame_idx % 300 == 0 or frame_idx == total_frames:
            print(f"已处理: {frame_idx}/{total_frames} 帧 ({100 * frame_idx / total_frames:.1f}%)")

    # 键盘控制：Q 退出，空格 暂停/继续
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print(f"用户中断于第 {frame_idx}/{total_frames} 帧")
        break
    elif key == ord(" "):
        paused = not paused

# 清理资源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"处理完成! 输出视频: {OUTPUT_VIDEO}")
