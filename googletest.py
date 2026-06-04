import cv2
import numpy as np


WEIGHTS = "backup/yolov4-tiny-custom_final.weights"
CONFIG  = "custom_cfg/yolov4-tiny-custom.cfg"

# 1. Load COCO class names
classes = []
with open("custom_cfg/custom.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 2. Build and initialize the network
# Provide paths to your downloaded .cfg and .weights files
net = cv2.dnn.readNetFromDarknet(CONFIG, WEIGHTS)

# Optional: Enable CUDA if OpenCV is compiled with GPU support
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Get output layer names from the YOLO architecture
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# 3. Open Video Stream (0 for default webcam, or provide a 'video.mp4' path)
cap = cv2.VideoCapture(2)

# Set confidence and NMS thresholds
CONFIDENCE_THRESHOLD = 0.1
NMS_THRESHOLD = 0.1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Perform object detection
    classes_detected, confidences, boxes = model.detect(
        frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
    )

    # 5. Draw bounding boxes on the frame
    for (class_id, score, box) in zip(classes_detected, confidences, boxes):
        # Unpack box coordinates
        x, y, w, h = box
        
        # Define color and label format
        color = (0, 255, 0)
        label = f"{classes[class_id]}: {score:.2f}"
        
        # Draw bounding rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Add class label text above the box
        cv2.putText(
            frame, label, (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    # Display the output
    cv2.imshow("YOLOv4-Tiny Detection", frame)

    # Press 'q' to exit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

