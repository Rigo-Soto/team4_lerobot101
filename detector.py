"""
This script if for testing the docker with the weights defined by the YOLO algorithhm
If this file has made it to the repository please delete it

- Arthwwr
"""
import cv2
import numpy as np
import time

CLASS_NAMES = ["base", "column"]  # match your training labels
CONF_THRESHOLD = 0.45
NMS_THRESHOLD = 0.4

# AI BULLSHIT
class YOLODetector:
    def __init__(self, weights: str, config: str, input_size=(416, 416)):
        self.net = cv2.dnn.readNetFromDarknet(config, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.input_size = input_size
        self.out_layers = [
            self.net.getLayerNames()[i - 1]
            for i in self.net.getUnconnectedOutLayers()
        ]

    def detect(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, self.input_size, swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.out_layers)

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for det in output:
                scores = det[5:]
                cid = int(np.argmax(scores))
                conf = float(scores[cid])
                if conf < CONF_THRESHOLD:
                    continue
                cx, cy, bw, bh = (det[:4] * [w, h, w, h]).astype(int)
                x, y = cx - bw // 2, cy - bh // 2
                boxes.append([x, y, bw, bh])
                confidences.append(conf)
                class_ids.append(cid)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        detections = {}
        for i in (indices.flatten() if len(indices) else []):
            name = CLASS_NAMES[class_ids[i]]
            x, y, bw, bh = boxes[i]
            detections[name] = {
                "bbox":       [x, y, bw, bh],          # pixels
                "centroid":   [x + bw/2, y + bh/2],    # pixels
                "confidence": confidences[i],
                "area":       bw * bh,
            }
        return detections  # keys: "base", "column" (absent if not detected)


if __name__ == '__main__':
    detect = YOLODetector('./backup/yolov4-tiny-custom_final.weights', 'custom_cfg/yolov4-tiny-custom.cfg')
    capt = cv2.VideoCapture(2)

    while capt.isOpened():
        _, frame = capt.read()
        if frame is None:
            raise Exception('Error while reading image')
        detected = detect.detect(frame)
        print(detected)

        time.sleep(1)
