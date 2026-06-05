from yolo_extract import YOLODetector
import cv2
import os
import time
from pathlib import Path

WORK_DIR = Path(".").resolve()
CFG_PATH = WORK_DIR / "custom_cfg/yolov4-tiny-custom.cfg"
WEIGHTS_PATH = WORK_DIR / "backup/yolov4-tiny-custom_final.weights"

def main():
	# detector = YOLODetector('backup/yolov4-tiny-custom_final.weights', 'custom_cfg/yolov4-tiny-custom.cfg', use_gpu=False, conf_threshold=0.1)

	cfg = CFG_PATH
	weights = WEIGHTS_PATH
	detector = YOLODetector("backup/yolov4-tiny-custom_final.weights", "custom_cfg/yolov4-tiny-custom.cfg", conf_threshold=0.2)

	cap = cv2.VideoCapture(2)

	while cap.isOpened():
		ret,frame = cap.read()

		if frame is None:
			print("noframe")
			continue	
		
		cv2.imshow("papu", frame)

		if hasattr(frame, "numpy"): 
             # Convert CHW float tensor → HWC uint8 for OpenCV
			 frame = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
		
		detected = detector.detect(frame)
		overlay = detector.draw_overlay(frame,detected)
		cv2.imshow("papu", overlay)
		cv2.waitKey(1)
		time.sleep(1/30) 

if __name__ == '__main__':
	main()
