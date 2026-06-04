from ctypes import *
import cv2
import numpy as np
import darknet_detect as darknet
from yolo_extract import _frame_to_darknet_image

net = darknet.load_darknet_network()

frame = cv2.imread("/home/angel/images.jpeg")
img, data_np = _frame_to_darknet_image(frame)

print("predict", flush=True)
darknet.predict_image(net, img)

print("boxes", flush=True)
dets, n = darknet.get_network_boxes_safe(net, frame.shape[1], frame.shape[0], 0.45, 0.5)

print("OK", n, dets, flush=True)
darknet.free_detections(dets, n)
