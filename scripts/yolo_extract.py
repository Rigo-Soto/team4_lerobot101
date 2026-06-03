"""
Image feature extractio utilities

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
    """
    Thin wrapper around OpenCV DNN for Tiny YOLOv4 inference.
 
    Returns a dict keyed by class name (e.g. "base", "column"), each value
    being a sub-dict with bbox, centroid, confidence, and area.
    """
 
    def __init__(
        self,
        weights: str,
        config: str,
        input_size: tuple = (416, 416),
        conf_threshold: float = 0.45,
        nms_threshold: float = 0.4,
        class_names: list[str] = None,
        use_gpu: bool = True,
    ):
        if not Path(weights).exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        if not Path(config).exists():
            raise FileNotFoundError(f"YOLO config not found: {config}")
 
        self.net = cv2.dnn.readNetFromDarknet(config, weights)
        if use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
 
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.class_names = class_names or []
        self._out_layer_names = [
            self.net.getLayerNames()[i - 1]
            for i in self.net.getUnconnectedOutLayers()
        ]
        logging.info(f"[YOLO] Loaded model: {weights}")
 
    def detect(self, frame: np.ndarray) -> dict[str, dict]:
        """
        Run inference on a single BGR or RGB frame.
 
        Returns:
            {
                "base":   {"bbox": [x,y,w,h], "centroid": [cx,cy], "confidence": float, "area": int},
                "column": {...},
                ...
            }
            Missing detections are simply absent from the dict.
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, self.input_size, swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self._out_layer_names)
 
        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for det in output:
                scores = det[5:]
                cid = int(np.argmax(scores))
                conf = float(scores[cid])
                if conf < self.conf_threshold:
                    continue
                cx_n, cy_n, bw_n, bh_n = det[:4]
                bw = int(bw_n * w)
                bh = int(bh_n * h)
                x  = int(cx_n * w - bw / 2)
                y  = int(cy_n * h - bh / 2)
                boxes.append([x, y, bw, bh])
                confidences.append(conf)
                class_ids.append(cid)
 
        if not boxes:
            return {}
 
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_threshold, self.nms_threshold
        )
        indices = indices.flatten() if len(indices) else []
 
        # Keep only the highest-confidence detection per class
        best: dict[str, dict] = {}
        for i in indices:
            cid = class_ids[i]
            if cid >= len(self.class_names):
                continue
            name = self.class_names[cid]
            conf = confidences[i]
            if name not in best or conf > best[name]["confidence"]:
                x, y, bw, bh = boxes[i]
                best[name] = {
                    "bbox":       [x, y, bw, bh],
                    "centroid":   [x + bw / 2, y + bh / 2],
                    "confidence": conf,
                    "area":       bw * bh,
                }
        return best
 
    def draw_overlay(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of the frame."""
        out = frame.copy()
        colours = {"base": (0, 255, 120), "column": (0, 160, 255)}
        for name, det in detections.items():
            x, y, bw, bh = det["bbox"]
            colour = colours.get(name, (200, 200, 200))
            cv2.rectangle(out, (x, y), (x + bw, y + bh), colour, 2)
            label = f"{name} {det['confidence']:.2f}"
            cv2.putText(out, label, (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        return out

def extract_feature_vector(
    detections: dict,
    frame_shape: tuple,        # (H, W)
    depth_map: np.ndarray = None,  # optional, from RealSense
) -> np.ndarray:
    """
    Returns a 16-dim vector regardless of which objects are detected.
    Missing detections are zero-filled (handled by the policy's validity flag).
    """
    H, W = frame_shape[:2]
    feat = np.zeros(16, dtype=np.float32)

    for idx, cls in enumerate(["base", "column"]):
        base_i = idx * 7           # 7 features per class
        det = detections.get(cls)
        if det is None:
            feat[base_i + 6] = 0.0  # validity flag = 0 (missing)
            continue

        x, y, bw, bh = det["bbox"]
        cx, cy = det["centroid"]

        # Normalise to [0, 1]
        feat[base_i + 0] = cx / W                   # centroid x
        feat[base_i + 1] = cy / H                   # centroid y
        feat[base_i + 2] = bw / W                   # relative width
        feat[base_i + 3] = bh / H                   # relative height
        feat[base_i + 4] = det["confidence"]        # confidence
        feat[base_i + 5] = (bw * bh) / (W * H)     # relative area
        feat[base_i + 6] = 1.0                      # validity flag

    # Indices 14–15: relative pose between column and base
    if "base" in detections and "column" in detections:
        bcx, bcy = detections["base"]["centroid"]
        ccx, ccy = detections["column"]["centroid"]
        feat[14] = (ccx - bcx) / W   # Δx (column relative to base)
        feat[15] = (ccy - bcy) / H   # Δy

    return feat


