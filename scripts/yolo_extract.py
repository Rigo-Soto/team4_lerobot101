#!/usr/bin/env python3
"""
Image feature extraction utilities using pjreddie/darknet through ctypes.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from ctypes import (
    Structure,
    POINTER,
    pointer,
    c_int,
    c_float,
    c_void_p,
)

from darknet_detect import load_darknet_network
from darknet_detect import BOX,DETECTION,IMAGE
import darknet_detect as darknet


CLASS_NAMES = ["base", "column"]  # Debe coincidir con tu .names / entrenamiento
CONF_THRESHOLD = 0.45
NMS_THRESHOLD = 0.4




def _resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path.resolve()

    if base_dir is None:
        base_dir = Path(".").resolve()

    return (base_dir / path).resolve()


def _frame_to_darknet_image(frame: np.ndarray) -> tuple[IMAGE, np.ndarray]:
    """
    Convierte un frame BGR de OpenCV a IMAGE de Darknet.

    Importante:
    - Darknet espera float32 en formato CHW.
    - OpenCV usa BGR, por eso convertimos a RGB.
    - Regresamos también el arreglo numpy para mantener viva la memoria.
    """

    if frame is None:
        raise ValueError("El frame es None")

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Frame inválido. Se esperaba HxWx3, llegó: {frame.shape}")

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape

    chw = rgb.transpose(2, 0, 1)
    data_np = np.ascontiguousarray(chw, dtype=np.float32) / 255.0
    data_ptr = data_np.ctypes.data_as(POINTER(c_float))

    darknet_image = IMAGE(w=w, h=h, c=c, data=data_ptr)

    return darknet_image, data_np


class YOLODetector:
    """
    Wrapper para Tiny YOLO usando libdarknet.so de pjreddie/darknet.

    Returns:
        {
            "base": {
                "bbox": [x, y, w, h],
                "centroid": [cx, cy],
                "confidence": float,
                "area": int,
            },
            "column": {...},
        }
    """

    def __init__(
        self,
        weights: str,
        config: str,
        class_names: list[str] | None = None,
        conf_threshold: float = CONF_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
        gpu_id: int = 0,
    ):
        work_dir = Path(".").resolve()

        cfg_path = _resolve_path(config, work_dir)
        weights_path = _resolve_path(weights, work_dir)

        self.class_names = class_names if class_names is not None else CLASS_NAMES
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)

        self.net = load_darknet_network(
            cfg_path=cfg_path,
            weights_path=weights_path,
            gpu_id=gpu_id,
        )


        logging.info(f"[YOLO] Loaded Darknet model")
        logging.info(f"[YOLO] cfg: {cfg_path}")
        logging.info(f"[YOLO] weights: {weights_path}")
        logging.info(f"[YOLO] classes: {self.class_names}")



    def detect(self, frame: np.ndarray) -> dict[str, dict]:
        h, w = frame.shape[:2]
    
        darknet_image, data_np = _frame_to_darknet_image(frame)
    
        # Mantener viva la memoria del frame convertido
        _keep_alive = data_np
    
        num = c_int(0)
        pnum = pointer(num)
        null_map = POINTER(c_int)()

        darknet.predict_image(self.net, darknet_image)

        dets,num,pnum = darknet.get_network_boxes_safe(
            self.net,
            w,
            h,
            (self.conf_threshold),
            (0.5),
        )

    
        num_dets = pnum[0]
    
        if not dets or num_dets == 0:
            return {}
    
        try:
            darknet.do_nms_sort(
                dets,
                num_dets,
                len(self.class_names),
                c_float(self.nms_threshold),
            )
    
            best: dict[str, dict] = {}
    
            for i in range(num_dets):
                det = dets[i]
    
                for cid, name in enumerate(self.class_names):
                    conf = float(det.prob[cid])
    
                    if conf < self.conf_threshold:
                        continue
    
                    bbox = det.bbox
    
                    bw = int(round(bbox.w))
                    bh = int(round(bbox.h))
                    x = int(round(bbox.x - bbox.w / 2))
                    y = int(round(bbox.y - bbox.h / 2))
    
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))
                    bw = max(1, min(bw, w - x))
                    bh = max(1, min(bh, h - y))
    
                    if name not in best or conf > best[name]["confidence"]:
                        best[name] = {
                            "bbox": [x, y, bw, bh],
                            "centroid": [x + bw / 2, y + bh / 2],
                            "confidence": conf,
                            "area": bw * bh,
                        }

            return best
    
        finally:

            darknet.free_detections(dets, num_dets)

    def draw_overlay(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        """
        Dibuja cajas y etiquetas.
        """

        out = frame.copy()
        colours = {
            "base": (0, 255, 120),
            "column": (0, 160, 255),
        }

        for name, det in detections.items():
            x, y, bw, bh = det["bbox"]
            colour = colours.get(name, (200, 200, 200))

            cv2.rectangle(out, (x, y), (x + bw, y + bh), colour, 2)

            label = f"{name} {det['confidence']:.2f}"
            cv2.putText(
                out,
                label,
                (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                colour,
                2,
            )

        return out


def extract_feature_vector(
    detections: dict,
    frame_shape: tuple,
    depth_map: np.ndarray = None,
) -> np.ndarray:
    """
    Returns a 16-dim vector regardless of which objects are detected.
    Missing detections are zero-filled.
    """

    H, W = frame_shape[:2]
    feat = np.zeros(16, dtype=np.float32)

    for idx, cls in enumerate(["base", "column"]):
        base_i = idx * 7
        det = detections.get(cls)

        if det is None:
            feat[base_i + 6] = 0.0
            continue

        x, y, bw, bh = det["bbox"]
        cx, cy = det["centroid"]

        feat[base_i + 0] = cx / W
        feat[base_i + 1] = cy / H
        feat[base_i + 2] = bw / W
        feat[base_i + 3] = bh / H
        feat[base_i + 4] = det["confidence"]
        feat[base_i + 5] = (bw * bh) / (W * H)
        feat[base_i + 6] = 1.0

    if "base" in detections and "column" in detections:
        bcx, bcy = detections["base"]["centroid"]
        ccx, ccy = detections["column"]["centroid"]
        feat[14] = (ccx - bcx) / W
        feat[15] = (ccy - bcy) / H

    return feat
