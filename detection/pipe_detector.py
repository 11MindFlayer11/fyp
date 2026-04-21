"""
detection/pipe_detector.py

YOLOv8-based pipe detection.

Returns a list of PipeDetection objects, each containing:
  - bounding box (x1, y1, x2, y2) in image pixels
  - pixel centre (cx, cy)
  - confidence score
  - class name
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from config.settings import (
    YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_IOU_THRESHOLD,
    YOLO_DEVICE, YOLO_PIPE_CLASS_ID, YOLO_INPUT_SIZE,
)

log = logging.getLogger(__name__)


@dataclass
class PipeDetection:
    """A single pipe detection in one frame."""
    bbox:         np.ndarray       # [x1, y1, x2, y2] in image pixels (float32)
    confidence:   float
    class_id:     int
    class_name:   str
    pixel_center: np.ndarray = field(init=False)   # (cx, cy)

    def __post_init__(self):
        self.pixel_center = np.array([
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        ])

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height


class PipeDetector:
    """
    Wraps a YOLOv8 model for pipe detection.

    If the model file does not exist, a warning is logged and the detector
    falls back to a stub that returns empty lists (useful for testing the rest
    of the pipeline without weights).
    """

    def __init__(self, model_path: str = YOLO_MODEL_PATH):
        self._model = None
        self._names: dict = {}
        self._load_model(model_path)

    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO
            if not os.path.exists(model_path):
                log.warning(
                    f"YOLO model not found at '{model_path}'. "
                    "Detector will return empty results until model is provided."
                )
                return
            self._model = YOLO(model_path)
            self._model.to(YOLO_DEVICE)
            self._names = self._model.names
            log.info(f"YOLO model loaded: '{model_path}' on {YOLO_DEVICE}")
        except ImportError:
            log.error("ultralytics not installed. Run: pip install ultralytics")
        except Exception as exc:
            log.error(f"Failed to load YOLO model: {exc}")

    # ──────────────────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> List[PipeDetection]:
        """
        Run inference on `frame_bgr` and return a list of PipeDetection.
        Thread-safe (model inference is GIL-protected by PyTorch).
        """
        if self._model is None:
            return []

        results = self._model.predict(
            source=frame_bgr,
            conf=YOLO_CONFIDENCE,
            iou=YOLO_IOU_THRESHOLD,
            imgsz=YOLO_INPUT_SIZE,
            verbose=False,
            stream=False,
        )

        detections: List[PipeDetection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())

                # Filter to pipe class if specified
                if YOLO_PIPE_CLASS_ID is not None and cls_id != YOLO_PIPE_CLASS_ID:
                    continue

                conf      = float(box.conf[0].item())
                bbox      = box.xyxy[0].cpu().numpy().astype(np.float32)
                cls_name  = self._names.get(cls_id, f"cls_{cls_id}")

                detections.append(PipeDetection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                ))

        log.debug(f"Detected {len(detections)} pipe(s)")
        return detections

    # ──────────────────────────────────────────────────────────────────────────

    def draw_detections(self, frame_bgr: np.ndarray,
                        detections: List[PipeDetection],
                        world_coords: Optional[List[Optional[np.ndarray]]] = None
                        ) -> np.ndarray:
        """
        Annotate frame with bounding boxes, confidence, and optional world coords.
        Returns a new annotated image (does not modify input).
        """
        vis = frame_bgr.copy()

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox.astype(int)
            cx, cy = det.pixel_center.astype(int)

            # Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 50), 2)

            # Label
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(vis, label,
                        (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 50), 2)

            # Pixel centre cross
            cv2.drawMarker(vis, (cx, cy), (0, 255, 255),
                           cv2.MARKER_CROSS, 16, 2)

            # World coordinates overlay
            if world_coords and i < len(world_coords) and world_coords[i] is not None:
                wc = world_coords[i]
                coord_str = f"W({wc[0]:.3f}, {wc[1]:.3f}, {wc[2]:.3f})"
                cv2.putText(vis, coord_str,
                            (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 0), 1)

        return vis

    @property
    def ready(self) -> bool:
        return self._model is not None
