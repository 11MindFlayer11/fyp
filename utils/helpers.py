"""
utils/helpers.py

Shared utility functions used across the pipeline.
"""

import logging
import os
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from config.settings import (
    WORKSPACE_X_MIN, WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN, WORKSPACE_Y_MAX,
    WORKSPACE_Z_MIN, WORKSPACE_Z_MAX,
    LOG_DIR, LOG_LEVEL,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(name: str = "crane_vision") -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(name)
    log.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    # File handler
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3
    ) if _has_rotating_handler() else logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


def _has_rotating_handler():
    try:
        from logging import handlers   # noqa
        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────────────────────────────────────
#  Workspace validation
# ──────────────────────────────────────────────────────────────────────────────

def is_in_workspace(xyz: np.ndarray) -> bool:
    """Return True if world coordinate falls inside the configured bounds."""
    if xyz is None or len(xyz) < 3:
        return False
    return (WORKSPACE_X_MIN <= xyz[0] <= WORKSPACE_X_MAX and
            WORKSPACE_Y_MIN <= xyz[1] <= WORKSPACE_Y_MAX and
            WORKSPACE_Z_MIN <= xyz[2] <= WORKSPACE_Z_MAX)


def clamp_to_workspace(xyz: np.ndarray) -> np.ndarray:
    """Clamp coordinate to workspace bounds (does not modify in-place)."""
    out = xyz.copy()
    out[0] = float(np.clip(out[0], WORKSPACE_X_MIN, WORKSPACE_X_MAX))
    out[1] = float(np.clip(out[1], WORKSPACE_Y_MIN, WORKSPACE_Y_MAX))
    out[2] = float(np.clip(out[2], WORKSPACE_Z_MIN, WORKSPACE_Z_MAX))
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  FPS counter
# ──────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    """Rolling-window FPS estimator."""

    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)

    def tick(self):
        self._times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray,
             fps: float,
             pose_valid: bool,
             num_pipes: int,
             target_xyz: Optional[np.ndarray],
             command_count: int) -> np.ndarray:
    """Overlay a heads-up display on the frame. Returns modified copy."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Semi-transparent top bar
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    pose_col = (0, 220, 0) if pose_valid else (0, 60, 220)
    pose_str = "POSE OK" if pose_valid else "NO POSE"

    cv2.putText(vis, f"FPS: {fps:5.1f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(vis, pose_str, (120, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, pose_col, 2)
    cv2.putText(vis, f"Pipes: {num_pipes}", (240, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(vis, f"Cmds: {command_count}", (350, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    if target_xyz is not None:
        tgt_str = (f"TARGET → X:{target_xyz[0]:+.3f}  "
                   f"Y:{target_xyz[1]:+.3f}  "
                   f"Z:{target_xyz[2]:+.3f} m")
        cv2.putText(vis, tgt_str, (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 220), 1)

    return vis


def draw_workspace_overlay(frame: np.ndarray,
                            pose_estimator,
                            K: np.ndarray,
                            dist: np.ndarray) -> np.ndarray:
    """
    Project the workspace boundary rectangle onto the image plane
    so operators can see the crane's working area.
    """
    if not pose_estimator.is_valid():
        return frame

    corners_world = np.array([
        [WORKSPACE_X_MIN, WORKSPACE_Y_MIN, 0],
        [WORKSPACE_X_MAX, WORKSPACE_Y_MIN, 0],
        [WORKSPACE_X_MAX, WORKSPACE_Y_MAX, 0],
        [WORKSPACE_X_MIN, WORKSPACE_Y_MAX, 0],
    ], dtype=np.float64)

    rvec, _ = cv2.Rodrigues(pose_estimator.R_world_to_cam)
    tvec    = pose_estimator.t_world_to_cam

    img_pts, _ = cv2.projectPoints(corners_world, rvec, tvec, K, dist)
    img_pts    = img_pts.reshape(-1, 2).astype(int)

    vis = frame.copy()
    for i in range(4):
        pt1 = tuple(img_pts[i])
        pt2 = tuple(img_pts[(i + 1) % 4])
        cv2.line(vis, pt1, pt2, (180, 100, 0), 2)

    cv2.putText(vis, "WORKSPACE", tuple(img_pts[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 100, 0), 1)
    return vis


# ──────────────────────────────────────────────────────────────────────────────
#  ArUco board generation (utility, not runtime)
# ──────────────────────────────────────────────────────────────────────────────

def generate_aruco_marker(marker_id: int, size_px: int = 400,
                           save_path: Optional[str] = None) -> np.ndarray:
    """
    Generate and optionally save a single ArUco marker image.
    Useful for printing floor markers.
    """
    from config.settings import ARUCO_DICT
    _DICT_MAP = {
        "DICT_4X4_50":   cv2.aruco.DICT_4X4_50,
        "DICT_5X5_250":  cv2.aruco.DICT_5X5_250,
        "DICT_6X6_250":  cv2.aruco.DICT_6X6_250,
        "DICT_ARUCO_ORIG": cv2.aruco.DICT_ARUCO_ORIGINAL,
    }
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        _DICT_MAP.get(ARUCO_DICT, cv2.aruco.DICT_5X5_250)
    )
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
    if save_path:
        cv2.imwrite(save_path, marker_img)
    return marker_img
