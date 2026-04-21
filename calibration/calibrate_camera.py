"""
calibration/calibrate_camera.py

Offline checkerboard camera calibration.
Run this ONCE before deploying the system.

Usage:
    python -m calibration.calibrate_camera --collect   # collect frames interactively
    python -m calibration.calibrate_camera --calibrate # compute and save intrinsics
    python -m calibration.calibrate_camera --full      # collect + calibrate in one step
"""

import argparse
import glob
import logging
import os
import sys
import time

import cv2
import numpy as np

# Allow running as a script from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    CALIBRATION_FILE, CALIBRATION_IMAGES_DIR,
    CHECKERBOARD_SIZE, CHECKERBOARD_SQUARE_MM,
    MIN_CALIBRATION_IMAGES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def collect_calibration_images(save_dir: str = CALIBRATION_IMAGES_DIR,
                                target_count: int = 30) -> int:
    """
    Open camera and let the user capture frames with SPACE.
    Saves frames to `save_dir`. Returns number of saved images.
    """
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    saved = 0
    log.info("=== CALIBRATION IMAGE COLLECTION ===")
    log.info(f"Target: {target_count} frames | Checkerboard: {CHECKERBOARD_SIZE}")
    log.info("Controls:  SPACE = capture  |  Q = quit")

    while saved < target_count:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, CHECKERBOARD_SIZE, corners, found)
            cv2.putText(display, "Board found! Press SPACE to capture",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        else:
            cv2.putText(display, "No board detected",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(display, f"Captured: {saved}/{target_count}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Calibration Image Collector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and found:
            fname = os.path.join(save_dir, f"calib_{saved:03d}.png")
            cv2.imwrite(fname, frame)
            saved += 1
            log.info(f"Saved {fname}")
            time.sleep(0.3)  # brief pause to avoid duplicates
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    log.info(f"Collection complete. {saved} images saved to '{save_dir}'")
    return saved


def compute_calibration(images_dir: str = CALIBRATION_IMAGES_DIR,
                         output_file: str = CALIBRATION_FILE) -> dict:
    """
    Compute camera matrix and distortion coefficients from saved checkerboard images.
    Saves results to `output_file` (.npz). Returns calibration dict.
    """
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")) +
                         glob.glob(os.path.join(images_dir, "*.jpeg")))
    if len(image_paths) < MIN_CALIBRATION_IMAGES:
        raise ValueError(
            f"Need at least {MIN_CALIBRATION_IMAGES} calibration images, "
            f"found {len(image_paths)} in '{images_dir}'"
        )

    cols, rows = CHECKERBOARD_SIZE
    sq = CHECKERBOARD_SQUARE_MM / 1000.0   # convert mm → metres

    # Prepare 3-D object points for one checkerboard
    obj_pts_template = np.zeros((rows * cols, 3), np.float32)
    obj_pts_template[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq

    obj_points = []   # 3-D points in world space
    img_points = []   # 2-D points in image space
    img_shape  = None

    good = 0
    for path in image_paths:
        img  = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]   # (width, height)

        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)
        if not found:
            log.debug(f"Board not found in {path}")
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        obj_points.append(obj_pts_template)
        img_points.append(corners_refined)
        good += 1
        log.info(f"  ✓ {os.path.basename(path)}")

    if good < MIN_CALIBRATION_IMAGES:
        raise ValueError(
            f"Only {good} valid boards detected; need {MIN_CALIBRATION_IMAGES}."
        )

    log.info(f"Running calibration on {good} valid images …")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None
    )

    log.info(f"Reprojection RMS error: {rms:.4f} px  (target < 1.0)")
    log.info(f"Camera matrix K:\n{K}")
    log.info(f"Distortion coefficients: {dist.ravel()}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    np.savez(output_file,
             camera_matrix=K,
             dist_coeffs=dist,
             rms_error=rms,
             image_size=np.array(img_shape))
    log.info(f"Calibration saved to '{output_file}'")
    return {"camera_matrix": K, "dist_coeffs": dist, "rms": rms}


def load_calibration(calibration_file: str = CALIBRATION_FILE) -> tuple:
    """
    Load camera matrix and distortion coefficients from .npz file.
    Returns (camera_matrix, dist_coeffs).
    """
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(
            f"Calibration file '{calibration_file}' not found. "
            "Run: python -m calibration.calibrate_camera --full"
        )
    data = np.load(calibration_file)
    K    = data["camera_matrix"]
    dist = data["dist_coeffs"]
    log.info(f"Loaded calibration from '{calibration_file}'  "
             f"(RMS={float(data.get('rms_error', -1)):.4f})")
    return K, dist


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect",   action="store_true",
                       help="Interactively collect calibration images")
    group.add_argument("--calibrate", action="store_true",
                       help="Compute calibration from existing images")
    group.add_argument("--full",      action="store_true",
                       help="Collect images then calibrate")
    parser.add_argument("--images-dir", default=CALIBRATION_IMAGES_DIR)
    parser.add_argument("--output",     default=CALIBRATION_FILE)
    parser.add_argument("--count",      type=int, default=30)
    args = parser.parse_args()

    if args.collect or args.full:
        collect_calibration_images(args.images_dir, args.count)
    if args.calibrate or args.full:
        compute_calibration(args.images_dir, args.output)
