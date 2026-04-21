"""
main.py — Vision-Based Roof Gantry Crane System
================================================
Real-time pipeline:
  Frame → Undistort → ArUco Pose → YOLO Detect → Pixel→World → Track → Select → Command

Usage:
    python main.py                        # normal operation
    python main.py --no-display           # headless / server mode
    python main.py --record output.mp4    # save annotated video
    python main.py --dry-run              # force serial off (no hardware)
    python main.py --generate-markers     # print ArUco markers to ./output/markers/
"""

import argparse
import logging
import os
import signal
import sys
import time

import cv2
import numpy as np

# ── Project imports ────────────────────────────────────────────────────────────
from calibration.calibrate_camera import load_calibration
from config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, CAMERA_FPS,
    DISPLAY_WINDOW, SAVE_ANNOTATED_VIDEO, ANNOTATED_VIDEO_PATH,
    PRINT_COORDS, ARUCO_MARKER_IDS, OUTPUT_DIR,
)
from control.crane_controller import CraneController
from detection.pipe_detector import PipeDetector
from localization.aruco_pose import ArucoPoseEstimator
from tracking.pipe_tracker import PipeTracker
from tracking.target_selector import TargetSelector
from utils.helpers import (
    FPSCounter, draw_hud, draw_workspace_overlay, generate_aruco_marker,
    is_in_workspace,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("crane_vision.main")

# ── Graceful shutdown ──────────────────────────────────────────────────────────
_RUNNING = True

def _handle_signal(sig, frame):
    global _RUNNING
    log.info("Shutdown signal received — stopping…")
    _RUNNING = False

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run(args):
    global _RUNNING

    # ── Output directory ───────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Calibration ───────────────────────────────────────────────────────────
    log.info("Loading camera calibration…")
    K, dist = load_calibration()

    # ── Subsystem init ────────────────────────────────────────────────────────
    pose_estimator = ArucoPoseEstimator(K, dist)
    pipe_detector  = PipeDetector()
    tracker        = PipeTracker()
    selector       = TargetSelector()
    fps_counter    = FPSCounter(window=30)

    # ── Camera ────────────────────────────────────────────────────────────────
    log.info(f"Opening camera {CAMERA_INDEX} at {FRAME_WIDTH}×{FRAME_HEIGHT}")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    if not cap.isOpened():
        log.error(f"Cannot open camera index {CAMERA_INDEX}")
        sys.exit(1)

    # ── Optional video writer ─────────────────────────────────────────────────
    video_writer = None
    save_video = args.record or (SAVE_ANNOTATED_VIDEO and not args.no_display)
    if save_video:
        vpath = args.record if args.record else ANNOTATED_VIDEO_PATH
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(vpath, fourcc, CAMERA_FPS,
                                       (FRAME_WIDTH, FRAME_HEIGHT))
        log.info(f"Recording to '{vpath}'")

    # ── Serial controller ─────────────────────────────────────────────────────
    serial_enabled = not args.dry_run
    controller = CraneController(enabled=serial_enabled)
    controller.open()

    display = DISPLAY_WINDOW and not args.no_display

    # ── MAIN LOOP ─────────────────────────────────────────────────────────────
    log.info("Pipeline running. Press Q in the window (or Ctrl-C) to stop.")
    frame_idx = 0

    while _RUNNING:
        ret, frame = cap.read()
        if not ret:
            log.warning("Failed to grab frame — retrying…")
            time.sleep(0.05)
            continue

        fps_counter.tick()
        frame_idx += 1

        # ── 1. Undistort ───────────────────────────────────────────────────────
        frame_ud = cv2.undistort(frame, K, dist)

        # ── 2. ArUco pose estimation ───────────────────────────────────────────
        pose_valid = pose_estimator.process_frame(frame_ud)

        # ── 3. YOLO pipe detection ─────────────────────────────────────────────
        detections = pipe_detector.detect(frame_ud) if pipe_detector.ready else []

        # ── 4. Pixel → World coordinates ──────────────────────────────────────
        world_coords = []
        for det in detections:
            if pose_valid:
                cx, cy = det.pixel_center
                wc = pose_estimator.pixel_to_world_floor(cx, cy)
                if wc is not None and is_in_workspace(wc):
                    world_coords.append(wc)
                else:
                    world_coords.append(None)
            else:
                world_coords.append(None)

        # ── 5. Track ───────────────────────────────────────────────────────────
        tracks = tracker.update(detections, world_coords)

        # ── 6. Select target ──────────────────────────────────────────────────
        target = selector.select(tracks)
        target_xyz = target.smoothed_world if target else None

        # ── 7. Send crane command (stable target only) ────────────────────────
        if target is not None and target_xyz is not None and not controller.is_busy:
            if PRINT_COORDS:
                log.info(
                    f"TARGET id={target.track_id:3d} | "
                    f"X={target_xyz[0]:+.4f}  Y={target_xyz[1]:+.4f}  Z={target_xyz[2]:+.4f} m"
                )
            controller.send_pickup_command(target_xyz)

        # ── 8. Visualise ───────────────────────────────────────────────────────
        if display or save_video:
            vis = pose_estimator.draw_markers(frame_ud)
            vis = draw_workspace_overlay(vis, pose_estimator, K, dist)

            # Draw all live tracks
            for track in tracker.all_tracks():
                sw = track.smoothed_world
                vis = pipe_detector.draw_detections(
                    vis,
                    [type("D", (), {"bbox": track.bbox,
                                    "confidence": track.confidence,
                                    "class_name": f"pipe#{track.track_id}",
                                    "pixel_center": track.smoothed_pixel if track.smoothed_pixel is not None else np.array([0.0, 0.0])})()],
                    [sw],
                )
                # Stability indicator
                if track.is_stable:
                    sp = track.smoothed_pixel
                    if sp is not None:
                        cv2.circle(vis, tuple(sp.astype(int)), 20, (0, 255, 0), 3)

            # Target highlight
            if target is not None and target.smoothed_pixel is not None:
                sp = target.smoothed_pixel.astype(int)
                cv2.drawMarker(vis, tuple(sp), (0, 0, 255),
                               cv2.MARKER_STAR, 30, 3)
                cv2.putText(vis, "PICKUP TARGET", (sp[0] + 12, sp[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            vis = draw_hud(vis, fps_counter.fps, pose_valid,
                           len(tracker.all_tracks()), target_xyz,
                           controller.command_count)

            if display:
                cv2.imshow("Crane Vision System", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    _RUNNING = False
                elif key == ord('r'):
                    tracker.reset()
                    log.info("Tracker reset by user")

            if save_video and video_writer:
                video_writer.write(vis)

    # ── Cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    controller.close()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    log.info(f"Shutdown complete. Processed {frame_idx} frames.")


# ──────────────────────────────────────────────────────────────────────────────
#  MARKER GENERATION UTILITY
# ──────────────────────────────────────────────────────────────────────────────

import os

def generate_markers(output_dir: str = "output/markers"):
    os.makedirs(output_dir, exist_ok=True)

    # === SETTINGS ===
    dpi = 300
    marker_size_cm = 12

    # convert cm → pixels
    marker_size_px = int((marker_size_cm / 2.54) * dpi)

    # 20% white border
    border_size = int(0.2 * marker_size_px)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

    for mid in ARUCO_MARKER_IDS:
        path = os.path.join(output_dir, f"marker_{mid:02d}.png")

        # generate marker
        marker = cv2.aruco.generateImageMarker(
            aruco_dict,
            mid,
            marker_size_px
        )

        # create white canvas
        full_size = marker_size_px + 2 * border_size
        canvas = 255 * np.ones((full_size, full_size), dtype=np.uint8)

        # place marker in center
        canvas[
            border_size:border_size + marker_size_px,
            border_size:border_size + marker_size_px
        ] = marker

        cv2.imwrite(path, canvas)
        log.info(f"  Saved marker {mid} → {path}")

    log.info(f"Generated {len(ARUCO_MARKER_IDS)} markers ({marker_size_cm} cm) in '{output_dir}'")

# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vision-Based Roof Gantry Crane — Real-Time Pipeline"
    )
    parser.add_argument("--no-display",       action="store_true",
                        help="Run headless (no OpenCV window)")
    parser.add_argument("--dry-run",          action="store_true",
                        help="Disable serial output (safe hardware-free mode)")
    parser.add_argument("--record",           metavar="FILE",
                        help="Save annotated video to FILE")
    parser.add_argument("--generate-markers", action="store_true",
                        help="Print ArUco marker images to output/markers/ and exit")
    args = parser.parse_args()

    if args.generate_markers:
        generate_markers()
        sys.exit(0)

    run(args)
