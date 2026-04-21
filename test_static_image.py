"""
test_static_image.py — Test 3D Visualization on Static Image
=============================================================
Loads a static image, runs ArUco pose estimation, detects pipes (e.g., pen),
projects to world coordinates, and visualizes the results.

Usage:
    python test_static_image.py path/to/image.png [--output output.png] [--no-display]
"""

import argparse
import cv2
import numpy as np
import sys

# Project imports
from calibration.calibrate_camera import load_calibration
from config.settings import ARUCO_MARKER_IDS, WORKSPACE_X_MIN, WORKSPACE_X_MAX, WORKSPACE_Y_MIN, WORKSPACE_Y_MAX
from detection.pipe_detector import PipeDetector
from localization.aruco_pose import ArucoPoseEstimator
from utils.helpers import draw_workspace_overlay, draw_hud, is_in_workspace

def main():
    parser = argparse.ArgumentParser(description="Test 3D visualization on static image")
    parser.add_argument("image_path", help="Path to the static image file")
    parser.add_argument("--output", help="Path to save annotated image (optional)")
    parser.add_argument("--no-display", action="store_true", help="Don't display window")
    args = parser.parse_args()

    # Load calibration
    print("Loading camera calibration...")
    try:
        K, dist = load_calibration()
    except Exception as e:
        print(f"Error loading calibration: {e}")
        sys.exit(1)

    # Load image
    print(f"Loading image: {args.image_path}")
    frame = cv2.imread(args.image_path)
    if frame is None:
        print(f"Error: Cannot load image {args.image_path}")
        sys.exit(1)

    # Undistort
    frame_ud = cv2.undistort(frame, K, dist)

    # Initialize pose estimator
    pose_estimator = ArucoPoseEstimator(K, dist)

    # Detect and visualize marker centers
    print("Detecting ArUco marker centers...")
    img_centers, world_centers, marker_ids = pose_estimator.detect_marker_centers(frame_ud)
    print(f"Detected {len(marker_ids)} markers: {marker_ids}")
    if len(world_centers) > 0:
        print(f"World marker positions:\n{world_centers}")

    # Fit plane to marker centers
    if len(world_centers) >= 3:
        plane_normal, plane_dist = pose_estimator.fit_plane_from_points(world_centers)
        print(f"Fitted plane: normal={plane_normal}, distance={plane_dist}")

    # Process frame for pose using plane method
    print("Processing ArUco markers for pose estimation (plane method)...")
    pose_valid = pose_estimator.process_frame_with_plane(frame_ud)
    print(f"Pose valid: {pose_valid}")
    if pose_valid:
        cam_pos = pose_estimator.camera_position_world()
        print(f"Camera position (world): {cam_pos}")

    # Draw markers and centers with rectangle
    annotated = pose_estimator.draw_marker_centers_and_rectangle(frame_ud.copy())
    annotated = pose_estimator.draw_markers(annotated)

    # Draw workspace overlay
    if pose_valid:
        annotated = draw_workspace_overlay(annotated, pose_estimator, K, dist)

    # Pipe detection (for pen)
    print("Detecting pipes...")
    pipe_detector = PipeDetector()
    detections = pipe_detector.detect(frame_ud) if pipe_detector.ready else []
    print(f"Detected {len(detections)} pipes")

    # Convert to world coordinates
    world_coords = []
    for det in detections:
        if pose_valid:
            cx, cy = det.pixel_center
            wc = pose_estimator.pixel_to_world_floor(cx, cy)
            if wc is not None and is_in_workspace(wc):
                world_coords.append(wc)
                print(f"Pipe at pixel ({cx:.1f}, {cy:.1f}) -> world {wc}")
                # Draw world coords on image
                cv2.putText(annotated, f"World: {wc[0]:.2f}, {wc[1]:.2f}",
                           (int(cx), int(cy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            else:
                world_coords.append(None)
        else:
            world_coords.append(None)

    # Draw HUD
    fps = 0  # Static image, no FPS
    target_xyz = world_coords[0] if world_coords and world_coords[0] is not None else None
    command_count = 0  # No controller in test
    annotated = draw_hud(annotated, fps, pose_valid, len(detections), target_xyz, command_count)

    # Save or display
    if args.output:
        cv2.imwrite(args.output, annotated)
        print(f"Saved annotated image to {args.output}")

    if not args.no_display:
        cv2.imshow("3D Visualization Test", annotated)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Test completed.")

if __name__ == "__main__":
    main()