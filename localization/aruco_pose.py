"""
localization/aruco_pose.py

ArUco-marker-based camera pose estimation.

Responsibilities:
  1. Detect ArUco markers in each frame.
  2. Estimate camera pose (R, t) in world coordinates using the known
     floor positions of the markers (PnP).
  3. Expose helpers to project pixel → world coordinates on the floor plane.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    ARUCO_DICT, ARUCO_MARKER_SIZE_M, ARUCO_WORLD_POSITIONS,
    MIN_MARKERS_FOR_POSE,
)

log = logging.getLogger(__name__)

# ── Build ArUco detector once ──────────────────────────────────────────────────

_DICT_MAP = {
    "DICT_4X4_50":     cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":    cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250":    cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50":     cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100":    cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250":    cv2.aruco.DICT_5X5_250,
    "DICT_6X6_250":    cv2.aruco.DICT_6X6_250,
    "DICT_ARUCO_ORIG": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

def _build_detector() -> cv2.aruco.ArucoDetector:
    dict_id = _DICT_MAP.get(ARUCO_DICT)
    if dict_id is None:
        raise ValueError(f"Unknown ArUco dict: '{ARUCO_DICT}'")
    aruco_dict   = cv2.aruco.getPredefinedDictionary(dict_id)
    aruco_params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

_DETECTOR = _build_detector()


class ArucoPoseEstimator:
    """
    Detects ArUco markers and estimates the camera pose each frame using solvePnP.

    The floor plane is Z = 0 in world coordinates.
    """

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self.K    = camera_matrix   # (3,3)
        self.dist = dist_coeffs     # (1,5) or (5,1) etc.

        # Half side length used to compute 4 corner positions of each marker
        self._hs = ARUCO_MARKER_SIZE_M / 2.0

        # Latest valid pose
        self.R_world_to_cam: Optional[np.ndarray] = None   # (3,3)
        self.t_world_to_cam: Optional[np.ndarray] = None   # (3,1)
        self.R_cam_to_world: Optional[np.ndarray] = None
        self.t_cam_to_world: Optional[np.ndarray] = None
        self.last_marker_ids: List[int] = []

    # ──────────────────────────────────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray) -> bool:
        """
        Detect markers and update pose.
        Returns True if a valid pose was obtained.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = _DETECTOR.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            log.debug("No ArUco markers detected")
            return False

        ids = ids.flatten().tolist()
        self.last_marker_ids = ids

        # ── Collect world ↔ image point correspondences ───────────────────────
        world_pts = []
        image_pts = []

        for marker_id, corners in zip(ids, corners_list):
            if marker_id not in ARUCO_WORLD_POSITIONS:
                continue  # Unknown marker
            centre_world = ARUCO_WORLD_POSITIONS[marker_id]
            hs = self._hs

            # 4 corners of the marker in world frame (Z=0 floor)
            # Convention: top-left, top-right, bottom-right, bottom-left
            marker_world_corners = np.array([
                centre_world + np.array([-hs,  hs, 0.0]),
                centre_world + np.array([ hs,  hs, 0.0]),
                centre_world + np.array([ hs, -hs, 0.0]),
                centre_world + np.array([-hs, -hs, 0.0]),
            ], dtype=np.float64)

            # Detected image corners (shape: (4, 2))
            img_corners = corners[0].astype(np.float64)

            world_pts.append(marker_world_corners)
            image_pts.append(img_corners)

        if len(world_pts) < MIN_MARKERS_FOR_POSE:
            log.debug(f"Only {len(world_pts)} known markers visible (need {MIN_MARKERS_FOR_POSE})")
            return False

        world_pts = np.vstack(world_pts)   # (N*4, 3)
        image_pts = np.vstack(image_pts)   # (N*4, 2)

        # ── solvePnP ──────────────────────────────────────────────────────────
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            world_pts, image_pts, self.K, self.dist,
            iterationsCount=200,
            reprojectionError=4.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            log.warning("solvePnPRansac failed")
            return False

        # Refine with inliers only
        if inliers is not None and len(inliers) >= 8:
            idx = inliers.flatten()
            cv2.solvePnPRefineLM(
                world_pts[idx], image_pts[idx],
                self.K, self.dist, rvec, tvec
            )

        R, _ = cv2.Rodrigues(rvec)
        self.R_world_to_cam = R
        self.t_world_to_cam = tvec.reshape(3, 1)

        # Inverse: camera → world
        self.R_cam_to_world = R.T
        self.t_cam_to_world = -R.T @ tvec.reshape(3, 1)

        log.debug(f"Pose updated | markers={len(world_pts)//4} | "
                  f"cam_pos={self.camera_position_world()}")
        return True

    # ──────────────────────────────────────────────────────────────────────────

    def pixel_to_world_floor(self, px: float, py: float) -> Optional[np.ndarray]:
        """
        Project a pixel (px, py) onto the Z=0 world floor plane.
        Returns np.array([X, Y, Z]) in world metres, or None if pose unavailable.

        Math:
          p_cam = K_inv @ [px, py, 1]^T
          Ray in world: d = R_cam_to_world @ p_cam
          Origin in world: o = t_cam_to_world
          Intersect with Z=0: t = -o[2] / d[2]
          point = o + t * d
        """
        if self.R_cam_to_world is None:
            return None

        # Undistort the pixel first
        pt_undist = cv2.undistortPoints(
            np.array([[[px, py]]], dtype=np.float64),
            self.K, self.dist, P=self.K
        ).reshape(2)

        # Normalised camera ray
        K_inv = np.linalg.inv(self.K)
        ray_cam = K_inv @ np.array([pt_undist[0], pt_undist[1], 1.0])

        # Transform ray to world frame
        ray_world = self.R_cam_to_world @ ray_cam
        origin    = self.t_cam_to_world.flatten()   # camera centre in world

        # Intersect with Z=0 plane
        if abs(ray_world[2]) < 1e-9:
            log.debug("Ray nearly parallel to floor — cannot intersect")
            return None

        t = -origin[2] / ray_world[2]
        if t < 0:
            log.debug("Intersection behind camera")
            return None

        point = origin + t * ray_world
        return point   # shape (3,)

    # ──────────────────────────────────────────────────────────────────────────

    def camera_position_world(self) -> Optional[np.ndarray]:
        """Return camera centre in world coordinates."""
        if self.t_cam_to_world is None:
            return None
        return self.t_cam_to_world.flatten()

    def is_valid(self) -> bool:
        return self.R_cam_to_world is not None

    # ──────────────────────────────────────────────────────────────────────────

    def draw_markers(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draw detected markers and pose axes onto `frame_bgr` (in-place copy)."""
        vis = frame_bgr.copy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = _DETECTOR.detectMarkers(gray)

        if ids is None:
            return vis

        cv2.aruco.drawDetectedMarkers(vis, corners_list, ids)

        # Draw individual rvec/tvec per marker for visual reference
        for marker_id, corners in zip(ids.flatten(), corners_list):
            if marker_id not in ARUCO_WORLD_POSITIONS:
                continue
            ok, rvec_m, tvec_m = cv2.solvePnP(
                self._single_marker_world_corners(marker_id),
                corners[0].astype(np.float64),
                self.K, self.dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if ok:
                cv2.drawFrameAxes(vis, self.K, self.dist,
                                  rvec_m, tvec_m, ARUCO_MARKER_SIZE_M * 0.5)
        return vis

    def _single_marker_world_corners(self, marker_id: int) -> np.ndarray:
        c = ARUCO_WORLD_POSITIONS[marker_id]
        hs = self._hs
        return np.array([
            c + [-hs,  hs, 0], c + [ hs,  hs, 0],
            c + [ hs, -hs, 0], c + [-hs, -hs, 0],
        ], dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────────────
    # PLANE-BASED POSE ESTIMATION (Alternative method)
    # ──────────────────────────────────────────────────────────────────────────

    def detect_marker_centers(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Detect ArUco marker centers in both image and world coordinates.
        
        Returns:
            (image_centers, world_centers, marker_ids)
            - image_centers: (N, 2) array of pixel centers
            - world_centers: (N, 3) array of world coordinates (on Z=0 floor)
            - marker_ids: list of detected marker IDs
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = _DETECTOR.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3), []

        ids = ids.flatten().tolist()
        image_centers = []
        world_centers = []
        valid_ids = []

        for marker_id, corners in zip(ids, corners_list):
            if marker_id not in ARUCO_WORLD_POSITIONS:
                continue  # Unknown marker

            # Image center: average of 4 corners
            img_center = corners[0].mean(axis=0)
            image_centers.append(img_center)

            # World center: use config position
            world_center = ARUCO_WORLD_POSITIONS[marker_id]
            world_centers.append(world_center)
            valid_ids.append(marker_id)

        return (
            np.array(image_centers, dtype=np.float64),
            np.array(world_centers, dtype=np.float64),
            valid_ids,
        )

    def fit_plane_from_points(self, world_points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Fit a plane to the given world points using least-squares.
        
        For points on Z=0 floor plane (normal = [0, 0, 1]), this should 
        confirm the floor plane equation.
        
        Returns:
            (plane_normal, plane_distance)
            - plane_normal: (3,) normalized normal vector
            - plane_distance: scalar distance from origin
            Returns (None, None) if fitting fails
        """
        if len(world_points) < 3:
            log.debug("Need at least 3 points to fit a plane")
            return None, None

        # Center the points
        centroid = world_points.mean(axis=0)
        centered = world_points - centroid

        # SVD to find plane normal
        _, _, V = np.linalg.svd(centered)
        plane_normal = V[-1]  # Last singular vector (minimum variance)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # Ensure normal points "up" (positive Z)
        if plane_normal[2] < 0:
            plane_normal = -plane_normal

        # Plane distance (d in ax + by + cz + d = 0)
        plane_distance = -np.dot(plane_normal, centroid)

        return plane_normal, plane_distance

    def process_frame_with_plane(self, frame_bgr: np.ndarray) -> bool:
        """
        Alternative pose estimation using marker centers and plane fitting.
        Detects marker centers, fits a plane to them, and computes camera pose.
        
        This is more robust when full marker corners are not visible.
        Returns True if a valid pose was obtained.
        """
        image_centers, world_centers, marker_ids = self.detect_marker_centers(frame_bgr)

        if len(marker_ids) < MIN_MARKERS_FOR_POSE:
            log.debug(f"Only {len(marker_ids)} known markers visible (need {MIN_MARKERS_FOR_POSE})")
            return False

        # Fit plane to world marker centers
        plane_normal, plane_distance = self.fit_plane_from_points(world_centers)
        if plane_normal is None:
            log.warning("Could not fit plane to marker centers")
            return False

        log.debug(f"Fitted plane: normal={plane_normal}, distance={plane_distance}")

        # Use solvePnP with marker centers instead of corners
        success, rvec, tvec = cv2.solvePnP(
            world_centers, image_centers, self.K, self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            log.warning("solvePnP with marker centers failed")
            return False

        # Refine
        cv2.solvePnPRefineLM(world_centers, image_centers, self.K, self.dist, rvec, tvec)

        R, _ = cv2.Rodrigues(rvec)
        self.R_world_to_cam = R
        self.t_world_to_cam = tvec.reshape(3, 1)

        # Inverse: camera → world
        self.R_cam_to_world = R.T
        self.t_cam_to_world = -R.T @ tvec.reshape(3, 1)

        self.last_marker_ids = marker_ids
        log.debug(f"Pose updated (plane method) | markers={len(marker_ids)} | "
                  f"cam_pos={self.camera_position_world()}")
        return True

    def draw_marker_centers_and_rectangle(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Draw detected marker centers and connect them with lines to form a rectangle.
        Also annotate with marker IDs.
        """
        vis = frame_bgr.copy()
        image_centers, _, marker_ids = self.detect_marker_centers(frame_bgr)

        if len(marker_ids) == 0:
            return vis

        # Draw centers as circles
        for (cx, cy), mid in zip(image_centers, marker_ids):
            cv2.circle(vis, (int(cx), int(cy)), 8, (0, 255, 0), -1)
            cv2.putText(vis, f"ID:{mid}", (int(cx) + 10, int(cy)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw rectangle connecting centers (in marker ID order)
        if len(image_centers) >= 2:
            # Sort by marker ID for consistent order: 0, 3, 4, 5
            sorted_pairs = sorted(zip(marker_ids, image_centers), key=lambda x: x[0])
            sorted_centers = np.array([p[1] for p in sorted_pairs], dtype=np.int32)

            # Draw polygon connecting the centers
            if len(sorted_centers) >= 3:
                cv2.polylines(vis, [sorted_centers], isClosed=True, color=(255, 0, 0), thickness=2)

        return vis
