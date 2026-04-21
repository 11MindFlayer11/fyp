"""
tests/test_pipeline.py

Unit tests for core pipeline components.
Run with:  pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestWorkspaceBounds:
    def test_inside(self):
        from utils.helpers import is_in_workspace
        assert is_in_workspace(np.array([1.0, 0.75, 0.025]))

    def test_outside_x(self):
        from utils.helpers import is_in_workspace
        assert not is_in_workspace(np.array([-1.0, 0.5, 0.0]))

    def test_outside_y(self):
        from utils.helpers import is_in_workspace
        assert not is_in_workspace(np.array([1.0, 5.0, 0.0]))

    def test_boundary(self):
        from utils.helpers import is_in_workspace
        from config.settings import WORKSPACE_X_MIN, WORKSPACE_Y_MIN
        assert is_in_workspace(np.array([WORKSPACE_X_MIN, WORKSPACE_Y_MIN, 0.0]))

    def test_none_returns_false(self):
        from utils.helpers import is_in_workspace
        assert not is_in_workspace(None)


class TestClampWorkspace:
    def test_clamp_high_x(self):
        from utils.helpers import clamp_to_workspace
        from config.settings import WORKSPACE_X_MAX
        result = clamp_to_workspace(np.array([99.0, 0.5, 0.0]))
        assert result[0] == pytest.approx(WORKSPACE_X_MAX)

    def test_no_clamp_needed(self):
        from utils.helpers import clamp_to_workspace
        pt = np.array([1.0, 0.5, 0.02])
        result = clamp_to_workspace(pt)
        np.testing.assert_array_almost_equal(result, pt)


# ──────────────────────────────────────────────────────────────────────────────
#  FPS Counter
# ──────────────────────────────────────────────────────────────────────────────

class TestFPSCounter:
    def test_zero_before_ticks(self):
        from utils.helpers import FPSCounter
        f = FPSCounter()
        assert f.fps == 0.0

    def test_approx_fps(self):
        import time
        from utils.helpers import FPSCounter
        f = FPSCounter(window=10)
        for _ in range(10):
            f.tick()
            time.sleep(0.01)
        assert 80 < f.fps < 120   # roughly 100 fps with 10ms sleeps


# ──────────────────────────────────────────────────────────────────────────────
#  IoU helper
# ──────────────────────────────────────────────────────────────────────────────

class TestBboxIoU:
    def test_perfect_overlap(self):
        from tracking.pipe_tracker import _bbox_iou
        a = np.array([0, 0, 10, 10], dtype=float)
        assert _bbox_iou(a, a) == pytest.approx(1.0)

    def test_no_overlap(self):
        from tracking.pipe_tracker import _bbox_iou
        a = np.array([0, 0, 5, 5], dtype=float)
        b = np.array([10, 10, 20, 20], dtype=float)
        assert _bbox_iou(a, b) == pytest.approx(0.0)

    def test_half_overlap(self):
        from tracking.pipe_tracker import _bbox_iou
        a = np.array([0, 0, 4, 4], dtype=float)
        b = np.array([2, 0, 6, 4], dtype=float)
        iou = _bbox_iou(a, b)
        # intersection = 2×4=8, union = 16+16-8=24  → iou = 1/3
        assert iou == pytest.approx(1/3, rel=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
#  Tracker basics
# ──────────────────────────────────────────────────────────────────────────────

class _FakeDet:
    """Minimal detection object for tracker tests."""
    def __init__(self, x1, y1, x2, y2, conf=0.9):
        self.bbox         = np.array([x1, y1, x2, y2], dtype=float)
        self.confidence   = conf
        self.pixel_center = np.array([(x1+x2)/2, (y1+y2)/2])


class TestPipeTracker:
    def test_single_pipe_assigned_id(self):
        from tracking.pipe_tracker import PipeTracker
        tracker = PipeTracker()
        det = _FakeDet(100, 100, 200, 200)
        tracks = tracker.update([det], [np.array([1.0, 0.5, 0.0])])
        assert len(tracks) == 1

    def test_same_pipe_same_id(self):
        from tracking.pipe_tracker import PipeTracker
        tracker = PipeTracker()
        det1 = _FakeDet(100, 100, 200, 200)
        t1 = tracker.update([det1], [np.array([1.0, 0.5, 0.0])])
        tid = list(t1.keys())[0]

        det2 = _FakeDet(102, 102, 202, 202)   # slight movement
        t2 = tracker.update([det2], [np.array([1.01, 0.51, 0.0])])
        assert tid in t2

    def test_empty_frame_ages_out(self):
        from tracking.pipe_tracker import PipeTracker
        tracker = PipeTracker()
        det = _FakeDet(100, 100, 200, 200)
        tracker.update([det], [np.array([1.0, 0.5, 0.0])])
        # Send many empty frames
        for _ in range(PipeTracker.MAX_MISSING_FRAMES + 2):
            tracks = tracker.update([], [])
        assert len(tracks) == 0

    def test_smoothed_world_uses_z_offset(self):
        from tracking.pipe_tracker import PipeTracker
        from config.settings import PIPE_RADIUS_M
        tracker = PipeTracker()
        det = _FakeDet(100, 100, 200, 200)
        wc = np.array([1.0, 0.5, 0.0])
        for _ in range(6):
            tracks = tracker.update([det], [wc])
        track = list(tracks.values())[0]
        sw = track.smoothed_world
        assert sw is not None
        assert sw[2] == pytest.approx(PIPE_RADIUS_M, rel=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
#  ArUco pose estimator (geometry only, no camera needed)
# ──────────────────────────────────────────────────────────────────────────────

class TestArucoPoseGeometry:
    """
    Test pixel_to_world_floor with a synthetic, known camera pose.
    Camera placed directly above origin at height H, looking straight down.
    """

    HEIGHT = 2.0   # metres above floor

    def _make_estimator(self):
        from localization.aruco_pose import ArucoPoseEstimator

        H = self.HEIGHT
        f = 800.0   # focal length in pixels

        K = np.array([[f, 0, 320],
                      [0, f, 240],
                      [0, 0,   1]], dtype=np.float64)
        dist = np.zeros((1, 5))

        est = ArucoPoseEstimator(K, dist)

        # Camera looks straight down: Rcam_to_world flips Y and Z
        # world_to_cam: rotation that maps world Z upwards into camera -Z
        R = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1],
        ], dtype=np.float64)
        t = np.array([[0], [0], [H]], dtype=np.float64)   # cam at (0,0,H)

        est.R_world_to_cam = R
        est.t_world_to_cam = t
        est.R_cam_to_world = R.T
        est.t_cam_to_world = -R.T @ t
        return est, K, f

    def test_principal_point_maps_to_origin(self):
        est, K, f = self._make_estimator()
        # Principal point in image → world (0, 0, 0)
        pt = est.pixel_to_world_floor(320.0, 240.0)
        assert pt is not None
        np.testing.assert_allclose(pt[:2], [0.0, 0.0], atol=1e-6)

    def test_offset_pixel_correct_scale(self):
        est, K, f = self._make_estimator()
        H = self.HEIGHT
        # A pixel 80px to the right of principal point:
        # expected world X = H * (80 / f)
        expected_x = H * (80.0 / f)
        pt = est.pixel_to_world_floor(320.0 + 80.0, 240.0)
        assert pt is not None
        assert abs(pt[0] - expected_x) < 0.01


# ──────────────────────────────────────────────────────────────────────────────
#  Controller (dry-run, no hardware)
# ──────────────────────────────────────────────────────────────────────────────

class TestCraneController:
    def test_dry_run_send_returns_true(self):
        from control.crane_controller import CraneController
        ctrl = CraneController(enabled=False)
        ctrl.open()
        result = ctrl.send_pickup_command(np.array([1.0, 0.5, 0.025]))
        assert result is True
        assert ctrl.command_count == 1
        ctrl.close()

    def test_context_manager(self):
        from control.crane_controller import CraneController
        with CraneController(enabled=False) as ctrl:
            assert ctrl.send_pickup_command(np.array([0.5, 0.5, 0.02]))
