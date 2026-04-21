"""
tracking/pipe_tracker.py

Temporal tracking and coordinate smoothing for detected pipes.

Features:
  - Identity-preserving tracking via IoU-based assignment
  - Moving-average smoothing over a configurable window
  - Stability detection: flags pipes whose position has been consistent
    for N consecutive frames (ready for pickup)
  - Tracks age-out (removes tracks not seen for several frames)
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    SMOOTHING_WINDOW, STABILITY_FRAMES, STABILITY_TOLERANCE_M,
    MAX_TRACKED_PIPES, WORKSPACE_X_MIN, WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN, WORKSPACE_Y_MAX, WORKSPACE_Z_MIN, WORKSPACE_Z_MAX,
    PIPE_RADIUS_M,
)

log = logging.getLogger(__name__)


@dataclass
class TrackedPipe:
    """State maintained for one continuously tracked pipe."""
    track_id:   int
    bbox:       np.ndarray        # latest image bounding box [x1,y1,x2,y2]
    confidence: float

    # History buffers
    world_history:  deque = field(default_factory=lambda: deque(maxlen=SMOOTHING_WINDOW))
    pixel_history:  deque = field(default_factory=lambda: deque(maxlen=SMOOTHING_WINDOW))

    # Stability
    stable_frames:  int = 0
    is_stable:      bool = False

    # Book-keeping
    frames_since_seen: int = 0
    total_frames:      int = 0

    # ── computed properties ────────────────────────────────────────────────────

    @property
    def smoothed_world(self) -> Optional[np.ndarray]:
        """Moving-average world position."""
        if not self.world_history:
            return None
        pts = [p for p in self.world_history if p is not None]
        if not pts:
            return None
        avg = np.mean(pts, axis=0)
        avg[2] += PIPE_RADIUS_M   # Z-offset: grab centre of pipe cross-section
        return avg

    @property
    def smoothed_pixel(self) -> Optional[np.ndarray]:
        if not self.pixel_history:
            return None
        return np.mean(list(self.pixel_history), axis=0)

    def is_in_workspace(self) -> bool:
        pt = self.smoothed_world
        if pt is None:
            return False
        return (WORKSPACE_X_MIN <= pt[0] <= WORKSPACE_X_MAX and
                WORKSPACE_Y_MIN <= pt[1] <= WORKSPACE_Y_MAX and
                WORKSPACE_Z_MIN <= pt[2] <= WORKSPACE_Z_MAX)


class PipeTracker:
    """
    IoU-based multi-object tracker with world-coordinate smoothing.

    Each frame:
      1. Receive raw detections + world coords
      2. Match to existing tracks (Hungarian-lite: greedy IoU)
      3. Update matched tracks, create new ones, age-out lost ones
    """

    _next_id = 0
    MAX_MISSING_FRAMES = 10   # frames before a track is dropped

    def __init__(self):
        self._tracks: Dict[int, TrackedPipe] = {}

    # ──────────────────────────────────────────────────────────────────────────

    def update(self,
               detections,                          # List[PipeDetection]
               world_coords: List[Optional[np.ndarray]]
               ) -> Dict[int, TrackedPipe]:
        """
        Process one frame of detections + world coords.
        Returns the current dict of live tracks {track_id: TrackedPipe}.
        """
        # Age existing tracks
        for t in self._tracks.values():
            t.frames_since_seen += 1

        # ── Match detections to tracks via IoU ────────────────────────────────
        matched_track_ids = set()
        matched_det_ids   = set()

        if self._tracks and detections:
            iou_matrix = self._compute_iou_matrix(
                [d.bbox for d in detections],
                [t.bbox for t in self._tracks.values()]
            )
            track_ids = list(self._tracks.keys())

            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = iou_matrix.max()
                if max_iou < 0.25:   # minimum IoU to consider a match
                    break
                di, ti = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                track_id = track_ids[ti]

                self._update_track(self._tracks[track_id],
                                   detections[di],
                                   world_coords[di])
                matched_track_ids.add(track_id)
                matched_det_ids.add(di)

                # Suppress row and column
                iou_matrix[di, :] = -1
                iou_matrix[:, ti] = -1

        # ── Create new tracks for unmatched detections ────────────────────────
        for di, (det, wc) in enumerate(zip(detections, world_coords)):
            if di in matched_det_ids:
                continue
            if len(self._tracks) >= MAX_TRACKED_PIPES:
                continue
            new_id = PipeTracker._next_id
            PipeTracker._next_id += 1
            track = TrackedPipe(track_id=new_id, bbox=det.bbox.copy(),
                                confidence=det.confidence)
            self._update_track(track, det, wc)
            self._tracks[new_id] = track
            log.debug(f"New track {new_id}")

        # ── Remove stale tracks ───────────────────────────────────────────────
        stale = [tid for tid, t in self._tracks.items()
                 if t.frames_since_seen > self.MAX_MISSING_FRAMES]
        for tid in stale:
            log.debug(f"Dropping track {tid}")
            del self._tracks[tid]

        return dict(self._tracks)

    # ──────────────────────────────────────────────────────────────────────────

    def _update_track(self, track: TrackedPipe, det, world_coord) -> None:
        track.bbox       = det.bbox.copy()
        track.confidence = det.confidence
        track.frames_since_seen = 0
        track.total_frames += 1

        track.pixel_history.append(det.pixel_center.copy())
        track.world_history.append(
            world_coord.copy() if world_coord is not None else None
        )

        # Stability check
        sw = track.smoothed_world
        if sw is not None and len(track.world_history) >= STABILITY_FRAMES:
            recent = [p for p in list(track.world_history)[-STABILITY_FRAMES:]
                      if p is not None]
            if len(recent) == STABILITY_FRAMES:
                spread = np.max(np.std(recent, axis=0)[:2])   # XY std-dev
                if spread < STABILITY_TOLERANCE_M:
                    track.stable_frames += 1
                    if track.stable_frames >= STABILITY_FRAMES:
                        if not track.is_stable:
                            log.info(f"Track {track.track_id} is STABLE at {sw}")
                        track.is_stable = True
                    return
        # Reset stability if moved
        track.stable_frames = 0
        track.is_stable = False

    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_iou_matrix(bboxes_a: List[np.ndarray],
                             bboxes_b: List[np.ndarray]) -> np.ndarray:
        """(|A| x |B|) IoU matrix."""
        matrix = np.zeros((len(bboxes_a), len(bboxes_b)))
        for i, a in enumerate(bboxes_a):
            for j, b in enumerate(bboxes_b):
                matrix[i, j] = _bbox_iou(a, b)
        return matrix

    def stable_tracks(self) -> List[TrackedPipe]:
        return [t for t in self._tracks.values() if t.is_stable]

    def all_tracks(self) -> List[TrackedPipe]:
        return [t for t in self._tracks.values() if t.frames_since_seen == 0]

    def reset(self):
        self._tracks.clear()


# ──────────────────────────────────────────────────────────────────────────────

def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
