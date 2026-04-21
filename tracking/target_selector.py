"""
tracking/target_selector.py

Decides WHICH tracked pipe to pick up next.

Strategies:
  "closest"        → closest stable pipe to crane home position
  "first_detected" → lowest track_id (oldest)
  "manual"         → no automatic selection; caller provides target_id
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from config.settings import (
    TARGET_SELECTION_STRATEGY, CRANE_HOME_POSITION
)
from tracking.pipe_tracker import TrackedPipe

log = logging.getLogger(__name__)


class TargetSelector:
    """Selects the next pickup target from live tracks."""

    def __init__(self, strategy: str = TARGET_SELECTION_STRATEGY):
        self.strategy  = strategy
        self._override_id: Optional[int] = None   # for "manual" mode

    # ──────────────────────────────────────────────────────────────────────────

    def select(self, tracks: Dict[int, TrackedPipe]) -> Optional[TrackedPipe]:
        """
        Choose a target pipe. Only considers stable pipes with valid world coords.
        Returns a TrackedPipe or None if no suitable candidate exists.
        """
        candidates = [
            t for t in tracks.values()
            if t.is_stable
            and t.smoothed_world is not None
            and t.is_in_workspace()
        ]

        if not candidates:
            return None

        if self.strategy == "closest":
            return self._closest(candidates)

        elif self.strategy == "first_detected":
            return min(candidates, key=lambda t: t.track_id)

        elif self.strategy == "manual":
            if self._override_id is not None:
                matching = [t for t in candidates if t.track_id == self._override_id]
                return matching[0] if matching else None
            return None

        else:
            log.warning(f"Unknown strategy '{self.strategy}', defaulting to 'closest'")
            return self._closest(candidates)

    def set_manual_target(self, track_id: int):
        """Set a specific track_id for manual selection mode."""
        self._override_id = track_id
        self.strategy = "manual"

    def _closest(self, candidates: List[TrackedPipe]) -> TrackedPipe:
        return min(candidates,
                   key=lambda t: np.linalg.norm(
                       t.smoothed_world[:2] - CRANE_HOME_POSITION[:2]
                   ))
