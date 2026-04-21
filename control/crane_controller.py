"""
control/crane_controller.py

Sends world-coordinate pickup commands to the crane controller hardware
over a serial port.

Protocol (simple CSV per line):
  OUT  →  "x,y,z\n"   e.g. "1.2340,0.7850,0.0250\n"
  IN   ←  "ACK\n" | "BUSY\n" | "ERR\n"

Thread-safe: uses a lock around the serial port access.
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

from config.settings import (
    SERIAL_PORT, SERIAL_BAUD, SERIAL_TIMEOUT,
    SERIAL_ENABLED, COMMAND_FORMAT,
)

log = logging.getLogger(__name__)


class CraneController:
    """
    Manages serial communication with the crane motor controller.
    Use as a context manager or call .open() / .close() manually.
    """

    RESPONSE_TIMEOUT = 5.0   # seconds to wait for ACK

    def __init__(self, port: str = SERIAL_PORT,
                 baud: int = SERIAL_BAUD,
                 enabled: bool = SERIAL_ENABLED):
        self._port    = port
        self._baud    = baud
        self._enabled = enabled
        self._serial  = None
        self._lock    = threading.Lock()
        self._busy    = False
        self._last_command: Optional[np.ndarray] = None
        self._command_count = 0

    # ──────────────────────────────────────────────────────────────────────────

    def open(self) -> bool:
        if not self._enabled:
            log.info("Serial disabled (SERIAL_ENABLED=False) — running in dry-run mode")
            return True
        try:
            import serial
            self._serial = serial.Serial(
                self._port, self._baud, timeout=SERIAL_TIMEOUT
            )
            time.sleep(2.0)   # wait for Arduino reset
            log.info(f"Serial opened: {self._port} @ {self._baud} baud")
            return True
        except ImportError:
            log.error("pyserial not installed. Run: pip install pyserial")
        except Exception as exc:
            log.error(f"Failed to open serial port '{self._port}': {exc}")
        return False

    def close(self):
        if self._serial and self._serial.is_open:
            self._serial.close()
            log.info("Serial port closed")

    # ──────────────────────────────────────────────────────────────────────────

    def send_pickup_command(self, world_xyz: np.ndarray) -> bool:
        """
        Send a pickup command for the given world position.
        Returns True if the command was accepted (or if in dry-run mode).

        world_xyz: np.array([X, Y, Z]) in metres
        """
        if self._busy:
            log.debug("Controller busy — command queued/dropped")
            return False

        x, y, z = float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])
        cmd = COMMAND_FORMAT.format(x=x, y=y, z=z)

        self._command_count += 1
        self._last_command = world_xyz.copy()

        if not self._enabled or self._serial is None:
            log.info(f"[DRY-RUN] Crane command #{self._command_count}: {cmd.strip()}")
            return True

        with self._lock:
            try:
                self._serial.write(cmd.encode("ascii"))
                self._serial.flush()
                log.info(f"Sent command #{self._command_count}: {cmd.strip()}")

                # Wait for acknowledgement
                self._busy = True
                deadline = time.time() + self.RESPONSE_TIMEOUT
                while time.time() < deadline:
                    line = self._serial.readline().decode("ascii", errors="ignore").strip()
                    if not line:
                        continue
                    log.debug(f"Crane reply: {line}")
                    if line.upper() == "ACK":
                        self._busy = False
                        return True
                    elif line.upper() in ("BUSY", "ERR"):
                        self._busy = False
                        log.warning(f"Crane responded: {line}")
                        return False

                log.warning("Timed out waiting for crane ACK")
                self._busy = False
                return False

            except Exception as exc:
                log.error(f"Serial write error: {exc}")
                self._busy = False
                return False

    # ──────────────────────────────────────────────────────────────────────────

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def command_count(self) -> int:
        return self._command_count
