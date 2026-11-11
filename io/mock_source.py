"""M2: Mock data source producing realistic 4-channel telemetry for offline development.

Simulates the STM32F446RE UART telemetry stream at ~20 Hz:
    - RL in [10.0, 150.0] Ohms
    - XL in [-100.0, 100.0] Ohms
    - V1_man in [0.0, 3.3] Volts
    - V2_man in [0.0, 3.3] Volts
    - t (Unix epoch timestamp in seconds)

Produces smooth sinusoidal sweeps with periodic random detuning disturbances to
emulate physical antenna handling, enabling M3 (RF Physics), M4 (RL Agent),
and M5 (UI Dashboard) to develop without physical hardware.
"""

from __future__ import annotations

import math
import random
import threading
import time
from typing import Any

# Import filters & constants with fallback strategies
try:
    from io.filters import RL_MAX, RL_MIN, V_MAX, V_MIN, XL_MAX, XL_MIN
except (ImportError, AttributeError):
    try:
        from .filters import RL_MAX, RL_MIN, V_MAX, V_MIN, XL_MAX, XL_MIN
    except (ImportError, AttributeError):
        from filters import RL_MAX, RL_MIN, V_MAX, V_MIN, XL_MAX, XL_MIN


class MockSource:
    """Mock telemetry source matching SerialReader's public interface.

    Runs an internal background thread that generates realistic time-varying
    measurements at a configurable rate (default 20 Hz). Provides thread-safe
    data retrieval via `get_latest()`.
    """

    def __init__(
        self,
        update_rate: float = 20.0,
        enable_noise: bool = True,
        enable_disturbances: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize MockSource.

        Args:
            update_rate: Telemetry generation frequency in Hz (default 20.0).
            enable_noise: Whether to overlay small Gaussian measurement jitter.
            enable_disturbances: Whether to periodically inject sudden detuning steps.
            seed: Optional seed for pseudo-random reproducibility in testing.
        """
        self.update_rate = float(update_rate) if update_rate > 0 else 20.0
        self.dt = 1.0 / self.update_rate
        self.enable_noise = enable_noise
        self.enable_disturbances = enable_disturbances

        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # State tracking for smooth variations and detuning offsets
        self._start_time = time.time()
        self._sim_time = 0.0
        self._rl_offset = 0.0
        self._xl_offset = 0.0

        # Nominal initial state (50 Ohm matched load, 1.65V mid-scale varactor bias)
        self._latest_data: dict[str, float] = {
            "RL": 50.0,
            "XL": 0.0,
            "V1_man": 1.65,
            "V2_man": 1.65,
            "t": time.time(),
        }

    def start(self) -> None:
        """Start the background simulation thread.

        If already running, this is a no-op.
        """
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="MockSourceThread",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the background simulation thread cleanly.

        Args:
            timeout: Maximum seconds to wait for worker thread to terminate.
        """
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    def is_connected(self) -> bool:
        """Check whether the mock source is actively producing data.

        Returns:
            True if the worker thread is running, False otherwise.
        """
        return self._thread is not None and self._thread.is_alive()

    def generate_step(self, timestamp: float | None = None) -> dict[str, float]:
        """Generate a single time-stepped measurement.

        Can be used synchronously or by the background worker loop.

        Args:
            timestamp: Optional epoch timestamp to record. Defaults to time.time().

        Returns:
            Dictionary matching the public API contract:
            {"RL": float, "XL": float, "V1_man": float, "V2_man": float, "t": float}
        """
        now = time.time() if timestamp is None else float(timestamp)
        t = now - self._start_time

        # Environmental detuning events (simulating hand capacitance or load shift)
        if self.enable_disturbances:
            # ~2% chance per frame (at 20 Hz, roughly once every 2.5 - 5 seconds)
            if self._rng.random() < 0.02:
                self._rl_offset += self._rng.uniform(-25.0, 25.0)
                self._xl_offset += self._rng.uniform(-35.0, 35.0)

            # Slow decay of disturbance offsets back toward equilibrium
            self._rl_offset *= 0.995
            self._xl_offset *= 0.995

        # Measurement noise jitter
        noise_rl = self._rng.gauss(0.0, 0.4) if self.enable_noise else 0.0
        noise_xl = self._rng.gauss(0.0, 0.4) if self.enable_noise else 0.0
        noise_v1 = self._rng.gauss(0.0, 0.01) if self.enable_noise else 0.0
        noise_v2 = self._rng.gauss(0.0, 0.01) if self.enable_noise else 0.0

        # Multi-frequency smooth sinusoidal variations
        # RL: Base 80 Ohms, sweeps across [10, 150]
        rl_raw = (
            80.0
            + 45.0 * math.sin(2.0 * math.pi * 0.08 * t)
            + 15.0 * math.sin(2.0 * math.pi * 0.02 * t + 0.5)
            + self._rl_offset
            + noise_rl
        )
        rl = max(RL_MIN, min(RL_MAX, rl_raw))

        # XL: Center 0 Ohms, sweeps across [-100, 100]
        xl_raw = (
            0.0
            + 65.0 * math.sin(2.0 * math.pi * 0.06 * t + 1.2)
            + 20.0 * math.cos(2.0 * math.pi * 0.015 * t)
            + self._xl_offset
            + noise_xl
        )
        xl = max(XL_MIN, min(XL_MAX, xl_raw))

        # V1: Manual tuning voltage [0.0, 3.3] V
        v1_raw = (
            1.65
            + 1.2 * math.sin(2.0 * math.pi * 0.04 * t)
            + 0.3 * math.sin(2.0 * math.pi * 0.1 * t + 0.3)
            + noise_v1
        )
        v1 = max(V_MIN, min(V_MAX, v1_raw))

        # V2: Manual tuning voltage [0.0, 3.3] V
        v2_raw = (
            1.65
            + 1.2 * math.cos(2.0 * math.pi * 0.035 * t + 0.9)
            + 0.3 * math.cos(2.0 * math.pi * 0.09 * t)
            + noise_v2
        )
        v2 = max(V_MIN, min(V_MAX, v2_raw))

        return {
            "RL": float(rl),
            "XL": float(xl),
            "V1_man": float(v1),
            "V2_man": float(v2),
            "t": float(now),
        }

    def get_latest(self) -> dict[str, float]:
        """Return the latest generated telemetry frame.

        Thread-safe copy of the latest measurement.

        Returns:
            Dictionary with keys: 'RL', 'XL', 'V1_man', 'V2_man', 't'.
        """
        with self._lock:
            return {
                "RL": float(self._latest_data["RL"]),
                "XL": float(self._latest_data["XL"]),
                "V1_man": float(self._latest_data["V1_man"]),
                "V2_man": float(self._latest_data["V2_man"]),
                "t": float(self._latest_data["t"]),
            }

    def _worker_loop(self) -> None:
        """Background generator thread running at target update_rate."""
        while not self._stop_event.is_set():
            frame = self.generate_step()
            with self._lock:
                self._latest_data = frame

            # Rate-limited sleep: wakes up early if stop() is called
            self._stop_event.wait(timeout=self.dt)
