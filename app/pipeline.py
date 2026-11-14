"""M5: Real-time integration pipeline (IO -> Physics -> RL Agent -> UI).

Orchestrates the live Hardware-in-the-Loop telemetry loop:
    1. Ingestion: Reads 4-pot telemetry from STM32 UART (SerialReader) or MockSource
    2. Mode Arbitrator:
       - 'Manual (Pots)': V1, V2 driven by physical pots 3 & 4
       - 'Auto (RL)': V1, V2 autonomously driven by DQN agent inference with
         limit-cycle escape, boundary-hit recovery, and best-so-far retention
    3. Physics Engine: Simulates Pi-network (Z_in, Gamma, VSWR) via physics.pi_network
    4. Diagnostics: Update rates (Hz), packet counts, latency age, suboptimality gap,
       and breadcrumb trajectory trail
"""

from __future__ import annotations

import collections
import dataclasses
import math
import sys
import time
from pathlib import Path
from typing import Any, Deque, List, Optional, Set, Tuple

import numpy as np

# Ensure project root and io directory are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IO_DIR = PROJECT_ROOT / "io"
for p in (str(PROJECT_ROOT), str(IO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# M2 Imports
try:
    from serial_reader import SerialReader, detect_serial_port, get_latest
    from mock_source import MockSource
except ImportError:
    try:
        from io.serial_reader import SerialReader, detect_serial_port, get_latest
        from io.mock_source import MockSource
    except ImportError:
        SerialReader = None        # type: ignore[assignment,misc]
        MockSource = None          # type: ignore[assignment,misc]
        detect_serial_port = None  # type: ignore[assignment]

# M3 Imports
from physics.varactor import capacitance
from physics.pi_network import compute, grid_search_match
from physics.smith import impedance_to_gamma, gamma_to_impedance

# M4 Imports
try:
    from rl.agent import load_agent, InferenceAgent
    from rl.env import ACTION_NAMES, _clamp_voltage
except ImportError:
    load_agent = None          # type: ignore[assignment]
    InferenceAgent = None      # type: ignore[assignment,misc]
    ACTION_NAMES = {0: "V1+", 1: "V1-", 2: "V2+", 3: "V2-", 4: "Hold"}

    def _clamp_voltage(val: float) -> float:
        return float(np.clip(val, 0.0, 3.3))

import config


# ============================================================================
# Action and Recovery Helpers (M4 Algorithm Integration)
# ============================================================================

def apply_voltage_action(v1: float, v2: float, action: int) -> Tuple[float, float]:
    """Return clamped candidate (V1, V2) after applying discrete action 0..4."""
    v_step = getattr(config, "V_STEP", 0.05)
    if action == 0:
        return _clamp_voltage(v1 + v_step), v2
    if action == 1:
        return _clamp_voltage(v1 - v_step), v2
    if action == 2:
        return v1, _clamp_voltage(v2 + v_step)
    if action == 3:
        return v1, _clamp_voltage(v2 - v_step)
    return v1, v2


def select_cycle_escape_action(
    rl: float,
    xl: float,
    v1: float,
    v2: float,
    blocked_action: int,
    visited_states: Set[Tuple[float, float]],
) -> Optional[Tuple[int, float, float, dict[str, Any]]]:
    """Evaluate non-cycling moving actions via physics and pick lowest next-state VSWR."""
    candidates = []
    for alt_action in (0, 1, 2, 3):
        if alt_action == blocked_action:
            continue
        cand_v1, cand_v2 = apply_voltage_action(v1, v2, alt_action)
        if (cand_v1 == v1 and cand_v2 == v2) or (cand_v1, cand_v2) in visited_states:
            continue
        cand_sim = compute(rl, xl, cand_v1, cand_v2)
        candidates.append((float(cand_sim["vswr"]), alt_action, cand_v1, cand_v2, cand_sim))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, best_action, best_v1, best_v2, best_sim = candidates[0]
    return best_action, best_v1, best_v2, best_sim


def run_rl_tuning(
    agent: Any,
    rl: float,
    xl: float,
    v1_init: float,
    v2_init: float,
    max_steps: int = 140,
    success_vswr: float = 1.20,
) -> dict[str, Any]:
    """Execute a greedy RL rollout with cycle avoidance, boundary recovery, and best-so-far retention."""
    v1 = _clamp_voltage(v1_init)
    v2 = _clamp_voltage(v2_init)
    cur_sim = compute(rl, xl, v1, v2)
    initial_vswr = float(cur_sim["vswr"])

    best_v1, best_v2, best_sim = v1, v2, cur_sim
    visited_states: Set[Tuple[float, float]] = {(v1, v2)}
    recent_states: Deque[Tuple[float, float]] = collections.deque([(v1, v2)], maxlen=8)
    cycle_detected = False
    two_state_cycle_detected = False
    boundary_fallback_attempted = False
    boundary_fallback_succeeded = False
    steps_taken = 0
    stop_reason = "max_steps"
    trajectory: List[dict[str, Any]] = []

    for _ in range(max_steps):
        vswr = float(cur_sim["vswr"])
        gamma = cur_sim["gamma"]
        trajectory.append({"V1": v1, "V2": v2, "gamma": gamma, "vswr": vswr})

        if vswr < success_vswr:
            stop_reason = "target_reached"
            break

        obs = np.array([gamma.real, gamma.imag, v1, v2], dtype=np.float32)
        action = int(agent.act(obs))
        if action == 4:
            stop_reason = "hold"
            break

        cand_v1, cand_v2 = apply_voltage_action(v1, v2, action)

        # 1. Boundary-hit recovery: try alternative legal move only if it strictly lowers VSWR
        if cand_v1 == v1 and cand_v2 == v2:
            boundary_fallback_attempted = True
            escape = select_cycle_escape_action(
                rl=rl, xl=xl, v1=v1, v2=v2, blocked_action=action, visited_states=visited_states
            )
            if escape is not None and float(escape[3]["vswr"]) < vswr:
                boundary_fallback_succeeded = True
                _, next_v1, next_v2, next_sim = escape
            else:
                stop_reason = "boundary_hit"
                break
        # 2. Limit cycle detection & escape
        elif (cand_v1, cand_v2) in visited_states:
            cycle_detected = True
            if len(recent_states) >= 2 and (cand_v1, cand_v2) == recent_states[-2]:
                two_state_cycle_detected = True
            escape = select_cycle_escape_action(
                rl=rl, xl=xl, v1=v1, v2=v2, blocked_action=action, visited_states=visited_states
            )
            if escape is None:
                stop_reason = "cycle_unbreakable"
                break
            _, next_v1, next_v2, next_sim = escape
        else:
            next_v1, next_v2 = cand_v1, cand_v2
            next_sim = compute(rl, xl, next_v1, next_v2)

        steps_taken += 1
        v1, v2, cur_sim = next_v1, next_v2, next_sim
        visited_states.add((v1, v2))
        recent_states.append((v1, v2))

        if float(cur_sim["vswr"]) < float(best_sim["vswr"]):
            best_v1, best_v2, best_sim = v1, v2, cur_sim
    else:
        stop_reason = "target_reached" if float(cur_sim["vswr"]) < success_vswr else "max_steps"

    cycle_escaped = bool(cycle_detected and stop_reason != "cycle_unbreakable")
    return {
        "best_v1": float(best_v1),
        "best_v2": float(best_v2),
        "vswr": float(best_sim["vswr"]),
        "gamma": best_sim["gamma"],
        "gamma_abs": float(best_sim["gamma_abs"]),
        "Zin": best_sim["Zin"],
        "initial_vswr": initial_vswr,
        "final_v1": float(v1),
        "final_v2": float(v2),
        "final_vswr": float(cur_sim["vswr"]),
        "steps": steps_taken,
        "cycle_detected": cycle_detected,
        "cycle_escaped": cycle_escaped,
        "two_state_cycle_detected": two_state_cycle_detected,
        "boundary_fallback_attempted": boundary_fallback_attempted,
        "boundary_fallback_succeeded": boundary_fallback_succeeded,
        "stop_reason": stop_reason,
        "trajectory": trajectory,
    }


# ============================================================================
# Pipeline Snapshot Dataclass
# ============================================================================

@dataclasses.dataclass
class PipelineSnapshot:
    """Immutable state snapshot for thread-safe UI rendering without emojis."""

    timestamp: float
    mode: str                     # "Manual (Pots)" or "Auto (RL)"
    source_type: str              # "Mock" or "Serial"
    connection_status: str        # "Streaming (Serial)", "Streaming (Mock)", "Disconnected"
    error_message: Optional[str]

    # Raw telemetry from hardware/mock
    RL: float                     # Load resistance (Ohms)
    XL: float                     # Load reactance (Ohms)
    V1_man: float                 # Pot 3 voltage (Volts)
    V2_man: float                 # Pot 4 voltage (Volts)

    # Active voltages applied to varactors
    V1_active: float              # Active bias voltage V1 (Volts)
    V2_active: float              # Active bias voltage V2 (Volts)
    C1_pF: float                  # Varactor C1 capacitance (pF)
    C2_pF: float                  # Varactor C2 capacitance (pF)

    # Physics engine calculations
    Z_in: complex                 # Input impedance looking into Pi-network (Ohms)
    gamma_in: complex             # Reflection coefficient Gamma_in
    gamma_abs: float              # Magnitude |Gamma_in|
    vswr: float                   # Voltage Standing Wave Ratio
    return_loss_db: float         # Return Loss in dB
    gamma_load: complex           # Antenna load reflection coefficient Gamma_load

    # Match classification & Ground Truth Benchmark
    match_status: str             # "MATCHED", "DIFF. LOAD (NEAR-OPT)", "ACCEPTABLE", "SUBOPTIMAL"
    is_matched: bool              # True if VSWR <= 1.20
    vswr_opt: float               # Theoretical optimal VSWR via grid search
    v1_opt: float                 # Theoretical optimal V1 (Volts)
    v2_opt: float                 # Theoretical optimal V2 (Volts)
    gamma_opt: complex            # Theoretical optimal Gamma*
    delta_vswr: float             # Suboptimality gap: vswr - vswr_opt

    # RL Agent diagnostics
    last_action_id: int           # 0..4
    last_action_name: str         # "V1+", "V1-", "V2+", "V2-", "Hold"
    agent_loaded: bool            # True if model.pt is loaded
    steps_in_auto: int            # Steps taken in current Auto rollout
    cycle_escaped: bool           # True if a limit cycle was detected and broken
    boundary_recovered: bool      # True if boundary fallback was activated

    # Trajectory breadcrumbs for Smith chart (oldest -> newest)
    trail: list[complex]

    # Performance diagnostics
    packet_count: int
    fps: float                    # Measured loop frequency in Hz
    last_update_age: float        # Seconds elapsed since last data update


# ============================================================================
# Pipeline Controller
# ============================================================================

class PipelineController:
    """Coordinates the dataflow between M1/M2, M3, M4, and M5."""

    def __init__(
        self,
        source_type: str = "Mock",
        mode: str = "Manual (Pots)",
        model_path: str | Path = "models/model.pt",
        trail_max_len: int = 35,
        serial_port: Optional[str] = None,
        baud_rate: int = 115200,
        max_rollout_steps: int = 140,
    ) -> None:
        self.source_type = source_type
        self.mode = mode
        self.model_path = Path(model_path)
        self.trail_max_len = trail_max_len
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.max_rollout_steps = max_rollout_steps

        # Data sources
        self._mock_source: Optional[Any] = None
        self._serial_reader: Optional[Any] = None

        # RL Agent
        self.agent: Optional[Any] = None
        self.agent_loaded: bool = False
        self._load_agent()

        # Internal active voltages
        self._v1_auto: float = 1.65
        self._v2_auto: float = 1.65
        self._last_action_id: int = 4
        self._last_action_name: str = "Hold"
        self._steps_in_auto: int = 0
        self._cycle_escaped: bool = False
        self._boundary_recovered: bool = False

        # Diagnostics & Metrics
        self.packet_count: int = 0
        self.last_update_time: float = time.time()
        self.connection_status: str = "Initializing"
        self.error_message: Optional[str] = None
        self._trail: Deque[complex] = collections.deque(maxlen=self.trail_max_len)

        # FPS calculation tracking
        self._last_tick_time: float = time.time()
        self._fps: float = 20.0
        self._dt_history: Deque[float] = collections.deque(maxlen=20)

        # Baseline caching
        self._cached_rl: float = -9999.0
        self._cached_xl: float = -9999.0
        self._cached_baseline: Optional[dict[str, Any]] = None

        # Initialize data source
        self._init_source()

    def _load_agent(self) -> None:
        """Load the trained Double DQN model checkpoint."""
        if load_agent is None:
            self.agent = None
            self.agent_loaded = False
            self.error_message = "M4 module (rl.agent) not imported."
            return

        resolved_path = PROJECT_ROOT / self.model_path
        if not resolved_path.exists():
            resolved_path = Path(self.model_path)

        if resolved_path.exists():
            try:
                self.agent = load_agent(resolved_path)
                self.agent_loaded = True
            except Exception as exc:
                self.agent = None
                self.agent_loaded = False
                self.error_message = f"Model load error: {exc}"
        else:
            self.agent = None
            self.agent_loaded = False
            self.error_message = f"Model checkpoint not found: {resolved_path}"

    def _init_source(self) -> None:
        """Instantiate and start the selected telemetry data source."""
        self._stop_sources()

        if self.source_type == "Mock":
            if MockSource is not None:
                try:
                    self._mock_source = MockSource(update_rate=20.0, enable_noise=True, enable_disturbances=True)
                    self._mock_source.start()
                    self.connection_status = "Streaming (Mock)"
                    self.error_message = None
                except Exception as exc:
                    self.connection_status = "Error"
                    self.error_message = f"Failed to start MockSource: {exc}"
            else:
                self.connection_status = "Fallback"
                self.error_message = "MockSource class not available."
        elif self.source_type == "Serial":
            if SerialReader is not None:
                target_port = self.serial_port
                if target_port is None and detect_serial_port is not None:
                    try:
                        target_port = detect_serial_port()
                    except Exception:
                        target_port = None

                try:
                    self._serial_reader = SerialReader(port=target_port, baudrate=self.baud_rate)
                    self._serial_reader.start()
                    # Check connection state
                    is_conn = False
                    if hasattr(self._serial_reader, "is_connected"):
                        is_conn = (
                            self._serial_reader.is_connected()
                            if callable(self._serial_reader.is_connected)
                            else bool(self._serial_reader.is_connected)
                        )
                    self.connection_status = "Streaming (Serial)" if is_conn else "Disconnected"
                    self.error_message = None
                except Exception as exc:
                    self.connection_status = "Disconnected"
                    self.error_message = f"Serial connection failed: {exc}"
            else:
                self.connection_status = "Fallback"
                self.error_message = "SerialReader class not available."

    def _stop_sources(self) -> None:
        """Safely stop and join background streaming threads."""
        if self._mock_source is not None:
            try:
                self._mock_source.stop()
            except Exception:
                pass
            self._mock_source = None

        if self._serial_reader is not None:
            try:
                self._serial_reader.stop()
            except Exception:
                pass
            self._serial_reader = None

    def set_source(self, source_type: str, serial_port: Optional[str] = None) -> None:
        """Switch telemetry source between 'Mock' and 'Serial'."""
        if self.source_type != source_type or self.serial_port != serial_port:
            self.source_type = source_type
            self.serial_port = serial_port
            self._init_source()

    def set_mode(self, mode: str) -> None:
        """Toggle operating mode: 'Manual (Pots)' or 'Auto (RL)'."""
        if mode != self.mode:
            self.mode = mode
            self._steps_in_auto = 0

    def reset_trail(self) -> None:
        """Clear the historical trajectory breadcrumbs."""
        self._trail.clear()

    def step(
        self,
        override_data: Optional[dict[str, float]] = None,
        agent_steps_per_tick: int = 1,
        auto_full_rollout: bool = True,
    ) -> PipelineSnapshot:
        """Execute one complete integration pipeline cycle.

        Parameters
        ----------
        override_data : dict, optional
            Direct manual injection of telemetry for testing or UI manual slider mode.
        agent_steps_per_tick : int, optional
            Number of micro-steps to step the RL agent per cycle if not full rollout (default: 1).
        auto_full_rollout : bool, optional
            Whether Auto mode executes a full M4 tuning rollout to reach the best match (default: True).

        Returns
        -------
        PipelineSnapshot
            The full state of the network and diagnostics for the current cycle.
        """
        now = time.time()
        dt = now - self._last_tick_time
        self._last_tick_time = now
        if dt > 0:
            self._dt_history.append(dt)
            self._fps = 1.0 / (sum(self._dt_history) / len(self._dt_history))

        # 1. Acquire Telemetry
        raw_data: Optional[dict[str, float]] = None
        if override_data is not None:
            raw_data = override_data
        elif self.source_type == "Mock" and self._mock_source is not None:
            raw_data = self._mock_source.get_latest()
        elif self.source_type == "Serial" and self._serial_reader is not None:
            raw_data = self._serial_reader.get_latest()
            is_conn = False
            if hasattr(self._serial_reader, "is_connected"):
                is_conn = (
                    self._serial_reader.is_connected()
                    if callable(self._serial_reader.is_connected)
                    else bool(self._serial_reader.is_connected)
                )
            if is_conn:
                self.connection_status = "Streaming (Serial)"
            else:
                self.connection_status = "Disconnected"
                if hasattr(self._serial_reader, "get_connection_status"):
                    stat = self._serial_reader.get_connection_status()
                    if isinstance(stat, dict) and stat.get("last_error"):
                        self.error_message = str(stat["last_error"])

        if raw_data is None or not isinstance(raw_data, dict):
            raw_data = {
                "RL": 13.5,
                "XL": 69.5,
                "V1_man": 1.65,
                "V2_man": 1.65,
                "t": now,
            }
            if self.connection_status not in ("Disconnected", "Error"):
                self.connection_status = "Fallback (Default)"

        RL = float(raw_data.get("RL", 13.5))
        XL = float(raw_data.get("XL", 69.5))
        V1_man = float(raw_data.get("V1_man", 1.65))
        V2_man = float(raw_data.get("V2_man", 1.65))
        t_sample = float(raw_data.get("t", now))

        self.packet_count += 1
        self.last_update_time = t_sample
        last_update_age = max(0.0, now - t_sample)

        # 2. Physics Ground Truth Optimum (Cached by disturbance coordinates)
        if (
            self._cached_baseline is None
            or abs(RL - self._cached_rl) > 0.05
            or abs(XL - self._cached_xl) > 0.05
        ):
            self._cached_baseline = grid_search_match(RL, XL, grid_size=34)
            self._cached_rl = RL
            self._cached_xl = XL

        baseline = self._cached_baseline
        v1_opt = float(baseline["best_V1"])
        v2_opt = float(baseline["best_V2"])
        vswr_opt = float(baseline["best_vswr"])
        gamma_opt = complex(baseline["gamma"])

        # 3. Determine Active Voltages based on Mode
        if self.mode == "Manual (Pots)":
            V1_active = _clamp_voltage(V1_man)
            V2_active = _clamp_voltage(V2_man)
            self._v1_auto = V1_active
            self._v2_auto = V2_active
            action_id = 4
            action_name = "Hold (Manual)"
            steps_taken = 0
            cycle_escaped = False
            boundary_recovered = False
        else:
            # Auto (RL) Mode
            if self.agent_loaded and self.agent is not None:
                if auto_full_rollout:
                    rollout = run_rl_tuning(
                        agent=self.agent,
                        rl=RL,
                        xl=XL,
                        v1_init=self._v1_auto,
                        v2_init=self._v2_auto,
                        max_steps=self.max_rollout_steps,
                        success_vswr=1.20,
                    )
                    V1_active = rollout["best_v1"]
                    V2_active = rollout["best_v2"]
                    steps_taken = rollout["steps"]
                    cycle_escaped = rollout["cycle_escaped"]
                    boundary_recovered = rollout["boundary_fallback_succeeded"]
                    action_id = 4 if rollout["stop_reason"] == "target_reached" else 0
                    action_name = f"Auto ({rollout['stop_reason']}, {steps_taken} steps)"

                    # Append intermediate rollout trajectory to trail
                    for pt in rollout.get("trajectory", []):
                        self._trail.append(pt["gamma"])
                else:
                    # Incremental micro-stepping per tick
                    V1_active = self._v1_auto
                    V2_active = self._v2_auto
                    steps_taken = 0
                    cycle_escaped = False
                    boundary_recovered = False
                    for _ in range(max(1, agent_steps_per_tick)):
                        sim_mid = compute(RL, XL, V1_active, V2_active)
                        g_mid = sim_mid["gamma"]
                        obs = np.array([g_mid.real, g_mid.imag, V1_active, V2_active], dtype=np.float32)
                        action_id = int(self.agent.act(obs))
                        action_name = ACTION_NAMES.get(action_id, "Hold")
                        V1_active, V2_active = apply_voltage_action(V1_active, V2_active, action_id)
                        steps_taken += 1

                self._v1_auto = V1_active
                self._v2_auto = V2_active
                self._steps_in_auto += steps_taken
                self._cycle_escaped = cycle_escaped
                self._boundary_recovered = boundary_recovered
            else:
                # Theoretical grid-search fallback if model.pt is missing
                V1_active = v1_opt
                V2_active = v2_opt
                self._v1_auto = V1_active
                self._v2_auto = V2_active
                action_id = 4
                action_name = "Auto (GridSearch Fallback)"
                steps_taken = 0
                cycle_escaped = False
                boundary_recovered = False

        self._last_action_id = action_id
        self._last_action_name = action_name

        # 4. Final RF Physics Engine Evaluation
        sim = compute(RL, XL, V1_active, V2_active)
        Z_in = sim["Zin"]
        gamma_in = sim["gamma"]
        gamma_abs = sim["gamma_abs"]
        vswr = sim["vswr"]
        return_loss_db = float(-20.0 * np.log10(max(1e-6, gamma_abs)))
        gamma_load = impedance_to_gamma(complex(RL, XL))

        # Suboptimality gap
        delta_vswr = max(0.0, vswr - vswr_opt)

        # Match Status Classification (per plan_m5.md)
        if vswr <= 1.20:
            match_status = "MATCHED"
            is_matched = True
        elif vswr_opt >= 1.50 and delta_vswr <= 0.15:
            match_status = "DIFF. LOAD (NEAR-OPT)"
            is_matched = True
        elif vswr <= 1.50:
            match_status = "ACCEPTABLE"
            is_matched = False
        else:
            match_status = "SUBOPTIMAL"
            is_matched = False

        # Varactor Capacitances
        C1_pF = float(capacitance(V1_active) * 1e12)
        C2_pF = float(capacitance(V2_active) * 1e12)

        # Append final point to trajectory
        self._trail.append(gamma_in)

        return PipelineSnapshot(
            timestamp=now,
            mode=self.mode,
            source_type=self.source_type,
            connection_status=self.connection_status,
            error_message=self.error_message,
            RL=RL,
            XL=XL,
            V1_man=V1_man,
            V2_man=V2_man,
            V1_active=V1_active,
            V2_active=V2_active,
            C1_pF=C1_pF,
            C2_pF=C2_pF,
            Z_in=Z_in,
            gamma_in=gamma_in,
            gamma_abs=gamma_abs,
            vswr=vswr,
            return_loss_db=return_loss_db,
            gamma_load=gamma_load,
            match_status=match_status,
            is_matched=is_matched,
            vswr_opt=vswr_opt,
            v1_opt=v1_opt,
            v2_opt=v2_opt,
            gamma_opt=gamma_opt,
            delta_vswr=delta_vswr,
            last_action_id=action_id,
            last_action_name=action_name,
            agent_loaded=self.agent_loaded,
            steps_in_auto=self._steps_in_auto,
            cycle_escaped=self._cycle_escaped,
            boundary_recovered=self._boundary_recovered,
            trail=list(self._trail),
            packet_count=self.packet_count,
            fps=self._fps,
            last_update_age=last_update_age,
        )

    def close(self) -> None:
        """Release streaming threads and device resources."""
        self._stop_sources()
