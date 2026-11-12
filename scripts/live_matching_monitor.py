#!/usr/bin/env python3
"""Live Cross-Module Action Monitor (M1 USART -> M2 -> M3 -> M4).

Demonstrates the live end-to-end operation in real time:
1. Ingests raw 4-channel ADC telemetry from USART (or MockSource if offline).
2. Converts raw 12-bit ADC counts into calibrated physical units:
   - ADC 1 -> Load Resistance RL (Ohms)
   - ADC 2 -> Load Reactance XL (Ohms)
   - ADC 3 -> Manual Varactor Bias V1 (Volts)
   - ADC 4 -> Manual Varactor Bias V2 (Volts)
3. Evaluates unmitigated RF state (initial Gamma and VSWR).
4. Solves for the exact theoretical optimal varactor voltages (V1*, V2*)
   using the M3 physics engine (grid_search_match).
5. Runs the trained M4 Double DQN RL agent (models/model.pt) in Python-side
   simulation starting from the initial state to predict/tune virtual
   varactor voltages (V1_rl, V2_rl) — no voltages are sent to the STM32.
6. Compares physics ground-truth optimal vs RL best-so-far voltages and VSWR,
   calculating voltage error and matching accuracy metrics live.

Usage:
    # Live hardware (auto-detect STM32 or fallback to mock)
    python scripts/live_matching_monitor.py

    # Explicit serial port
    python scripts/live_matching_monitor.py --port COM3

    # Force mock simulation mode
    python scripts/live_matching_monitor.py --mock

    # Run for a fixed number of frames with a custom rollout horizon
    python scripts/live_matching_monitor.py --count 10 --max-steps 70
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IO_DIR = PROJECT_ROOT / "io"
if str(IO_DIR) not in sys.path:
    sys.path.insert(0, str(IO_DIR))

# Ensure coexistence with stdlib io
import io as stdlib_io

if hasattr(stdlib_io, "__path__"):
    if str(IO_DIR) not in stdlib_io.__path__:
        stdlib_io.__path__.append(str(IO_DIR))
else:
    stdlib_io.__path__ = [str(IO_DIR)]

import numpy as np

# Module imports
import config
from filters import (
    ADC_MAX,
    ADC_VREF,
    RL_MAX,
    RL_MIN,
    V_MAX,
    V_MIN,
    XL_MAX,
    XL_MIN,
    adc_to_physical,
)
from serial_reader import (
    HAS_PYSERIAL,
    SerialReader,
    detect_serial_port,
    parse_adc_line,
)
from mock_source import MockSource
from physics.pi_network import compute, grid_search_match
from rl.agent import load_agent, InferenceAgent
from rl.env import ACTION_NAMES, _clamp_voltage


# ---------------------------------------------------------------------------
# RL Autonomous Tuning Engine
# ---------------------------------------------------------------------------
def _apply_voltage_action(v1: float, v2: float, action: int) -> tuple[float, float]:
    """Return clamped candidate (V1, V2) after applying discrete action 0..4."""
    if action == 0:
        return _clamp_voltage(v1 + config.V_STEP), v2
    if action == 1:
        return _clamp_voltage(v1 - config.V_STEP), v2
    if action == 2:
        return v1, _clamp_voltage(v2 + config.V_STEP)
    if action == 3:
        return v1, _clamp_voltage(v2 - config.V_STEP)
    return v1, v2


def _select_cycle_escape_action(
    rl: float,
    xl: float,
    v1: float,
    v2: float,
    blocked_action: int,
    visited_states: set[tuple[float, float]],
) -> Optional[tuple[int, float, float, dict[str, Any]]]:
    """Evaluate non-cycling moving actions via physics and pick lowest next-state VSWR."""
    candidates: list[tuple[float, int, float, float, dict[str, Any]]] = []
    for alt_action in (0, 1, 2, 3):
        if alt_action == blocked_action:
            continue
        cand_v1, cand_v2 = _apply_voltage_action(v1, v2, alt_action)
        # Skip boundary no-ops and already-visited states to prevent re-entering cycles.
        if (cand_v1 == v1 and cand_v2 == v2) or (cand_v1, cand_v2) in visited_states:
            continue
        cand_sim = compute(rl, xl, cand_v1, cand_v2)
        candidates.append(
            (float(cand_sim["vswr"]), alt_action, cand_v1, cand_v2, cand_sim)
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    _, best_action, best_v1, best_v2, best_sim = candidates[0]
    return best_action, best_v1, best_v2, best_sim


def run_rl_tuning(
    agent: InferenceAgent,
    rl: float,
    xl: float,
    v1_init: float,
    v2_init: float,
    max_steps: int = 70,
    success_vswr: float = 1.20,
) -> dict[str, Any]:
    """Run greedy RL agent rollout in Python simulation with cycle avoidance.

    Tracks all visited `(V1, V2)` states and the best-so-far configuration
    encountered across the rollout. When the agent's greedy action would revisit
    a recently visited state (including two-state oscillations), alternative
    unvisited moving actions are compared via the M3 physics model without
    modifying the Q-network. Regardless of stop reason, the returned ``v1``,
    ``v2``, and ``vswr`` represent the best-so-far configuration, while
    ``final_*`` fields retain the terminal trajectory state for comparison.

    Args:
        agent: Trained inference agent.
        rl: Load resistance in Ohms.
        xl: Load reactance in Ohms.
        v1_init: Initial V1 bias voltage in Volts.
        v2_init: Initial V2 bias voltage in Volts.
        max_steps: Maximum action iterations (must be >= 1, default: 70).
        success_vswr: Target VSWR threshold for early termination.

    Returns:
        dict containing best-so-far configuration, terminal state metrics,
        initial VSWR, step count, cycle-detection status, and stop reason.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1; got {max_steps!r}.")

    v1 = _clamp_voltage(v1_init)
    v2 = _clamp_voltage(v2_init)

    cur_sim = compute(rl, xl, v1, v2)
    initial_vswr = float(cur_sim["vswr"])

    best_v1, best_v2, best_sim = v1, v2, cur_sim
    visited_states: set[tuple[float, float]] = {(v1, v2)}
    recent_states: deque[tuple[float, float]] = deque([(v1, v2)], maxlen=8)
    cycle_detected = False
    two_state_cycle_detected = False
    cycle_events = 0
    steps_taken = 0
    stop_reason = "max_steps"
    t0 = time.perf_counter()

    for _ in range(max_steps):
        vswr = float(cur_sim["vswr"])
        gamma = cur_sim["gamma"]

        # Goal reached: stop immediately before taking any unnecessary action.
        if vswr < success_vswr:
            stop_reason = "target_reached"
            break

        obs = np.array([gamma.real, gamma.imag, v1, v2], dtype=np.float32)
        action = int(agent.act(obs))

        # Hold action
        if action == 4:
            stop_reason = "hold"
            break

        cand_v1, cand_v2 = _apply_voltage_action(v1, v2, action)

        # Preserve boundary-limit handling
        if cand_v1 == v1 and cand_v2 == v2:
            stop_reason = "boundary_hit"
            break

        # Detect limit cycle / short oscillation before repeating a visited state
        if (cand_v1, cand_v2) in visited_states:
            cycle_detected = True
            cycle_events += 1
            if len(recent_states) >= 2 and (cand_v1, cand_v2) == recent_states[-2]:
                two_state_cycle_detected = True

            escape = _select_cycle_escape_action(
                rl=rl,
                xl=xl,
                v1=v1,
                v2=v2,
                blocked_action=action,
                visited_states=visited_states,
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
        stop_reason = (
            "target_reached"
            if float(cur_sim["vswr"]) < success_vswr
            else "max_steps"
        )

    t_eval_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "v1": float(best_v1),
        "v2": float(best_v2),
        "vswr": float(best_sim["vswr"]),
        "gamma_abs": float(best_sim["gamma_abs"]),
        "gamma": best_sim["gamma"],
        "initial_vswr": initial_vswr,
        "final_v1": float(v1),
        "final_v2": float(v2),
        "final_vswr": float(cur_sim["vswr"]),
        "final_gamma_abs": float(cur_sim["gamma_abs"]),
        "final_gamma": cur_sim["gamma"],
        "best_v1": float(best_v1),
        "best_v2": float(best_v2),
        "best_vswr": float(best_sim["vswr"]),
        "best_gamma_abs": float(best_sim["gamma_abs"]),
        "best_gamma": best_sim["gamma"],
        "steps": steps_taken,
        "cycle_detected": cycle_detected,
        "two_state_cycle_detected": two_state_cycle_detected,
        "cycle_events": cycle_events,
        "stop_reason": stop_reason,
        "eval_ms": t_eval_ms,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _color(text: str, code: str) -> str:
    """Return ANSI colored string if stdout is a tty."""
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text


def green(text: str) -> str:
    return _color(text, "32")


def yellow(text: str) -> str:
    return _color(text, "33")


def red(text: str) -> str:
    return _color(text, "31")


def cyan(text: str) -> str:
    return _color(text, "36")


def bold(text: str) -> str:
    return _color(text, "1")


def _pad_color(text: str, width: int, color_fn: Callable[[str], str]) -> str:
    """Pad plain text to fixed display width before applying optional ANSI color."""
    return color_fn(f"{text:<{width}}")


# ---------------------------------------------------------------------------
# Main Monitor Loop
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live USART -> Physics -> RL Agent Accuracy Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--port",
        "-p",
        type=str,
        default=None,
        help="Serial port (e.g. COM3, COM4, /dev/ttyACM0). Auto-detects if omitted.",
    )
    parser.add_argument(
        "--baud",
        "-b",
        type=int,
        default=config.BAUD,
        help=f"UART baud rate (default: {config.BAUD})",
    )
    parser.add_argument(
        "--mock",
        "-m",
        action="store_true",
        help="Force simulated MockSource telemetry even if serial ports are present.",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=0,
        help="Number of telemetry frames to evaluate before exiting (0 = continuous).",
    )
    parser.add_argument(
        "--rate",
        "-r",
        type=float,
        default=4.0,
        help="Display refresh rate limit in Hz (default: 4.0 Hz for readable streaming).",
    )
    parser.add_argument(
        "--v-init",
        choices=["telemetry", "center"],
        default="telemetry",
        help="Initial voltage for RL tuning: 'telemetry' uses measured V1/V2 from pot; 'center' uses 1.65V/1.65V.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=70,
        help="Maximum RL rollout steps per tuning cycle (default: 70).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/model.pt",
        help="Path to trained RL model checkpoint (default: models/model.pt).",
    )
    args = parser.parse_args()

    if args.max_steps < 1:
        print(red(f"[ERROR] --max-steps must be >= 1; got {args.max_steps}."))
        return 1

    # 1. Load trained RL agent
    model_path = PROJECT_ROOT / args.model
    if not model_path.exists():
        print(red(f"[ERROR] Trained model checkpoint not found at: {model_path}"))
        return 1

    print(bold("=" * 112))
    print(bold("  AUTONOMOUS IMPEDANCE MATCHING NETWORK — LIVE ACCURACY MONITOR"))
    print(bold("  M1 (Firmware) -> M2 (UART/ADC) -> M3 (Physics) -> M4 (RL Agent, Python Simulation)"))
    print(bold("=" * 112))
    print(f"Loading trained RL model from {cyan(str(model_path))} ...")
    agent = load_agent(str(model_path), device="cpu")
    print(green(f"  Model loaded: state_dim={agent.state_dim}, action_dim={agent.action_dim}"))

    # 2. Determine and initialize telemetry source
    is_live_serial = False
    source: Any = None
    serial_port: Optional[str] = None

    if not args.mock and HAS_PYSERIAL:
        serial_port = args.port or detect_serial_port()

    if serial_port and not args.mock:
        print(f"Targeting serial port: {cyan(serial_port)} at {args.baud} baud...")
        reader = SerialReader(port=serial_port, baudrate=args.baud)
        try:
            reader.start()
            source = reader
            is_live_serial = True
            print(green(f"  Connected to live hardware on {serial_port}!"))
        except Exception as exc:
            print(yellow(f"  [WARN] Failed to open {serial_port} ({exc}). Falling back to MockSource."))
            source = None

    if source is None:
        print(cyan("Using simulated telemetry generator (io.MockSource)..."))
        mock = MockSource(update_rate=20.0, enable_noise=True, enable_disturbances=True, seed=42)
        mock.start()
        source = mock

    mode_label = f"LIVE UART ({serial_port})" if is_live_serial else "MOCK SIMULATION (~20Hz)"
    print(f"Telemetry Mode : {bold(mode_label)}")
    print(f"RL Init Voltage: {bold(args.v_init)}")
    print(f"Max RL Steps   : {bold(str(args.max_steps))}")
    print(f"Target VSWR    : {green('< 1.20')}")
    print("Press Ctrl+C to stop and display summary statistics.\n")

    # Table Header
    print("┌" + "─" * 110 + "┐")
    print(
        f"│ {'#':<3} │ {'MEASURED LOAD':<13} │ {'PHYSICS EXACT':<17} │ {'RL BEST-SO-FAR':<17} │ {'ROLLOUT TRACE':<20} │ {'ACCURACY / MATCH':<23} │"
    )
    print(
        f"│ {'':<3} │ {'RL(Ω)  XL(Ω)':<13} │ {'V1*  V2*   VSWR*':<17} │ {'V1   V2    BEST':<17} │ {'INIT   FINAL  STP C':<20} │ {'ΔV1   ΔV2   STATUS':<23} │"
    )
    print("├" + "─" * 110 + "┤")

    # Metrics aggregation
    frame_count = 0
    success_count = 0
    cycle_frames = 0
    total_vswr_gap = 0.0
    total_init_vswr = 0.0
    total_final_vswr = 0.0
    total_best_vswr = 0.0
    total_v1_err = 0.0
    total_v2_err = 0.0
    total_rl_steps = 0
    t_min_interval = 1.0 / max(0.1, args.rate) if args.rate > 0 else 0.0
    last_display_time = 0.0

    try:
        while True:
            # Poll latest measurement
            raw_data = source.get_latest_raw() if hasattr(source, "get_latest_raw") else {}
            data = source.get_latest()

            now = time.time()
            if t_min_interval > 0 and (now - last_display_time) < t_min_interval:
                time.sleep(0.02)
                continue
            last_display_time = now

            rl = float(data["RL"])
            xl = float(data["XL"])
            v1_meas = float(data["V1_man"])
            v2_meas = float(data["V2_man"])

            # 3. Physics Engine Ground Truth (grid_search_match)
            phys_opt = grid_search_match(rl, xl, grid_size=34)
            v1_opt = float(phys_opt["best_V1"])
            v2_opt = float(phys_opt["best_V2"])
            vswr_opt = float(phys_opt["best_vswr"])

            # 4. RL Agent Autonomous Prediction (Python simulation)
            v1_start = v1_meas if args.v_init == "telemetry" else 1.65
            v2_start = v2_meas if args.v_init == "telemetry" else 1.65
            rl_res = run_rl_tuning(
                agent=agent,
                rl=rl,
                xl=xl,
                v1_init=v1_start,
                v2_init=v2_start,
                max_steps=args.max_steps,
                success_vswr=1.20,
            )
            v1_rl = float(rl_res["best_v1"])
            v2_rl = float(rl_res["best_v2"])
            vswr_best = float(rl_res["best_vswr"])
            vswr_final = float(rl_res["final_vswr"])
            vswr_init = float(rl_res["initial_vswr"])
            steps = int(rl_res["steps"])
            cycle_detected = bool(rl_res["cycle_detected"])

            # 5. Accuracy & Error Calculations (evaluated on returned best-so-far state)
            dv1 = abs(v1_rl - v1_opt)
            dv2 = abs(v2_rl - v2_opt)
            vswr_gap = max(0.0, vswr_best - vswr_opt)

            # Match status categorization
            if vswr_best < 1.20:
                status_str = _pad_color("MATCHED", 11, green)
                success_count += 1
            elif vswr_opt >= 1.5:
                status_str = _pad_color("DIFF. LOAD", 11, yellow)
                if abs(vswr_best - vswr_opt) < 0.15:
                    success_count += 1
            else:
                status_str = _pad_color("SUBOPTIMAL", 11, red)

            # Running metrics
            frame_count += 1
            if cycle_detected:
                cycle_frames += 1
            total_v1_err += dv1
            total_v2_err += dv2
            total_vswr_gap += vswr_gap
            total_init_vswr += vswr_init
            total_final_vswr += vswr_final
            total_best_vswr += vswr_best
            total_rl_steps += steps

            # Format row
            load_str = f"{rl:5.1f} {xl:+6.1f}"
            phys_str = f"{v1_opt:4.2f} {v2_opt:4.2f} {vswr_opt:6.2f}"
            rl_str = f"{v1_rl:4.2f} {v2_rl:4.2f} {vswr_best:6.2f}"
            trace_str = (
                f"{vswr_init:6.2f} {vswr_final:6.2f} {steps:3d} "
                f"{'Y' if cycle_detected else 'N'}"
            )
            err_str = f"{dv1:4.2f}  {dv2:4.2f}  {status_str}"

            print(
                f"│ {frame_count:3d} │ {load_str:<13} │ {phys_str:<17} │ {rl_str:<17} │ {trace_str:<20} │ {err_str} │"
            )

            if args.count > 0 and frame_count >= args.count:
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n" + yellow("[INFO] Interrupted by user."))
    finally:
        source.stop()

    print("└" + "─" * 110 + "┘")

    # 6. Final Summary Statistics
    if frame_count > 0:
        avg_v1_err = total_v1_err / frame_count
        avg_v2_err = total_v2_err / frame_count
        avg_vswr_gap = total_vswr_gap / frame_count
        avg_init_vswr = total_init_vswr / frame_count
        avg_final_vswr = total_final_vswr / frame_count
        avg_best_vswr = total_best_vswr / frame_count
        avg_steps = total_rl_steps / frame_count
        match_rate = (success_count / frame_count) * 100.0
        cycle_rate = (cycle_frames / frame_count) * 100.0

        print("\n" + bold("=" * 112))
        print(bold("                                  EVALUATION ACCURACY SUMMARY"))
        print(bold("=" * 112))
        print(f"  Total Frames Evaluated     : {bold(str(frame_count))} (max_steps={args.max_steps})")
        print(f"  Match Success Rate         : {green(f'{match_rate:.1f}%')} (best-so-far VSWR < 1.20 or near-optimal)")
        print(f"  Average Mean Voltage Error : ΔV1 = {cyan(f'{avg_v1_err:.3f} V')}, ΔV2 = {cyan(f'{avg_v2_err:.3f} V')}")
        print(
            f"  Average VSWR Progression   : Initial = {cyan(f'{avg_init_vswr:.4f}')}, "
            f"Final (end) = {cyan(f'{avg_final_vswr:.4f}')}, Best-So-Far (returned) = {green(f'{avg_best_vswr:.4f}')}"
        )
        print(f"  Average Suboptimality Gap  : {cyan(f'{avg_vswr_gap:.4f}')} (VSWR_best - VSWR_opt)")
        print(f"  Average RL Action Steps    : {cyan(f'{avg_steps:.1f}')} steps per tuning cycle")
        print(f"  Cycles Detected & Escaped  : {cyan(f'{cycle_frames}/{frame_count} ({cycle_rate:.1f}%)')}")
        print("  Control Mode Note          : Tuning runs purely in Python simulation (voltages are not sent to STM32).")
        print("=" * 112 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
