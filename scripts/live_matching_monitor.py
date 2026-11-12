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
5. Runs the trained M4 Double DQN RL agent (models/model.pt) starting from
   the initial state to predict/tune the final varactor voltages (V1_rl, V2_rl).
6. Compares physics ground-truth optimal vs RL predicted voltages and VSWR,
   calculating voltage error and matching accuracy metrics live.

Usage:
    # Live hardware (auto-detect STM32 or fallback to mock)
    python scripts/live_matching_monitor.py

    # Explicit serial port
    python scripts/live_matching_monitor.py --port COM3

    # Force mock simulation mode
    python scripts/live_matching_monitor.py --mock

    # Run for a fixed number of frames (e.g. 10 frames)
    python scripts/live_matching_monitor.py --count 10
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

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
def run_rl_tuning(
    agent: InferenceAgent,
    rl: float,
    xl: float,
    v1_init: float,
    v2_init: float,
    max_steps: int = 40,
    success_vswr: float = 1.20,
) -> dict[str, Any]:
    """Run greedy RL agent rollout to autonomous convergence.

    Args:
        agent: Trained inference agent.
        rl: Load resistance in Ohms.
        xl: Load reactance in Ohms.
        v1_init: Initial V1 bias voltage in Volts.
        v2_init: Initial V2 bias voltage in Volts.
        max_steps: Maximum action iterations.
        success_vswr: Target VSWR threshold for early termination.

    Returns:
        dict containing final voltages, VSWR, steps taken, and trajectory summary.
    """
    v1 = _clamp_voltage(v1_init)
    v2 = _clamp_voltage(v2_init)

    visited_states: set[tuple[float, float]] = {(v1, v2)}
    steps_taken = 0
    t0 = time.perf_counter()

    for step in range(max_steps):
        sim = compute(rl, xl, v1, v2)
        vswr = float(sim["vswr"])
        gamma = sim["gamma"]

        # Goal reached
        if vswr < success_vswr:
            break

        obs = np.array([gamma.real, gamma.imag, v1, v2], dtype=np.float32)
        action = int(agent.act(obs))
        steps_taken += 1

        # Hold action
        if action == 4:
            break

        v1_prev, v2_prev = v1, v2
        if action == 0:
            v1 = _clamp_voltage(v1 + config.V_STEP)
        elif action == 1:
            v1 = _clamp_voltage(v1 - config.V_STEP)
        elif action == 2:
            v2 = _clamp_voltage(v2 + config.V_STEP)
        elif action == 3:
            v2 = _clamp_voltage(v2 - config.V_STEP)

        # Detect boundary limit hit
        if v1 == v1_prev and v2 == v2_prev:
            break

        # Detect limit cycle / oscillation
        if (v1, v2) in visited_states:
            break
        visited_states.add((v1, v2))

    t_eval_ms = (time.perf_counter() - t0) * 1000.0
    final_sim = compute(rl, xl, v1, v2)
    return {
        "v1": float(v1),
        "v2": float(v2),
        "vswr": float(final_sim["vswr"]),
        "gamma_abs": float(final_sim["gamma_abs"]),
        "gamma": final_sim["gamma"],
        "steps": steps_taken,
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
        "--model",
        type=str,
        default="models/model.pt",
        help="Path to trained RL model checkpoint (default: models/model.pt).",
    )
    args = parser.parse_args()

    # 1. Load trained RL agent
    model_path = PROJECT_ROOT / args.model
    if not model_path.exists():
        print(red(f"[ERROR] Trained model checkpoint not found at: {model_path}"))
        return 1

    print(bold("=" * 86))
    print(bold("  AUTONOMOUS IMPEDANCE MATCHING NETWORK — LIVE ACCURACY MONITOR"))
    print(bold("  M1 (Firmware) -> M2 (UART/ADC) -> M3 (Physics) -> M4 (RL Agent)"))
    print(bold("=" * 86))
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
    print(f"Target VSWR    : {green('< 1.20')}")
    print("Press Ctrl+C to stop and display summary statistics.\n")

    # Table Header
    print("┌" + "─" * 84 + "┐")
    print(
        f"│ {'#':<3} │ {'MEASURED LOAD':<15} │ {'PHYSICS EXACT':<16} │ {'RL PREDICTED':<17} │ {'ACCURACY / MATCH':<20} │"
    )
    print(
        f"│     │ {'RL(Ω)   XL(Ω)':<15} │ {'V1*   V2*  VSWR*':<16} │ {'V1   V2   VSWR':<17} │ {'ΔV1   ΔV2   STATUS':<20} │"
    )
    print("├" + "─" * 84 + "┤")

    # Metrics aggregation
    frame_count = 0
    success_count = 0
    total_vswr_gap = 0.0
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

            # 4. RL Agent Autonomous Prediction
            v1_start = v1_meas if args.v_init == "telemetry" else 1.65
            v2_start = v2_meas if args.v_init == "telemetry" else 1.65
            rl_res = run_rl_tuning(
                agent=agent,
                rl=rl,
                xl=xl,
                v1_init=v1_start,
                v2_init=v2_start,
                max_steps=35,
                success_vswr=1.20,
            )
            v1_rl = rl_res["v1"]
            v2_rl = rl_res["v2"]
            vswr_rl = rl_res["vswr"]
            steps = rl_res["steps"]

            # 5. Accuracy & Error Calculations
            dv1 = abs(v1_rl - v1_opt)
            dv2 = abs(v2_rl - v2_opt)
            vswr_gap = max(0.0, vswr_rl - vswr_opt)

            # Match status categorization
            is_matched = vswr_rl < 1.20 or abs(vswr_rl - vswr_opt) < 0.05
            if vswr_rl < 1.20:
                status_str = green("MATCHED")
                success_count += 1
            elif vswr_opt >= 1.5:
                status_str = yellow("DIFF. LOAD")
                if abs(vswr_rl - vswr_opt) < 0.15:
                    success_count += 1
            else:
                status_str = red("SUBOPTIMAL")

            # Running metrics
            frame_count += 1
            total_v1_err += dv1
            total_v2_err += dv2
            total_vswr_gap += vswr_gap
            total_rl_steps += steps

            # Format row
            load_str = f"{rl:5.1f} {xl:+6.1f}"
            phys_str = f"{v1_opt:4.2f} {v2_opt:4.2f} {vswr_opt:5.2f}"
            rl_str = f"{v1_rl:4.2f} {v2_rl:4.2f} {vswr_rl:5.2f}"
            err_str = f"{dv1:4.2f}  {dv2:4.2f}  {status_str}"

            print(
                f"│ {frame_count:3d} │ {load_str:<15} │ {phys_str:<16} │ {rl_str:<17} │ {err_str:<29} │"
            )

            if args.count > 0 and frame_count >= args.count:
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n" + yellow("[INFO] Interrupted by user."))
    finally:
        source.stop()

    print("└" + "─" * 84 + "┘")

    # 6. Final Summary Statistics
    if frame_count > 0:
        avg_v1_err = total_v1_err / frame_count
        avg_v2_err = total_v2_err / frame_count
        avg_vswr_gap = total_vswr_gap / frame_count
        avg_steps = total_rl_steps / frame_count
        match_rate = (success_count / frame_count) * 100.0

        print("\n" + bold("=" * 86))
        print(bold("                     EVALUATION ACCURACY SUMMARY"))
        print(bold("=" * 86))
        print(f"  Total Frames Evaluated     : {bold(str(frame_count))}")
        print(f"  Match Success Rate         : {green(f'{match_rate:.1f}%')} (VSWR < 1.20 or near-optimal)")
        print(f"  Average Mean Voltage Error : ΔV1 = {cyan(f'{avg_v1_err:.3f} V')}, ΔV2 = {cyan(f'{avg_v2_err:.3f} V')}")
        print(f"  Average Suboptimality Gap  : {cyan(f'{avg_vswr_gap:.4f}')} (VSWR_rl - VSWR_opt)")
        print(f"  Average RL Action Steps    : {cyan(f'{avg_steps:.1f}')} steps per tuning cycle")
        print("=" * 86 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
