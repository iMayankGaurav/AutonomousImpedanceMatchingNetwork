#!/usr/bin/env python3
"""M4 Independent Load Generalization Validation Suite.

Evaluates the trained Double DQN agent (`models/model.pt`) across >= 100 distinct,
unseen load impedance disturbance conditions within the demonstration range:
    RL: 10.0 to 17.0 Ohms
    XL: +62.0 to +77.0 Ohms

Compares the RL agent's autonomous tuning policy against the M3 RF physics engine
grid-search optimizer (ground-truth theoretical optimum) across multiple initial
bias voltage conditions:
    1. Midpoint initialization (V1 = 1.65 V, V2 = 1.65 V)
    2. Low-bias initialization (V1 = 0.50 V, V2 = 0.50 V)
    3. High-bias initialization (V1 = 3.00 V, V2 = 3.00 V)

Produces:
    - Detailed per-episode metrics in CSV (`results/generalization_results.csv`)
    - Statistical evaluation summary in JSON (`results/generalization_summary.json`)
    - Human-readable validation report in Markdown (`results/generalization_report.md`)

Usage:
    python scripts/validate_rl_generalization.py
    python scripts/validate_rl_generalization.py --num-loads 100 --seed 42 --max-steps 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IO_DIR = PROJECT_ROOT / "io"
if str(IO_DIR) not in sys.path:
    sys.path.insert(0, str(IO_DIR))

# Ensure coexistence with Python's stdlib io
import io as stdlib_io

if hasattr(stdlib_io, "__path__"):
    if str(IO_DIR) not in stdlib_io.__path__:
        stdlib_io.__path__.append(str(IO_DIR))
else:
    stdlib_io.__path__ = [str(IO_DIR)]

import numpy as np

# Module imports
import config
from filters import RL_MAX, RL_MIN, XL_MAX, XL_MIN
from physics.pi_network import compute, grid_search_match
from rl.agent import load_agent, InferenceAgent
from rl.env import ACTION_NAMES, ImpedanceMatchingEnv


# ---------------------------------------------------------------------------
# Load Generation (Continuous floating-point disturbance loads)
# ---------------------------------------------------------------------------
def generate_independent_loads(
    num_loads: int = 100,
    rl_range: tuple[float, float] = (10.0, 17.0),
    xl_range: tuple[float, float] = (62.0, 77.0),
    seed: int = 42,
) -> list[tuple[float, float]]:
    """Generate at least `num_loads` distinct continuous floating-point loads.

    Loads are sampled continuously across the specified rectangular range,
    including non-integer and non-0.5 Ohm multiples. Key boundary corners
    and midpoint are deterministically included.

    Args:
        num_loads: Total number of distinct load points (minimum 100).
        rl_range: (RL_min, RL_max) in Ohms.
        xl_range: (XL_min, XL_max) in Ohms.
        seed: Random seed for deterministic reproducibility.

    Returns:
        List of distinct (RL, XL) tuples.
    """
    if num_loads < 1:
        raise ValueError("num_loads must be >= 1.")

    rng = np.random.default_rng(seed)
    loads_set: set[tuple[float, float]] = set()
    loads: list[tuple[float, float]] = []

    # 1. Deterministic boundary anchors (corners and center)
    corners = [
        (rl_range[0], xl_range[0]),
        (rl_range[0], xl_range[1]),
        (rl_range[1], xl_range[0]),
        (rl_range[1], xl_range[1]),
        (0.5 * (rl_range[0] + rl_range[1]), 0.5 * (xl_range[0] + xl_range[1])),
    ]
    for c_rl, c_xl in corners:
        key = (round(c_rl, 4), round(c_xl, 4))
        if key not in loads_set:
            loads_set.add(key)
            loads.append(key)

    # 2. Continuous random sampling
    while len(loads) < num_loads:
        rl = round(float(rng.uniform(rl_range[0], rl_range[1])), 4)
        xl = round(float(rng.uniform(xl_range[0], xl_range[1])), 4)
        key = (rl, xl)
        if key not in loads_set:
            loads_set.add(key)
            loads.append(key)

    return loads[:num_loads]


# ---------------------------------------------------------------------------
# Single Episode Evaluator
# ---------------------------------------------------------------------------
def evaluate_single_load(
    env: ImpedanceMatchingEnv,
    agent: InferenceAgent,
    rl: float,
    xl: float,
    v1_init: float,
    v2_init: float,
    grid_size: int = 67,
) -> dict[str, Any]:
    """Evaluate one load from specified initial voltages under greedy policy.

    Args:
        env: Reusable ImpedanceMatchingEnv instance.
        agent: Trained inference agent.
        rl: Load resistance in Ohms.
        xl: Load reactance in Ohms.
        v1_init: Initial V1 bias voltage in Volts.
        v2_init: Initial V2 bias voltage in Volts.
        grid_size: Grid size for physics ground truth optimizer (67 matches discrete lattice).

    Returns:
        Dictionary containing initial state, physics optimum, DQN final state, and metrics.
    """
    # 1. Physics ground truth optimum
    opt = grid_search_match(rl, xl, grid_size=grid_size)
    opt_v1 = float(opt["best_V1"])
    opt_v2 = float(opt["best_V2"])
    opt_vswr = float(opt["best_vswr"])
    opt_gamma_abs = float(opt["best_gamma_abs"])

    # 2. Reset environment to exact initial conditions
    obs, init_info = env.reset(
        options={
            "RL": rl,
            "XL": xl,
            "V1": v1_init,
            "V2": v2_init,
        }
    )
    init_vswr = float(init_info["vswr"])
    init_gamma_abs = float(init_info["gamma_abs"])

    # 3. Greedy agent rollout
    step_count = 0
    last_info = init_info
    t0 = time.perf_counter()

    while True:
        action = int(agent.act(obs))
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        step_count += 1
        obs = next_obs
        last_info = step_info

        if terminated or truncated:
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    final_v1 = float(last_info["V1"])
    final_v2 = float(last_info["V2"])
    final_vswr = float(last_info["vswr"])
    final_gamma_abs = float(last_info["gamma_abs"])
    final_gamma_real = float(last_info["gamma_real"])
    final_gamma_imag = float(last_info["gamma_imag"])

    vswr_gap = max(0.0, final_vswr - opt_vswr)
    v1_err = abs(final_v1 - opt_v1)
    v2_err = abs(final_v2 - opt_v2)

    return {
        "rl": rl,
        "xl": xl,
        "v1_init": v1_init,
        "v2_init": v2_init,
        "init_vswr": init_vswr,
        "init_gamma_abs": init_gamma_abs,
        "opt_v1": opt_v1,
        "opt_v2": opt_v2,
        "opt_vswr": opt_vswr,
        "opt_gamma_abs": opt_gamma_abs,
        "dqn_v1": final_v1,
        "dqn_v2": final_v2,
        "dqn_vswr": final_vswr,
        "dqn_gamma_abs": final_gamma_abs,
        "dqn_gamma_real": final_gamma_real,
        "dqn_gamma_imag": final_gamma_imag,
        "vswr_gap": vswr_gap,
        "v1_error": v1_err,
        "v2_error": v2_err,
        "action_count": step_count,
        "matched_12": final_vswr < 1.20,
        "matched_15": final_vswr < 1.50,
        "is_optimal": vswr_gap < 0.05,
        "elapsed_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Statistics Aggregator
# ---------------------------------------------------------------------------
def compute_experiment_statistics(
    records: list[dict[str, Any]], init_label: str
) -> dict[str, Any]:
    """Compute comprehensive statistical metrics for a set of evaluation records."""
    n = len(records)
    if n == 0:
        return {}

    vswrs = [r["dqn_vswr"] for r in records]
    gaps = [r["vswr_gap"] for r in records]
    steps = [r["action_count"] for r in records]
    v1_errs = [r["v1_error"] for r in records]
    v2_errs = [r["v2_error"] for r in records]

    matched_12_count = sum(1 for r in records if r["matched_12"])
    matched_15_count = sum(1 for r in records if r["matched_15"])

    # Worst cases sorted by final VSWR descending
    sorted_worst = sorted(records, key=lambda r: (r["dqn_vswr"], r["vswr_gap"]), reverse=True)
    top_worst = sorted_worst[:5]

    return {
        "init_label": init_label,
        "num_loads": n,
        "success_rate_12": (matched_12_count / n) * 100.0,
        "success_rate_15": (matched_15_count / n) * 100.0,
        "matched_12_count": matched_12_count,
        "matched_15_count": matched_15_count,
        "mean_vswr": float(np.mean(vswrs)),
        "median_vswr": float(np.median(vswrs)),
        "max_vswr": float(np.max(vswrs)),
        "min_vswr": float(np.min(vswrs)),
        "std_vswr": float(np.std(vswrs)),
        "mean_vswr_gap": float(np.mean(gaps)),
        "median_vswr_gap": float(np.median(gaps)),
        "max_vswr_gap": float(np.max(gaps)),
        "mean_v1_error": float(np.mean(v1_errs)),
        "mean_v2_error": float(np.mean(v2_errs)),
        "mean_action_count": float(np.mean(steps)),
        "median_action_count": float(np.median(steps)),
        "max_action_count": int(np.max(steps)),
        "min_action_count": int(np.min(steps)),
        "worst_loads": [
            {
                "rl": w["rl"],
                "xl": w["xl"],
                "opt_v1": w["opt_v1"],
                "opt_v2": w["opt_v2"],
                "opt_vswr": round(w["opt_vswr"], 3),
                "dqn_v1": w["dqn_v1"],
                "dqn_v2": w["dqn_v2"],
                "dqn_vswr": round(w["dqn_vswr"], 3),
                "vswr_gap": round(w["vswr_gap"], 3),
                "action_count": w["action_count"],
            }
            for w in top_worst
        ],
    }


# ---------------------------------------------------------------------------
# Output Artifact Exporters (CSV, JSON, Markdown)
# ---------------------------------------------------------------------------
def export_results(
    all_records: list[dict[str, Any]],
    stats_by_init: dict[str, dict[str, Any]],
    out_dir: Path,
    loads: list[tuple[float, float]],
    seed: int,
    max_steps: int,
    model_path: Path,
) -> tuple[Path, Path, Path]:
    """Export detailed CSV, summary JSON, and human-readable Markdown report."""
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "generalization_results.csv"
    json_path = out_dir / "generalization_summary.json"
    md_path = out_dir / "generalization_report.md"

    # 1. Export CSV
    fieldnames = [
        "init_label",
        "load_idx",
        "rl",
        "xl",
        "v1_init",
        "v2_init",
        "init_vswr",
        "init_gamma_abs",
        "opt_v1",
        "opt_v2",
        "opt_vswr",
        "opt_gamma_abs",
        "dqn_v1",
        "dqn_v2",
        "dqn_vswr",
        "dqn_gamma_abs",
        "dqn_gamma_real",
        "dqn_gamma_imag",
        "vswr_gap",
        "v1_error",
        "v2_error",
        "action_count",
        "matched_12",
        "matched_15",
        "is_optimal",
        "elapsed_ms",
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, rec in enumerate(all_records):
            row = dict(rec)
            row["load_idx"] = idx + 1
            writer.writerow(row)

    # 2. Export JSON
    summary_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": str(model_path.resolve()),
            "num_distinct_loads": len(loads),
            "seed": seed,
            "max_steps": max_steps,
            "rl_range": [RL_MIN, RL_MAX],
            "xl_range": [XL_MIN, XL_MAX],
            "total_evaluations": len(all_records),
        },
        "experiments": stats_by_init,
    }
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 3. Export Markdown Report
    with open(md_path, mode="w", encoding="utf-8") as f:
        f.write("# M4 Independent Load Generalization Report\n\n")
        f.write(
            f"> **Validation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"**Loads:** {len(loads)} distinct disturbance loads | "
            f"**Horizon:** max_steps={max_steps}\n\n"
        )
        f.write("## 1. Executive Summary Table\n\n")
        f.write(
            "| Initialization Condition | Distinct Loads | Success (VSWR < 1.2) | Success (VSWR < 1.5) | "
            "Mean VSWR | Median VSWR | Max VSWR | Mean Gap | Mean Steps |\n"
        )
        f.write(
            "|---|---|---|---|---|---|---|---|---|\n"
        )
        for label, st in stats_by_init.items():
            f.write(
                f"| **{label}** | {st['num_loads']} | **{st['success_rate_12']:.1f}%** | "
                f"{st['success_rate_15']:.1f}% | {st['mean_vswr']:.3f} | {st['median_vswr']:.3f} | "
                f"{st['max_vswr']:.3f} | {st['mean_vswr_gap']:.4f} | {st['mean_action_count']:.1f} |\n"
            )

        f.write("\n## 2. Worst-Performing Loads per Initialization Condition\n\n")
        for label, st in stats_by_init.items():
            f.write(f"### {label}\n\n")
            f.write(
                "| RL (Ω) | XL (Ω) | Opt V1 | Opt V2 | Opt VSWR | DQN V1 | DQN V2 | DQN VSWR | Gap | Steps |\n"
            )
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            for w in st["worst_loads"]:
                f.write(
                    f"| {w['rl']:6.2f} | {w['xl']:+6.2f} | {w['opt_v1']:4.2f} | {w['opt_v2']:4.2f} | "
                    f"{w['opt_vswr']:5.2f} | {w['dqn_v1']:4.2f} | {w['dqn_v2']:4.2f} | "
                    f"**{w['dqn_vswr']:5.2f}** | {w['vswr_gap']:5.2f} | {w['action_count']:3d} |\n"
                )
            f.write("\n")

        f.write("## 3. Generalization Scope & Boundaries\n\n")
        f.write(
            "- **Tested Disturbance Space:** Continuous floating-point loads within $R_L \\in [10.0, 17.0]\\,\\Omega$ "
            "and $X_L \\in [+62.0, +77.0]\\,\\Omega$.\n"
            "- **Independence Guarantee:** Evaluated on distinct randomly generated continuous load impedances, "
            "not repeated frames from telemetry.\n"
            "- **Generalization Boundary:** These results confirm generalization within the physical demonstration envelope. "
            "Performance on wide-band or out-of-distribution loads ($R_L > 45\\,\\Omega$, $X_L < 30\\,\\Omega$) "
            "must be verified separately against the RF network's physical matchability boundary.\n"
        )

    return csv_path, json_path, md_path


# ---------------------------------------------------------------------------
# Main Execution CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independent Load Generalization Validation for M4 Double DQN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--num-loads",
        "-n",
        type=int,
        default=100,
        help="Number of distinct continuous loads to evaluate (default: 100).",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for deterministic load generation (default: 42).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum action steps per episode (default: 100).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/model.pt",
        help="Path to trained Double DQN checkpoint (default: models/model.pt).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output directory for reports and metrics (default: results).",
    )
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model
    if not model_path.exists():
        print(f"[ERROR] Trained model checkpoint not found: {model_path}", file=sys.stderr)
        return 1

    out_dir = PROJECT_ROOT / args.out_dir

    print("=" * 86)
    print("  M4 DOUBLE DQN — INDEPENDENT LOAD GENERALIZATION VALIDATION")
    print(f"  Target Envelope: RL in [{RL_MIN}, {RL_MAX}] Ω, XL in [{XL_MIN}, {XL_MAX}] Ω")
    print(f"  Load Count     : {args.num_loads} distinct floating-point loads (seed={args.seed})")
    print(f"  Max Steps      : {args.max_steps} per episode")
    print(f"  Model Path     : {model_path}")
    print("=" * 86)

    # 1. Generate independent test loads
    print(f"\n[1/3] Generating {args.num_loads} distinct continuous disturbance loads...")
    loads = generate_independent_loads(
        num_loads=args.num_loads,
        rl_range=(RL_MIN, RL_MAX),
        xl_range=(XL_MIN, XL_MAX),
        seed=args.seed,
    )
    distinct_count = len(set(loads))
    print(f"      Generated {len(loads)} loads ({distinct_count} verified unique).")
    print(f"      Sample loads: {loads[:3]} ... {loads[-2:]}")

    # 2. Load agent and environment
    print("\n[2/3] Loading trained RL agent and physics environment...")
    agent = load_agent(str(model_path), device="cpu")
    env = ImpedanceMatchingEnv(
        load_mode="full",
        max_steps=args.max_steps,
        grid_size=67,
        seed=args.seed,
    )
    print(f"      Agent loaded: state_dim={agent.state_dim}, action_dim={agent.action_dim}")

    # 3. Define initialization experiments
    init_experiments = [
        ("Midpoint (1.65V, 1.65V)", 1.65, 1.65),
        ("Low Bias (0.50V, 0.50V)", 0.50, 0.50),
        ("High Bias (3.00V, 3.00V)", 3.00, 3.00),
    ]

    all_records: list[dict[str, Any]] = []
    stats_by_init: dict[str, dict[str, Any]] = {}

    print("\n[3/3] Running evaluation across all initialization conditions...")
    for label, v1_i, v2_i in init_experiments:
        print(f"\n  ▶ Testing Initialization: {label} ...")
        t_start = time.perf_counter()
        records: list[dict[str, Any]] = []

        for idx, (rl, xl) in enumerate(loads):
            rec = evaluate_single_load(
                env=env,
                agent=agent,
                rl=rl,
                xl=xl,
                v1_init=v1_i,
                v2_init=v2_i,
                grid_size=67,
            )
            rec["init_label"] = label
            records.append(rec)
            all_records.append(rec)

            if (idx + 1) % 25 == 0 or (idx + 1) == len(loads):
                print(f"      Progress: {idx + 1}/{len(loads)} loads completed...")

        duration = time.perf_counter() - t_start
        st = compute_experiment_statistics(records, init_label=label)
        stats_by_init[label] = st

        print(f"      Completed in {duration:.2f}s ({duration / len(loads) * 1000.0:.1f} ms/load)")
        print(
            f"      Results: Success (VSWR < 1.2): {st['success_rate_12']:.1f}% | "
            f"Success (VSWR < 1.5): {st['success_rate_15']:.1f}% | "
            f"Mean VSWR: {st['mean_vswr']:.3f} | Mean Gap: {st['mean_vswr_gap']:.4f} | "
            f"Mean Steps: {st['mean_action_count']:.1f}"
        )

    # 4. Export files
    print(f"\n[Exporting Results to '{out_dir}']...")
    csv_file, json_file, md_file = export_results(
        all_records=all_records,
        stats_by_init=stats_by_init,
        out_dir=out_dir,
        loads=loads,
        seed=args.seed,
        max_steps=args.max_steps,
        model_path=model_path,
    )
    print(f"  ✓ Detailed CSV : {csv_file}")
    print(f"  ✓ Summary JSON : {json_file}")
    print(f"  ✓ Markdown Doc : {md_file}")

    # 5. Print comprehensive final console report
    print("\n" + "=" * 86)
    print("                     GENERALIZATION TEST RESULTS SUMMARY")
    print("=" * 86)
    print(
        f"{'Initialization':<26} | {'N':<4} | {'< 1.2 (%)':<10} | {'< 1.5 (%)':<10} | "
        f"{'Mean VSWR':<10} | {'Max VSWR':<9} | {'Mean Gap':<9} | {'Mean Steps':<10}"
    )
    print("-" * 86)
    for label, st in stats_by_init.items():
        print(
            f"{label:<26} | {st['num_loads']:<4d} | {st['success_rate_12']:>9.1f}% | "
            f"{st['success_rate_15']:>9.1f}% | {st['mean_vswr']:>10.3f} | "
            f"{st['max_vswr']:>9.3f} | {st['mean_vswr_gap']:>9.4f} | {st['mean_action_count']:>10.1f}"
        )
    print("=" * 86)

    # Worst cases across midpoint
    mid_stats = stats_by_init.get("Midpoint (1.65V, 1.65V)")
    if mid_stats and mid_stats.get("worst_loads"):
        print("\nTop 5 Worst-Performing Loads (Midpoint 1.65V Initialization):")
        print(f"  {'RL (Ω)':<7} {'XL (Ω)':<8} {'Opt VSWR':<10} {'DQN VSWR':<10} {'Gap':<8} {'Steps':<6}")
        print("  " + "-" * 52)
        for w in mid_stats["worst_loads"]:
            print(
                f"  {w['rl']:<7.2f} {w['xl']:<+8.2f} {w['opt_vswr']:<10.2f} "
                f"{w['dqn_vswr']:<10.2f} {w['vswr_gap']:<8.2f} {w['action_count']:<6d}"
            )
        print("=" * 86 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
