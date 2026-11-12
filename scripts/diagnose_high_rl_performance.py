#!/usr/bin/env python3
"""M4 High Resistance Region Performance Diagnostic Script.

Investigates performance deterioration near the upper resistance boundary (RL = 16-17 Ohms)
observed during live UART tests:
1. Evaluates >= 100 deterministic distinct loads across four resistance bands:
   - Band 1: 10.0 to 12.0 Ohms
   - Band 2: 12.0 to 14.0 Ohms
   - Band 3: 14.0 to 16.0 Ohms
   - Band 4: 16.0 to 17.0 Ohms
2. Compares DQN final VSWR against physics grid-search optimum.
3. Tests both standard live-monitor horizon (max_steps = 35) and extended horizon (max_steps = 100)
   to distinguish step-truncation artifacts from true policy failure.
4. Identifies failure modes (step budget exhaustion, limit cycle oscillation, boundary pinning).
5. Exports per-load CSV (`results/high_rl_investigation_results.csv`) and
   summary JSON (`results/high_rl_investigation_summary.json`).

Usage:
    python scripts/diagnose_high_rl_performance.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

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
from physics.pi_network import compute, grid_search_match
from rl.agent import load_agent, InferenceAgent
from rl.env import ACTION_NAMES, ImpedanceMatchingEnv, _clamp_voltage

BANDS = [
    ("10-12 Ω", 10.0, 12.0),
    ("12-14 Ω", 12.0, 14.0),
    ("14-16 Ω", 14.0, 16.0),
    ("16-17 Ω", 16.0, 17.0),
]


def generate_stratified_loads(
    loads_per_band: int = 25,
    xl_range: tuple[float, float] = (62.0, 77.0),
    seed: int = 42,
) -> list[tuple[str, float, float]]:
    """Generate deterministic, stratified continuous loads across all 4 resistance bands."""
    rng = np.random.default_rng(seed)
    stratified: list[tuple[str, float, float]] = []
    global_set: set[tuple[float, float]] = set()

    for label, r_min, r_max in BANDS:
        band_count = 0
        # Anchors: band center and midpoint of corners to avoid exact boundary sharing
        anchors = [
            (r_min + 0.05, xl_range[0] + 0.1),
            (r_min + 0.05, xl_range[1] - 0.1),
            (r_max - 0.05, xl_range[0] + 0.1),
            (r_max - 0.05, xl_range[1] - 0.1),
            (0.5 * (r_min + r_max), 0.5 * (xl_range[0] + xl_range[1])),
        ]
        for a_rl, a_xl in anchors:
            key = (round(a_rl, 4), round(a_xl, 4))
            if key not in global_set and band_count < loads_per_band:
                global_set.add(key)
                stratified.append((label, key[0], key[1]))
                band_count += 1

        while band_count < loads_per_band:
            rl = round(float(rng.uniform(r_min, r_max)), 4)
            xl = round(float(rng.uniform(xl_range[0], xl_range[1])), 4)
            key = (rl, xl)
            if key not in global_set:
                global_set.add(key)
                stratified.append((label, rl, xl))
                band_count += 1

    return stratified


def run_rollout(
    agent: InferenceAgent,
    rl: float,
    xl: float,
    v1_init: float = 1.65,
    v2_init: float = 1.65,
    max_steps: int = 100,
) -> dict[str, Any]:
    """Execute greedy rollout and capture trajectory diagnostic metrics."""
    v1 = _clamp_voltage(v1_init)
    v2 = _clamp_voltage(v2_init)

    visited: set[tuple[float, float]] = {(v1, v2)}
    traj: list[tuple[float, float, float, int]] = []
    oscillation_detected = False
    boundary_pinning_detected = False
    premature_hold_detected = False
    step_at_success: int | None = None

    vswr_at_35 = None
    v1_at_35 = None
    v2_at_35 = None

    for s in range(max_steps):
        sim = compute(rl, xl, v1, v2)
        vswr = float(sim["vswr"])
        gamma = sim["gamma"]

        if s == 35:
            vswr_at_35 = vswr
            v1_at_35 = v1
            v2_at_35 = v2

        if vswr < 1.20 and step_at_success is None:
            step_at_success = s

        # Terminate if matched
        if vswr < 1.20:
            break

        obs = np.array([gamma.real, gamma.imag, v1, v2], dtype=np.float32)
        action = int(agent.act(obs))
        traj.append((v1, v2, vswr, action))

        if action == 4:
            premature_hold_detected = True
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

        # Boundary hit check
        if v1 == v1_prev and v2 == v2_prev:
            boundary_pinning_detected = True
            break

        # Oscillation check
        if (v1, v2) in visited:
            oscillation_detected = True
            break
        visited.add((v1, v2))

    final_sim = compute(rl, xl, v1, v2)
    final_vswr = float(final_sim["vswr"])

    if vswr_at_35 is None:
        vswr_at_35 = final_vswr
        v1_at_35 = v1
        v2_at_35 = v2

    total_steps = len(traj)
    matched_at_35 = (step_at_success is not None and step_at_success <= 35) or (vswr_at_35 < 1.20)
    matched_at_final = final_vswr < 1.20

    # Determine failure mode if failed at max_steps=35
    failure_mode_35 = "None (Matched)"
    if not matched_at_35:
        if matched_at_final:
            failure_mode_35 = "Step Truncation (Needed >35 steps)"
        elif oscillation_detected:
            failure_mode_35 = "Limit Cycle Oscillation"
        elif boundary_pinning_detected:
            failure_mode_35 = "Boundary Pinning"
        elif premature_hold_detected:
            failure_mode_35 = "Premature Hold"
        else:
            failure_mode_35 = "Suboptimal Convergence"

    return {
        "v1_final": v1,
        "v2_final": v2,
        "vswr_final": final_vswr,
        "gamma_abs_final": float(final_sim["gamma_abs"]),
        "steps_total": total_steps,
        "matched_final": matched_at_final,
        "v1_at_35": v1_at_35,
        "v2_at_35": v2_at_35,
        "vswr_at_35": vswr_at_35,
        "matched_at_35": matched_at_35,
        "failure_mode_35": failure_mode_35,
        "oscillation": oscillation_detected,
        "boundary_pinning": boundary_pinning_detected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 High Resistance Region Diagnostic")
    parser.add_argument("--loads-per-band", type=int, default=25, help="Loads per resistance band (default: 25)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--out-dir", type=str, default="results", help="Output directory (default: results)")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = PROJECT_ROOT / "models" / "model.pt"
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
        return 1

    print("=" * 86)
    print("  M4 PERFORMANCE INVESTIGATION: UPPER RESISTANCE BOUNDARY (RL = 16-17 Ω)")
    print("=" * 86)
    print(f"Generating {args.loads_per_band * 4} stratified loads across 4 bands (seed={args.seed})...")

    loads = generate_stratified_loads(loads_per_band=args.loads_per_band, seed=args.seed)
    print(f"Total loads generated: {len(loads)} (distinct: {len(set((l[1], l[2]) for l in loads))})")

    agent = load_agent(str(model_path), device="cpu")
    print(f"Model loaded: {agent.model_path}")

    # Inspect normalization buffers
    input_center = agent.online_net.input_center.cpu().numpy()
    input_scale = agent.online_net.input_scale.cpu().numpy()
    print(f"Model normalization: center={input_center}, scale={input_scale} (verified identical to training)")

    per_load_results: list[dict[str, Any]] = []
    band_records: dict[str, list[dict[str, Any]]] = {b[0]: [] for b in BANDS}

    t0 = time.perf_counter()
    for idx, (band, rl, xl) in enumerate(loads):
        opt = grid_search_match(rl, xl, grid_size=67)
        opt_v1 = float(opt["best_V1"])
        opt_v2 = float(opt["best_V2"])
        opt_vswr = float(opt["best_vswr"])

        rollout = run_rollout(agent, rl, xl, v1_init=1.65, v2_init=1.65, max_steps=100)

        manhattan_dist_steps = round(abs(opt_v1 - 1.65) / 0.05 + abs(opt_v2 - 1.65) / 0.05)
        vswr_gap_final = max(0.0, rollout["vswr_final"] - opt_vswr)
        vswr_gap_35 = max(0.0, rollout["vswr_at_35"] - opt_vswr)

        rec = {
            "load_idx": idx + 1,
            "band": band,
            "rl": rl,
            "xl": xl,
            "opt_v1": opt_v1,
            "opt_v2": opt_v2,
            "opt_vswr": opt_vswr,
            "manhattan_dist_steps": manhattan_dist_steps,
            "dqn_v1_35": rollout["v1_at_35"],
            "dqn_v2_35": rollout["v2_at_35"],
            "dqn_vswr_35": rollout["vswr_at_35"],
            "matched_35": rollout["matched_at_35"],
            "vswr_gap_35": vswr_gap_35,
            "failure_mode_35": rollout["failure_mode_35"],
            "dqn_v1_final": rollout["v1_final"],
            "dqn_v2_final": rollout["v2_final"],
            "dqn_vswr_final": rollout["vswr_final"],
            "matched_final": rollout["matched_final"],
            "steps_total": rollout["steps_total"],
            "vswr_gap_final": vswr_gap_final,
            "oscillation": rollout["oscillation"],
            "boundary_pinning": rollout["boundary_pinning"],
        }
        per_load_results.append(rec)
        band_records[band].append(rec)

    elapsed = time.perf_counter() - t0
    print(f"Evaluated {len(loads)} loads in {elapsed:.2f}s ({elapsed/len(loads)*1000.0:.1f} ms/load)\n")

    # Aggregate Band Statistics
    band_summaries: dict[str, Any] = {}
    print("=" * 86)
    print("                     PER-BAND PERFORMANCE COMPARISON")
    print("=" * 86)
    print(
        f"{'Resistance Band':<15} | {'N':<4} | {'@35 Steps':<11} | {'@100 Steps':<11} | "
        f"{'Mean VSWR@35':<13} | {'Mean VSWR@100':<13} | {'Mean Gap@100':<12} | {'Req. Steps':<10}"
    )
    print("-" * 86)

    for band, _, _ in BANDS:
        recs = band_records[band]
        n = len(recs)
        sr35 = sum(1 for r in recs if r["matched_35"]) / n * 100.0
        sr100 = sum(1 for r in recs if r["matched_final"]) / n * 100.0
        mean_vswr35 = float(np.mean([r["dqn_vswr_35"] for r in recs]))
        mean_vswr100 = float(np.mean([r["dqn_vswr_final"] for r in recs]))
        mean_gap100 = float(np.mean([r["vswr_gap_final"] for r in recs]))
        mean_req_steps = float(np.mean([r["manhattan_dist_steps"] for r in recs]))
        mean_actual_steps = float(np.mean([r["steps_total"] for r in recs]))

        # Count failure modes at 35 steps
        trunc_count = sum(1 for r in recs if "Step Truncation" in r["failure_mode_35"])
        osc_count = sum(1 for r in recs if r["oscillation"])

        band_summaries[band] = {
            "num_loads": n,
            "success_rate_at_35_steps": round(sr35, 1),
            "success_rate_at_100_steps": round(sr100, 1),
            "mean_vswr_at_35_steps": round(mean_vswr35, 4),
            "mean_vswr_at_100_steps": round(mean_vswr100, 4),
            "mean_optimality_gap_at_100_steps": round(mean_gap100, 4),
            "mean_theoretical_manhattan_steps": round(mean_req_steps, 1),
            "mean_actual_steps_taken": round(mean_actual_steps, 1),
            "truncation_failures_at_35": trunc_count,
            "oscillation_failures": osc_count,
        }

        print(
            f"{band:<15} | {n:<4d} | {sr35:>9.1f}%  | {sr100:>9.1f}%  | "
            f"{mean_vswr35:>13.3f} | {mean_vswr100:>13.3f} | {mean_gap100:>12.4f} | {mean_req_steps:>10.1f}"
        )
    print("=" * 86)

    # Identify Worst-Performing Loads (Top 5 Overall by final VSWR gap)
    worst_loads = sorted(per_load_results, key=lambda r: r["vswr_gap_final"], reverse=True)[:5]
    print("\nTop 5 Worst-Performing Loads (Highest Optimality Gap):")
    print(f"  {'RL (Ω)':<7} {'XL (Ω)':<8} {'Band':<10} {'Opt V1,V2':<12} {'Opt VSWR':<10} {'DQN V1,V2':<12} {'DQN VSWR':<10} {'Gap':<8} {'Mode'}")
    print("  " + "-" * 88)
    for w in worst_loads:
        opt_v_str = f"{w['opt_v1']:.2f}, {w['opt_v2']:.2f}"
        dqn_v_str = f"{w['dqn_v1_final']:.2f}, {w['dqn_v2_final']:.2f}"
        print(
            f"  {w['rl']:<7.2f} {w['xl']:<+8.2f} {w['band']:<10} {opt_v_str:<12} {w['opt_vswr']:<10.3f} "
            f"{dqn_v_str:<12} {w['dqn_vswr_final']:<10.3f} {w['vswr_gap_final']:<8.3f} {w['failure_mode_35']}"
        )
    print("=" * 86 + "\n")

    # Export CSV
    csv_path = out_dir / "high_rl_investigation_results.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_load_results[0].keys()))
        writer.writeheader()
        writer.writerows(per_load_results)
    print(f"Saved detailed per-load results to: {csv_path}")

    # Export JSON
    json_path = out_dir / "high_rl_investigation_summary.json"
    summary_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": str(model_path.resolve()),
            "total_loads": len(loads),
            "seed": args.seed,
            "bands": [b[0] for b in BANDS],
        },
        "band_performance": band_summaries,
        "worst_loads": [
            {
                "rl": w["rl"],
                "xl": w["xl"],
                "band": w["band"],
                "opt_v1": w["opt_v1"],
                "opt_v2": w["opt_v2"],
                "opt_vswr": round(w["opt_vswr"], 3),
                "dqn_v1": w["dqn_v1_final"],
                "dqn_v2": w["dqn_v2_final"],
                "dqn_vswr": round(w["dqn_vswr_final"], 3),
                "vswr_gap": round(w["vswr_gap_final"], 3),
                "steps": w["steps_total"],
                "failure_mode_35": w["failure_mode_35"],
            }
            for w in worst_loads
        ],
    }
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary JSON to: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
