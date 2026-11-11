"""M4: Held-out evaluation, baseline comparison, failure analysis, and plots.

Evaluates the trained Double DQN model (`models/model.pt`) without updating any
weights across the full 350-load held-out matchable split (`build_load_catalog`
with split_seed=42), comparing:
1. Random-action baseline (same initial states and load sequence)
2. Trained Double DQN agent (greedy policy, epsilon=0.0, torch.no_grad())
3. M3 brute-force grid-search reference optimizer (non-sequential oracle)

All outputs (CSV, JSON, markdown report, and plots) are saved inside `rl/`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "mplconfig_rl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from physics.pi_network import compute, grid_search_match
from rl.dqn import DoubleDQNAgent
from rl.env import ACTION_NAMES, DEFAULT_GRID_SIZE, ImpedanceMatchingEnv
from rl.train import (
    DEFAULT_FINAL_MODEL_PATH,
    DEFAULT_FIGURES_DIR,
    DEFAULT_LOGS_DIR,
    build_load_catalog,
    get_non_overwriting_path,
)

DEFAULT_TRAINING_METRICS_CSV = DEFAULT_LOGS_DIR / "double_dqn_metrics.csv"

RL_REGIONS: tuple[tuple[str, float, float], ...] = (
    ("10.0-14.5", 10.0, 15.0),
    ("15.0-19.5", 15.0, 20.0),
    ("20.0-24.5", 20.0, 25.0),
    ("25.0-30.0+", 25.0, float("inf")),
)

XL_REGIONS: tuple[tuple[str, float, float], ...] = (
    ("<55.0", float("-inf"), 55.0),
    ("55.0-64.5", 55.0, 65.0),
    ("65.0-74.5", 65.0, 75.0),
    ("75.0+", 75.0, float("inf")),
)


def _snapshot_weights_sha256(agent: DoubleDQNAgent) -> str:
    """Compute a SHA-256 digest of online and target network parameters."""
    h = hashlib.sha256()
    for net in (agent.online_net, agent.target_net):
        for param in net.parameters():
            h.update(param.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _finite_vswr_for_stats(vswr: float, cap: float = 1000.0) -> float:
    return float(vswr) if math.isfinite(vswr) else cap


def _generate_initial_voltages(
    num_loads: int,
    eval_seed: int,
) -> list[tuple[float, float]]:
    """Sample deterministic initial (V1, V2) pairs on the discrete 0.05 V lattice."""
    rng = np.random.default_rng(eval_seed)
    lattice = np.round(
        np.arange(config.V_MIN, config.V_MAX + 1e-9, config.V_STEP), 4
    )
    pairs: list[tuple[float, float]] = []
    for _ in range(num_loads):
        v1 = float(lattice[int(rng.integers(0, len(lattice)))])
        v2 = float(lattice[int(rng.integers(0, len(lattice)))])
        pairs.append((v1, v2))
    return pairs


def _verify_holdout_separation(
    train_loads: Sequence[tuple[float, float]],
    holdout_loads: Sequence[tuple[float, float]],
    training_metrics_csv: Optional[Path] = None,
) -> dict[str, Any]:
    """Verify holdout uniqueness and zero overlap with training pool and logged runs."""
    train_set = set(train_loads)
    holdout_set = set(holdout_loads)
    pool_overlap = train_set & holdout_set

    csv_overlap_count: Optional[int] = None
    csv_unique_loads: Optional[int] = None
    if training_metrics_csv is not None and training_metrics_csv.exists():
        with training_metrics_csv.open("r", newline="", encoding="utf-8") as f:
            csv_loads = {
                (round(float(row["RL"]), 4), round(float(row["XL"]), 4))
                for row in csv.DictReader(f)
            }
        csv_unique_loads = len(csv_loads)
        csv_overlap_count = len(csv_loads & holdout_set)

    if len(holdout_loads) != len(holdout_set):
        raise RuntimeError(
            f"Duplicate loads detected in holdout set: {len(holdout_loads)} vs {len(holdout_set)} unique."
        )
    if pool_overlap:
        raise RuntimeError(
            f"Train/holdout contamination detected: {len(pool_overlap)} overlapping loads."
        )
    if csv_overlap_count is not None and csv_overlap_count > 0:
        raise RuntimeError(
            f"Training CSV overlap detected with holdout loads: {csv_overlap_count} loads."
        )

    return {
        "train_pool_size": len(train_loads),
        "holdout_size": len(holdout_loads),
        "holdout_unique_count": len(holdout_set),
        "train_holdout_overlap": len(pool_overlap),
        "training_csv_unique_loads": csv_unique_loads,
        "training_csv_holdout_overlap": csv_overlap_count,
    }


def _run_sequential_episode(
    env: ImpedanceMatchingEnv,
    policy_fn: Callable[[np.ndarray], int],
    approach_name: str,
    load_idx: int,
    rl: float,
    xl: float,
    v1_init: float,
    v2_init: float,
    record_trajectory: bool = False,
) -> dict[str, Any]:
    """Run one sequential rollout from fixed (RL, XL, V1_init, V2_init) and record diagnostics."""
    obs, init_info = env.reset(options={"RL": rl, "XL": xl, "V1": v1_init, "V2": v2_init})
    initial_gamma_abs = float(init_info["gamma_abs"])
    initial_vswr = float(init_info["vswr"])
    best_gamma_abs = float(init_info["best_gamma_abs"])
    best_vswr = float(init_info["best_vswr"])
    best_v1 = float(init_info["best_V1"])
    best_v2 = float(init_info["best_V2"])

    action_counts = [0, 0, 0, 0, 0]
    boundary_hits = 0
    hold_steps = 0
    immediate_reversals = 0
    revisited_state_steps = 0
    total_reward = 0.0
    inference_time_s = 0.0
    prev_action: Optional[int] = None
    prev_v1 = float(init_info["V1"])
    prev_v2 = float(init_info["V2"])
    visited_states = {(prev_v1, prev_v2)}

    trajectory: list[dict[str, Any]] = []
    if record_trajectory:
        trajectory.append(
            {
                "step": 0,
                "action": None,
                "action_name": "INIT",
                "V1": prev_v1,
                "V2": prev_v2,
                "gamma_abs": initial_gamma_abs,
                "vswr": initial_vswr,
                "gamma_real": float(init_info["gamma_real"]),
                "gamma_imag": float(init_info["gamma_imag"]),
                "boundary_hit": False,
            }
        )

    last_info = init_info
    steps = 0
    t_ep_start = time.perf_counter()

    while True:
        t_inf0 = time.perf_counter()
        action = int(policy_fn(obs))
        inference_time_s += time.perf_counter() - t_inf0

        next_obs, reward, terminated, truncated, step_info = env.step(action)
        steps += 1
        total_reward += float(reward)
        action_counts[action] += 1

        next_v1 = float(step_info["V1"])
        next_v2 = float(step_info["V2"])

        is_boundary_hit = (action in (0, 1, 2, 3)) and (
            next_v1 == prev_v1 and next_v2 == prev_v2
        )
        if is_boundary_hit:
            boundary_hits += 1
        if action == 4:
            hold_steps += 1
        if prev_action is not None and (
            (prev_action == 0 and action == 1)
            or (prev_action == 1 and action == 0)
            or (prev_action == 2 and action == 3)
            or (prev_action == 3 and action == 2)
        ):
            immediate_reversals += 1
        if (next_v1, next_v2) in visited_states:
            revisited_state_steps += 1
        visited_states.add((next_v1, next_v2))

        if record_trajectory:
            trajectory.append(
                {
                    "step": steps,
                    "action": action,
                    "action_name": ACTION_NAMES[action],
                    "V1": next_v1,
                    "V2": next_v2,
                    "gamma_abs": float(step_info["gamma_abs"]),
                    "vswr": float(step_info["vswr"]),
                    "gamma_real": float(step_info["gamma_real"]),
                    "gamma_imag": float(step_info["gamma_imag"]),
                    "boundary_hit": is_boundary_hit,
                }
            )

        prev_action = action
        prev_v1 = next_v1
        prev_v2 = next_v2
        obs = next_obs
        last_info = step_info

        if terminated or truncated:
            break

    exec_time_s = time.perf_counter() - t_ep_start
    final_gamma_abs = float(last_info["gamma_abs"])
    final_vswr = float(last_info["vswr"])
    is_matched = bool(last_info["is_matched"])

    # Ineffective steps: staying stationary (boundary clamp or premature Hold) or oscillation reversals
    ineffective_actions = boundary_hits + hold_steps + immediate_reversals

    record: dict[str, Any] = {
        "approach": approach_name,
        "load_index": load_idx,
        "RL": float(rl),
        "XL": float(xl),
        "initial_V1": float(v1_init),
        "initial_V2": float(v2_init),
        "initial_gamma_abs": initial_gamma_abs,
        "initial_vswr": initial_vswr,
        "final_V1": float(last_info["V1"]),
        "final_V2": float(last_info["V2"]),
        "final_gamma_abs": final_gamma_abs,
        "final_vswr": final_vswr,
        "best_V1": best_v1,
        "best_V2": best_v2,
        "best_gamma_abs": best_gamma_abs,
        "best_vswr": best_vswr,
        "suboptimality_gap": float(final_gamma_abs - best_gamma_abs),
        "steps": steps,
        "is_matched": is_matched,
        "vswr_lt_2": bool(final_vswr < 2.0),
        "vswr_lt_1_5": bool(final_vswr < 1.5),
        "vswr_lt_1_2": bool(final_vswr < 1.2),
        "terminated": bool(last_info["terminated"]),
        "truncated": bool(last_info["truncated"]),
        "total_reward": float(total_reward),
        "execution_time_s": float(exec_time_s),
        "execution_time_ms": float(exec_time_s * 1000.0),
        "inference_time_ms": float(inference_time_s * 1000.0),
        "count_V1_plus": action_counts[0],
        "count_V1_minus": action_counts[1],
        "count_V2_plus": action_counts[2],
        "count_V2_minus": action_counts[3],
        "count_Hold": action_counts[4],
        "boundary_hits": boundary_hits,
        "hold_steps": hold_steps,
        "immediate_reversals": immediate_reversals,
        "revisited_state_steps": revisited_state_steps,
        "ineffective_actions": ineffective_actions,
    }
    if record_trajectory:
        record["trajectory"] = trajectory
    return record


def _run_grid_search_reference(
    load_idx: int,
    rl: float,
    xl: float,
    v1_init: float,
    v2_init: float,
    grid_size: int = DEFAULT_GRID_SIZE,
    success_vswr: float = 1.2,
    match_tolerance: float = 0.015,
) -> dict[str, Any]:
    """Run M3 brute-force 2D grid search as a non-sequential reference optimum."""
    init_sim = compute(rl, xl, v1_init, v2_init)
    init_gamma_abs = float(init_sim["gamma_abs"])
    init_vswr = float(init_sim["vswr"])

    t0 = time.perf_counter()
    res = grid_search_match(rl, xl, grid_size=grid_size)
    exec_time_s = time.perf_counter() - t0

    best_gamma_abs = float(res["best_gamma_abs"])
    best_vswr = float(res["best_vswr"])
    best_v1 = round(float(res["best_V1"]), 4)
    best_v2 = round(float(res["best_V2"]), 4)

    if best_vswr < success_vswr:
        is_matched = True
    elif best_vswr < 2.0:
        is_matched = best_gamma_abs <= best_gamma_abs + match_tolerance
    else:
        is_matched = False

    return {
        "approach": "grid_search",
        "load_index": load_idx,
        "RL": float(rl),
        "XL": float(xl),
        "initial_V1": float(v1_init),
        "initial_V2": float(v2_init),
        "initial_gamma_abs": init_gamma_abs,
        "initial_vswr": init_vswr,
        "final_V1": best_v1,
        "final_V2": best_v2,
        "final_gamma_abs": best_gamma_abs,
        "final_vswr": best_vswr,
        "best_V1": best_v1,
        "best_V2": best_v2,
        "best_gamma_abs": best_gamma_abs,
        "best_vswr": best_vswr,
        "suboptimality_gap": 0.0,
        "steps": None,
        "grid_evaluations": grid_size * grid_size,
        "is_matched": is_matched,
        "vswr_lt_2": bool(best_vswr < 2.0),
        "vswr_lt_1_5": bool(best_vswr < 1.5),
        "vswr_lt_1_2": bool(best_vswr < 1.2),
        "terminated": is_matched,
        "truncated": False,
        "total_reward": None,
        "execution_time_s": float(exec_time_s),
        "execution_time_ms": float(exec_time_s * 1000.0),
        "inference_time_ms": float(exec_time_s * 1000.0),
        "count_V1_plus": None,
        "count_V1_minus": None,
        "count_V2_plus": None,
        "count_V2_minus": None,
        "count_Hold": None,
        "boundary_hits": None,
        "hold_steps": None,
        "immediate_reversals": None,
        "revisited_state_steps": None,
        "ineffective_actions": None,
    }


def _summarize_approach(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics across a list of per-load records."""
    n = len(records)
    if n == 0:
        return {"count": 0}

    vswrs = np.array([_finite_vswr_for_stats(float(r["final_vswr"])) for r in records], dtype=np.float64)
    gammas = np.array([float(r["final_gamma_abs"]) for r in records], dtype=np.float64)
    subopt = np.array([float(r["suboptimality_gap"]) for r in records], dtype=np.float64)
    exec_ms = np.array([float(r["execution_time_ms"]) for r in records], dtype=np.float64)
    inf_ms = np.array([float(r["inference_time_ms"]) for r in records], dtype=np.float64)

    matched = [r for r in records if r["is_matched"]]
    timeouts = [r for r in records if r["truncated"]]
    is_sequential = records[0]["steps"] is not None

    summary: dict[str, Any] = {
        "count": n,
        "matched_count": len(matched),
        "timeout_count": len(timeouts),
        "is_matched_rate": float(len(matched) / n),
        "vswr_lt_2_rate": float(sum(1 for r in records if r["vswr_lt_2"]) / n),
        "vswr_lt_1_5_rate": float(sum(1 for r in records if r["vswr_lt_1_5"]) / n),
        "vswr_lt_1_2_rate": float(sum(1 for r in records if r["vswr_lt_1_2"]) / n),
        "final_vswr_mean": float(np.mean(vswrs)),
        "final_vswr_median": float(np.median(vswrs)),
        "final_vswr_min": float(np.min(vswrs)),
        "final_vswr_max": float(np.max(vswrs)),
        "final_vswr_p95": float(np.percentile(vswrs, 95)),
        "final_gamma_abs_mean": float(np.mean(gammas)),
        "final_gamma_abs_median": float(np.median(gammas)),
        "suboptimality_gap_mean": float(np.mean(subopt)),
        "suboptimality_gap_median": float(np.median(subopt)),
        "suboptimality_gap_max": float(np.max(subopt)),
        "exec_time_ms_mean": float(np.mean(exec_ms)),
        "exec_time_ms_total": float(np.sum(exec_ms)),
        "inference_time_ms_mean": float(np.mean(inf_ms)),
    }

    if is_sequential:
        all_steps = np.array([int(r["steps"]) for r in records], dtype=np.float64)
        matched_steps = (
            np.array([int(r["steps"]) for r in matched], dtype=np.float64)
            if matched
            else np.array([], dtype=np.float64)
        )
        total_steps = int(np.sum(all_steps))
        action_totals = {
            ACTION_NAMES[0]: int(sum(int(r["count_V1_plus"]) for r in records)),
            ACTION_NAMES[1]: int(sum(int(r["count_V1_minus"]) for r in records)),
            ACTION_NAMES[2]: int(sum(int(r["count_V2_plus"]) for r in records)),
            ACTION_NAMES[3]: int(sum(int(r["count_V2_minus"]) for r in records)),
            ACTION_NAMES[4]: int(sum(int(r["count_Hold"]) for r in records)),
        }
        boundary_hits = int(sum(int(r["boundary_hits"]) for r in records))
        hold_steps = int(sum(int(r["hold_steps"]) for r in records))
        reversals = int(sum(int(r["immediate_reversals"]) for r in records))
        revisits = int(sum(int(r["revisited_state_steps"]) for r in records))
        ineffective = int(sum(int(r["ineffective_actions"]) for r in records))

        summary.update(
            {
                "steps_all_mean": float(np.mean(all_steps)),
                "steps_all_median": float(np.median(all_steps)),
                "steps_to_match_mean": float(np.mean(matched_steps)) if matched else None,
                "steps_to_match_median": float(np.median(matched_steps)) if matched else None,
                "total_steps": total_steps,
                "action_counts": action_totals,
                "action_fractions": {
                    k: float(v / max(1, total_steps)) for k, v in action_totals.items()
                },
                "boundary_hits_total": boundary_hits,
                "boundary_hit_rate": float(boundary_hits / max(1, total_steps)),
                "episodes_with_boundary_hit": int(
                    sum(1 for r in records if int(r["boundary_hits"]) > 0)
                ),
                "hold_steps_total": hold_steps,
                "hold_rate": float(hold_steps / max(1, total_steps)),
                "immediate_reversals_total": reversals,
                "immediate_reversal_rate": float(reversals / max(1, total_steps)),
                "revisited_state_steps_total": revisits,
                "revisited_state_rate": float(revisits / max(1, total_steps)),
                "ineffective_actions_total": ineffective,
                "ineffective_action_rate": float(ineffective / max(1, total_steps)),
            }
        )
    else:
        summary.update(
            {
                "steps_all_mean": None,
                "steps_all_median": None,
                "steps_to_match_mean": None,
                "steps_to_match_median": None,
                "grid_evaluations_per_load": int(records[0]["grid_evaluations"]),
            }
        )

    return summary


def _summarize_by_regions(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Break down DQN performance across RL bins, XL bins, and achievable best_vswr regime."""

    def _bin_summary(
        key: str, bins: Sequence[tuple[str, float, float]]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for label, lo, hi in bins:
            sub = [r for r in records if lo <= float(r[key]) < hi]
            if not sub:
                continue
            stats = _summarize_approach(sub)
            stats["region"] = label
            stats["min_val"] = float(min(float(r[key]) for r in sub))
            stats["max_val"] = float(max(float(r[key]) for r in sub))
            out.append(stats)
        return out

    strict_sub = [r for r in records if float(r["best_vswr"]) < 1.2]
    tol_sub = [r for r in records if float(r["best_vswr"]) >= 1.2]
    regime_stats = []
    for label, sub in (("best_vswr < 1.2", strict_sub), ("1.2 <= best_vswr < 1.5", tol_sub)):
        if sub:
            s = _summarize_approach(sub)
            s["region"] = label
            regime_stats.append(s)

    return {
        "by_RL": _bin_summary("RL", RL_REGIONS),
        "by_XL": _bin_summary("XL", XL_REGIONS),
        "by_best_vswr_regime": regime_stats,
    }


def _investigate_late_training_and_holdout_failures(
    dqn_records: Sequence[dict[str, Any]],
    training_metrics_csv: Optional[Path] = DEFAULT_TRAINING_METRICS_CSV,
) -> dict[str, Any]:
    """Compare late-training timeouts (final third of training) with held-out failures."""
    late_train_timeouts: list[dict[str, Any]] = []
    if training_metrics_csv is not None and training_metrics_csv.exists():
        with training_metrics_csv.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            late_start_ep = (2 * len(rows)) // 3 + 1
            for row in rows:
                ep = int(row["episode"])
                if ep >= late_start_ep and row["is_matched"] not in ("1", "True", "true"):
                    late_train_timeouts.append(
                        {
                            "episode": ep,
                            "RL": float(row["RL"]),
                            "XL": float(row["XL"]),
                            "final_vswr": float(row["final_vswr"]),
                            "final_gamma_abs": float(row["final_gamma_abs"]),
                            "episode_length": int(row["episode_length"]),
                            "suboptimality_gap": float(row["suboptimality_gap"]),
                        }
                    )

    holdout_failures = [r for r in dqn_records if not r["is_matched"]]
    holdout_near_miss = [
        r for r in holdout_failures if float(r["final_vswr"]) < 1.5
    ]
    holdout_high_vswr = [
        r for r in holdout_failures if float(r["final_vswr"]) >= 1.5
    ]

    near_timeout_proximity: dict[str, Any] = {}
    if late_train_timeouts:
        train_coords = np.array(
            [[t["RL"], t["XL"]] for t in late_train_timeouts], dtype=np.float64
        )
        near_records: list[dict[str, Any]] = []
        far_records: list[dict[str, Any]] = []
        for r in dqn_records:
            pt = np.array([float(r["RL"]), float(r["XL"])], dtype=np.float64)
            min_dist = float(np.min(np.linalg.norm(train_coords - pt, axis=1)))
            if min_dist <= 2.5:
                near_records.append(r)
            else:
                far_records.append(r)

        near_timeout_proximity = {
            "late_train_timeout_RL_range": [
                float(np.min(train_coords[:, 0])),
                float(np.max(train_coords[:, 0])),
            ],
            "late_train_timeout_XL_range": [
                float(np.min(train_coords[:, 1])),
                float(np.max(train_coords[:, 1])),
            ],
            "holdout_near_2_5_ohms": _summarize_approach(near_records),
            "holdout_far_gt_2_5_ohms": _summarize_approach(far_records),
        }

    return {
        "late_training_timeout_count": len(late_train_timeouts),
        "late_training_timeouts": late_train_timeouts,
        "holdout_failure_count": len(holdout_failures),
        "holdout_near_miss_vswr_lt_1_5_count": len(holdout_near_miss),
        "holdout_high_vswr_ge_1_5_count": len(holdout_high_vswr),
        "holdout_failures": [
            {
                "load_index": r["load_index"],
                "RL": r["RL"],
                "XL": r["XL"],
                "initial_V1": r["initial_V1"],
                "initial_V2": r["initial_V2"],
                "initial_vswr": r["initial_vswr"],
                "final_V1": r["final_V1"],
                "final_V2": r["final_V2"],
                "best_V1": r["best_V1"],
                "best_V2": r["best_V2"],
                "final_vswr": r["final_vswr"],
                "best_vswr": r["best_vswr"],
                "suboptimality_gap": r["suboptimality_gap"],
                "steps": r["steps"],
                "boundary_hits": r["boundary_hits"],
                "immediate_reversals": r["immediate_reversals"],
                "revisited_state_steps": r["revisited_state_steps"],
                "hold_steps": r["hold_steps"],
                "action_counts": {
                    "V1+": r["count_V1_plus"],
                    "V1-": r["count_V1_minus"],
                    "V2+": r["count_V2_plus"],
                    "V2-": r["count_V2_minus"],
                    "Hold": r["count_Hold"],
                },
            }
            for r in holdout_failures
        ],
        "proximity_to_late_training_timeouts": near_timeout_proximity,
    }


def _select_representative_examples(
    dqn_records: Sequence[dict[str, Any]],
    random_records: Sequence[dict[str, Any]],
    grid_records: Sequence[dict[str, Any]],
    dqn_env: ImpedanceMatchingEnv,
    agent: DoubleDQNAgent,
) -> dict[str, Any]:
    """Select concrete successful and unsuccessful examples and replay trajectories for plots."""
    matched = [r for r in dqn_records if r["is_matched"]]
    failed = [r for r in dqn_records if not r["is_matched"]]

    def _replay(rec: dict[str, Any]) -> dict[str, Any]:
        traj_rec = _run_sequential_episode(
            env=dqn_env,
            policy_fn=lambda obs: agent.select_action(obs, epsilon=0.0),
            approach_name="double_dqn",
            load_idx=int(rec["load_index"]),
            rl=float(rec["RL"]),
            xl=float(rec["XL"]),
            v1_init=float(rec["initial_V1"]),
            v2_init=float(rec["initial_V2"]),
            record_trajectory=True,
        )
        idx = int(rec["load_index"])
        traj_rec["random_final_vswr"] = random_records[idx]["final_vswr"]
        traj_rec["grid_best_vswr"] = grid_records[idx]["final_vswr"]
        return traj_rec

    examples: list[tuple[str, dict[str, Any]]] = []
    if matched:
        # 1. Clean fast match from noticeable mismatch
        candidates_fast = [
            r for r in matched if float(r["initial_vswr"]) >= 2.0 and int(r["steps"]) >= 4
        ] or matched
        fast_best = min(
            candidates_fast,
            key=lambda r: (float(r["final_vswr"]), int(r["steps"])),
        )
        examples.append(("Successful Fast Match", fast_best))

        # 2. Boundary-tolerance or challenging match (1.2 <= best_vswr < 1.5 or slower convergence)
        tol_matches = [
            r for r in matched if float(r["best_vswr"]) >= 1.2 and int(r["load_index"]) != int(fast_best["load_index"])
        ]
        if tol_matches:
            tol_matches.sort(key=lambda r: int(r["steps"]), reverse=True)
            chosen_tol = tol_matches[len(tol_matches) // 2]
            examples.append(("Successful Boundary-Regime Match", chosen_tol))

    if failed:
        # Failed episodes sorted from worst final VSWR down
        failed_sorted = sorted(
            failed,
            key=lambda r: float(r["final_vswr"]),
            reverse=True,
        )
        examples.append(("Unsuccessful Timeout (Worst VSWR)", failed_sorted[0]))
        if len(failed_sorted) > 1:
            examples.append(("Unsuccessful Timeout (Near-Miss / Cycle)", failed_sorted[-1]))
    elif matched:
        # Fallback if zero failures exist: show slowest/highest-VSWR match
        slowest = max(matched, key=lambda r: (float(r["final_vswr"]), int(r["steps"])))
        examples.append(("Hardest Match (Slowest / Highest VSWR)", slowest))

    replayed = []
    for label, rec in examples:
        rep = _replay(rec)
        rep["example_label"] = label
        actions = [pt["action_name"] for pt in rep["trajectory"] if pt["action"] is not None]
        if len(actions) <= 25:
            rep["action_sequence_preview"] = " -> ".join(actions)
        else:
            rep["action_sequence_preview"] = (
                " -> ".join(actions[:12])
                + f" -> ... ({len(actions) - 20} steps) ... -> "
                + " -> ".join(actions[-8:])
            )
        replayed.append(rep)

    return {
        "successful_samples": [r for r in replayed if r["is_matched"]],
        "unsuccessful_samples": [r for r in replayed if not r["is_matched"]],
        "all_replayed": replayed,
    }


def generate_evaluation_plots(
    dqn_records: Sequence[dict[str, Any]],
    random_records: Sequence[dict[str, Any]],
    grid_records: Sequence[dict[str, Any]],
    summary_by_approach: dict[str, dict[str, Any]],
    region_breakdown: dict[str, Any],
    timeout_investigation: dict[str, Any],
    representative_examples: dict[str, Any],
    figures_dir: Path,
) -> list[str]:
    """Generate clean comparison, distribution, region, diagnostic, and trajectory plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    approaches = ["random", "double_dqn", "grid_search"]
    labels = ["Random Baseline", "Double DQN", "Grid Search (Ref)"]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c"]

    # 1. Approach comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=130)

    metric_keys = [
        ("is_matched_rate", "env is_matched"),
        ("vswr_lt_2_rate", "VSWR < 2.0"),
        ("vswr_lt_1_5_rate", "VSWR < 1.5"),
        ("vswr_lt_1_2_rate", "VSWR < 1.2"),
    ]
    x = np.arange(len(metric_keys))
    width = 0.26
    for i, (app, lbl, col) in enumerate(zip(approaches, labels, colors)):
        vals = [100.0 * summary_by_approach[app][mk] for mk, _ in metric_keys]
        axes[0].bar(x + (i - 1) * width, vals, width, label=lbl, color=col)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([lbl for _, lbl in metric_keys], rotation=15)
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("Held-Out Loads (%)")
    axes[0].set_title("Match & Threshold Success Rates")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)

    x2 = np.arange(len(approaches))
    means = [summary_by_approach[a]["final_vswr_mean"] for a in approaches]
    medians = [summary_by_approach[a]["final_vswr_median"] for a in approaches]
    w2 = 0.35
    axes[1].bar(x2 - w2 / 2, means, w2, label="Mean VSWR", color="#ff7f0e")
    axes[1].bar(x2 + w2 / 2, medians, w2, label="Median VSWR", color="#1f77b4")
    axes[1].axhline(1.5, color="#555555", linestyle="--", linewidth=1.0, label="VSWR = 1.5")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(["Random", "Double DQN", "Grid Search"])
    axes[1].set_ylabel("Final VSWR (log scale)")
    axes[1].set_title("Mean vs Median Final VSWR")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)

    seq_apps = ["random", "double_dqn"]
    seq_labels = ["Random Baseline", "Double DQN"]
    mean_steps_all = [summary_by_approach[a]["steps_all_mean"] for a in seq_apps]
    mean_steps_match = [
        summary_by_approach[a]["steps_to_match_mean"] or 0.0 for a in seq_apps
    ]
    x3 = np.arange(len(seq_apps))
    axes[2].bar(x3 - w2 / 2, mean_steps_all, w2, label="All Loads", color="#9467bd")
    axes[2].bar(x3 + w2 / 2, mean_steps_match, w2, label="Matched Loads", color="#2ca02c")
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels(seq_labels)
    axes[2].set_ylabel("Steps per Episode")
    axes[2].set_title("Sequential Steps (Grid Search = 4,489 evals)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    p1 = get_non_overwriting_path(figures_dir / "eval_approach_comparison.png")
    fig.savefig(p1)
    plt.close(fig)
    saved_paths.append(str(p1))

    # 2. Final VSWR ECDF + zoomed distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=130)
    for app, lbl, col in zip(approaches, labels, colors):
        recs = {"random": random_records, "double_dqn": dqn_records, "grid_search": grid_records}[app]
        vals = np.sort([_finite_vswr_for_stats(float(r["final_vswr"])) for r in recs])
        cdf = 100.0 * np.arange(1, len(vals) + 1) / len(vals)
        axes[0].plot(np.clip(vals, 1.0, 20.0), cdf, label=lbl, color=col, linewidth=1.8)
        axes[1].plot(vals, cdf, label=lbl, color=col, linewidth=1.8)

    for vline in (1.2, 1.5, 2.0):
        axes[0].axvline(vline, color="#444444", linestyle="--", linewidth=0.9)
        axes[1].axvline(vline, color="#444444", linestyle="--", linewidth=0.9)
    axes[0].set_xlim(1.0, 10.0)
    axes[0].set_xlabel("Final VSWR (clipped at 10)")
    axes[0].set_ylabel("Cumulative Held-Out Loads (%)")
    axes[0].set_title("Final VSWR Cumulative Distribution (Wide)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlim(1.0, 2.2)
    axes[1].set_xlabel("Final VSWR")
    axes[1].set_ylabel("Cumulative Held-Out Loads (%)")
    axes[1].set_title("Matching Zone Zoom (VSWR 1.0 to 2.2)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    p2 = get_non_overwriting_path(figures_dir / "eval_vswr_distribution.png")
    fig.savefig(p2)
    plt.close(fig)
    saved_paths.append(str(p2))

    # 3. Region map + RL / XL bin breakdown
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=130)
    dqn_rl = np.array([float(r["RL"]) for r in dqn_records])
    dqn_xl = np.array([float(r["XL"]) for r in dqn_records])
    dqn_vswr = np.array([min(3.0, float(r["final_vswr"])) for r in dqn_records])
    dqn_match = np.array([bool(r["is_matched"]) for r in dqn_records])

    sc = axes[0].scatter(
        dqn_rl[dqn_match],
        dqn_xl[dqn_match],
        c=dqn_vswr[dqn_match],
        cmap="viridis",
        vmin=1.0,
        vmax=2.0,
        s=26,
        alpha=0.85,
        label="Held-out Matched",
    )
    if np.any(~dqn_match):
        axes[0].scatter(
            dqn_rl[~dqn_match],
            dqn_xl[~dqn_match],
            marker="x",
            color="red",
            s=55,
            linewidths=1.8,
            label="Held-out Failed",
        )
    late_timeouts = timeout_investigation.get("late_training_timeouts", [])
    if late_timeouts:
        axes[0].scatter(
            [t["RL"] for t in late_timeouts],
            [t["XL"] for t in late_timeouts],
            marker="^",
            facecolors="none",
            edgecolors="#ff6600",
            s=75,
            linewidths=1.3,
            label="Late-Train Timeouts",
        )
    fig.colorbar(sc, ax=axes[0], label="DQN Final VSWR")
    axes[0].set_xlabel("Load Resistance RL (Ohms)")
    axes[0].set_ylabel("Load Reactance XL (Ohms)")
    axes[0].set_title("Held-Out Impedance Plane & Failures")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[0].grid(True, alpha=0.3)

    rl_bins = region_breakdown["by_RL"]
    axes[1].bar(
        [b["region"] for b in rl_bins],
        [100.0 * b["is_matched_rate"] for b in rl_bins],
        color="#1f77b4",
    )
    axes[1].set_ylim(0, 108)
    axes[1].set_ylabel("Match Success Rate (%)")
    axes[1].set_title("DQN Success Rate by RL Region (Ohms)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(True, axis="y", alpha=0.3)

    xl_bins = region_breakdown["by_XL"]
    axes[2].bar(
        [b["region"] for b in xl_bins],
        [100.0 * b["is_matched_rate"] for b in xl_bins],
        color="#2ca02c",
    )
    axes[2].set_ylim(0, 108)
    axes[2].set_ylabel("Match Success Rate (%)")
    axes[2].set_title("DQN Success Rate by XL Region (Ohms)")
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    p3 = get_non_overwriting_path(figures_dir / "eval_region_heatmap.png")
    fig.savefig(p3)
    plt.close(fig)
    saved_paths.append(str(p3))

    # 4. Action distribution & boundary/ineffective diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), dpi=130)
    act_keys = ["V1+", "V1-", "V2+", "V2-", "Hold"]
    dqn_matched = [r for r in dqn_records if r["is_matched"]]
    dqn_failed = [r for r in dqn_records if not r["is_matched"]]
    matched_sum = _summarize_approach(dqn_matched) if dqn_matched else None
    failed_sum = _summarize_approach(dqn_failed) if dqn_failed else None

    x_act = np.arange(len(act_keys))
    series = [("Random", summary_by_approach["random"], "#7f7f7f"), ("DQN All", summary_by_approach["double_dqn"], "#1f77b4")]
    if failed_sum is not None:
        series.append(("DQN Failed", failed_sum, "#d62728"))
    w_act = 0.8 / len(series)
    for idx_s, (lbl, sdict, col) in enumerate(series):
        fracs = [100.0 * sdict["action_fractions"][k] for k in act_keys]
        axes[0].bar(
            x_act + (idx_s - (len(series) - 1) / 2.0) * w_act,
            fracs,
            w_act,
            label=lbl,
            color=col,
        )
    axes[0].set_xticks(x_act)
    axes[0].set_xticklabels(act_keys)
    axes[0].set_ylabel("Action Share (%)")
    axes[0].set_title("Action Distribution")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)

    diag_keys = [
        ("boundary_hit_rate", "Boundary Hit"),
        ("hold_rate", "Hold"),
        ("immediate_reversal_rate", "1-Step Reversal"),
        ("revisited_state_rate", "Revisited (V1,V2)"),
    ]
    x_diag = np.arange(len(diag_keys))
    diag_series = [("DQN All", summary_by_approach["double_dqn"], "#1f77b4")]
    if matched_sum is not None:
        diag_series.append(("DQN Matched", matched_sum, "#2ca02c"))
    if failed_sum is not None:
        diag_series.append(("DQN Failed", failed_sum, "#d62728"))
    w_d = 0.8 / len(diag_series)
    for idx_s, (lbl, sdict, col) in enumerate(diag_series):
        vals = [100.0 * sdict[dk] for dk, _ in diag_keys]
        axes[1].bar(
            x_diag + (idx_s - (len(diag_series) - 1) / 2.0) * w_d,
            vals,
            w_d,
            label=lbl,
            color=col,
        )
    axes[1].set_xticks(x_diag)
    axes[1].set_xticklabels([dl for _, dl in diag_keys], rotation=12)
    axes[1].set_ylabel("Share of Steps (%)")
    axes[1].set_title("Boundary Hits & Repeated Ineffective Actions")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    p4 = get_non_overwriting_path(figures_dir / "eval_action_diagnostics.png")
    fig.savefig(p4)
    plt.close(fig)
    saved_paths.append(str(p4))

    # 5. Representative trajectories (Voltage trajectory + VSWR vs step)
    samples = representative_examples.get("all_replayed", [])[:4]
    if samples:
        nrows = len(samples)
        fig, axes = plt.subplots(nrows, 2, figsize=(12.5, 3.4 * nrows), dpi=125)
        if nrows == 1:
            axes = np.array([axes])
        for row_i, rep in enumerate(samples):
            traj = rep["trajectory"]
            v1s = [pt["V1"] for pt in traj]
            v2s = [pt["V2"] for pt in traj]
            steps = [pt["step"] for pt in traj]
            vswrs = [min(25.0, pt["vswr"]) for pt in traj]

            ax_v = axes[row_i, 0]
            ax_v.plot(v1s, v2s, "-o", markersize=2.5, color="#1f77b4", linewidth=1.2)
            ax_v.scatter([v1s[0]], [v2s[0]], color="#ff7f0e", s=65, zorder=5, label="Start")
            ax_v.scatter([v1s[-1]], [v2s[-1]], color="#d62728", marker="s", s=55, zorder=5, label="Final")
            ax_v.scatter([rep["best_V1"]], [rep["best_V2"]], color="#2ca02c", marker="*", s=120, zorder=6, label="Grid Opt")
            ax_v.set_xlim(-0.1, 3.4)
            ax_v.set_ylim(-0.1, 3.4)
            ax_v.set_xlabel("V1 (V)")
            ax_v.set_ylabel("V2 (V)")
            ax_v.set_title(
                f"{rep['example_label']} | RL={rep['RL']} XL={rep['XL']} ({rep['steps']} steps)"
            )
            ax_v.legend(fontsize=7, loc="best")
            ax_v.grid(True, alpha=0.3)

            ax_s = axes[row_i, 1]
            ax_s.plot(steps, vswrs, color="#1f77b4", linewidth=1.5, label="DQN VSWR")
            ax_s.axhline(rep["best_vswr"], color="#2ca02c", linestyle=":", linewidth=1.2, label=f"Grid Best ({rep['best_vswr']:.3f})")
            ax_s.axhline(1.2, color="#777777", linestyle="--", linewidth=0.9, label="VSWR=1.2")
            ax_s.axhline(1.5, color="#999999", linestyle="-.", linewidth=0.9, label="VSWR=1.5")
            ax_s.set_xlabel("Step")
            ax_s.set_ylabel("VSWR (capped at 25)")
            ax_s.set_title(
                f"VSWR: {rep['initial_vswr']:.2f} -> {rep['final_vswr']:.3f} | matched={rep['is_matched']}"
            )
            ax_s.legend(fontsize=7, loc="best")
            ax_s.grid(True, alpha=0.3)

        fig.tight_layout()
        p5 = get_non_overwriting_path(figures_dir / "eval_sample_trajectories.png")
        fig.savefig(p5)
        plt.close(fig)
        saved_paths.append(str(p5))

    return saved_paths


def write_markdown_report(summary: dict[str, Any], out_path: Path) -> None:
    """Write a clear, structured Markdown evaluation report inside `rl/logs/`."""
    by_app = summary["approaches"]
    dqn = by_app["double_dqn"]
    rnd = by_app["random"]
    grd = by_app["grid_search"]
    t_inv = summary["timeout_investigation"]
    integrity = summary["integrity"]

    lines = [
        "# Held-Out Evaluation Report — Double DQN Impedance Matching",
        "",
        "## 1. Evaluation Protocol & Integrity Checks",
        f"- **Model Evaluated:** `{summary['model_path']}` (device: `{summary['device']}`, training update steps: `{summary['model_update_steps']}`)",
        f"- **Held-Out Test Loads:** `{integrity['holdout_size']}` unique matchable loads (`best_vswr < 1.5`), strictly disjoint from the `{integrity['train_pool_size']}` training loads (`overlap={integrity['train_holdout_overlap']}`) and from all `{integrity['training_csv_unique_loads']}` loads sampled in training (`overlap={integrity['training_csv_holdout_overlap']}`).",
        f"- **Reproducible Seeds & Horizon:** `eval_seed={summary['eval_seed']}`, `max_steps={summary['max_steps']}`. Identical `(RL, XL, initial_V1, initial_V2)` starting states across sequential controllers.",
        f"- **Weight Immutability Check:** SHA-256 identical before/after evaluation (`weights_unchanged={integrity['weights_unchanged']}`).",
        "",
        "## 2. Approach Comparison (350 Held-Out Loads)",
        "",
        "| Metric | Random Baseline | Double DQN | Grid Search (Reference Optimum) |",
        "|---|---:|---:|---:|",
        f"| `is_matched` (Env Definition) | {100*rnd['is_matched_rate']:.2f}% ({rnd['matched_count']}/{rnd['count']}) | {100*dqn['is_matched_rate']:.2f}% ({dqn['matched_count']}/{dqn['count']}) | {100*grd['is_matched_rate']:.2f}% ({grd['matched_count']}/{grd['count']}) |",
        f"| VSWR < 2.0 | {100*rnd['vswr_lt_2_rate']:.2f}% | {100*dqn['vswr_lt_2_rate']:.2f}% | {100*grd['vswr_lt_2_rate']:.2f}% |",
        f"| VSWR < 1.5 | {100*rnd['vswr_lt_1_5_rate']:.2f}% | {100*dqn['vswr_lt_1_5_rate']:.2f}% | {100*grd['vswr_lt_1_5_rate']:.2f}% |",
        f"| VSWR < 1.2 | {100*rnd['vswr_lt_1_2_rate']:.2f}% | {100*dqn['vswr_lt_1_2_rate']:.2f}% | {100*grd['vswr_lt_1_2_rate']:.2f}% |",
        f"| Final VSWR (Mean / Median) | {rnd['final_vswr_mean']:.3f} / {rnd['final_vswr_median']:.3f} | {dqn['final_vswr_mean']:.3f} / {dqn['final_vswr_median']:.3f} | {grd['final_vswr_mean']:.3f} / {grd['final_vswr_median']:.3f} |",
        f"| Final VSWR (Min / Max) | {rnd['final_vswr_min']:.3f} / {rnd['final_vswr_max']:.3f} | {dqn['final_vswr_min']:.3f} / {dqn['final_vswr_max']:.3f} | {grd['final_vswr_min']:.3f} / {grd['final_vswr_max']:.3f} |",
        f"| Mean Final `|Gamma|` | {rnd['final_gamma_abs_mean']:.4f} | {dqn['final_gamma_abs_mean']:.4f} | {grd['final_gamma_abs_mean']:.4f} |",
        f"| Mean Suboptimality Gap | {rnd['suboptimality_gap_mean']:.4f} | {dqn['suboptimality_gap_mean']:.4f} | 0.0000 |",
        f"| Mean Steps (All / Matched) | {rnd['steps_all_mean']:.2f} / {rnd['steps_to_match_mean'] or 0.0:.2f} | {dqn['steps_all_mean']:.2f} / {dqn['steps_to_match_mean'] or 0.0:.2f} | N/A (4,489 lattice evals) |",
        f"| Mean Execution Time (ms/load) | {rnd['exec_time_ms_mean']:.3f} | {dqn['exec_time_ms_mean']:.3f} | {grd['exec_time_ms_mean']:.3f} |",
        "",
        "> **Important Distinction:** Grid search is an **offline, open-loop reference optimization method** that assumes full analytical knowledge of `ZL = RL + jXL` and evaluates all 4,489 voltage grid points simultaneously in vectorized NumPy. Double DQN is a **closed-loop sequential controller** operating step-by-step on measured `[Re(Gamma), Im(Gamma), V1, V2]` without knowing `RL` or `XL`.",
        "",
        "## 3. Double DQN Regional Breakdown",
        "### By Load Resistance (`RL`)",
        "| `RL` Region (Ohms) | Loads | `is_matched` | VSWR < 1.5 | Mean VSWR | Mean Steps (All) | Mean Subopt Gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b in summary["region_breakdown"]["by_RL"]:
        lines.append(
            f"| {b['region']} | {b['count']} | {100*b['is_matched_rate']:.1f}% | {100*b['vswr_lt_1_5_rate']:.1f}% | {b['final_vswr_mean']:.3f} | {b['steps_all_mean']:.1f} | {b['suboptimality_gap_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "### By Load Reactance (`XL`)",
            "| `XL` Region (Ohms) | Loads | `is_matched` | VSWR < 1.5 | Mean VSWR | Mean Steps (All) | Mean Subopt Gap |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for b in summary["region_breakdown"]["by_XL"]:
        lines.append(
            f"| {b['region']} | {b['count']} | {100*b['is_matched_rate']:.1f}% | {100*b['vswr_lt_1_5_rate']:.1f}% | {b['final_vswr_mean']:.3f} | {b['steps_all_mean']:.1f} | {b['suboptimality_gap_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 4. Action Usage, Boundaries & Failure Investigation",
            f"- **DQN Action Fractions:** {json.dumps(dqn['action_fractions'])}",
            f"- **Boundary Hits:** {dqn['boundary_hits_total']} / {dqn['total_steps']} steps ({100*dqn['boundary_hit_rate']:.2f}% across {dqn['episodes_with_boundary_hit']} episodes)",
            f"- **Immediate Reversals:** {dqn['immediate_reversals_total']} steps ({100*dqn['immediate_reversal_rate']:.2f}%)",
            f"- **Revisited `(V1, V2)` States:** {dqn['revisited_state_steps_total']} steps ({100*dqn['revisited_state_rate']:.2f}%)",
            f"- **Late-Training Timeouts Observed in CSV:** {t_inv['late_training_timeout_count']} episodes",
            f"- **Held-Out Failures (`not is_matched`):** {t_inv['holdout_failure_count']} / {dqn['count']} ({t_inv['holdout_near_miss_vswr_lt_1_5_count']} with VSWR < 1.5, {t_inv['holdout_high_vswr_ge_1_5_count']} with VSWR >= 1.5)",
            "",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_held_out(
    model_path: Path = DEFAULT_FINAL_MODEL_PATH,
    max_steps: int = 140,
    match_vswr_threshold: float = 1.5,
    success_vswr: float = 1.2,
    match_tolerance: float = 0.015,
    grid_size: int = DEFAULT_GRID_SIZE,
    split_seed: int = 42,
    eval_seed: int = 100,
    logs_dir: Path = DEFAULT_LOGS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    training_metrics_csv: Optional[Path] = DEFAULT_TRAINING_METRICS_CSV,
    generate_plots: bool = True,
    device: Optional[str] = "cpu",
    max_loads: Optional[int] = None,
) -> dict[str, Any]:
    """Evaluate `models/model.pt`, Random baseline, and Grid Search on identical held-out loads."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    catalog = build_load_catalog(
        vswr_threshold=match_vswr_threshold,
        split_seed=split_seed,
        grid_size=grid_size,
    )
    train_loads: list[tuple[float, float]] = catalog["train_loads"]
    holdout_loads: list[tuple[float, float]] = catalog["holdout_loads"]
    integrity = _verify_holdout_separation(
        train_loads=train_loads,
        holdout_loads=holdout_loads,
        training_metrics_csv=training_metrics_csv,
    )

    if max_loads is not None:
        holdout_loads = holdout_loads[: int(max_loads)]

    agent = DoubleDQNAgent.load(model_path, device=device)
    agent.online_net.eval()
    weights_before = _snapshot_weights_sha256(agent)

    initial_voltages = _generate_initial_voltages(len(holdout_loads), eval_seed=eval_seed)

    dqn_env = ImpedanceMatchingEnv(
        load_mode="matchable",
        match_vswr_threshold=match_vswr_threshold,
        success_vswr=success_vswr,
        match_tolerance=match_tolerance,
        max_steps=max_steps,
        grid_size=grid_size,
        random_initial_v=False,
        load_pool=holdout_loads,
        seed=eval_seed,
    )
    rand_env = ImpedanceMatchingEnv(
        load_mode="matchable",
        match_vswr_threshold=match_vswr_threshold,
        success_vswr=success_vswr,
        match_tolerance=match_tolerance,
        max_steps=max_steps,
        grid_size=grid_size,
        random_initial_v=False,
        load_pool=holdout_loads,
        seed=eval_seed,
    )

    dqn_records: list[dict[str, Any]] = []
    random_records: list[dict[str, Any]] = []
    grid_records: list[dict[str, Any]] = []

    for idx, ((rl, xl), (v1_init, v2_init)) in enumerate(
        zip(holdout_loads, initial_voltages)
    ):
        dqn_rec = _run_sequential_episode(
            env=dqn_env,
            policy_fn=lambda obs: agent.select_action(obs, epsilon=0.0),
            approach_name="double_dqn",
            load_idx=idx,
            rl=rl,
            xl=xl,
            v1_init=v1_init,
            v2_init=v2_init,
        )
        load_rand_rng = np.random.default_rng(eval_seed + 10_000 + idx)
        rand_rec = _run_sequential_episode(
            env=rand_env,
            policy_fn=lambda _obs, r=load_rand_rng: int(r.integers(0, 5)),
            approach_name="random",
            load_idx=idx,
            rl=rl,
            xl=xl,
            v1_init=v1_init,
            v2_init=v2_init,
        )
        grid_rec = _run_grid_search_reference(
            load_idx=idx,
            rl=rl,
            xl=xl,
            v1_init=v1_init,
            v2_init=v2_init,
            grid_size=grid_size,
            success_vswr=success_vswr,
            match_tolerance=match_tolerance,
        )
        dqn_records.append(dqn_rec)
        random_records.append(rand_rec)
        grid_records.append(grid_rec)

    weights_after = _snapshot_weights_sha256(agent)
    if weights_before != weights_after:
        raise RuntimeError("Model weights mutated during evaluation!")
    integrity["weights_unchanged"] = True
    integrity["weights_sha256"] = weights_after
    integrity["evaluated_loads_count"] = len(dqn_records)
    integrity["evaluated_unique_loads"] = len(
        {(r["RL"], r["XL"]) for r in dqn_records}
    )
    if integrity["evaluated_loads_count"] != integrity["evaluated_unique_loads"]:
        raise RuntimeError("Held-out loads were not evaluated uniquely.")
    if max_loads is None and integrity["evaluated_loads_count"] != integrity["holdout_size"]:
        raise RuntimeError(
            f"Expected {integrity['holdout_size']} held-out evaluations, got {integrity['evaluated_loads_count']}."
        )

    summary_by_approach = {
        "random": _summarize_approach(random_records),
        "double_dqn": _summarize_approach(dqn_records),
        "grid_search": _summarize_approach(grid_records),
    }
    region_breakdown = _summarize_by_regions(dqn_records)
    timeout_investigation = _investigate_late_training_and_holdout_failures(
        dqn_records=dqn_records,
        training_metrics_csv=training_metrics_csv,
    )
    representative_examples = _select_representative_examples(
        dqn_records=dqn_records,
        random_records=random_records,
        grid_records=grid_records,
        dqn_env=dqn_env,
        agent=agent,
    )

    logs_dir = Path(logs_dir)
    figures_dir = Path(figures_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = get_non_overwriting_path(logs_dir / "eval_per_load_results.csv")
    all_rows = dqn_records + random_records + grid_records
    fieldnames = list(dqn_records[0].keys()) + ["grid_evaluations"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            out_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out_row)

    plots: list[str] = []
    if generate_plots:
        plots = generate_evaluation_plots(
            dqn_records=dqn_records,
            random_records=random_records,
            grid_records=grid_records,
            summary_by_approach=summary_by_approach,
            region_breakdown=region_breakdown,
            timeout_investigation=timeout_investigation,
            representative_examples=representative_examples,
            figures_dir=figures_dir,
        )

    summary = {
        "status": "completed",
        "model_path": str(model_path),
        "model_update_steps": int(agent.update_steps),
        "device": str(agent.device),
        "max_steps": max_steps,
        "match_vswr_threshold": match_vswr_threshold,
        "success_vswr": success_vswr,
        "match_tolerance": match_tolerance,
        "grid_size": grid_size,
        "split_seed": split_seed,
        "eval_seed": eval_seed,
        "integrity": integrity,
        "approaches": summary_by_approach,
        "region_breakdown": region_breakdown,
        "timeout_investigation": timeout_investigation,
        "representative_examples": {
            "successful_samples": [
                {k: v for k, v in s.items() if k != "trajectory"}
                for s in representative_examples["successful_samples"]
            ],
            "unsuccessful_samples": [
                {k: v for k, v in s.items() if k != "trajectory"}
                for s in representative_examples["unsuccessful_samples"]
            ],
        },
        "csv_path": str(csv_path),
        "plots": plots,
    }

    report_path = get_non_overwriting_path(logs_dir / "eval_report.md")
    write_markdown_report(summary, report_path)
    summary["report_path"] = str(report_path)

    json_path = get_non_overwriting_path(logs_dir / "eval_summary.json")
    summary["summary_json_path"] = str(json_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained Double DQN against Random baseline and Grid Search."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_FINAL_MODEL_PATH)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=100)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-loads", type=int, default=None)
    args = parser.parse_args(argv)

    summary = evaluate_held_out(
        model_path=args.model_path,
        max_steps=args.max_steps,
        split_seed=args.split_seed,
        eval_seed=args.eval_seed,
        generate_plots=not args.no_plots,
        max_loads=args.max_loads,
    )
    print(json.dumps(summary["approaches"], indent=2))
    print(f"Saved summary JSON: {summary['summary_json_path']}")
    print(f"Saved CSV results : {summary['csv_path']}")
    print(f"Saved MD report   : {summary['report_path']}")
    print(f"Saved plots       : {len(summary['plots'])}")


if __name__ == "__main__":
    main()
