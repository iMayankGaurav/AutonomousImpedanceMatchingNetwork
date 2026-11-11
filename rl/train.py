"""M4: Double DQN training pipeline for autonomous impedance matching.

Trains a DoubleDQNAgent on ImpedanceMatchingEnv using loads verified as
matchable by M3 grid_search_match, logs per-episode metrics and reward
components, tracks unmatchable loads separately, saves periodic checkpoints,
handles KeyboardInterrupt safely, generates training plots in `rl/figures/`,
and saves the final model without overwriting existing checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

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
from physics.pi_network import grid_search_match
from rl.dqn import DoubleDQNAgent, compute_epsilon
from rl.env import DEFAULT_GRID_SIZE, ImpedanceMatchingEnv

RL_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS_DIR = RL_DIR / "checkpoints"
DEFAULT_LOGS_DIR = RL_DIR / "logs"
DEFAULT_FIGURES_DIR = RL_DIR / "figures"
DEFAULT_FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "model.pt"


def set_global_seeds(seed: int) -> None:
    """Seed Python random, NumPy, and PyTorch for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_non_overwriting_path(target_path: Path) -> Path:
    """Return target_path if unused; otherwise append _v2, _v3, ... without overwriting."""
    if not target_path.exists():
        return target_path
    stem = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_v{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def build_load_catalog(
    vswr_threshold: float = 1.5,
    rl_step: float = 0.5,
    xl_step: float = 0.5,
    full_scan_step: float = 2.5,
    holdout_fraction: float = 0.2,
    split_seed: int = 42,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> dict[str, Any]:
    """Identify matchable loads for training/holdout and separately catalog unmatchable loads.

    Uses M3 `grid_search_match` at `grid_size` (default 67 -> 0.05 V lattice).
    """
    rl_low, rl_high = config.RL_RANGE
    xl_low, xl_high = config.XL_RANGE

    # Fine scan over the region containing all matchable loads for 100 MHz / 150 nH
    scan_rl_high = min(rl_high, 45.0)
    scan_xl_low = max(xl_low, 30.0)
    scan_xl_high = min(xl_high, 95.0)
    rls_fine = np.arange(rl_low, scan_rl_high + 1e-9, rl_step)
    xls_fine = np.arange(scan_xl_low, scan_xl_high + 1e-9, xl_step)

    matchable: list[tuple[float, float]] = []
    unmatchable_envelope: list[dict[str, float]] = []

    for rl in rls_fine:
        for xl in xls_fine:
            res = grid_search_match(float(rl), float(xl), grid_size=grid_size)
            coord = (round(float(rl), 4), round(float(xl), 4))
            if res["best_vswr"] < vswr_threshold:
                matchable.append(coord)
            else:
                unmatchable_envelope.append(
                    {
                        "RL": coord[0],
                        "XL": coord[1],
                        "best_gamma_abs": round(float(res["best_gamma_abs"]), 6),
                        "best_vswr": round(float(res["best_vswr"]), 6)
                        if math.isfinite(res["best_vswr"])
                        else float("inf"),
                    }
                )

    if len(matchable) < 10:
        raise RuntimeError(
            f"Too few matchable loads ({len(matchable)}) found for threshold={vswr_threshold}."
        )

    # Full-range coarse scan to track unmatchable loads across entire RL_RANGE x XL_RANGE
    rls_full = np.arange(rl_low, rl_high + 1e-9, full_scan_step)
    xls_full = np.arange(xl_low, xl_high + 1e-9, full_scan_step)
    full_total = 0
    full_achievable_vswr2 = 0
    full_achievable_thresh = 0
    full_unmatchable_vswr2: list[dict[str, float]] = []

    for rl in rls_full:
        for xl in xls_full:
            full_total += 1
            res = grid_search_match(float(rl), float(xl), grid_size=grid_size)
            bv = float(res["best_vswr"])
            if bv < 2.0:
                full_achievable_vswr2 += 1
            else:
                full_unmatchable_vswr2.append(
                    {
                        "RL": round(float(rl), 4),
                        "XL": round(float(xl), 4),
                        "best_gamma_abs": round(float(res["best_gamma_abs"]), 6),
                        "best_vswr": round(bv, 6) if math.isfinite(bv) else float("inf"),
                    }
                )
            if bv < vswr_threshold:
                full_achievable_thresh += 1

    rng = np.random.default_rng(split_seed)
    indices = np.arange(len(matchable))
    rng.shuffle(indices)
    n_holdout = max(1, int(round(len(matchable) * holdout_fraction)))
    holdout_idx = set(indices[:n_holdout].tolist())

    train_loads = [matchable[i] for i in range(len(matchable)) if i not in holdout_idx]
    holdout_loads = [matchable[i] for i in range(len(matchable)) if i in holdout_idx]

    return {
        "train_loads": train_loads,
        "holdout_loads": holdout_loads,
        "unmatchable_envelope_count": len(unmatchable_envelope),
        "unmatchable_envelope_sample": unmatchable_envelope[:20],
        "full_range_scan": {
            "step_ohms": full_scan_step,
            "total_loads": full_total,
            "achievable_vswr_lt_2": full_achievable_vswr2,
            "achievable_vswr_lt_threshold": full_achievable_thresh,
            "unmatchable_vswr_ge_2_count": len(full_unmatchable_vswr2),
            "unmatchable_fraction": round(len(full_unmatchable_vswr2) / max(1, full_total), 4),
            "unmatchable_loads_sample": full_unmatchable_vswr2[:25],
            "unmatchable_best_vswr_median": round(
                float(np.median([x["best_vswr"] for x in full_unmatchable_vswr2])), 4
            )
            if full_unmatchable_vswr2
            else None,
        },
    }


def _rolling_mean(values: Sequence[float], window: int = 50) -> float:
    if not values:
        return 0.0
    return float(np.mean(values[-window:]))


def _rolling_series(values: Sequence[float], window: int = 50) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = float(np.mean(arr[start : i + 1]))
    return out


def generate_training_plots(
    records: Sequence[dict[str, Any]],
    figures_dir: Path,
    prefix: str = "train",
    rolling_window: int = 50,
) -> list[str]:
    """Generate and save all 6 training plots plus an overview inside `rl/figures/`."""
    if not records:
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    episodes = np.array([int(r["episode"]) for r in records], dtype=np.int32)
    rewards = np.array([float(r["total_reward"]) for r in records], dtype=np.float64)
    rolling_rewards = _rolling_series(rewards, window=rolling_window)
    vswrs = np.array(
        [float(r["final_vswr"]) if r["final_vswr"] != "inf" else 50.0 for r in records],
        dtype=np.float64,
    )
    success_rate = 100.0 * _rolling_series(
        [int(r["is_matched"]) for r in records], window=rolling_window
    )
    epsilons = np.array([float(r["epsilon"]) for r in records], dtype=np.float64)
    losses = np.array(
        [float(r["mean_loss"]) if r["mean_loss"] != "" else np.nan for r in records],
        dtype=np.float64,
    )

    saved_paths: list[str] = []

    def _save_single(name: str, y: np.ndarray, title: str, ylabel: str, color: str, mask_nan: bool = False, hline: Optional[float] = None) -> None:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        if mask_nan:
            valid = np.isfinite(y)
            ax.plot(episodes[valid], y[valid], color=color, linewidth=1.5)
        else:
            ax.plot(episodes, y, color=color, linewidth=1.3)
        if hline is not None:
            ax.axhline(hline, color="#555555", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = get_non_overwriting_path(figures_dir / f"{prefix}_{name}.png")
        fig.savefig(out_path)
        plt.close(fig)
        saved_paths.append(str(out_path))

    _save_single("episode_reward", rewards, "Episode Reward", "Total Reward", "#1f77b4")
    _save_single(
        "rolling_average_reward",
        rolling_rewards,
        f"Rolling Average Reward (window={rolling_window})",
        "Mean Reward",
        "#ff7f0e",
    )
    _save_single("final_vswr", vswrs, "Final Episode VSWR", "Final VSWR", "#2ca02c", hline=1.2)
    _save_single(
        "success_rate",
        success_rate,
        f"Rolling Match Success Rate (window={rolling_window})",
        "Success Rate (%)",
        "#9467bd",
    )
    _save_single("epsilon_decay", epsilons, "Epsilon Decay Schedule", "Epsilon", "#8c564b")
    _save_single(
        "training_loss",
        losses,
        "Mean Huber Training Loss per Episode",
        "Huber Loss",
        "#d62728",
        mask_nan=True,
    )

    # Combined 3x2 overview plot
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), dpi=120)
    panels = [
        (axes[0, 0], rewards, "Episode Reward", "Reward", "#1f77b4", False, None),
        (axes[0, 1], rolling_rewards, f"Rolling Average Reward (w={rolling_window})", "Mean Reward", "#ff7f0e", False, None),
        (axes[1, 0], vswrs, "Final VSWR", "VSWR", "#2ca02c", False, 1.2),
        (axes[1, 1], success_rate, f"Success Rate (w={rolling_window})", "Success (%)", "#9467bd", False, None),
        (axes[2, 0], epsilons, "Epsilon Decay", "Epsilon", "#8c564b", False, None),
        (axes[2, 1], losses, "Training Loss", "Huber Loss", "#d62728", True, None),
    ]
    for ax, y, title, ylabel, color, mask_nan, hline in panels:
        if mask_nan:
            valid = np.isfinite(y)
            ax.plot(episodes[valid], y[valid], color=color, linewidth=1.3)
        else:
            ax.plot(episodes, y, color=color, linewidth=1.2)
        if hline is not None:
            ax.axhline(hline, color="#555555", linestyle="--", linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    overview_path = get_non_overwriting_path(figures_dir / f"{prefix}_overview.png")
    fig.savefig(overview_path)
    plt.close(fig)
    saved_paths.append(str(overview_path))

    return saved_paths


def train_double_dqn(
    num_episodes: int = 1000,
    max_steps: int = 100,
    lr: float = 0.001,
    gamma: float = 0.99,
    buffer_capacity: int = 50_000,
    batch_size: int = 64,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 6_000,
    warmup_steps: int = 500,
    target_update_freq: int = 250,
    hidden_dims: tuple[int, int] = (128, 128),
    huber_beta: float = 1.0,
    max_grad_norm: float = 10.0,
    match_vswr_threshold: float = 1.5,
    success_vswr: float = 1.2,
    checkpoint_interval: int = 200,
    seed: int = 42,
    final_model_path: Path = DEFAULT_FINAL_MODEL_PATH,
    checkpoints_dir: Path = DEFAULT_CHECKPOINTS_DIR,
    logs_dir: Path = DEFAULT_LOGS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    run_name: str = "double_dqn",
    log_interval: int = 50,
    generate_plots: bool = True,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Run Double DQN training with safe interrupt handling, periodic checkpoints, and plots."""
    set_global_seeds(seed)

    catalog = build_load_catalog(
        vswr_threshold=match_vswr_threshold,
        split_seed=seed,
    )
    train_loads: list[tuple[float, float]] = catalog["train_loads"]
    holdout_loads: list[tuple[float, float]] = catalog["holdout_loads"]

    env = ImpedanceMatchingEnv(
        load_mode="matchable",
        match_vswr_threshold=match_vswr_threshold,
        success_vswr=success_vswr,
        max_steps=max_steps,
        random_initial_v=True,
        load_pool=train_loads,
        seed=seed,
    )
    agent = DoubleDQNAgent(
        state_dim=env.observation_space_shape[0],
        action_dim=env.action_space_n,
        hidden_dims=hidden_dims,
        lr=lr,
        gamma=gamma,
        huber_beta=huber_beta,
        max_grad_norm=max_grad_norm,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        device=device,
        seed=seed,
    )

    training_config = {
        "num_episodes": num_episodes,
        "max_steps": max_steps,
        "lr": lr,
        "gamma": gamma,
        "buffer_capacity": buffer_capacity,
        "batch_size": batch_size,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "eps_decay_steps": eps_decay_steps,
        "warmup_steps": warmup_steps,
        "target_update_freq": target_update_freq,
        "hidden_dims": list(hidden_dims),
        "huber_beta": huber_beta,
        "max_grad_norm": max_grad_norm,
        "match_vswr_threshold": match_vswr_threshold,
        "success_vswr": success_vswr,
        "checkpoint_interval": checkpoint_interval,
        "seed": seed,
        "device": str(agent.device),
    }

    total_env_steps = 0
    episode_records: list[dict[str, Any]] = []
    success_history: list[float] = []
    reward_history: list[float] = []
    final_gamma_history: list[float] = []
    length_history: list[float] = []
    loss_history: list[float] = []
    periodic_checkpoints: list[str] = []

    nan_or_inf_events = 0
    max_observed_grad_norm = 0.0
    interrupted = False
    ep_epsilon = eps_start

    checkpoints_dir = Path(checkpoints_dir)
    logs_dir = Path(logs_dir)
    figures_dir = Path(figures_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    try:
        for ep in range(1, num_episodes + 1):
            obs, info = env.reset()
            ep_reward = 0.0
            comp_sums = {
                "progress": 0.0,
                "mismatch": 0.0,
                "vswr_penalty": 0.0,
                "step_cost": 0.0,
                "success_bonus": 0.0,
            }
            ep_losses: list[float] = []
            ep_grad_norms: list[float] = []
            ep_q_means: list[float] = []
            ep_steps = 0

            while True:
                ep_epsilon = compute_epsilon(
                    total_env_steps, eps_start, eps_end, eps_decay_steps
                )
                action = agent.select_action(obs, epsilon=ep_epsilon)
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                total_env_steps += 1
                ep_steps += 1

                agent.store_transition(obs, action, reward, next_obs, terminated)

                if total_env_steps >= warmup_steps:
                    upd = agent.update()
                    if upd is not None:
                        loss_val = float(upd["loss"])
                        gn_val = float(upd["grad_norm"])
                        mq_val = float(upd["mean_q"])
                        if not (
                            math.isfinite(loss_val)
                            and math.isfinite(gn_val)
                            and math.isfinite(mq_val)
                        ):
                            nan_or_inf_events += 1
                            raise RuntimeError(
                                f"Non-finite training stat at ep {ep}: {upd}"
                            )
                        ep_losses.append(loss_val)
                        ep_grad_norms.append(gn_val)
                        ep_q_means.append(mq_val)
                        max_observed_grad_norm = max(max_observed_grad_norm, gn_val)

                ep_reward += reward
                rc = step_info["reward_components"]
                for key in comp_sums:
                    comp_sums[key] += float(rc[key])

                obs = next_obs
                info = step_info
                if terminated or truncated:
                    break

            matched = bool(info["is_matched"])
            final_gamma = float(info["gamma_abs"])
            final_vswr = float(info["vswr"])
            mean_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
            mean_grad_norm = float(np.mean(ep_grad_norms)) if ep_grad_norms else 0.0
            mean_q = float(np.mean(ep_q_means)) if ep_q_means else 0.0

            success_history.append(1.0 if matched else 0.0)
            reward_history.append(ep_reward)
            final_gamma_history.append(final_gamma)
            length_history.append(float(ep_steps))
            if ep_losses:
                loss_history.append(mean_loss)

            rolling_success = _rolling_mean(success_history, 50)
            record = {
                "episode": ep,
                "total_env_steps": total_env_steps,
                "epsilon": round(ep_epsilon, 6),
                "total_reward": round(ep_reward, 6),
                "reward_progress": round(comp_sums["progress"], 6),
                "reward_mismatch": round(comp_sums["mismatch"], 6),
                "reward_vswr_penalty": round(comp_sums["vswr_penalty"], 6),
                "reward_step_cost": round(comp_sums["step_cost"], 6),
                "reward_success_bonus": round(comp_sums["success_bonus"], 6),
                "final_gamma_abs": round(final_gamma, 6),
                "final_vswr": round(final_vswr, 6) if math.isfinite(final_vswr) else "inf",
                "episode_length": ep_steps,
                "is_matched": int(matched),
                "rolling_success_50": round(rolling_success, 4),
                "mean_loss": round(mean_loss, 6) if math.isfinite(mean_loss) else "",
                "mean_grad_norm": round(mean_grad_norm, 6),
                "mean_q": round(mean_q, 6),
                "suboptimality_gap": round(float(info["suboptimality_gap"]), 6),
                "RL": round(float(info["RL"]), 4),
                "XL": round(float(info["XL"]), 4),
            }
            episode_records.append(record)

            if log_interval > 0 and (ep % log_interval == 0 or ep == 1 or ep == num_episodes):
                loss_str = f"{mean_loss:.4f}" if math.isfinite(mean_loss) else "warmup"
                print(
                    f"[ep {ep:4d}/{num_episodes} | step {total_env_steps:6d} | eps {ep_epsilon:.3f}] "
                    f"R={_rolling_mean(reward_history, 50):6.2f} | "
                    f"|Gamma|={_rolling_mean(final_gamma_history, 50):.3f} | "
                    f"VSWR={final_vswr:.3f} | len={_rolling_mean(length_history, 50):5.1f} | "
                    f"succ50={100*rolling_success:5.1f}% | loss={loss_str}"
                )

            if checkpoint_interval > 0 and ep % checkpoint_interval == 0 and ep < num_episodes:
                periodic_path = get_non_overwriting_path(
                    checkpoints_dir / f"{run_name}_ep{ep:04d}.pt"
                )
                agent.save(
                    periodic_path,
                    overwrite=False,
                    metadata={
                        "episode": ep,
                        "total_env_steps": total_env_steps,
                        "seed": seed,
                        "rolling_success_50": rolling_success,
                    },
                )
                periodic_checkpoints.append(str(periodic_path))

    except KeyboardInterrupt:
        interrupted = True
        print("\n[Interrupted] KeyboardInterrupt caught; saving partial logs and checkpoint safely...")

    # Save CSV metrics, unmatchable load report, plots, and model safely
    saved_csv_path = ""
    if episode_records:
        safe_csv = get_non_overwriting_path(logs_dir / f"{run_name}_metrics.csv")
        fieldnames = list(episode_records[0].keys())
        with safe_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_records)
        saved_csv_path = str(safe_csv)

    unmatchable_report_path = get_non_overwriting_path(
        logs_dir / f"{run_name}_unmatchable_loads.json"
    )
    with unmatchable_report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "match_vswr_threshold": match_vswr_threshold,
                "unmatchable_envelope_count": catalog["unmatchable_envelope_count"],
                "unmatchable_envelope_sample": catalog["unmatchable_envelope_sample"],
                "full_range_scan": catalog["full_range_scan"],
            },
            f,
            indent=2,
        )

    saved_plots: list[str] = []
    if generate_plots and episode_records:
        saved_plots = generate_training_plots(
            episode_records, figures_dir=figures_dir, prefix=run_name
        )

    if interrupted:
        target_save = get_non_overwriting_path(
            checkpoints_dir / f"{run_name}_interrupted.pt"
        )
    else:
        target_save = get_non_overwriting_path(Path(final_model_path))

    agent.save(
        target_save,
        overwrite=False,
        metadata={
            "status": "interrupted" if interrupted else "completed",
            "completed_episodes": len(episode_records),
            "total_env_steps": total_env_steps,
            "seed": seed,
            "lr": lr,
            "gamma": gamma,
            "buffer_capacity": buffer_capacity,
            "batch_size": batch_size,
            "eps_decay_steps": eps_decay_steps,
            "target_update_freq": target_update_freq,
            "final_rolling_success_50": _rolling_mean(success_history, 50),
        },
    )

    completed = len(episode_records)
    first_w = min(50, max(1, completed))
    last_w = min(50, max(1, completed))
    summary = {
        "status": "interrupted" if interrupted else "completed",
        "config": training_config,
        "completed_episodes": completed,
        "total_env_steps": total_env_steps,
        "train_load_count": len(train_loads),
        "holdout_load_count": len(holdout_loads),
        "unmatchable_envelope_count": catalog["unmatchable_envelope_count"],
        "full_range_unmatchable_count": catalog["full_range_scan"]["unmatchable_vswr_ge_2_count"],
        "full_range_total_scanned": catalog["full_range_scan"]["total_loads"],
        "final_model_path": str(target_save),
        "periodic_checkpoints": periodic_checkpoints,
        "metrics_csv_path": saved_csv_path,
        "unmatchable_report_path": str(unmatchable_report_path),
        "plots": saved_plots,
        "nan_or_inf_events": nan_or_inf_events,
        "max_observed_grad_norm": round(max_observed_grad_norm, 6),
        "early_window": {
            "episodes": f"1-{first_w}",
            "mean_reward": round(float(np.mean(reward_history[:first_w])), 4)
            if reward_history
            else None,
            "mean_final_gamma_abs": round(
                float(np.mean(final_gamma_history[:first_w])), 4
            )
            if final_gamma_history
            else None,
            "mean_episode_length": round(float(np.mean(length_history[:first_w])), 2)
            if length_history
            else None,
            "success_rate": round(float(np.mean(success_history[:first_w])), 4)
            if success_history
            else None,
        },
        "late_window": {
            "episodes": f"{max(1, completed - last_w + 1)}-{completed}",
            "mean_reward": round(float(np.mean(reward_history[-last_w:])), 4)
            if reward_history
            else None,
            "mean_final_gamma_abs": round(
                float(np.mean(final_gamma_history[-last_w:])), 4
            )
            if final_gamma_history
            else None,
            "mean_episode_length": round(float(np.mean(length_history[-last_w:])), 2)
            if length_history
            else None,
            "success_rate": round(float(np.mean(success_history[-last_w:])), 4)
            if success_history
            else None,
            "mean_loss": round(float(np.mean(loss_history[-last_w:])), 6)
            if loss_history
            else None,
        },
        "overall_success_rate": round(float(np.mean(success_history)), 4)
        if success_history
        else None,
        "final_epsilon": round(ep_epsilon, 6),
    }

    summary_path = get_non_overwriting_path(logs_dir / f"{run_name}_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_json_path"] = str(summary_path)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train Double DQN for Pi-network matching.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-capacity", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=6_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--target-update-freq", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="double_dqn")
    parser.add_argument("--output-model", type=Path, default=DEFAULT_FINAL_MODEL_PATH)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    summary = train_double_dqn(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        lr=args.lr,
        gamma=args.gamma,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        warmup_steps=args.warmup_steps,
        target_update_freq=args.target_update_freq,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        final_model_path=args.output_model,
        run_name=args.run_name,
        generate_plots=not args.no_plots,
    )
    print("\nTraining Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
