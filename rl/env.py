"""M4: Gymnasium-compatible RL environment for RF Pi-network impedance matching.

Wraps the Module 3 RF physics engine (`physics.pi_network`) and shared hardware
constants (`config`) without duplicating physics equations.

Observation space (Section 4.4 of Project.md):
    np.ndarray of shape (4,), dtype float32:
        [gamma.real, gamma.imag, V1, V2]

Action space (discrete, 5 actions):
    0: Increase V1 by 0.05 V
    1: Decrease V1 by 0.05 V
    2: Increase V2 by 0.05 V
    3: Decrease V2 by 0.05 V
    4: Hold (keep V1 and V2 constant)
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

import config
from physics.pi_network import compute, grid_search_match

ACTION_NAMES: dict[int, str] = {
    0: "V1+",
    1: "V1-",
    2: "V2+",
    3: "V2-",
    4: "Hold",
}

# Discrete lattice size aligned with V_STEP = 0.05 V over [0.0, 3.3] V (67 points)
DEFAULT_GRID_SIZE: int = int(round((config.V_MAX - config.V_MIN) / config.V_STEP)) + 1


def _clamp_voltage(voltage: float) -> float:
    """Validate finiteness and clamp voltage strictly within [V_MIN, V_MAX]."""
    val = float(voltage)
    if not math.isfinite(val):
        raise ValueError(f"Voltage must be finite; got {voltage!r}.")
    clamped = float(np.clip(val, config.V_MIN, config.V_MAX))
    return round(clamped, 4)


def identify_matchable_loads(
    rl_values: Sequence[float],
    xl_values: Sequence[float],
    vswr_threshold: float = 1.5,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> list[tuple[float, float, dict[str, Any]]]:
    """Identify (RL, XL) loads with best_vswr < vswr_threshold via M3 grid_search_match."""
    matchable: list[tuple[float, float, dict[str, Any]]] = []
    for rl in rl_values:
        for xl in xl_values:
            res = grid_search_match(float(rl), float(xl), grid_size=grid_size)
            if res["best_vswr"] < vswr_threshold:
                matchable.append((float(rl), float(xl), res))
    return matchable


class ImpedanceMatchingEnv:
    """Gymnasium-style environment for autonomous Pi-network impedance matching."""

    action_space_n: int = 5
    observation_space_shape: tuple[int, ...] = (4,)
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        load_mode: str = "matchable",
        matchable_prob: float = 0.9,
        match_vswr_threshold: float = 1.5,
        success_vswr: float = 1.2,
        match_tolerance: float = 0.015,
        max_steps: int = 120,
        grid_size: int = DEFAULT_GRID_SIZE,
        random_initial_v: bool = True,
        load_pool: Optional[Sequence[tuple[float, float]]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if load_mode not in ("matchable", "full", "mixed"):
            raise ValueError(
                f"Invalid load_mode={load_mode!r}; expected 'matchable', 'full', or 'mixed'."
            )
        if not (0.0 <= matchable_prob <= 1.0):
            raise ValueError("matchable_prob must be in [0.0, 1.0].")
        if match_vswr_threshold <= 1.0:
            raise ValueError("match_vswr_threshold must be > 1.0.")
        if success_vswr <= 1.0:
            raise ValueError("success_vswr must be > 1.0.")
        if match_tolerance < 0.0:
            raise ValueError("match_tolerance must be >= 0.0.")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2.")

        self.load_mode = load_mode
        self.matchable_prob = float(matchable_prob)
        self.match_vswr_threshold = float(match_vswr_threshold)
        self.success_vswr = float(success_vswr)
        self.match_tolerance = float(match_tolerance)
        self.max_steps = int(max_steps)
        self.grid_size = int(grid_size)
        self.random_initial_v = bool(random_initial_v)
        self.load_pool: Optional[list[tuple[float, float]]] = (
            [(float(rl), float(xl)) for rl, xl in load_pool]
            if load_pool is not None
            else None
        )
        if self.load_pool is not None and len(self.load_pool) == 0:
            raise ValueError("load_pool cannot be empty when provided.")

        self.np_random = np.random.default_rng(seed)
        self._voltage_grid = np.round(
            np.arange(config.V_MIN, config.V_MAX + 1e-9, config.V_STEP), 4
        )

        # Episode state variables
        self.RL: float = 50.0
        self.XL: float = 0.0
        self.V1: float = 1.65
        self.V2: float = 1.65
        self.step_count: int = 0
        self.terminated: bool = False
        self.truncated: bool = False
        self._episode_started: bool = False
        self.prev_gamma_abs: float = 0.0
        self.best_ref: dict[str, Any] = {}
        self.last_sim: dict[str, Any] = {}

    def _sample_random_voltage(self) -> float:
        idx = int(self.np_random.integers(0, len(self._voltage_grid)))
        return float(self._voltage_grid[idx])

    def _sample_full_load(self) -> tuple[float, float]:
        rl = float(self.np_random.uniform(config.RL_RANGE[0], config.RL_RANGE[1]))
        xl = float(self.np_random.uniform(config.XL_RANGE[0], config.XL_RANGE[1]))
        return rl, xl

    def _sample_matchable_load(
        self, max_attempts: int = 500
    ) -> tuple[float, float, dict[str, Any]]:
        """Sample a load verified by M3 grid_search_match to meet match_vswr_threshold."""
        rl_low, rl_high = config.RL_RANGE
        xl_low, xl_high = config.XL_RANGE
        prop_rl_high = min(rl_high, 45.0)
        prop_xl_low = max(xl_low, 30.0)
        prop_xl_high = min(xl_high, 95.0)

        for _ in range(max_attempts):
            rl = float(self.np_random.uniform(rl_low, prop_rl_high))
            xl = float(self.np_random.uniform(prop_xl_low, prop_xl_high))
            ref = grid_search_match(rl, xl, grid_size=self.grid_size)
            if ref["best_vswr"] < self.match_vswr_threshold:
                return rl, xl, ref

        for _ in range(max_attempts):
            rl, xl = self._sample_full_load()
            ref = grid_search_match(rl, xl, grid_size=self.grid_size)
            if ref["best_vswr"] < self.match_vswr_threshold:
                return rl, xl, ref

        raise RuntimeError(
            f"Unable to sample a load with best_vswr < {self.match_vswr_threshold} "
            f"within {2 * max_attempts} attempts."
        )

    def _select_episode_load(
        self, rl_opt: Optional[float], xl_opt: Optional[float]
    ) -> tuple[float, float, dict[str, Any]]:
        if rl_opt is not None and xl_opt is not None:
            rl, xl = float(rl_opt), float(xl_opt)
            self._validate_load(rl, xl)
            ref = grid_search_match(rl, xl, grid_size=self.grid_size)
            return rl, xl, ref

        if rl_opt is not None or xl_opt is not None:
            raise ValueError("Provide both 'RL' and 'XL' in options, or neither.")

        if self.load_pool is not None:
            idx = int(self.np_random.integers(0, len(self.load_pool)))
            rl, xl = self.load_pool[idx]
            self._validate_load(rl, xl)
            ref = grid_search_match(rl, xl, grid_size=self.grid_size)
            return rl, xl, ref

        if self.load_mode == "full":
            rl, xl = self._sample_full_load()
            ref = grid_search_match(rl, xl, grid_size=self.grid_size)
            return rl, xl, ref

        if self.load_mode == "matchable":
            return self._sample_matchable_load()

        # mixed mode
        if float(self.np_random.random()) < self.matchable_prob:
            return self._sample_matchable_load()
        rl, xl = self._sample_full_load()
        ref = grid_search_match(rl, xl, grid_size=self.grid_size)
        return rl, xl, ref

    @staticmethod
    def _validate_load(rl: float, xl: float) -> None:
        if not (math.isfinite(rl) and math.isfinite(xl)):
            raise ValueError(f"Load RL and XL must be finite; got RL={rl}, XL={xl}.")
        if rl < 0.0:
            raise ValueError(f"Passive load resistance RL must be >= 0; got RL={rl}.")

    def _build_observation(self, sim_result: dict[str, Any]) -> np.ndarray:
        gamma = sim_result["gamma"]
        obs = np.array(
            [float(gamma.real), float(gamma.imag), self.V1, self.V2],
            dtype=np.float32,
        )
        if obs.shape != self.observation_space_shape or not np.all(np.isfinite(obs)):
            raise RuntimeError(f"Invalid or non-finite observation produced: {obs!r}")
        return obs

    def _check_matched(self, gamma_abs: float, vswr: float) -> bool:
        """Check whether the agent satisfied the load-aware matching criterion."""
        best_vswr = float(self.best_ref["best_vswr"])
        best_gamma_abs = float(self.best_ref["best_gamma_abs"])

        if best_vswr < self.success_vswr:
            return bool(vswr < self.success_vswr)
        if best_vswr < 2.0:
            return bool(gamma_abs <= best_gamma_abs + self.match_tolerance)
        # Physically unachievable loads (best_vswr >= 2.0) never falsely terminate as matched
        return False

    def _build_info(
        self,
        sim_result: dict[str, Any],
        is_matched: bool,
        action: Optional[int] = None,
        reward_components: Optional[dict[str, float]] = None,
    ) -> dict[str, Any]:
        gamma = sim_result["gamma"]
        gamma_abs = float(sim_result["gamma_abs"])
        vswr = float(sim_result["vswr"])
        best_gamma_abs = float(self.best_ref["best_gamma_abs"])
        best_vswr = float(self.best_ref["best_vswr"])

        info: dict[str, Any] = {
            "RL": self.RL,
            "XL": self.XL,
            "V1": self.V1,
            "V2": self.V2,
            "Zin": sim_result["Zin"],
            "gamma": gamma,
            "gamma_real": float(gamma.real),
            "gamma_imag": float(gamma.imag),
            "gamma_abs": gamma_abs,
            "vswr": vswr,
            "best_V1": float(self.best_ref["best_V1"]),
            "best_V2": float(self.best_ref["best_V2"]),
            "best_gamma_abs": best_gamma_abs,
            "best_vswr": best_vswr,
            "is_load_achievable": bool(best_vswr < 2.0),
            "suboptimality_gap": float(gamma_abs - best_gamma_abs),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "is_matched": bool(is_matched),
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        if action is not None:
            info["action"] = action
            info["action_name"] = ACTION_NAMES[action]
        if reward_components is not None:
            info["reward_components"] = reward_components
        return info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset episode state following Gymnasium conventions: returns (obs, info)."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        opts = options or {}
        self.RL, self.XL, self.best_ref = self._select_episode_load(
            opts.get("RL"), opts.get("XL")
        )

        if "V1" in opts and opts["V1"] is not None:
            self.V1 = _clamp_voltage(opts["V1"])
        elif self.random_initial_v:
            self.V1 = self._sample_random_voltage()
        else:
            self.V1 = _clamp_voltage(0.5 * (config.V_MIN + config.V_MAX))

        if "V2" in opts and opts["V2"] is not None:
            self.V2 = _clamp_voltage(opts["V2"])
        elif self.random_initial_v:
            self.V2 = self._sample_random_voltage()
        else:
            self.V2 = _clamp_voltage(0.5 * (config.V_MIN + config.V_MAX))

        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self._episode_started = True

        self.last_sim = compute(self.RL, self.XL, self.V1, self.V2)
        gamma_abs = float(self.last_sim["gamma_abs"])
        if not math.isfinite(gamma_abs):
            raise RuntimeError(f"Non-finite gamma_abs from physics.compute: {gamma_abs}")
        self.prev_gamma_abs = gamma_abs

        obs = self._build_observation(self.last_sim)
        is_matched = self._check_matched(gamma_abs, float(self.last_sim["vswr"]))
        info = self._build_info(self.last_sim, is_matched=is_matched)
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step environment by one discrete action: returns (obs, reward, terminated, truncated, info)."""
        if not self._episode_started:
            raise RuntimeError("Environment must be reset() before calling step().")
        if self.terminated or self.truncated:
            raise RuntimeError("Episode has ended; call reset() before calling step().")

        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action must be an integer in 0..4; got {action!r}.")
        action_int = int(action)
        if action_int not in ACTION_NAMES:
            raise ValueError(f"Invalid action {action_int}; expected an integer in 0..4.")

        if action_int == 0:
            self.V1 = _clamp_voltage(self.V1 + config.V_STEP)
        elif action_int == 1:
            self.V1 = _clamp_voltage(self.V1 - config.V_STEP)
        elif action_int == 2:
            self.V2 = _clamp_voltage(self.V2 + config.V_STEP)
        elif action_int == 3:
            self.V2 = _clamp_voltage(self.V2 - config.V_STEP)
        # action_int == 4: Hold (V1, V2 unchanged)

        self.step_count += 1
        self.last_sim = compute(self.RL, self.XL, self.V1, self.V2)
        gamma_abs = float(self.last_sim["gamma_abs"])
        vswr = float(self.last_sim["vswr"])
        if not math.isfinite(gamma_abs):
            raise RuntimeError(f"Non-finite gamma_abs from physics.compute: {gamma_abs}")

        best_gamma_abs = float(self.best_ref["best_gamma_abs"])
        is_achievable = float(self.best_ref["best_vswr"]) < 2.0
        is_matched = self._check_matched(gamma_abs, vswr)

        # Component-wise reward with unachievable-load safeguards
        progress_reward = 20.0 * (self.prev_gamma_abs - gamma_abs)
        mismatch_penalty = -0.5 * max(0.0, gamma_abs - best_gamma_abs)
        vswr_penalty = -0.5 if (is_achievable and vswr > 2.0) else 0.0
        step_cost = -0.1
        success_bonus = 50.0 if is_matched else 0.0
        total_reward = float(
            progress_reward
            + mismatch_penalty
            + vswr_penalty
            + step_cost
            + success_bonus
        )

        reward_components = {
            "progress": float(progress_reward),
            "mismatch": float(mismatch_penalty),
            "vswr_penalty": float(vswr_penalty),
            "step_cost": float(step_cost),
            "success_bonus": float(success_bonus),
            "total": total_reward,
        }
        if not all(math.isfinite(val) for val in reward_components.values()):
            raise RuntimeError(f"Non-finite reward computed: {reward_components!r}")

        self.terminated = bool(is_matched)
        self.truncated = bool((not self.terminated) and (self.step_count >= self.max_steps))
        self.prev_gamma_abs = gamma_abs

        obs = self._build_observation(self.last_sim)
        info = self._build_info(
            self.last_sim,
            is_matched=is_matched,
            action=action_int,
            reward_components=reward_components,
        )
        return obs, total_reward, self.terminated, self.truncated, info
