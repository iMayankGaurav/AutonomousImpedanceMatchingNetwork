"""Unit tests for the Gymnasium-style rl/env.py impedance matching environment."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from physics.pi_network import compute, grid_search_match
from rl.env import ImpedanceMatchingEnv


class TestImpedanceMatchingEnv(unittest.TestCase):
    def test_initialization_and_gymnasium_reset_contract(self):
        env = ImpedanceMatchingEnv(load_mode="matchable", max_steps=80, seed=42)
        self.assertEqual(env.action_space_n, 5)
        self.assertEqual(env.observation_space_shape, (4,))
        self.assertEqual(env.max_steps, 80)

        out = env.reset(seed=42, options={"RL": 10.0, "XL": 50.0, "V1": 1.25, "V2": 2.40})
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        obs, info = out

        self.assertEqual(obs.shape, (4,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertEqual(info["RL"], 10.0)
        self.assertEqual(info["XL"], 50.0)
        self.assertEqual(info["V1"], 1.25)
        self.assertEqual(info["V2"], 2.40)
        self.assertEqual(info["step_count"], 0)
        self.assertFalse(info["terminated"])
        self.assertFalse(info["truncated"])

    def test_invalid_init_and_input_validation(self):
        with self.assertRaises(ValueError):
            ImpedanceMatchingEnv(load_mode="invalid")
        with self.assertRaises(ValueError):
            ImpedanceMatchingEnv(max_steps=0)
        with self.assertRaises(ValueError):
            ImpedanceMatchingEnv(load_pool=[])

        env = ImpedanceMatchingEnv(load_mode="full", seed=1)
        # step() before reset() must fail
        with self.assertRaises(RuntimeError):
            env.step(4)

        # Invalid load values
        with self.assertRaises(ValueError):
            env.reset(options={"RL": -5.0, "XL": 10.0})
        with self.assertRaises(ValueError):
            env.reset(options={"RL": float("nan"), "XL": 10.0})
        with self.assertRaises(ValueError):
            env.reset(options={"RL": 20.0})  # Only one of RL/XL supplied

        # Invalid voltages and actions
        with self.assertRaises(ValueError):
            env.reset(options={"RL": 20.0, "XL": 10.0, "V1": float("inf"), "V2": 1.0})

        env.reset(options={"RL": 20.0, "XL": 10.0, "V1": 1.0, "V2": 1.0})
        for bad_action in [-1, 5, 1.5, True, "0"]:
            with self.assertRaises(ValueError):
                env.step(bad_action)  # type: ignore[arg-type]

    def test_all_five_actions_and_physics_alignment(self):
        env = ImpedanceMatchingEnv(load_mode="full", seed=7)
        obs, _ = env.reset(options={"RL": 15.0, "XL": 60.0, "V1": 1.50, "V2": 2.00})

        # Action 0: V1 + 0.05
        obs, reward, term, trunc, info = env.step(0)
        self.assertAlmostEqual(info["V1"], 1.55, places=6)
        self.assertAlmostEqual(info["V2"], 2.00, places=6)
        self.assertEqual(info["action_name"], "V1+")
        sim = compute(15.0, 60.0, 1.55, 2.00)
        np.testing.assert_allclose(
            obs, [sim["gamma"].real, sim["gamma"].imag, 1.55, 2.00], atol=1e-6
        )

        # Action 1: V1 - 0.05
        obs, _, _, _, info = env.step(1)
        self.assertAlmostEqual(info["V1"], 1.50, places=6)
        self.assertEqual(info["action_name"], "V1-")

        # Action 2: V2 + 0.05
        obs, _, _, _, info = env.step(2)
        self.assertAlmostEqual(info["V2"], 2.05, places=6)
        self.assertEqual(info["action_name"], "V2+")

        # Action 3: V2 - 0.05
        obs, _, _, _, info = env.step(3)
        self.assertAlmostEqual(info["V2"], 2.00, places=6)
        self.assertEqual(info["action_name"], "V2-")

        # Action 4: Hold
        obs_before = obs.copy()
        obs, reward, term, trunc, info = env.step(4)
        np.testing.assert_array_equal(obs, obs_before)
        self.assertEqual(info["action_name"], "Hold")
        self.assertAlmostEqual(info["reward_components"]["progress"], 0.0, places=7)

    def test_voltage_boundary_clamping(self):
        env = ImpedanceMatchingEnv(load_mode="full", seed=3)
        # Explicit out-of-range voltages clamp to [0.0, 3.3]
        obs, info = env.reset(options={"RL": 25.0, "XL": 40.0, "V1": -1.0, "V2": 5.0})
        self.assertEqual(info["V1"], config.V_MIN)
        self.assertEqual(info["V2"], config.V_MAX)

        # Stepping below V_MIN and above V_MAX stays clamped
        obs, _, _, _, info = env.step(1)  # V1- at 0.0
        self.assertEqual(info["V1"], 0.0)
        self.assertEqual(obs[2], np.float32(0.0))

        obs, _, _, _, info = env.step(2)  # V2+ at 3.3
        self.assertEqual(info["V2"], 3.3)
        self.assertEqual(obs[3], np.float32(3.3))

    def test_reward_calculation_and_components(self):
        # Achievable load RL=10, XL=50 matches near V1=3.3, V2=1.90
        env = ImpedanceMatchingEnv(load_mode="matchable", success_vswr=1.2, seed=5)
        _, info0 = env.reset(options={"RL": 10.0, "XL": 50.0, "V1": 3.3, "V2": 1.75})
        prev_gamma = info0["gamma_abs"]

        # Step closer (V2: 1.75 -> 1.80): VSWR ~ 1.2312 (> 1.2), non-terminal progress
        _, r1, term1, trunc1, info1 = env.step(2)
        rc1 = info1["reward_components"]
        expected_prog = 20.0 * (prev_gamma - info1["gamma_abs"])
        self.assertAlmostEqual(rc1["progress"], expected_prog, places=6)
        self.assertLessEqual(rc1["mismatch"], 0.0)
        self.assertEqual(rc1["vswr_penalty"], 0.0)
        self.assertEqual(rc1["step_cost"], -0.1)
        self.assertEqual(rc1["success_bonus"], 0.0)
        self.assertAlmostEqual(
            r1,
            rc1["progress"]
            + rc1["mismatch"]
            + rc1["vswr_penalty"]
            + rc1["step_cost"]
            + rc1["success_bonus"],
            places=6,
        )
        self.assertFalse(term1)
        self.assertFalse(trunc1)

        # Step onto match (V2: 1.80 -> 1.85): VSWR ~ 1.1457 < 1.2 -> terminal match
        _, r2, term2, trunc2, info2 = env.step(2)
        rc2 = info2["reward_components"]
        self.assertTrue(term2)
        self.assertFalse(trunc2)
        self.assertTrue(info2["is_matched"])
        self.assertEqual(rc2["success_bonus"], 50.0)
        self.assertGreater(r2, 45.0)

        # Stepping after termination raises RuntimeError
        with self.assertRaises(RuntimeError):
            env.step(4)

        # Also verify vswr_penalty fires on an achievable load when VSWR > 2.0
        env.reset(options={"RL": 10.0, "XL": 50.0, "V1": 0.0, "V2": 0.0})
        _, _, _, _, info_high = env.step(4)
        self.assertGreater(info_high["vswr"], 2.0)
        self.assertEqual(info_high["reward_components"]["vswr_penalty"], -0.5)

    def test_unachievable_load_no_misleading_vswr_penalty_or_false_match(self):
        # RL=50, XL=0 at default 100 MHz / 150 nH has best_vswr ~ 8.19 (unachievable)
        env = ImpedanceMatchingEnv(load_mode="full", max_steps=4, seed=9)
        _, info0 = env.reset(options={"RL": 50.0, "XL": 0.0, "V1": 3.3, "V2": 3.3})
        self.assertFalse(info0["is_load_achievable"])

        for _ in range(4):
            obs, reward, term, trunc, info = env.step(4)
            self.assertTrue(np.all(np.isfinite(obs)))
            self.assertTrue(np.isfinite(reward))
            self.assertEqual(info["reward_components"]["vswr_penalty"], 0.0)
            self.assertEqual(info["reward_components"]["success_bonus"], 0.0)
            self.assertAlmostEqual(info["reward_components"]["mismatch"], 0.0, places=6)
            self.assertFalse(term)

        self.assertTrue(trunc)
        with self.assertRaises(RuntimeError):
            env.step(4)

    def test_pure_reactive_load_no_nan_or_inf_observation(self):
        env = ImpedanceMatchingEnv(load_mode="full", max_steps=3, seed=10)
        obs, info0 = env.reset(options={"RL": 0.0, "XL": 50.0, "V1": 1.0, "V2": 1.0})
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertEqual(info0["vswr"], float("inf"))

        obs, reward, term, trunc, info = env.step(0)
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertTrue(np.isfinite(reward))
        self.assertEqual(info["vswr"], float("inf"))

    def test_multiple_randomly_sampled_loads_and_episode_immutability(self):
        env = ImpedanceMatchingEnv(
            load_mode="matchable", match_vswr_threshold=1.5, max_steps=10, seed=123
        )
        seen_loads = set()
        for _ in range(20):
            obs, info0 = env.reset()
            self.assertTrue(np.all(np.isfinite(obs)))
            rl0, xl0 = info0["RL"], info0["XL"]
            seen_loads.add((round(rl0, 4), round(xl0, 4)))

            # Verify M3 grid_search_match confirms load is genuinely matchable
            gs = grid_search_match(rl0, xl0, grid_size=env.grid_size)
            self.assertLess(gs["best_vswr"], 1.5)
            self.assertAlmostEqual(info0["best_gamma_abs"], gs["best_gamma_abs"], places=7)

            for step_i in range(10):
                obs, reward, term, trunc, info = env.step(step_i % 5)
                self.assertEqual(info["RL"], rl0)
                self.assertEqual(info["XL"], xl0)
                self.assertTrue(np.all(np.isfinite(obs)))
                self.assertTrue(np.isfinite(reward))
                if term or trunc:
                    break

        self.assertEqual(len(seen_loads), 20)

    def test_reproducibility_with_seed(self):
        env_a = ImpedanceMatchingEnv(load_mode="mixed", seed=999)
        env_b = ImpedanceMatchingEnv(load_mode="mixed", seed=999)
        obs_a, info_a = env_a.reset(seed=999)
        obs_b, info_b = env_b.reset(seed=999)
        np.testing.assert_array_equal(obs_a, obs_b)
        self.assertEqual((info_a["RL"], info_a["XL"]), (info_b["RL"], info_b["XL"]))

        for action in [0, 2, 1, 3, 4]:
            sa = env_a.step(action)
            sb = env_b.step(action)
            np.testing.assert_array_equal(sa[0], sb[0])
            self.assertEqual(sa[1:4], sb[1:4])


if __name__ == "__main__":
    unittest.main()
