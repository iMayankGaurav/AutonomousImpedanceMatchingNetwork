"""Unit tests for rl/agent.py greedy inference interface."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.agent import InferenceAgent, load_agent
from rl.dqn import DoubleDQNAgent
from rl.env import ImpedanceMatchingEnv


MODEL_PATH = PROJECT_ROOT / "models" / "model.pt"


class TestInferenceAgent(unittest.TestCase):
    def test_load_trained_model_and_frozen_eval_state(self):
        agent = load_agent(MODEL_PATH)
        self.assertIsInstance(agent, InferenceAgent)
        self.assertEqual(agent.state_dim, 4)
        self.assertEqual(agent.action_dim, 5)
        self.assertFalse(agent.online_net.training)
        self.assertTrue(agent.online_net.normalize_inputs)
        self.assertEqual(agent.metadata.get("completed_episodes"), 1000)
        self.assertGreater(agent.update_steps, 0)
        for param in agent.online_net.parameters():
            self.assertFalse(param.requires_grad)

    def test_default_load_agent_path_and_project_md_api(self):
        agent = load_agent()
        state = [0.2, -0.35, 1.65, 1.65]
        action = agent.act(state)
        self.assertIs(type(action), int)
        self.assertIn(action, {0, 1, 2, 3, 4})

    def test_matches_training_agent_predictions_and_normalization(self):
        inf_agent = load_agent(MODEL_PATH, device="cpu")
        full_agent = DoubleDQNAgent.load(MODEL_PATH, device="cpu")

        env = ImpedanceMatchingEnv(max_steps=20, seed=123)
        obs, _ = env.reset()
        states = [obs]
        for _ in range(5):
            act = inf_agent.act(obs)
            obs, _, terminated, truncated, _ = env.step(act)
            states.append(obs)
            if terminated or truncated:
                break

        for s in states:
            self.assertEqual(inf_agent.act(s), full_agent.act(s))
            np.testing.assert_allclose(inf_agent.q_values(s), full_agent.q_values(s), atol=1e-6)

    def test_deterministic_repeated_predictions_and_input_containers(self):
        agent = load_agent(MODEL_PATH, device="cpu")
        base_state = [0.42, -0.18, 2.10, 0.95]
        variants = [
            base_state,
            tuple(base_state),
            np.array(base_state, dtype=np.float32),
            np.array(base_state, dtype=np.float64),
        ]

        expected_action = agent.act(base_state)
        expected_q = agent.q_values(base_state)
        for _ in range(20):
            for variant in variants:
                act = agent.act(variant)
                self.assertIs(type(act), int)
                self.assertEqual(act, expected_action)
                np.testing.assert_allclose(agent.q_values(variant), expected_q, atol=1e-6)

    def test_no_weight_or_file_mutation_during_inference(self):
        file_bytes_before = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        agent = load_agent(MODEL_PATH, device="cpu")
        params_before = [p.detach().clone() for p in agent.online_net.parameters()]

        rng = np.random.default_rng(7)
        for _ in range(50):
            s = np.array(
                [
                    float(rng.uniform(-0.9, 0.9)),
                    float(rng.uniform(-0.9, 0.9)),
                    float(rng.uniform(0.0, 3.3)),
                    float(rng.uniform(0.0, 3.3)),
                ],
                dtype=np.float32,
            )
            act = agent.act(s)
            self.assertIn(act, {0, 1, 2, 3, 4})

        for p_before, p_after in zip(params_before, agent.online_net.parameters()):
            self.assertTrue(torch.equal(p_before, p_after))
        file_bytes_after = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        self.assertEqual(file_bytes_before, file_bytes_after)

    def test_missing_or_invalid_model_paths_and_corrupted_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with self.assertRaises(FileNotFoundError):
                load_agent(tmp / "does_not_exist.pt")
            with self.assertRaises(ValueError):
                load_agent(tmp)
            bad_file = tmp / "corrupt.pt"
            bad_file.write_text("not a checkpoint", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_agent(bad_file)
            incomplete_ckpt = tmp / "incomplete.pt"
            torch.save({"metadata": {}}, incomplete_ckpt)
            with self.assertRaises(ValueError):
                load_agent(incomplete_ckpt)

        with self.assertRaises(TypeError):
            load_agent(None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            load_agent("   ")

    def test_invalid_states_rejected_clearly(self):
        agent = load_agent(MODEL_PATH, device="cpu")
        invalid_states = [
            None,
            "0.1,0.2,1.0,1.0",
            {"gamma_re": 0.1},
            [0.1, 0.2, 1.0],
            [0.1, 0.2, 1.0, 1.0, 0.5],
            np.zeros((1, 4), dtype=np.float32),
            np.array(1.0, dtype=np.float32),
            [True, False, 1.0, 1.0],
            np.array([True, False, True, False]),
            [0.1 + 0.2j, 0.0, 1.0, 1.0],
            ["0.1", 0.2, 1.0, 1.0],
            [float("nan"), 0.0, 1.0, 1.0],
            [0.0, float("inf"), 1.0, 1.0],
            [0.0, 0.0, float("-inf"), 1.0],
        ]
        for bad in invalid_states:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    agent.act(bad)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

