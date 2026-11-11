"""Unit tests for rl/dqn.py (Stage 2: Double DQN verification)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.dqn import (
    DoubleDQNAgent,
    QNetwork,
    ReplayBuffer,
    compute_epsilon,
    resolve_device,
)


class TestDoubleDQN(unittest.TestCase):
    def test_qnetwork_architecture_and_dimensions(self):
        net = QNetwork(state_dim=4, action_dim=5, hidden_dims=(128, 128))
        linear_layers = [m for m in net.net if isinstance(m, torch.nn.Linear)]
        relu_layers = [m for m in net.net if isinstance(m, torch.nn.ReLU)]
        self.assertEqual(len(linear_layers), 3)
        self.assertEqual(len(relu_layers), 2)
        self.assertEqual((linear_layers[0].in_features, linear_layers[0].out_features), (4, 128))
        self.assertEqual((linear_layers[1].in_features, linear_layers[1].out_features), (128, 128))
        self.assertEqual((linear_layers[2].in_features, linear_layers[2].out_features), (128, 5))

        x_single = torch.tensor([0.2, -0.3, 1.65, 2.5], dtype=torch.float32)
        out_single = net(x_single)
        self.assertEqual(out_single.shape, (1, 5))

        x_batch = torch.randn(16, 4, dtype=torch.float32)
        out_batch = net(x_batch)
        self.assertEqual(out_batch.shape, (16, 5))

        with self.assertRaises(ValueError):
            net(torch.randn(8, 6))

    def test_action_selection_and_no_grad_inference(self):
        agent = DoubleDQNAgent(state_dim=4, action_dim=5, seed=42)
        state = np.array([0.1, -0.2, 1.5, 2.0], dtype=np.float32)

        # Greedy action must match argmax of q_values and never create grad history
        q_vals = agent.q_values(state)
        self.assertEqual(q_vals.shape, (5,))
        greedy = agent.select_action(state, epsilon=0.0)
        self.assertEqual(greedy, int(np.argmax(q_vals)))
        self.assertEqual(agent.act(state), greedy)
        for p in agent.online_net.parameters():
            self.assertIsNone(p.grad)

        # Pure exploration (epsilon=1.0) visits all 5 actions
        explored = {agent.select_action(state, epsilon=1.0) for _ in range(100)}
        self.assertEqual(explored, {0, 1, 2, 3, 4})

        # Step-based epsilon decay check
        self.assertAlmostEqual(compute_epsilon(0, 1.0, 0.05, 1000), 1.0)
        self.assertAlmostEqual(compute_epsilon(500, 1.0, 0.05, 1000), 0.525)
        self.assertAlmostEqual(compute_epsilon(2000, 1.0, 0.05, 1000), 0.05)

    def test_replay_buffer_storage_wraparound_and_sampling(self):
        buf = ReplayBuffer(capacity=5, state_dim=4, seed=7)
        self.assertEqual(len(buf), 0)

        for i in range(8):
            s = np.array([0.1 * i, -0.1 * i, 1.0, 2.0], dtype=np.float32)
            ns = s + 0.05
            buf.add(s, i % 5, float(i), ns, i == 7)

        self.assertEqual(len(buf), 5)
        states, actions, rewards, next_states, terms = buf.sample(3)
        self.assertEqual(states.shape, (3, 4))
        self.assertEqual(actions.shape, (3,))
        self.assertEqual(rewards.shape, (3,))
        self.assertEqual(next_states.shape, (3, 4))
        self.assertEqual(terms.shape, (3,))
        # Oldest 3 entries (rewards 0, 1, 2) were overwritten by 5, 6, 7
        self.assertTrue(np.all(rewards >= 3.0))

        with self.assertRaises(ValueError):
            buf.sample(10)
        with self.assertRaises(ValueError):
            buf.add([float("nan"), 0.0, 1.0, 1.0], 0, 0.0, [0.0, 0.0, 1.0, 1.0], False)

    def test_double_dqn_target_uses_online_argmax_and_target_value(self):
        """Construct controlled Q outputs where Standard DQN != Double DQN target."""
        agent = DoubleDQNAgent(state_dim=4, action_dim=5, gamma=0.9, seed=1)

        # Force deterministic bias-only Q-values on both networks for easy exact verification:
        # online_net Q(s') = [1, 10, 2, 3, 4] -> online argmax is action 1!
        # target_net Q(s') = [100, 5, 6, 7, 8] -> standard DQN max would pick 100 (action 0),
        # whereas Double DQN picks online argmax (action 1) and evaluates target_net[1] = 5!
        with torch.no_grad():
            for net, vals in [
                (agent.online_net, [1.0, 10.0, 2.0, 3.0, 4.0]),
                (agent.target_net, [100.0, 5.0, 6.0, 7.0, 8.0]),
            ]:
                for layer in net.net:
                    if isinstance(layer, torch.nn.Linear):
                        layer.weight.zero_()
                        layer.bias.zero_()
                net.net[-1].bias.copy_(torch.tensor(vals, dtype=torch.float32))

        rewards = torch.tensor([2.0, 2.0], dtype=torch.float32, device=agent.device)
        next_states = torch.zeros((2, 4), dtype=torch.float32, device=agent.device)
        terminateds = torch.tensor([0.0, 1.0], dtype=torch.float32, device=agent.device)

        targets = agent.compute_double_dqn_targets(rewards, next_states, terminateds)
        # Non-terminal: 2.0 + 0.9 * 5.0 = 6.5 (NOT 2.0 + 0.9 * 100.0 = 92.0!)
        # Terminal: 2.0 + 0.9 * 0 * 5.0 = 2.0
        self.assertAlmostEqual(float(targets[0].item()), 6.5, places=5)
        self.assertAlmostEqual(float(targets[1].item()), 2.0, places=5)

    def test_training_update_gradient_clipping_and_target_sync(self):
        agent = DoubleDQNAgent(
            state_dim=4,
            action_dim=5,
            batch_size=8,
            target_update_freq=3,
            max_grad_norm=1.0,
            seed=12,
        )
        # Not enough samples yet -> returns None
        self.assertIsNone(agent.update())

        rng = np.random.default_rng(12)
        for i in range(20):
            s = rng.uniform(-0.5, 0.5, size=4).astype(np.float32)
            s[2:] = np.clip(s[2:] + 1.5, 0.0, 3.3)
            ns = s.copy()
            ns[0] *= 0.8
            agent.store_transition(s, i % 5, 1.5, ns, i % 7 == 0)

        w_before = agent.online_net.net[0].weight.detach().clone()
        t_before = agent.target_net.net[0].weight.detach().clone()

        stats1 = agent.update()
        stats2 = agent.update()
        stats3 = agent.update()
        assert stats1 is not None and stats2 is not None and stats3 is not None

        self.assertFalse(stats1["target_synced"])
        self.assertFalse(stats2["target_synced"])
        self.assertTrue(stats3["target_synced"])  # 3rd update triggers target_update_freq=3
        self.assertTrue(np.isfinite(stats3["loss"]))
        self.assertFalse(torch.equal(agent.online_net.net[0].weight, w_before))
        self.assertFalse(torch.equal(agent.target_net.net[0].weight, t_before))
        for k, v in agent.online_net.state_dict().items():
            self.assertTrue(torch.equal(v, agent.target_net.state_dict()[k]))

    def test_safe_save_and_load_weights_only(self):
        agent = DoubleDQNAgent(state_dim=4, action_dim=5, seed=99)
        state = np.array([0.3, -0.4, 2.1, 0.9], dtype=np.float32)
        q_orig = agent.q_values(state)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_model.pt"
            agent.save(ckpt_path, overwrite=False, metadata={"episodes": 10})
            self.assertTrue(ckpt_path.exists())

            # Default overwrite=False refuses to overwrite existing file
            with self.assertRaises(FileExistsError):
                agent.save(ckpt_path, overwrite=False)

            loaded = DoubleDQNAgent.load(ckpt_path, device=resolve_device("auto"))
            q_loaded = loaded.q_values(state)
            np.testing.assert_allclose(q_orig, q_loaded, atol=1e-6)
            self.assertEqual(loaded.act(state), agent.act(state))
            self.assertEqual(loaded.online_net.training, False)
            self.assertEqual(loaded.target_net.training, False)


if __name__ == "__main__":
    unittest.main()

