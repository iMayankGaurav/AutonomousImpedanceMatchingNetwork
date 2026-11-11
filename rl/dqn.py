"""M4: Double Deep Q-Network (Double DQN), replay buffer, and checkpointing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

import config


def resolve_device(device: Optional[str | torch.device] = None) -> torch.device:
    """Resolve torch device, preferring CUDA when available unless explicitly specified."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but CUDA is not available.")
    return resolved


def compute_epsilon(
    env_step: int,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    decay_steps: int = 120_000,
) -> float:
    """Linearly decay epsilon by environment steps, clamped to [eps_end, eps_start]."""
    if env_step < 0:
        raise ValueError(f"env_step must be >= 0; got {env_step}.")
    if decay_steps <= 0:
        return float(eps_end)
    fraction = min(1.0, float(env_step) / float(decay_steps))
    return float(eps_start + fraction * (eps_end - eps_start))


class QNetwork(nn.Module):
    """Fully connected Q-network mapping observation state -> action Q-values."""

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 5,
        hidden_dims: Sequence[int] = (128, 128),
        normalize_inputs: bool = True,
    ) -> None:
        super().__init__()
        if state_dim < 1 or action_dim < 1:
            raise ValueError("state_dim and action_dim must be >= 1.")
        if len(hidden_dims) != 2 or any(h < 1 for h in hidden_dims):
            raise ValueError(f"hidden_dims must contain two positive integers; got {hidden_dims}.")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.normalize_inputs = bool(normalize_inputs)

        # Register input normalization buffers so inference accepts raw [gamma_re, gamma_im, V1, V2]
        if self.normalize_inputs and self.state_dim == 4:
            v_center = 0.5 * (config.V_MIN + config.V_MAX)
            v_half = 0.5 * (config.V_MAX - config.V_MIN)
            center = torch.tensor([0.0, 0.0, v_center, v_center], dtype=torch.float32)
            scale = torch.tensor([1.0, 1.0, v_half, v_half], dtype=torch.float32)
        else:
            center = torch.zeros(self.state_dim, dtype=torch.float32)
            scale = torch.ones(self.state_dim, dtype=torch.float32)
        self.register_buffer("input_center", center)
        self.register_buffer("input_scale", scale)

        h1, h2 = self.hidden_dims
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, self.action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2 or x.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected input shape (batch, {self.state_dim}); got {tuple(x.shape)}."
            )
        x_norm = (x - self.input_center) / self.input_scale
        return self.net(x_norm)


class ReplayBuffer:
    """Fixed-capacity circular experience replay buffer backed by NumPy arrays."""

    def __init__(
        self,
        capacity: int = 100_000,
        state_dim: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1; got {capacity}.")
        if state_dim < 1:
            raise ValueError(f"state_dim must be >= 1; got {state_dim}.")

        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.rng = np.random.default_rng(seed)

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.terminateds = np.zeros(self.capacity, dtype=np.float32)

        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        state: Sequence[float] | np.ndarray,
        action: int,
        reward: float,
        next_state: Sequence[float] | np.ndarray,
        terminated: bool,
    ) -> None:
        s_arr = np.asarray(state, dtype=np.float32)
        ns_arr = np.asarray(next_state, dtype=np.float32)
        if s_arr.shape != (self.state_dim,) or ns_arr.shape != (self.state_dim,):
            raise ValueError(
                f"Expected states of shape ({self.state_dim},); got {s_arr.shape} and {ns_arr.shape}."
            )
        if not (np.all(np.isfinite(s_arr)) and np.all(np.isfinite(ns_arr))):
            raise ValueError("Cannot store non-finite state or next_state in ReplayBuffer.")
        reward_f = float(reward)
        if not math.isfinite(reward_f):
            raise ValueError(f"Cannot store non-finite reward={reward!r} in ReplayBuffer.")
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action must be an integer; got {action!r}.")

        idx = self._pos
        self.states[idx] = s_arr
        self.actions[idx] = int(action)
        self.rewards[idx] = reward_f
        self.next_states[idx] = ns_arr
        self.terminateds[idx] = 1.0 if bool(terminated) else 0.0

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size}.")
        if batch_size > self._size:
            raise ValueError(
                f"Requested batch_size={batch_size} exceeds buffer size={self._size}."
            )

        indices = self.rng.choice(self._size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminateds[indices],
        )


class DoubleDQNAgent:
    """Double Deep Q-Network agent with target network, replay buffer, and Huber loss."""

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 5,
        hidden_dims: Sequence[int] = (128, 128),
        lr: float = 3e-4,
        gamma: float = 0.99,
        huber_beta: float = 1.0,
        max_grad_norm: float = 10.0,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        tau: float = 1.0,
        normalize_inputs: bool = True,
        device: Optional[str | torch.device] = None,
        seed: Optional[int] = None,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be > 0.")
        if not (0.0 <= gamma < 1.0):
            raise ValueError("gamma must be in [0.0, 1.0).")
        if huber_beta <= 0.0:
            raise ValueError("huber_beta must be > 0.")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be > 0.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if target_update_freq < 1:
            raise ValueError("target_update_freq must be >= 1.")
        if not (0.0 < tau <= 1.0):
            raise ValueError("tau must be in (0.0, 1.0].")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.huber_beta = float(huber_beta)
        self.max_grad_norm = float(max_grad_norm)
        self.buffer_capacity = int(buffer_capacity)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.tau = float(tau)
        self.normalize_inputs = bool(normalize_inputs)
        self.device = resolve_device(device)

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        self.rng = np.random.default_rng(seed)

        self.online_net = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            normalize_inputs=self.normalize_inputs,
        ).to(self.device)
        self.target_net = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            normalize_inputs=self.normalize_inputs,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss(beta=self.huber_beta)
        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_capacity,
            state_dim=self.state_dim,
            seed=seed,
        )
        self.update_steps: int = 0

    def _validate_state_array(self, state: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32)
        if arr.shape != (self.state_dim,):
            raise ValueError(f"Expected state shape ({self.state_dim},); got {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"State contains non-finite values: {arr!r}.")
        return arr

    def q_values(self, state: Sequence[float] | np.ndarray) -> np.ndarray:
        """Compute action Q-values without gradient tracking."""
        arr = self._validate_state_array(state)
        was_training = self.online_net.training
        self.online_net.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            q = self.online_net(tensor).squeeze(0).detach().cpu().numpy()
        if was_training:
            self.online_net.train()
        return q

    def select_action(
        self, state: Sequence[float] | np.ndarray, epsilon: float = 0.0
    ) -> int:
        """Epsilon-greedy action selection; greedy branch runs under torch.no_grad()."""
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0.0, 1.0]; got {epsilon}.")
        self._validate_state_array(state)

        if epsilon > 0.0 and float(self.rng.random()) < epsilon:
            return int(self.rng.integers(0, self.action_dim))

        q = self.q_values(state)
        return int(np.argmax(q))

    def act(self, state: Sequence[float] | np.ndarray) -> int:
        """Greedy inference interface compatible with Section 4.4 (`agent.act(state)`)."""
        return self.select_action(state, epsilon=0.0)

    def store_transition(
        self,
        state: Sequence[float] | np.ndarray,
        action: int,
        reward: float,
        next_state: Sequence[float] | np.ndarray,
        terminated: bool,
    ) -> None:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action must be an integer in 0..{self.action_dim - 1}; got {action!r}.")
        if not (0 <= int(action) < self.action_dim):
            raise ValueError(f"Action {action} out of range 0..{self.action_dim - 1}.")
        self.replay_buffer.add(state, int(action), reward, next_state, terminated)

    def compute_double_dqn_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminateds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Double DQN TD targets:
        1. Select argmax action on next_states using online_net.
        2. Evaluate that selected action using target_net.
        """
        with torch.no_grad():
            next_q_online = self.online_net(next_states)
            best_next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states).gather(1, best_next_actions).squeeze(1)
            targets = rewards + self.gamma * (1.0 - terminateds) * next_q_target
        return targets

    def sync_target_network(self, tau: Optional[float] = None) -> None:
        """Synchronize target_net from online_net (hard copy when tau=1.0, else Polyak)."""
        mix = self.tau if tau is None else float(tau)
        if mix == 1.0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            return
        with torch.no_grad():
            for target_p, online_p in zip(
                self.target_net.parameters(), self.online_net.parameters()
            ):
                target_p.data.mul_(1.0 - mix).add_(online_p.data, alpha=mix)

    def update(self, batch_size: Optional[int] = None) -> Optional[dict[str, float | bool]]:
        """Sample a minibatch and perform one Double DQN gradient step."""
        bs = self.batch_size if batch_size is None else int(batch_size)
        if len(self.replay_buffer) < bs:
            return None

        states_np, actions_np, rewards_np, next_states_np, terms_np = (
            self.replay_buffer.sample(bs)
        )
        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.from_numpy(actions_np).to(self.device)
        rewards = torch.from_numpy(rewards_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        terminateds = torch.from_numpy(terms_np).to(self.device)

        self.online_net.train()
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        targets = self.compute_double_dqn_targets(rewards, next_states, terminateds)

        if not (torch.isfinite(current_q).all() and torch.isfinite(targets).all()):
            raise RuntimeError("Non-finite Q-values or TD targets encountered during update.")

        loss = self.criterion(current_q, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss encountered: {loss.item()}")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_grad_norm).item()
        )
        self.optimizer.step()

        self.update_steps += 1
        target_synced = False
        if self.tau < 1.0:
            self.sync_target_network(self.tau)
            target_synced = True
        elif self.update_steps % self.target_update_freq == 0:
            self.sync_target_network(1.0)
            target_synced = True

        return {
            "loss": float(loss.item()),
            "mean_q": float(current_q.mean().item()),
            "mean_target": float(targets.mean().item()),
            "grad_norm": grad_norm,
            "target_synced": target_synced,
        }

    def save(
        self,
        path: str | Path,
        overwrite: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Save model weights and hyperparameters safely (only plain primitives + tensors)."""
        target_path = Path(path)
        if target_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing checkpoint: {target_path}"
            )
        target_path.parent.mkdir(parents=True, exist_ok=True)

        safe_metadata: dict[str, Any] = {}
        if metadata is not None:
            for k, v in metadata.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    safe_metadata[str(k)] = v
                else:
                    safe_metadata[str(k)] = str(v)

        checkpoint = {
            "format_version": 1,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dims": list(self.hidden_dims),
                "lr": self.lr,
                "gamma": self.gamma,
                "huber_beta": self.huber_beta,
                "max_grad_norm": self.max_grad_norm,
                "buffer_capacity": self.buffer_capacity,
                "batch_size": self.batch_size,
                "target_update_freq": self.target_update_freq,
                "tau": self.tau,
                "normalize_inputs": self.normalize_inputs,
            },
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_steps": self.update_steps,
            "metadata": safe_metadata,
        }
        torch.save(checkpoint, target_path)
        return target_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: Optional[str | torch.device] = None,
        seed: Optional[int] = None,
    ) -> "DoubleDQNAgent":
        """Load checkpoint safely with weights_only=True."""
        resolved_device = resolve_device(device)
        checkpoint = torch.load(path, map_location=resolved_device, weights_only=True)
        cfg = checkpoint["config"]
        agent = cls(
            state_dim=int(cfg["state_dim"]),
            action_dim=int(cfg["action_dim"]),
            hidden_dims=tuple(int(h) for h in cfg["hidden_dims"]),
            lr=float(cfg["lr"]),
            gamma=float(cfg["gamma"]),
            huber_beta=float(cfg["huber_beta"]),
            max_grad_norm=float(cfg["max_grad_norm"]),
            buffer_capacity=int(cfg["buffer_capacity"]),
            batch_size=int(cfg["batch_size"]),
            target_update_freq=int(cfg["target_update_freq"]),
            tau=float(cfg["tau"]),
            normalize_inputs=bool(cfg.get("normalize_inputs", True)),
            device=resolved_device,
            seed=seed,
        )
        agent.online_net.load_state_dict(checkpoint["online_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.update_steps = int(checkpoint.get("update_steps", 0))
        agent.online_net.eval()
        agent.target_net.eval()
        return agent
