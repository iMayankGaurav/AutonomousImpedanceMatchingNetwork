"""M4: Trained agent loading and deterministic greedy inference interface.

Implements the Module 4 -> Module 5 inference contract from Project.md Section 4.4:

    from rl.agent import load_agent

    agent = load_agent("models/model.pt")
    action = agent.act(state)  # state = [gamma_re, gamma_im, V1, V2]; action in {0..4}

Action encoding:
    0: Increase V1 by 0.05 V (V1+)
    1: Decrease V1 by 0.05 V (V1-)
    2: Increase V2 by 0.05 V (V2+)
    3: Decrease V2 by 0.05 V (V2-)
    4: Hold both voltages constant (Hold)

This module runs strictly in greedy evaluation mode (epsilon=0, torch.no_grad(),
online Q-network only) with frozen parameters and no replay buffer or training logic.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.dqn import QNetwork, resolve_device
from rl.env import ACTION_NAMES

DEFAULT_MODEL_PATH = Path("models/model.pt")


class InferenceAgent:
    """Read-only, greedy Double DQN inference wrapper for deployment."""

    action_names: dict[int, str] = dict(ACTION_NAMES)

    def __init__(
        self,
        online_net: QNetwork,
        device: torch.device,
        config_dict: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        update_steps: int = 0,
        model_path: Optional[Path] = None,
    ) -> None:
        self.online_net = online_net.to(device)
        self.online_net.eval()
        for param in self.online_net.parameters():
            param.requires_grad_(False)

        self.device = device
        self.state_dim: int = int(self.online_net.state_dim)
        self.action_dim: int = int(self.online_net.action_dim)
        self.config: dict[str, Any] = copy.deepcopy(config_dict or {})
        self.metadata: dict[str, Any] = copy.deepcopy(metadata or {})
        self.update_steps: int = int(update_steps)
        self.model_path: Optional[Path] = model_path

    def _validate_state(self, state: Sequence[float] | np.ndarray) -> np.ndarray:
        """Validate 4D finite state [gamma.real, gamma.imag, V1, V2]."""
        if state is None or isinstance(state, (str, bytes, dict)):
            raise ValueError(
                f"Expected a 1D sequence of {self.state_dim} finite numbers "
                f"[gamma.real, gamma.imag, V1, V2]; got {type(state).__name__}."
            )

        if isinstance(state, np.ndarray):
            if state.dtype == np.bool_ or state.dtype.kind not in ("i", "u", "f"):
                raise ValueError(
                    f"State array must have a real numeric dtype; got {state.dtype}."
                )
            arr = np.asarray(state, dtype=np.float32)
        else:
            if not isinstance(state, Sequence):
                try:
                    state_list = list(state)
                except TypeError as exc:
                    raise ValueError(
                        f"Expected an iterable of {self.state_dim} numbers; got {type(state).__name__}."
                    ) from exc
            else:
                state_list = list(state)

            if len(state_list) != self.state_dim:
                raise ValueError(
                    f"Expected state of length {self.state_dim} "
                    f"[gamma.real, gamma.imag, V1, V2]; got length {len(state_list)}."
                )
            for idx, item in enumerate(state_list):
                if isinstance(item, (bool, np.bool_)) or not isinstance(
                    item, (int, float, np.integer, np.floating)
                ):
                    raise ValueError(
                        f"State element at index {idx} must be a real number "
                        f"(not bool/complex/str); got {item!r}."
                    )
            arr = np.asarray(state_list, dtype=np.float32)

        if arr.ndim != 1 or arr.shape != (self.state_dim,):
            raise ValueError(
                f"Expected state shape ({self.state_dim},) representing "
                f"[gamma.real, gamma.imag, V1, V2]; got shape {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"State contains non-finite values (NaN or Inf): {arr!r}.")
        return arr

    def q_values(self, state: Sequence[float] | np.ndarray) -> np.ndarray:
        """Compute Q-values for all 5 actions under torch.no_grad() and eval()."""
        arr = self._validate_state(state)
        self.online_net.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            q = self.online_net(tensor).squeeze(0).detach().cpu().numpy()
        return q

    def act(self, state: Sequence[float] | np.ndarray) -> int:
        """Select deterministic greedy action ID in {0..4} for the given observation."""
        q = self.q_values(state)
        return int(np.argmax(q))


def load_agent(
    path: str | Path = DEFAULT_MODEL_PATH,
    device: Optional[str | torch.device] = None,
) -> InferenceAgent:
    """Load a trained Double DQN checkpoint for deterministic greedy inference.

    Parameters
    ----------
    path : str or Path, optional
        Path to the saved PyTorch checkpoint (default: ``"models/model.pt"``).
    device : str or torch.device, optional
        Target inference device (``None`` / ``"auto"``, ``"cpu"``, or ``"cuda"``).

    Returns
    -------
    InferenceAgent
        Frozen greedy inference agent exposing ``agent.act(state) -> int``.
    """
    if path is None or isinstance(path, bool) or not isinstance(path, (str, Path)):
        raise TypeError(f"Model path must be a str or Path; got {path!r}.")
    if isinstance(path, str) and not path.strip():
        raise ValueError("Model path cannot be empty.")

    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Trained model checkpoint not found: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"Model path must point to a file, not a directory: {candidate}")

    resolved_device = resolve_device(device)
    try:
        checkpoint = torch.load(
            candidate, map_location=resolved_device, weights_only=True
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to load valid model checkpoint from {candidate}: {exc}"
        ) from exc

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Invalid checkpoint format in {candidate}: expected dict, got {type(checkpoint).__name__}."
        )
    if "config" not in checkpoint or "online_state_dict" not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint in {candidate}: missing required 'config' or 'online_state_dict'."
        )

    cfg = checkpoint["config"]
    try:
        online_net = QNetwork(
            state_dim=int(cfg["state_dim"]),
            action_dim=int(cfg["action_dim"]),
            hidden_dims=tuple(int(h) for h in cfg["hidden_dims"]),
            normalize_inputs=bool(cfg.get("normalize_inputs", True)),
        )
        online_net.load_state_dict(checkpoint["online_state_dict"])
    except Exception as exc:
        raise ValueError(
            f"Incompatible model architecture or state_dict in {candidate}: {exc}"
        ) from exc

    return InferenceAgent(
        online_net=online_net,
        device=resolved_device,
        config_dict=cfg,
        metadata=checkpoint.get("metadata", {}),
        update_steps=int(checkpoint.get("update_steps", 0)),
        model_path=candidate,
    )


def _example_usage() -> None:
    """Minimal standalone usage example matching Project.md Section 4.4."""
    agent = load_agent("models/model.pt")
    sample_state = [0.25, -0.40, 1.65, 1.65]  # [gamma.real, gamma.imag, V1, V2]
    action = agent.act(sample_state)
    print(
        f"Loaded {agent.model_path} on {agent.device}; "
        f"state={sample_state} -> action={action} ({agent.action_names[action]})"
    )


if __name__ == "__main__":
    _example_usage()
