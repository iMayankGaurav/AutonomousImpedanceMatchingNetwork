# Module 4 (RL Agent) Integration Guide for M5

**Target Audience:** M5 (Dashboard & Integration) and M2 (Mock/Serial Data)  
**Inference Entry Point:** `rl/agent.py`  
**Trained Checkpoint:** `models/model.pt` (`294,637` bytes, `update_steps = 71721`)

---

## 1. Quick Start (`Project.md` Section 4.4 Contract)

```python
from pathlib import Path
from physics.pi_network import compute
from rl.agent import load_agent

# Resolve relative to repository root so launching from any working directory works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
agent = load_agent(PROJECT_ROOT / "models" / "model.pt", device="cpu")

# Raw, unnormalized 4D state: [gamma.real, gamma.imag, V1, V2]
sim = compute(RL=15.0, XL=60.0, V1=1.65, V2=1.65)
gamma = sim["gamma"]
state = [float(gamma.real), float(gamma.imag), 1.65, 1.65]

action = agent.act(state)  # int in {0, 1, 2, 3, 4}
print(action, agent.action_names[action])
```

### API Reference (`rl/agent.py`)
- `load_agent(path=DEFAULT_MODEL_PATH, device=None) -> InferenceAgent`
  - Loads weights strictly with `weights_only=True`, puts `online_net` in `.eval()`, freezes all parameters (`requires_grad_(False)`), and wraps predictions in `torch.no_grad()`.
  - **Call once at startup** (e.g., via `@st.cache_resource` in `app/dashboard.py` or inside `MatchingPipeline.__init__`), **never inside the 20 Hz refresh loop**.
  - Pass `device="cpu"` explicitly on laptops to avoid unnecessary CUDA probes.
- `agent.act(state) -> int`: Returns deterministic greedy action `argmax_a Q(s, a)` in `{0, 1, 2, 3, 4}` (`epsilon = 0`).
- `agent.q_values(state) -> np.ndarray`: Optional helper returning all 5 Q-values (useful for UI diagnostics).
- `agent.action_names`: `{0: "V1+", 1: "V1-", 2: "V2+", 3: "V2-", 4: "Hold"}`.

---

## 2. Constructing the Exact 4D Observation State

The trained policy expects a 1D, 4-element real finite sequence (`list`, `tuple`, or 1D `np.ndarray`):

```text
state = [gamma.real, gamma.imag, V1, V2]
```

| Index | Field | Source | Valid Range / Units |
|---:|---|---|---|
| `0` | `gamma.real` | `compute(RL, XL, V1, V2)["gamma"].real` | `[-1.0, 1.0]` (dimensionless) |
| `1` | `gamma.imag` | `compute(RL, XL, V1, V2)["gamma"].imag` | `[-1.0, 1.0]` (dimensionless) |
| `2` | `V1` | Current virtual bias for source-side `C1` | `[0.0, 3.3]` V (**raw volts**) |
| `3` | `V2` | Current virtual bias for load-side `C2` | `[0.0, 3.3]` V (**raw volts**) |

**Critical Integration Rules:**
1. **Do NOT normalize `V1, V2` before calling `agent.act()`:** `QNetwork.forward()` (`rl/dqn.py`) already applies frozen internal normalization buffers `(x - [0.0, 0.0, 1.65, 1.65]) / [1.0, 1.0, 1.65, 1.65]`. Passing pre-normalized values will corrupt predictions.
2. **Do NOT pass `RL`, `XL`, `Zin`, `vswr`, or capacitances:** The trained input dimension is fixed at `4`.
3. **Preserve channel ordering:** `V1` (index `2`) corresponds to source-side shunt varactor `C1`; `V2` (index `3`) corresponds to load-side shunt varactor `C2`.

---

## 3. Applying Actions to Virtual Voltages (`V1, V2`)

`agent.act(state)` only returns a discrete action ID; M5 (`app/pipeline.py`) must maintain and update the virtual `(V1_auto, V2_auto)` state using `config.py` constants (`V_MIN = 0.0`, `V_MAX = 3.3`, `V_STEP = 0.05`):

```python
import numpy as np
import config

def apply_rl_action(v1: float, v2: float, action: int) -> tuple[float, float]:
    if action == 0:
        v1 += config.V_STEP
    elif action == 1:
        v1 -= config.V_STEP
    elif action == 2:
        v2 += config.V_STEP
    elif action == 3:
        v2 -= config.V_STEP
    # action == 4: Hold

    v1 = round(float(np.clip(v1, config.V_MIN, config.V_MAX)), 4)
    v2 = round(float(np.clip(v2, config.V_MIN, config.V_MAX)), 4)
    return v1, v2
```

- **Simulation Only (No Physical Actuation):** Applying an RL action modifies Python's software variables `(V1_auto, V2_auto)` before calling `physics.pi_network.compute(RL, XL, V1_auto, V2_auto)`. The STM32 hardware only streams 4 potentiometer values one-way over UART (`PA4` is wired to Pot 3 ADC, and there is no DAC/varactor RF board).

---

## 4. Manual vs. Auto (RL) Mode Architecture

Keep Manual and Auto cleanly decoupled inside a per-session stateful controller (`app/pipeline.py`, stored in Streamlit `st.session_state`):

1. **Always poll M2:** On every UI tick (~`20 Hz`), read `sample = source.get_latest()` (`{"RL", "XL", "V1_man", "V2_man", "t"}`) so telemetry, connection status, and timestamp freshness remain live in both modes.
2. **Manual Mode:**
   - Set `(V1_eff, V2_eff) = (V1_man, V2_man)` straight from Pots 3 & 4.
   - Skip `agent.act()` completely so Manual mode has **zero runtime dependency** on `models/model.pt`.
   - Reset Auto tuning latches/counters while in Manual mode.
3. **Auto (RL) Mode:**
   - **Bumpless Transfer (`Manual -> Auto`):** Seed `(V1_auto, V2_auto)` from current `(V1_man, V2_man)` clamped to `[0.0, 3.3] V`. (The agent was trained from arbitrary initial voltages across `[0.0, 3.3] V`.)
   - Ignore Pots 3 & 4 (`V1_man, V2_man`) while Auto is active.
   - Execute bounded RL sub-steps on `(V1_auto, V2_auto)` (e.g., `1–10` steps per 50 ms frame for visible Smith chart animation, or up to `140` steps to converge within ~`16 ms` CPU time).
   - Set `(V1_eff, V2_eff) = (V1_auto, V2_auto)` and evaluate `compute(RL, XL, V1_eff, V2_eff)`.

---

## 5. Mandatory Runtime Safeguards for M5 (`app/pipeline.py`)

Held-out testing (`rl/logs/eval_summary.json`) showed `86.57%` exact match and `99.71%` `VSWR < 2.0`, while `47/350` episodes reached the `140`-step limit due to boundary-rail pinning (`20/47`), 2-state oscillation cycles (`24/47`), or `Hold` stalling (`3/47`). Additionally, live potentiometer ADC readings at `20 Hz` exhibit small LSB jitter.

Implement these 5 lightweight guards around the Auto control loop:

1. **Settled Latch + Load-Change Deadband (`~1.0 Ω`):**
   - Once `vswr < 1.2` (or a guard triggers), latch `"Matched / Held"` and keep `(V1_auto, V2_auto)` constant.
   - Only unlock and reset the step counter when `max(|RL - RL_locked|, |XL - XL_locked|) > 1.0 Ω` (or on user `"Re-tune"` click) so ADC LSB noise does not continuously re-trigger stepping.
2. **Per-Load Step Budget (`max_steps = 140`):**
   - Halt stepping when `steps_since_load_change >= 140` (mean convergence is `46.04` steps).
3. **Best-Seen Voltage Snapshot & Rollback:**
   - Track `(best_vswr_seen, best_V1_seen, best_V2_seen)` during the current tuning episode. If any guard halts stepping before `vswr < 1.2`, set `(V1_auto, V2_auto) = (best_V1_seen, best_V2_seen)`. This turns near-miss oscillation timeouts (`37/47` already under `VSWR < 1.5`) into clean stable matches.
4. **Boundary-Pinning / Repeated-No-Op Detector:**
   - If `(V1_auto, V2_auto)` does not change after applying `agent.act()` (either `action == 4` Hold or clamped at `0.0 V` / `3.3 V`) for `4` consecutive steps, snap to best-seen voltages and latch `"Settled (Boundary / Hold)"`.
5. **Short-Cycle Oscillation Detector:**
   - Store a ring buffer of the last `16` quantized `(V1_auto, V2_auto)` pairs. If a state is revisited `>= 2–3` times within that window, snap to `(best_V1_seen, best_V2_seen)` and latch `"Settled (Cycle Guard)"`.

---

## 6. Safe Error Handling

| Failure Condition | Expected Exception / Signal | Recommended M5 Behavior |
|---|---|---|
| Missing / unreadable / corrupt `models/model.pt` | `load_agent` raises `FileNotFoundError`, `ValueError`, `TypeError`, or `RuntimeError` | Set `agent = None`, display error status badge, keep **Manual** mode fully operational, disable/block Auto toggle |
| Non-finite or invalid telemetry (`NaN`, `Inf`, `RL < 0`, `bool`, wrong keys) | `agent.act` raises `ValueError` | Validate M2 values before `compute()`; catch `ValueError`/`RuntimeError` around `act()`, force `Hold` (`action = 4`), freeze last valid voltages |
| Total reflection (`gamma_abs >= 1.0`) | `compute()` returns `vswr = float("inf")` | Display `"∞"` in UI metric cards; `gamma.real, gamma.imag` remain finite so RL state checks do not break |
| Serial cable unplugged / stale stream | M2 reader stale timestamp / exception | Indicate disconnect age in status panel, hold last valid `(RL, XL)`, or let user switch to `Mock Source` without crashing RL state |

---

## 7. Matchable Load Region Notice for M2 & Demo Script (`Project.md` §7)

Under M3's verified RF circuit (`f = 100 MHz`, `L = 150 nH`, `C0 = 100 pF`), across `config.py`'s full `RL_RANGE (10–150 Ω) × XL_RANGE (-100..+100 Ω)`, **`96.62%` (`4,461 / 4,617`) of loads cannot physically achieve `VSWR < 2.0`** (for instance, `50 + j0 Ω` has best achievable `VSWR = 8.1889`).

All matchable loads (`VSWR < 1.5`) lie inside:
- **`RL ∈ [10.0, 29.0] Ω`**
- **`XL ∈ [43.5, 80.5] Ω`** (e.g., `15.0 + j60.0 Ω` matches to `VSWR = 1.1275`)

**Recommendation for M2 / M5:** Keep `config.py` and `physics/` untouched (modifying M3 constants invalidates `models/model.pt`), and provide a **"Matchable Demo Range"** toggle in `io/mock_source.py` / `app/dashboard.py` alongside a `"Full Range"` diagnostic indicator (`grid_search_match(RL, XL)["best_vswr"]` takes ~`0.5 ms` on load changes to warn when a user dials an unmatchable load).

