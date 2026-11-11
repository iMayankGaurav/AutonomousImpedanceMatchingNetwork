# Module 4 (RL Agent — Double DQN) Progress Report

**Module Owner:** Member 4 (RL Agent)  
**Model Artifact:** `models/model.pt` (`294,637` bytes, `update_steps = 71721`)  
**Status:** Complete, audited, and ready for M5 integration  

---

## 1. Project and M4 Objective

The Autonomous Impedance Matching Network demonstrates Hardware-in-the-Loop (HIL) adaptive RF impedance matching at `100 MHz`. An STM32 microcontroller streams four potentiometer readings over UART, while Python simulates a tunable RF Pi-network (`physics/pi_network.py`) and visualizes the complex reflection coefficient `Gamma` and `VSWR` on a live Smith Chart.

Module 4 (M4) delivers a trained Reinforcement Learning controller that autonomously adjusts virtual varactor reverse-bias voltages `(V1, V2)` step by step using only closed-loop RF reflection feedback—driving mismatched antenna loads toward `50 Ω` (`|Gamma| -> 0`, `VSWR -> 1.0`) without requiring analytical knowledge of load impedance `(RL, XL)` at inference time.

---

## 2. M4 Responsibilities and Scope

Per `Project.md` (Section 3 & Section 4.4), M4 owns:
- Gymnasium-compatible RL environment wrapping M3 physics (`rl/env.py`)
- Double Deep Q-Network agent, circular replay buffer, and safe checkpointing (`rl/dqn.py`)
- Reproducible training pipeline with logging, unmatchable-load accounting, checkpoints, and curves (`rl/train.py`)
- Held-out evaluation pipeline comparing Double DQN against random and grid-search baselines (`rl/evaluate.py`)
- Clean, deterministic, read-only inference API for M5 (`rl/agent.py`)
- Trained checkpoint (`models/model.pt`) and unit/integration tests (`rl/tests/`)

Shared/teammate files (`config.py`, `physics/`, `io/`, `app/`, `firmware/`, `hardware/`, `Project.md`) were left unmodified.

---

## 3. Implemented RL Architecture

| Artifact | Purpose |
|---|---|
| `rl/__init__.py` | Package initialization |
| `rl/env.py` | `ImpedanceMatchingEnv`, `ACTION_NAMES`, voltage clamping (`[0.0, 3.3] V`), matchable load discovery |
| `rl/dqn.py` | `QNetwork`, `ReplayBuffer`, `DoubleDQNAgent`, `compute_epsilon`, `resolve_device` |
| `rl/train.py` | Load catalog builder (80/20 train/holdout split), Double DQN training loop, CSV/JSON/PNG outputs |
| `rl/evaluate.py` | Held-out evaluation across 350 disjoint loads vs. Random and Grid Search baselines |
| `rl/agent.py` | `load_agent()` and `InferenceAgent` (`act()`, `q_values()`, strict input validation, `torch.no_grad()`) |
| `rl/tests/` | 28 unit and pipeline tests (`test_env.py`, `test_dqn.py`, `test_train.py`, `test_evaluate.py`, `test_agent.py`) |
| `models/model.pt` | Final trained Double DQN checkpoint (`294,637` bytes) |
| `rl/logs/` & `rl/figures/` | Verified training logs, evaluation CSV/JSON/Markdown reports, and 12 diagnostic PNG plots |

---

## 4. Environment State and Action Definitions (`rl/env.py`)

- **Physics Backend:** Calls `physics.pi_network.compute(RL, XL, V1, V2)` directly (`Z0 = 50.0 Ω`, `f = 100 MHz`, `L = 150 nH`, varactor `C(V) = 100 pF / sqrt(1 + V / 0.7)`).
- **Observation Space (`shape=(4,)`, `float32`):**
  `[gamma.real, gamma.imag, V1, V2]`
  - `gamma.real`, `gamma.imag ∈ [-1.0, 1.0]`
  - `V1` (source-side shunt `C1`) and `V2` (load-side shunt `C2`) in `[0.0, 3.3] V`
- **Action Space (`Discrete(5)`):**
  - `0`: `V1+` (`V1 += 0.05 V`)
  - `1`: `V1-` (`V1 -= 0.05 V`)
  - `2`: `V2+` (`V2 += 0.05 V`)
  - `3`: `V2-` (`V2 -= 0.05 V`)
  - `4`: `Hold` (`V1`, `V2` unchanged)
- **Boundary Enforcement:** `_clamp_voltage()` clips voltages to `[config.V_MIN, config.V_MAX]` (`[0.0, 3.3] V`) rounded to 4 decimal places.
- **Episode Load Handling:** Supports `"matchable"`, `"full"`, or `"mixed"` load modes. During training and evaluation, each episode fixes one `(RL, XL)` load and randomizes initial `(V1, V2)` across the `0.05 V` lattice (`67` points per axis).

---

## 5. Reward and Termination Approach

At every step in `ImpedanceMatchingEnv.step(action)`:

```text
total_reward = progress_reward + mismatch_penalty + vswr_penalty + step_cost + success_bonus
```

- `progress_reward = 20.0 * (prev_gamma_abs - gamma_abs)`
- `mismatch_penalty = -0.5 * max(0.0, gamma_abs - best_gamma_abs)`
- `vswr_penalty = -0.5` if `(best_vswr < 2.0 and vswr > 2.0)` else `0.0` (prevents unfair penalties on physically unachievable loads)
- `step_cost = -0.1`
- `success_bonus = +50.0` when `_check_matched()` is satisfied

**Load-Aware Termination (`_check_matched`):**
- If `best_vswr < 1.2`: episode succeeds when `vswr < 1.2`
- Else if `1.2 <= best_vswr < 2.0`: episode succeeds when `gamma_abs <= best_gamma_abs + 0.015`
- Else (`best_vswr >= 2.0`): never falsely terminates as matched; truncates at `max_steps`

---

## 6. Double DQN Design (`rl/dqn.py`)

- **Network (`QNetwork`):** Fully connected MLP `4 -> 128 (ReLU) -> 128 (ReLU) -> 5`.
- **Internal Input Normalization:** Buffers `input_center = [0.0, 0.0, 1.65, 1.65]` and `input_scale = [1.0, 1.0, 1.65, 1.65]` inside `QNetwork.forward()` normalize voltages to `[-1.0, 1.0]` automatically so callers pass raw `[gamma.real, gamma.imag, V1, V2]`.
- **Experience Replay (`ReplayBuffer`):** Circular NumPy buffer storing `(state, action, reward, next_state, terminated)` with uniform random minibatch sampling.
- **Double DQN Target Calculation:**
  - Online network selects maximizing next action: `a* = argmax_a Q_online(s', a)`
  - Target network evaluates that action: `target = r + gamma * (1 - terminated) * Q_target(s', a*)`
- **Optimization & Stability:** `SmoothL1Loss(beta=1.0)` (Huber loss), `Adam` optimizer, gradient norm clipping (`max_grad_norm = 10.0`), and periodic hard target-network synchronization (`tau = 1.0`).

---

## 7. Training Configuration

Final production run (`rl/logs/double_dqn_summary.json`):

| Parameter | Value |
|---|---|
| Episodes | `1,000` |
| Max steps per episode | `140` |
| Learning rate (`lr`) | `0.001` |
| Discount factor (`gamma`) | `0.99` |
| Replay buffer capacity | `50,000` |
| Batch size | `64` |
| Epsilon schedule | `1.0 -> 0.05` linearly over `25,000` env steps |
| Warmup steps | `500` |
| Target sync interval | Every `250` gradient updates |
| Hidden layers | `[128, 128]` |
| Loss / Gradient clip | SmoothL1 (`beta = 1.0`) / `10.0` |
| Matchable pool threshold | `best_vswr < 1.5` (`grid_size = 67`) |
| Seed / Device | `42` / `cpu` |

---

## 8. Training Run Results

- **Load Catalog (`build_load_catalog`):**
  - Identified `1,748` matchable loads (`best_vswr < 1.5`) at `0.5 Ω` resolution, split 80/20 (`split_seed=42`) into **`1,398` training loads** and **`350` strictly held-out test loads** (`overlap = 0`).
  - Separately cataloged `7,553` unmatchable points inside the search envelope and `4,461 / 4,617` (`96.62%`) unmatchable loads (`best_vswr >= 2.0`) across the full `config.py` `RL_RANGE × XL_RANGE` (`2.5 Ω` scan).
- **Execution Totals:**
  - Completed episodes: `1,000`
  - Total environment steps: `72,220`
  - Total gradient updates (`update_steps`): `71,721`
  - Final epsilon: `0.05`
  - `NaN` / `Inf` events: `0`
- **Learning Progression (`rl/logs/double_dqn_summary.json`):**

| Window | Episodes | Mean Reward | Mean Final `\|Gamma\|` | Mean Length | Match Success Rate | Mean Huber Loss |
|---|---:|---:|---:|---:|---:|---:|
| Early | `1–50` | `-90.9931` | `0.5431` | `128.56` | `18.0%` | — |
| Late | `951–1000` | `+26.2052` | `0.1235` | `61.06` | `94.0%` | `0.636786` |
| Overall | `1–1000` | — | — | — | `84.2%` | — |

Training plots are saved under `rl/figures/double_dqn_*.png`.

---

## 9. Held-Out Evaluation Methodology

Evaluation (`rl/evaluate.py`, summary in `rl/logs/eval_summary.json`) tested the untouched `models/model.pt` (`update_steps = 71721`) across all **`350` held-out loads**:
- **Zero Data Leakage:** `train_holdout_overlap = 0` and `training_csv_holdout_overlap = 0` (also verified disjoint from the `718` unique loads sampled during training).
- **Fair Comparison:** `eval_seed = 100`, `max_steps = 140`, identical `(RL, XL, initial_V1, initial_V2)` starting states across sequential controllers.
- **Weight Immutability Check:** `_snapshot_weights_sha256()` returned identical parameter-tensor SHA-256 `d9886cbad8cc358eb397ea6484032c1e65de70e75141796b67402b5973b7a215` before and after evaluation (`weights_unchanged = True`).

---

## 10. Results Against Random and Grid-Search Baselines

| Metric | Random Baseline | Double DQN | Grid Search (Reference Optimum) |
|---|---:|---:|---:|
| `is_matched` (Env Criterion) | `4.57%` (`16/350`) | **`86.57%` (`303/350`)** | `100.00%` (`350/350`) |
| `VSWR < 2.0` | `14.86%` (`52/350`) | **`99.71%` (`349/350`)** | `100.00%` (`350/350`) |
| `VSWR < 1.5` | `6.29%` (`22/350`) | **`90.00%` (`315/350`)** | `100.00%` (`350/350`) |
| `VSWR < 1.2` | `3.71%` (`13/350`) | **`43.14%` (`151/350`)** | `49.71%` (`174/350`) |
| Final VSWR (Mean / Median) | `28.0290` / `6.9674` | **`1.2994` / `1.2558`** | `1.2106` / `1.2088` |
| Mean Final `\|Gamma\|` | `0.6894` | **`0.1272`** | `0.0902` |
| Mean Suboptimality Gap (`\|Gamma\| - best`) | `0.5992` | **`0.0370`** | `0.0000` |
| Mean Steps (All / Matched) | `135.01` / `30.94` | **`58.66` / `46.04`** | N/A (`4,489` lattice evals) |
| Mean Execution Time (`ms / load`) | `8.494` | `15.808` (`10.487` inference) | `0.517` |

> **Important Context:** Grid Search is an **offline, open-loop reference optimization method** that assumes direct knowledge of `(RL, XL)` and evaluates all `67 × 67 = 4,489` `(V1, V2)` candidates simultaneously in vectorized NumPy. Double DQN is a **closed-loop sequential controller** acting solely from `[gamma.real, gamma.imag, V1, V2]` feedback without observing `RL` or `XL`.

---

## 11. Model Artifact and Inference Interface

- **Model File:** `models/model.pt` (`294,637` bytes, tracked in Git via `!models/model.pt` exception in `.gitignore`)
- **File SHA-256 (`sha256sum`):** `51d571a304f5cd405427715c241431aa8b4894f6eb9fff50aea56d80df3d1e1a`
- **Parameter-Tensor Digest (`_snapshot_weights_sha256`):** `d9886cbad8cc358eb397ea6484032c1e65de70e75141796b67402b5973b7a215`
- **Inference API (`rl/agent.py`):**
  ```python
  from rl.agent import load_agent

  agent = load_agent("models/model.pt", device="cpu")
  action = agent.act([gamma.real, gamma.imag, V1, V2])  # returns int in 0..4
  ```
- **Inference Guarantees:** Deterministic greedy selection (`epsilon = 0`), `online_net.eval()`, frozen parameters (`requires_grad_(False)`), `torch.no_grad()`, no training or replay state, and strict validation rejecting wrong dimensions, booleans, complex values, strings, or `NaN`/`Inf`.

---

## 12. Unit Test Results

Verified by executing `.venv/bin/python -B -m unittest discover -s rl/tests -p "test_*.py" -v`:

| Test File | Scope | Result |
|---|---|---|
| `rl/tests/test_agent.py` | `load_agent`, greedy `act()`, determinism, input/file error handling, frozen weights | `7 / 7` passed |
| `rl/tests/test_dqn.py` | `QNetwork` dims, action selection, `ReplayBuffer`, Double DQN target decoupling, target sync, save/load | `6 / 6` passed |
| `rl/tests/test_env.py` | Gymnasium contract, 5 actions, voltage clamping, reward components, pure reactive & unachievable load guards | `9 / 9` passed |
| `rl/tests/test_evaluate.py` | Held-out evaluation reproducibility, zero train/test overlap, non-overwriting outputs | `2 / 2` passed |
| `rl/tests/test_train.py` | Load catalog separation, non-overwriting checkpoints, `KeyboardInterrupt` safety, short pipeline run | `4 / 4` passed |
| **Total** | Full M4 RL Suite | **`28 / 28` passed (`OK` in `22.59s`)** |

---

## 13. Known Limitations and Failure Modes

1. **Held-Out Timeout Breakdown (`47 / 350 = 13.43%`):**
   - `37/47` timeouts already reached `VSWR < 1.5` (near-misses failing only the strict `< 1.2` or `0.015` tolerance); `10/47` ended with `VSWR >= 1.5`.
   - Root behaviors across all `47` timeouts: **`20/47` boundary-rail pinning** (repeating `V1+`/`V2+` at `3.3 V`, mostly `XL >= 75.0 Ω`), **`24/47` short oscillation cycles** (e.g., `V2+ <-> V2-` hovering near `VSWR ~ 1.20`), and **`3/47` `Hold` stalling**.
   - Action bias: policy uses `V2+` (`55.54%`) and `V1+` (`33.36%`) heavily relative to `V1-` (`2.01%`) and `V2-` (`7.85%`).
2. **Physical Physics Envelope Constraint:** Under M3's fixed defaults (`100 MHz`, `L = 150 nH`, `C0 = 100 pF`), **`96.62%` (`4,461 / 4,617`)** of `RL_RANGE (10–150 Ω) × XL_RANGE (-100..+100 Ω)` cannot physically achieve `VSWR < 2.0`. Even `RL = 50 Ω, XL = 0 Ω` has theoretical best `VSWR = 8.1889`. Matchable loads (`VSWR < 1.5`) reside inside **`RL ∈ [10.0, 29.0] Ω`, `XL ∈ [43.5, 80.5] Ω`**.
3. **Simulation-Only Voltage Actuation:** M4 updates software bias variables `(V1, V2)` evaluated by `physics/pi_network.py`; no physical STM32 DAC or physical RF varactor actuation hardware exists in the project.
4. **Runtime Guard Requirement:** Live M5 deployment must wrap `agent.act()` with a step cap (`140`), settled latch + pot-noise deadband, best-seen voltage rollback, and cycle/pinning detectors (detailed in `docs/M4_INTEGRATION_GUIDE.md`).

---

## 14. What M4 Has Delivered to Other Modules

- **For M5 (Dashboard & Integration):** Ready-to-call `load_agent()` and `agent.act()` inference interface (`rl/agent.py`), pre-trained `models/model.pt`, action map (`ACTION_NAMES`), and documented integration architecture (`docs/M4_INTEGRATION_GUIDE.md`).
- **For M2 / M5 Demo Planning:** Exact empirical bounds of the matchable load region (`RL ∈ [10.0, 29.0] Ω`, `XL ∈ [43.5, 80.5] Ω`) and unmatchable-load catalog (`rl/logs/double_dqn_unmatchable_loads.json`).
- **For M3 (Physics Engine):** Verified `physics.pi_network.compute()` and `grid_search_match()` across >100k simulation transitions and 350 held-out test cases with zero numerical instability (`0` NaN/Inf events).
- **For Team Documentation & QA:** Complete evaluation summary (`docs/M4_EVALUATION_REPORT.md`), CSV/JSON logs (`rl/logs/`), and 12 diagnostic figures (`rl/figures/`).

---

## 15. Current Completion Status

**M4 (Reinforcement Learning) is 100% complete.** All code, tests (`28/28` passing), training logs, evaluation baselines, figures, documentation, and tracked `models/model.pt` weights are verified and ready for M5 hardware-in-the-loop and dashboard integration.

