# Module 4 (RL Agent) Evaluation Report

**Evaluated Artifact:** `models/model.pt` (`294,637` bytes, `update_steps = 71721`)  
**Parameter-Tensor SHA-256:** `d9886cbad8cc358eb397ea6484032c1e65de70e75141796b67402b5973b7a215`  
**File SHA-256:** `51d571a304f5cd405427715c241431aa8b4894f6eb9fff50aea56d80df3d1e1a`  
**Source Artifacts:** `rl/logs/eval_summary.json`, `rl/logs/eval_per_load_results.csv`, `rl/logs/double_dqn_summary.json`, `rl/logs/double_dqn_unmatchable_loads.json`

---

## 1. Evaluation Protocol & Integrity Checks

1. **Strict Train / Held-Out Separation:**
   - Using `physics.pi_network.grid_search_match(..., grid_size=67)` (`0.05 V` lattice), `1,748` matchable loads (`best_vswr < 1.5`) were discovered at `0.5 Ω` resolution and split 80/20 (`split_seed = 42`):
     - **Training pool:** `1,398` loads
     - **Held-out test set:** `350` unique loads (`RL ∈ [10.0, 29.0] Ω`, `XL ∈ [43.5, 80.5] Ω`)
   - Verified overlap: `train_holdout_overlap = 0`, `training_csv_holdout_overlap = 0` (also completely disjoint from the `718` unique loads sampled during the `1,000`-episode training run).
2. **Controlled Paired Comparison:**
   - Deterministic `eval_seed = 100`, `max_steps = 140`, `match_vswr_threshold = 1.5`, `success_vswr = 1.2`, `match_tolerance = 0.015`.
   - Random baseline and Double DQN started from **identical `(RL, XL, initial_V1, initial_V2)` states** on all `350` loads.
3. **No Test-Time Training or Weight Mutation:**
   - Evaluated via `online_net.eval()` with greedy selection (`epsilon = 0`).
   - `_snapshot_weights_sha256()` matched `d9886cbad8cc358eb397ea6484032c1e65de70e75141796b67402b5973b7a215` before and after evaluation (`weights_unchanged = True`).

---

## 2. Approach Comparison (`350` Held-Out Matchable Loads)

| Metric | Random Baseline | Double DQN (Closed-Loop) | Grid Search (Reference Optimum) |
|---|---:|---:|---:|
| `is_matched` (Env Definition) | `4.57%` (`16/350`) | **`86.57%` (`303/350`)** | `100.00%` (`350/350`) |
| `VSWR < 2.0` | `14.86%` (`52/350`) | **`99.71%` (`349/350`)** | `100.00%` (`350/350`) |
| `VSWR < 1.5` | `6.29%` (`22/350`) | **`90.00%` (`315/350`)** | `100.00%` (`350/350`) |
| `VSWR < 1.2` | `3.71%` (`13/350`) | **`43.14%` (`151/350`)** | `49.71%` (`174/350`) |
| Final VSWR (Mean / Median) | `28.0290` / `6.9674` | **`1.2994` / `1.2558`** | `1.2106` / `1.2088` |
| Final VSWR (Min / Max / P95) | `1.1180` / `820.9333` / `108.7692` | **`1.1180` / `2.0041` / `1.5230`** | `1.0003` / `1.4987` / `1.4746` |
| Mean / Median Final `\|Gamma\|` | `0.6894` / `0.7490` | **`0.1272` / `0.1134`** | `0.0902` / `0.0945` |
| Mean Suboptimality Gap (`\|Gamma\| - best`) | `0.5992` | **`0.0370`** | `0.0000` |
| Mean Steps (All / Matched) | `135.01` / `30.94` | **`58.66` / `46.04`** | N/A (`4,489` lattice evals) |
| Median Steps (All / Matched) | `140.0` / `22.5` | **`49.0` / `45.0`** | N/A |
| Mean Exec Time (`ms / load`) | `8.494` (`0.514` infer) | `15.808` (`10.487` infer) | `0.517` |

### Methodological Clarification: Double DQN vs. Grid Search
- **Grid Search is an offline, open-loop reference optimization method**, not a deployable feedback controller: it assumes exact analytical knowledge of the hidden load `(RL, XL)` and sweeps all `67 × 67 = 4,489` `(V1, V2)` candidates simultaneously in vectorized NumPy.
- **Double DQN is an online, closed-loop controller** operating purely from live measurements `[gamma.real, gamma.imag, V1, V2]` via incremental `±0.05 V` adjustments without knowing `RL` or `XL`. It achieves `99.71%` `VSWR < 2.0` and sits within `+0.0370` mean `|Gamma|` of the exhaustive grid-search theoretical optimum.

---

## 3. Regional Performance Breakdown (Double DQN)

### By Load Resistance (`RL`)
| `RL` Region (`Ω`) | Loads | `is_matched` | `VSWR < 1.5` | `VSWR < 1.2` | Mean Final VSWR | Mean Steps (All) | Mean Subopt Gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| `10.0–14.5` | `129` | `86.8%` | `95.3%` | `65.1%` | `1.245` | `50.6` | `0.0511` |
| `15.0–19.5` | `105` | `88.6%` | `94.3%` | `58.1%` | `1.268` | `57.3` | `0.0417` |
| `20.0–24.5` | `76` | `77.6%` | `85.5%` | `7.9%` | `1.353` | `72.8` | `0.0184` |
| `25.0–30.0+` | `40` | `97.5%` | `70.0%` | `0.0%` | `1.457` | `61.4` | `0.0150` |

### By Load Reactance (`XL`)
| `XL` Region (`Ω`) | Loads | `is_matched` | `VSWR < 1.5` | `VSWR < 1.2` | Mean Final VSWR | Mean Steps (All) | Mean Subopt Gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| `< 55.0` | `34` | `91.2%` | `85.3%` | `26.5%` | `1.336` | `56.1` | `0.0189` |
| `55.0–64.5` | `88` | `85.2%` | `94.3%` | `42.0%` | `1.285` | `53.5` | `0.0411` |
| `65.0–74.5` | `148` | `91.2%` | `93.2%` | `58.1%` | `1.258` | `57.6` | `0.0449` |
| `75.0+` | `80` | `77.5%` | `81.2%` | `23.8%` | `1.376` | `67.3` | `0.0259` |

### By Physical Achievability Regime (`best_vswr`)
| Regime | Loads | `is_matched` | `VSWR < 2.0` | `VSWR < 1.5` | `VSWR < 1.2` | Mean Final VSWR | Mean Steps (All / Matched) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `best_vswr < 1.2` | `174` | `86.8%` (`151/174`) | `100.0%` | `100.0%` | `86.8%` | `1.191` | `53.6` / `40.5` |
| `1.2 <= best_vswr < 1.5` | `176` | `86.4%` (`152/176`) | `99.4%` | `80.1%` | `0.0%` | `1.407` | `63.6` / `51.6` |

---

## 4. Action Usage & Failure Diagnostics (`20,530` Total Steps)

- **Action Distribution:**
  - `V2+`: `55.54%` (`11,402` steps)
  - `V1+`: `33.36%` (`6,849` steps)
  - `V2-`: `7.85%` (`1,611` steps)
  - `V1-`: `2.01%` (`413` steps)
  - `Hold`: `1.24%` (`255` steps)
- **Step Diagnostics Across All 350 Episodes:**
  - Boundary hits (action clamped at `0.0 V` or `3.3 V`): `9.81%` (`2,013` steps across `20` episodes)
  - Immediate action reversals: `11.54%` (`2,370` steps)
  - Revisited `(V1, V2)` states: `22.59%` (`4,638` steps)
- **Timeout Failure Breakdown (`47 / 350 = 13.43%`):**
  - `37 / 47` timeouts reached `VSWR < 1.5`; only `10 / 47` ended at `VSWR >= 1.5` (single worst case `2.0041` on `RL = 22.5 Ω, XL = 78.5 Ω`).
  - **Boundary-Rail Pinning (`20 / 47`):** Agent repeats `V2+` after reaching `V2 = 3.3 V` (`18/20` occur in `XL >= 75.0 Ω`). Example: `load_index = 279` (`RL = 22.5 Ω, XL = 78.5 Ω`), `113` boundary hits, final `VSWR = 2.0041`.
  - **Two-State Oscillation Cycles (`24 / 47`):** Policy alternates opposite steps (e.g., `V2+ -> V2-`) near optimality without satisfying the strict termination bound. Example: `load_index = 104` (`RL = 13.5 Ω, XL = 60.5 Ω`), `105` reversals, hovers at `VSWR = 1.2022` (just above `1.2` threshold; grid best `1.0221`).
  - **Hold Stalling (`3 / 47`):** Policy selects `4` (`Hold`) prematurely (`53`, `82`, and `120` Hold steps = `255` total).
- **Representative Successes:**
  - `load_index = 48` (`RL = 11.0 Ω, XL = 76.5 Ω`): `VSWR 8.2243 -> 1.1211` in `67` steps (`gap = 0.000094`).
  - `load_index = 146` (`RL = 15.5 Ω, XL = 53.0 Ω`): `VSWR 3.4706 -> 1.4973` in `49` steps (`gap = 0.0124`).

---

## 5. Physics Envelope & Unmatchable Load Catalog

Across a `2.5 Ω` uniform scan of `config.py`'s `RL_RANGE (10–150 Ω) × XL_RANGE (-100..+100 Ω)` (`4,617` points, recorded in `rl/logs/double_dqn_unmatchable_loads.json`):
- **`4,461 / 4,617` (`96.62%`)** cannot physically reach `VSWR < 2.0` (`best_vswr` median `9.0009`), including `RL = 50.0 Ω, XL = 0.0 Ω` (`best_vswr = 8.1889`).
- Only `156` scanned loads satisfy `best_vswr < 2.0`, and `77` satisfy `best_vswr < 1.5`.
- Because `Double DQN` operates strictly closed-loop inside the matchable Pi-network manifold, M5 must apply step caps, cycle/pinning detection, and optional `grid_search_match()` feasibility indicators for out-of-envelope loads.

---

## 6. Saved Diagnostic Plots & Data Files

- **Training Curves (`rl/figures/`):** `double_dqn_overview.png`, `double_dqn_episode_reward.png`, `double_dqn_rolling_average_reward.png`, `double_dqn_final_vswr.png`, `double_dqn_success_rate.png`, `double_dqn_epsilon_decay.png`, `double_dqn_training_loss.png`
- **Evaluation Plots (`rl/figures/`):** `eval_approach_comparison.png`, `eval_vswr_distribution.png`, `eval_region_heatmap.png`, `eval_action_diagnostics.png`, `eval_sample_trajectories.png`
- **Logs (`rl/logs/`):** `double_dqn_metrics.csv`, `double_dqn_summary.json`, `double_dqn_unmatchable_loads.json`, `eval_per_load_results.csv`, `eval_summary.json`, `eval_report.md`

