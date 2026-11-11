# Held-Out Evaluation Report — Double DQN Impedance Matching

## 1. Evaluation Protocol & Integrity Checks
- **Model Evaluated:** `models/model.pt` (device: `cpu`, training update steps: `71721`)
- **Held-Out Test Loads:** `350` unique matchable loads (`best_vswr < 1.5`), strictly disjoint from the `1398` training loads (`overlap=0`) and from all `718` loads sampled in training (`overlap=0`).
- **Reproducible Seeds & Horizon:** `eval_seed=100`, `max_steps=140`. Identical `(RL, XL, initial_V1, initial_V2)` starting states across sequential controllers.
- **Weight Immutability Check:** SHA-256 identical before/after evaluation (`weights_unchanged=True`).

## 2. Approach Comparison (350 Held-Out Loads)

| Metric | Random Baseline | Double DQN | Grid Search (Reference Optimum) |
|---|---:|---:|---:|
| `is_matched` (Env Definition) | 4.57% (16/350) | 86.57% (303/350) | 100.00% (350/350) |
| VSWR < 2.0 | 14.86% | 99.71% | 100.00% |
| VSWR < 1.5 | 6.29% | 90.00% | 100.00% |
| VSWR < 1.2 | 3.71% | 43.14% | 49.71% |
| Final VSWR (Mean / Median) | 28.029 / 6.967 | 1.299 / 1.256 | 1.211 / 1.209 |
| Final VSWR (Min / Max) | 1.118 / 820.933 | 1.118 / 2.004 | 1.000 / 1.499 |
| Mean Final `|Gamma|` | 0.6894 | 0.1272 | 0.0902 |
| Mean Suboptimality Gap | 0.5992 | 0.0370 | 0.0000 |
| Mean Steps (All / Matched) | 135.01 / 30.94 | 58.66 / 46.04 | N/A (4,489 lattice evals) |
| Mean Execution Time (ms/load) | 8.494 | 15.808 | 0.517 |

> **Important Distinction:** Grid search is an **offline, open-loop reference optimization method** that assumes full analytical knowledge of `ZL = RL + jXL` and evaluates all 4,489 voltage grid points simultaneously in vectorized NumPy. Double DQN is a **closed-loop sequential controller** operating step-by-step on measured `[Re(Gamma), Im(Gamma), V1, V2]` without knowing `RL` or `XL`.

## 3. Double DQN Regional Breakdown
### By Load Resistance (`RL`)
| `RL` Region (Ohms) | Loads | `is_matched` | VSWR < 1.5 | Mean VSWR | Mean Steps (All) | Mean Subopt Gap |
|---|---:|---:|---:|---:|---:|---:|
| 10.0-14.5 | 129 | 86.8% | 95.3% | 1.245 | 50.6 | 0.0511 |
| 15.0-19.5 | 105 | 88.6% | 94.3% | 1.268 | 57.3 | 0.0417 |
| 20.0-24.5 | 76 | 77.6% | 85.5% | 1.353 | 72.8 | 0.0184 |
| 25.0-30.0+ | 40 | 97.5% | 70.0% | 1.457 | 61.4 | 0.0150 |

### By Load Reactance (`XL`)
| `XL` Region (Ohms) | Loads | `is_matched` | VSWR < 1.5 | Mean VSWR | Mean Steps (All) | Mean Subopt Gap |
|---|---:|---:|---:|---:|---:|---:|
| <55.0 | 34 | 91.2% | 85.3% | 1.336 | 56.1 | 0.0189 |
| 55.0-64.5 | 88 | 85.2% | 94.3% | 1.285 | 53.5 | 0.0411 |
| 65.0-74.5 | 148 | 91.2% | 93.2% | 1.258 | 57.6 | 0.0449 |
| 75.0+ | 80 | 77.5% | 81.2% | 1.376 | 67.3 | 0.0259 |

## 4. Action Usage, Boundaries & Failure Investigation
- **DQN Action Fractions:** {"V1+": 0.33360935216755966, "V1-": 0.020116902094495858, "V2+": 0.5553823672674135, "V2-": 0.07847053093034584, "Hold": 0.012420847540185095}
- **Boundary Hits:** 2013 / 20530 steps (9.81% across 20 episodes)
- **Immediate Reversals:** 2370 steps (11.54%)
- **Revisited `(V1, V2)` States:** 4638 steps (22.59%)
- **Late-Training Timeouts Observed in CSV:** 16 episodes
- **Held-Out Failures (`not is_matched`):** 47 / 350 (37 with VSWR < 1.5, 10 with VSWR >= 1.5)

