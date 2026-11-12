# M4 Independent Load Generalization Report

> **Validation Date:** 2026-11-12 13:21:37 | **Loads:** 100 distinct disturbance loads | **Horizon:** max_steps=100

## 1. Executive Summary Table

| Initialization Condition | Distinct Loads | Success (VSWR < 1.2) | Success (VSWR < 1.5) | Mean VSWR | Median VSWR | Max VSWR | Mean Gap | Mean Steps |
|---|---|---|---|---|---|---|---|---|
| **Midpoint (1.65V, 1.65V)** | 100 | **97.0%** | 100.0% | 1.174 | 1.185 | 1.222 | 0.1391 | 32.4 |
| **Low Bias (0.50V, 0.50V)** | 100 | **95.0%** | 100.0% | 1.186 | 1.190 | 1.248 | 0.1511 | 73.8 |
| **High Bias (3.00V, 3.00V)** | 100 | **90.0%** | 100.0% | 1.178 | 1.188 | 1.361 | 0.1431 | 21.0 |

## 2. Worst-Performing Loads per Initialization Condition

### Midpoint (1.65V, 1.65V)

| RL (Ω) | XL (Ω) | Opt V1 | Opt V2 | Opt VSWR | DQN V1 | DQN V2 | DQN VSWR | Gap | Steps |
|---|---|---|---|---|---|---|---|---|---|
|  14.14 | +62.34 | 3.30 | 2.70 |  1.03 | 2.30 | 2.70 | ** 1.22** |  0.20 | 100 |
|  17.00 | +62.00 | 3.30 | 2.75 |  1.18 | 3.15 | 2.75 | ** 1.21** |  0.03 | 100 |
|  10.00 | +77.00 | 1.05 | 3.30 |  1.20 | 1.10 | 3.30 | ** 1.20** |  0.00 | 100 |
|  11.97 | +71.89 | 1.55 | 3.10 |  1.03 | 1.65 | 3.00 | ** 1.20** |  0.17 |  27 |
|  15.23 | +65.94 | 3.05 | 2.90 |  1.02 | 2.20 | 2.85 | ** 1.20** |  0.18 |  35 |

### Low Bias (0.50V, 0.50V)

| RL (Ω) | XL (Ω) | Opt V1 | Opt V2 | Opt VSWR | DQN V1 | DQN V2 | DQN VSWR | Gap | Steps |
|---|---|---|---|---|---|---|---|---|---|
|  11.15 | +62.67 | 2.15 | 2.60 |  1.02 | 1.50 | 2.60 | ** 1.25** |  0.23 | 100 |
|  14.14 | +62.34 | 3.30 | 2.70 |  1.03 | 2.30 | 2.70 | ** 1.22** |  0.20 | 100 |
|  10.98 | +63.72 | 2.00 | 2.65 |  1.02 | 1.45 | 2.65 | ** 1.22** |  0.20 | 100 |
|  11.50 | +68.13 | 1.75 | 2.90 |  1.02 | 1.25 | 2.95 | ** 1.21** |  0.20 | 100 |
|  17.00 | +62.00 | 3.30 | 2.75 |  1.18 | 3.15 | 2.75 | ** 1.21** |  0.03 | 100 |

### High Bias (3.00V, 3.00V)

| RL (Ω) | XL (Ω) | Opt V1 | Opt V2 | Opt VSWR | DQN V1 | DQN V2 | DQN VSWR | Gap | Steps |
|---|---|---|---|---|---|---|---|---|---|
|  10.16 | +63.35 | 1.80 | 2.65 |  1.03 | 3.00 | 2.70 | ** 1.36** |  0.33 | 100 |
|  10.00 | +62.00 | 1.90 | 2.55 |  1.01 | 3.00 | 2.60 | ** 1.31** |  0.30 | 100 |
|  17.00 | +77.00 | 2.20 | 3.30 |  1.19 | 3.00 | 3.30 | ** 1.29** |  0.10 | 100 |
|  15.21 | +76.51 | 1.90 | 3.30 |  1.13 | 2.70 | 3.30 | ** 1.28** |  0.15 | 100 |
|  12.28 | +67.56 | 1.95 | 2.90 |  1.02 | 3.00 | 2.90 | ** 1.27** |  0.26 | 100 |

## 3. Generalization Scope & Boundaries

- **Tested Disturbance Space:** Continuous floating-point loads within $R_L \in [10.0, 17.0]\,\Omega$ and $X_L \in [+62.0, +77.0]\,\Omega$.
- **Independence Guarantee:** Evaluated on distinct randomly generated continuous load impedances, not repeated frames from telemetry.
- **Generalization Boundary:** These results confirm generalization within the physical demonstration envelope. Performance on wide-band or out-of-distribution loads ($R_L > 45\,\Omega$, $X_L < 30\,\Omega$) must be verified separately against the RF network's physical matchability boundary.
