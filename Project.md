# STM32-Driven Autonomous RL Impedance Matching Network

## Modular 5-Person Project Plan

**Hardware:** STM32 microcontroller (e.g., STM32F446RE), 4x 10kΩ potentiometers, PC/Laptop
**Software:** C (STM32 HAL), Python, PyTorch (DQN), Streamlit, Matplotlib
**Team Size:** 5 members, one module each
**Approach:** Modules are built in parallel against fixed interface contracts (Section 4), so nobody waits on anyone else.

---

## 1. Project Significance: What Problem Are We Solving?

### The Real-World Problem ("Environmental Detuning")
In RF communications (5G, IoT, radar), an antenna is designed to have a characteristic impedance ($Z_0$) of exactly $50\,\Omega$. When the environment changes (a hand grabs a smartphone, a drone flies near a metal structure), parasitic capacitance and inductance change the antenna impedance to a mismatched load ($Z_L$).

When $Z_L \neq Z_0$, RF energy reflects back into the transmitter instead of radiating out. This causes high Voltage Standing Wave Ratio (VSWR), wastes battery power, and drops data rates.

### The Solution: Dynamic $\pi$-Matching
A tunable $\pi$-matching network is placed between the transmitter and the antenna. It uses two variable capacitors (varactor diodes, controlled by bias voltages $V_1$ and $V_2$) and an inductor. By changing $V_1$ and $V_2$, we transform the broken $Z_L$ back into $50\,\Omega$.

### The Significance of This Project
Tuning $V_1$ and $V_2$ algebraically in real time is hard because of non-linear math and hardware parasitics. This project demonstrates a **Reinforcement Learning (RL) approach**.

To keep the demo robust, the RF physics is simulated mathematically on a laptop. An **STM32 microcontroller** reading **4 physical potentiometers** bridges the physical and virtual worlds, creating an interactive **"Human vs. AI" hardware-in-the-loop (HIL)** demo: tune the network by hand with knobs, then switch on the RL agent and watch it solve the Smith Chart puzzle instantly.

---

## 2. System Architecture (Unchanged)

```
 ┌──────────────────────┐   UART/USB    ┌─────────────────────────────────────────────┐
 │ STM32 + 4 Pots       │  CSV @ 20 Hz  │ Laptop (Python)                             │
 │ P1 R_L  P2 X_L       │ ────────────► │ serial → filter → physics ⇄ RL agent → UI   │
 │ P3 V1   P4 V2        │               │                       (Streamlit Smith Chart)│
 └──────────────────────┘               └─────────────────────────────────────────────┘
```

**Stage 1: Hardware Interface.** Four pots on four ADC pins:
* **Pot 1 ($R_L$ disturbance):** 0–3.3 V → 10–150 Ω
* **Pot 2 ($X_L$ disturbance):** 0–3.3 V → −100 to +100 Ω
* **Pot 3 ($V_1$ manual):** 0–3.3 V bias for varactor $C_1$
* **Pot 4 ($V_2$ manual):** 0–3.3 V bias for varactor $C_2$

The STM32 streams a line like `55.2,-10.5,1.2,2.4\n` to the laptop at ~20 Hz.

**Stage 2: Virtual RF Physics Engine.** Python forms $Z_L = R_L + jX_L$, maps $V_1, V_2$ to picofarads, models a 100 MHz $\pi$-network, and computes $Z_{in}$, $\Gamma$, and VSWR.

**Stage 3: Two Operating Modes.**
* **Manual:** physics uses $V_1, V_2$ straight from Pots 3 and 4.
* **Auto (RL):** Pots 3 and 4 are ignored; the DQN agent reads $\Gamma$ and drives $V_1, V_2$ to the center of the Smith chart.

**Stage 4: Visualization.** Streamlit dashboard with a live Smith Chart, target center (50 Ω), current mismatch dot, live VSWR, and a "Manual (Pots)" / "Auto (RL)" toggle.

---

## 3. Modules and Ownership

| Module | Owner | Topic | Main deliverable |
|---|---|---|---|
| **M1** | Member 1 | Hardware & STM32 Firmware | Pots wired; firmware streaming CSV over UART |
| **M2** | Member 2 | Serial Link & Signal Processing | `serial_reader` + filter + unit mapping + mock data source |
| **M3** | Member 3 | RF Physics Engine | `physics` library: varactor, π-network, Γ, VSWR |
| **M4** | Member 4 | RL Agent (DQN) | Gym-style env, training script, `model.pt` |
| **M5** | Member 5 | Dashboard, Integration & QA | Streamlit app, mode toggle, tests, demo, docs |

---

### M1: Hardware & STM32 Firmware (Member 1)
**Goal:** Reliable physical data source.

**Tasks**
1. Wire 4 pots to 3.3 V, GND, and four ADC pins (e.g., PA0–PA3). Add a 100 nF capacitor from each wiper to GND to reduce noise.
2. Configure 4-channel ADC (12-bit) with scan mode + DMA (circular) in STM32CubeIDE.
3. Convert raw 0–4095 values into a CSV line (raw or scaled, per the contract in Section 4) and transmit via UART (USB virtual COM / ST-LINK) at 20 Hz using a timer.
4. Add a heartbeat LED and a simple `PING`/`ID` response so Python can detect the board.
5. Document the pinout and wiring in `hardware/wiring.md`.

**Deliverables:** `firmware/` project, wiring diagram, 30-second capture of the stream.
**Definition of done:** Terminal shows 4 stable values at ~20 Hz; turning a pot changes only its own column.
**Depends on:** nothing. **Unblocks:** M2 (real data), M5 (live demo).

---

### M2: Serial Link & Signal Processing (Member 2)
**Goal:** Clean, physically meaningful inputs for the rest of the system, with or without hardware.

**Tasks**
1. Write `serial_reader.py` using `pyserial`: auto-detect the port, parse lines, tolerate malformed or partial lines, reconnect on unplug.
2. Apply a software low-pass filter (exponential moving average or a small median + EMA) to remove pot jitter.
3. Map values to physical bounds: $R_L \in [10,150]\,\Omega$, $X_L \in [-100,100]\,\Omega$, $V_1,V_2 \in [0,3.3]\,\text{V}$.
4. Build `mock_source.py` that produces the same data format (sine sweeps, random jumps) so M3, M4, and M5 can develop without the board.
5. Expose one clean API: `get_latest() -> dict` (see Section 4).

**Deliverables:** `io/serial_reader.py`, `io/mock_source.py`, `io/filters.py`, unit tests for the parser.
**Definition of done:** Same API works from real board or mock; bad lines never crash the reader.
**Depends on:** M1 format (agreed upfront in Section 4). **Unblocks:** M5.

---

### M3: RF Physics Engine (Member 3)
**Goal:** The mathematical heart: given $Z_L, V_1, V_2$, return $\Gamma$ and VSWR.

**Tasks**
1. Varactor model $C(V)$, e.g. $C(V) = C_0 / (1 + V/V_j)^{\gamma}$ with tunable constants; keep capacitance in a realistic pF range.
2. $\pi$-network at 100 MHz using complex arithmetic (suggested topology: shunt $C_2$ at the load, series $L$, shunt $C_1$ at the source):
   * $Y_2 = 1/Z_L + j\omega C_2$, $Z_3 = 1/Y_2 + j\omega L$, $Y_{in} = 1/Z_3 + j\omega C_1$, $Z_{in} = 1/Y_{in}$
   * $\Gamma = (Z_{in} - Z_0)/(Z_{in} + Z_0)$, $\text{VSWR} = (1+|\Gamma|)/(1-|\Gamma|)$
3. Choose component values (L, C range) so that most of the $Z_L$ range is matchable. Verify by brute-force grid search over $V_1, V_2$ and record the best achievable $|\Gamma|$ per $Z_L$.
4. Provide Smith Chart helper functions (Γ ↔ normalized impedance, grid arcs for plotting) for M5.
5. Write unit tests: $Z_L = 50\,\Omega$ with symmetric setting gives low $|\Gamma|$; $|\Gamma| \le 1$ always.

**Deliverables:** `physics/varactor.py`, `physics/pi_network.py`, `physics/smith.py`, `docs/physics_notes.md`, a "matchable region" plot.
**Definition of done:** `compute(RL, XL, V1, V2)` returns correct Γ and VSWR; grid search confirms a solution exists for the demo range.
**Depends on:** nothing. **Unblocks:** M4 (environment) and M5 (display).

---

### M4: RL Agent, DQN (Member 4)
**Goal:** A trained agent that autonomously drives $|\Gamma|$ toward 0.

**Tasks**
1. Build `env.py` (Gym-style `reset()` / `step(action)`) on top of M3's physics. Until M3 is ready, use a stub with the same signature.
2. State: $[\Gamma_{re}, \Gamma_{im}, V_1, V_2]$ (optionally $R_L, X_L$ normalized, if training shows it helps).
3. Actions (5): $V_1$ up, $V_1$ down, $V_2$ up, $V_2$ down (0.05 V steps), Hold.
4. Reward: incremental reward for reducing $|\Gamma|$, large penalty for VSWR > 2.0, +50 bonus for $|\Gamma| \approx 0$; small per-step cost so the agent is fast.
5. Train with DQN (replay buffer, target network, epsilon-greedy) over thousands of random $Z_L$ episodes. Save `model.pt`.
6. Evaluate: success rate, mean steps to VSWR < 1.5, comparison against random and a grid-search baseline. Save plots.
7. Provide `agent.py` with `load_agent(path)` and `act(state) -> action` for live inference.

**Deliverables:** `rl/env.py`, `rl/dqn.py`, `rl/train.py`, `rl/evaluate.py`, `models/model.pt`, training curves.
**Definition of done:** On random test loads, the agent reaches VSWR < 2 in the large majority of episodes within a bounded number of steps.
**Depends on:** M3 API (stub first). **Unblocks:** M5 (AI mode).

---

### M5: Dashboard, Integration & QA (Member 5)
**Goal:** The visual product and the glue that makes everything work together.

**Tasks**
1. Build the Streamlit app: live Smith Chart (Matplotlib grid), target center, current mismatch dot, optional trail of recent points, live VSWR / $|\Gamma|$ / $Z_{in}$ metrics.
2. Add "Manual (Pots)" / "Auto (RL)" toggle; in Auto mode, override Pots 3 and 4 with the agent's $V_1, V_2$.
3. Integrate: M2 data → M3 physics → M4 agent → display, in one refresh loop (about 10–20 Hz). Include a data source selector (Real board / Mock).
4. Add a status panel: serial connected, last update age, current mode.
5. Own QA: end-to-end test, failure handling (unplugged board, missing model), and `README.md` with run instructions.
6. Prepare and rehearse the demo script (Section 7) and slides.

**Deliverables:** `app/dashboard.py`, `app/smith_plot.py`, `README.md`, demo script.
**Definition of done:** One command launches the dashboard; the full Human vs AI demo runs without intervention.
**Depends on:** stubs from all modules at first, real modules as they land.

---

## 4. Interface Contracts (Agree on These First)

These let everyone work in parallel. Change them only with the whole team's agreement.

**4.1 Serial protocol (M1 → M2).** One ASCII line per sample, 20 Hz, 115200 baud:
```
<adc_RL>,<adc_XL>,<adc_V1>,<adc_V2>\n      e.g.  2048,1900,512,3000
```
Raw 12-bit ADC counts (0–4095). Scaling to physical units happens in Python (M2), so firmware stays simple.

**4.2 Input API (M2 → M5).**
```python
get_latest() -> {"RL": float, "XL": float, "V1_man": float, "V2_man": float, "t": float}
# RL, XL in ohms; V1_man, V2_man in volts; t = timestamp in seconds
```

**4.3 Physics API (M3 → M4, M5).**
```python
compute(RL, XL, V1, V2) -> {"Zin": complex, "gamma": complex, "gamma_abs": float, "vswr": float}
```

**4.4 Agent API (M4 → M5).**
```python
agent = load_agent("models/model.pt")
action = agent.act(state)   # state = [gamma_re, gamma_im, V1, V2]; action in {0..4}
```
Action encoding: 0 = V1+, 1 = V1−, 2 = V2+, 3 = V2−, 4 = Hold.

**4.5 Shared constants** live in `config.py`: `Z0 = 50`, `FREQ = 100e6`, `V_MIN = 0.0`, `V_MAX = 3.3`, `V_STEP = 0.05`, `RL_RANGE = (10, 150)`, `XL_RANGE = (-100, 100)`, `BAUD = 115200`.

---

## 5. Folder Structure

```
rl-impedance-matching/
├── README.md                    # How to install and run (M5)
├── Project.md                   # This file
├── requirements.txt             # pyserial, numpy, torch, streamlit, matplotlib, pytest
├── config.py                    # Shared constants (all)
│
├── firmware/                    # M1: STM32CubeIDE project
│   ├── Core/
│   │   ├── Inc/
│   │   └── Src/
│   │       └── main.c           # ADC+DMA, timer, UART streaming
│   └── README.md                # Build/flash notes
│
├── hardware/                    # M1
│   ├── wiring.md                # Pinout and wiring table
│   └── wiring_diagram.png
│
├── io/                          # M2: data link and processing
│   ├── __init__.py
│   ├── serial_reader.py         # pyserial reader, auto-detect, reconnect
│   ├── mock_source.py           # Fake data source (same API)
│   └── filters.py               # EMA/median filters, ADC→unit mapping
│
├── physics/                     # M3: RF simulation
│   ├── __init__.py
│   ├── varactor.py              # C(V) model
│   ├── pi_network.py            # Zin, Gamma, VSWR
│   └── smith.py                 # Smith chart math helpers
│
├── rl/                          # M4: reinforcement learning
│   ├── __init__.py
│   ├── env.py                   # Gym-style environment
│   ├── dqn.py                   # Network, replay buffer, agent class
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Metrics and baselines
│   └── agent.py                 # load_agent / act for inference
│
├── models/
│   └── model.pt                 # Trained weights (M4)
│
├── app/                         # M5: dashboard and integration
│   ├── dashboard.py             # Streamlit entry point
│   ├── smith_plot.py            # Matplotlib Smith chart drawing
│   └── pipeline.py              # io → physics → agent loop
│
├── tests/                       # Each owner tests own module; M5 adds e2e
│   ├── test_serial_parser.py    # M2
│   ├── test_physics.py          # M3
│   ├── test_env.py              # M4
│   └── test_end_to_end.py       # M5
│
├── docs/
│   ├── physics_notes.md         # M3
│   ├── training_report.md       # M4
│   ├── demo_script.md           # M5
│   └── figures/
│
└── data/                        # Optional: logged captures and test inputs
```

---

## 6. Collaboration Workflow

* **Git:** one repo, `main` stays working. Each member works on a branch named `m1-firmware`, `m2-io`, `m3-physics`, `m4-rl`, `m5-app`. Merge through pull requests reviewed by one other member.
* **Stub first:** on day one each owner pushes a stub implementing their API from Section 4 (returning dummy values), so every other module can import it immediately.
* **Integration checkpoints:**
  1. **Checkpoint A, contracts and stubs:** all APIs exist and import cleanly.
  2. **Checkpoint B, real modules:** M1 streams real data, M3 physics verified, M4 env runs on real physics.
  3. **Checkpoint C, full pipeline:** dashboard runs end-to-end with mock data, then with the board and trained model.
  4. **Checkpoint D, demo rehearsal:** whole team runs the script below at least twice.
* **Help pairs (to cover absences):** M1 ↔ M2 (data link), M3 ↔ M4 (physics/env), M5 supports whoever is behind.

---

## 7. Demo Script (Human vs AI)

1. Start the dashboard in **Manual** mode. The Smith chart dot sits near the center.
2. Twist **Pots 1 and 2** to create a wild mismatch (environmental detuning); watch VSWR spike.
3. Try to fix it by hand with **Pots 3 and 4**, and notice how hard the non-linear tuning is.
4. Flip the toggle to **Auto (RL)**. The agent drives the dot to the center and VSWR drops below 2 within moments.
5. Change the mismatch again with Pots 1 and 2 and show the agent re-tuning on its own.

---

## 8. Why This Architecture Excels for a College Project
1. **STM32 expertise:** ADC with DMA and UART streaming show real embedded-systems skills.
2. **Interactive hardware:** the 4-pot setup makes an abstract RL concept tactile.
3. **"Human vs AI" angle:** manual tuning of a non-linear network shows *why* RL is the right tool.
4. **Modular design:** clear interfaces let 5 people work in parallel, and each module can be tested and demoed on its own.

---

## 9. Risks and Fallbacks

| Risk | Fallback |
|---|---|
| STM32 board or toolchain trouble | Use `mock_source.py`; the rest of the system is unaffected |
| RL training does not converge | Simplify state, shrink the $Z_L$ range, add a curriculum; keep grid-search as a baseline to show |
| Some $Z_L$ values cannot be matched | M3 adjusts L/C ranges; restrict demo knobs to the matchable region |
| Dashboard too slow | Reduce refresh rate, draw only the latest point and a short trail |
| Serial noise or jitter | Stronger filtering in M2; hardware capacitors in M1 |
