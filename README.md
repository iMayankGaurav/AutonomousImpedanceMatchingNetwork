# STM32-Driven Autonomous RL Impedance Matching Network

[![Pytest Suite](https://img.shields.io/badge/pytest-passing-success.svg)](./tests/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/hardware-STM32F446RE%20%2B%204%20Pots-orange.svg)](./firmware/)

An autonomous, Reinforcement Learning (Double DQN) dynamic impedance matching system featuring **Hardware-in-the-Loop (HIL)** interaction. A 4-channel potentiometer interface on an **STM32 microcontroller** streams live antenna disturbances and manual tuning inputs over UART at ~20 Hz to a Python physics simulation and an interactive **Streamlit Smith Chart dashboard**.

---

## 1. System Architecture

```
 ┌──────────────────────┐   UART/USB    ┌─────────────────────────────────────────────┐
 │ STM32 + 4 Pots       │  CSV @ 20 Hz  │ Laptop (Python Host)                        │
 │ P1 R_L  P2 X_L       │ ────────────► │ serial → filter → physics ⇄ RL agent → UI   │
 │ P3 V1   P4 V2        │               │                       (Streamlit Smith Chart)│
 └──────────────────────┘               └─────────────────────────────────────────────┘
```

* **M1 (Firmware & Hardware):** STM32F446RE ADC with DMA (scan mode) + Timer-triggered UART streaming at 115200 baud.
* **M2 (Data Link & Filters):** Auto-reconnecting `SerialReader`, software median/EMA filtering, unit scaling, and realistic `MockSource`.
* **M3 (RF Physics Engine):** 100 MHz $\pi$-matching network ($Z_{in}$, $\Gamma$, VSWR), non-linear varactor diode $C(V)$ modeling, and Smith chart conformal transformations.
* **M4 (Reinforcement Learning):** Gymnasium-compatible environment and trained Double DQN agent (`models/model.pt`) selecting optimal discrete voltage steps ($V_1+, V_1-, V_2+, V_2-, \text{Hold}$).
* **M5 (Dashboard, Integration & QA):** Real-time integration pipeline (`app/pipeline.py`), Matplotlib-based Smith chart rendering (`app/smith_plot.py`), interactive Streamlit dashboard (`app/dashboard.py`), and end-to-end QA test suite (`tests/test_end_to_end.py`).

---

## 2. Installation & Quick Start

### A. Prerequisites
Ensure Python 3.10+ is installed. Clone the repository and enter the project directory:
```bash
git clone https://github.com/iMayankGaurav/AutonomousImpedanceMatchingNetwork.git
cd AutonomousImpedanceMatchingNetwork
```

### B. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies include: `streamlit`, `matplotlib`, `numpy`, `torch`, `pyserial`, `pytest`)*

### C. Launch the Dashboard
Run the Streamlit application with a single command:
```bash
streamlit run app/dashboard.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 3. Operating Modes & Live Demo

The dashboard supports two operating modes via the sidebar toggle:

1. **Manual (Pots) Mode:**
   - Varactor voltages $V_1, V_2$ are driven directly by physical Potentiometers 3 and 4 (or by the Mock sweep).
   - Test your intuition: try turning the knobs to steer the input impedance dot into the $50\,\Omega$ center circle ($\text{VSWR} < 2.0$). Notice how difficult non-linear 2D tuning is!

2. **Auto (RL) Mode:**
   - Overrides Pots 3 and 4. The trained Double DQN agent takes over, observing $[\text{Re}(\Gamma), \text{Im}(\Gamma), V_1, V_2]$ and iteratively stepping voltages to drive VSWR to $< 1.5$ within moments.
   - Sweep the disturbance pots (Pots 1 & 2) and watch the agent dynamically re-acquire impedance match in real time.

For full presentation talking points, refer to the [Live Demo Script](docs/demo_script.md).

---

## 4. Telemetry Source Selection

* **Mock (Simulated) [Default]:**
  Generates smooth continuous sinusoidal sweeps and sudden detuning disturbances without requiring physical hardware. Ideal for immediate offline evaluation and demos.
* **Real Board (STM32 UART):**
  Connect your STM32 Nucleo board via USB. Select your COM port in the sidebar and click **Connect / Reconnect**. The system automatically reads 12-bit ADC values formatted as `<adc_RL>,<adc_XL>,<adc_V1>,<adc_V2>\n` at 115200 baud.

---

## 5. Running the Test Suite

Run all unit and integration tests across all modules:
```bash
pytest tests/ -v
```

To run individual module test suites:
* **M2 Serial & Filters:** `pytest tests/test_serial_parser.py -v`
* **M3 RF Physics Engine:** `pytest tests/test_physics.py -v`
* **M4 RL Environment & Training:** `pytest rl/tests/ -v`
* **M5 End-to-End Pipeline & UI:** `pytest tests/test_end_to_end.py -v`

---

## 6. Project Layout

```
rl-impedance-matching/
├── README.md                    # Project overview & running instructions (M5)
├── Project.md                   # Complete specification & 5-person plan
├── requirements.txt             # Python dependencies
├── config.py                    # Shared constants & physical parameters
│
├── firmware/                    # M1: STM32CubeIDE project (HAL, ADC+DMA, UART)
│   ├── Core/
│   │   ├── Inc/
│   │   └── Src/
│   │       └── main.c           # ADC+DMA, timer, UART streaming
│   └── README.md                # Build/flash notes
│
├── hardware/                    # M1: Wiring & schematics
│   ├── wiring.md                # Pinout and wiring table
│   └── wiring_diagram.png
│
├── io/                          # M2: Data link & signal processing
│   ├── __init__.py
│   ├── serial_reader.py         # pyserial reader, auto-detect, reconnect
│   ├── mock_source.py           # Simulated hardware stream
│   └── filters.py               # EMA/median filters, ADC-to-unit mapping
│
├── physics/                     # M3: RF physics simulation engine
│   ├── __init__.py
│   ├── varactor.py              # Varactor diode C(V) model
│   ├── pi_network.py            # Pi-matching network (Zin, Gamma, VSWR)
│   └── smith.py                 # Smith chart mathematical helpers
│
├── rl/                          # M4: Reinforcement learning (DQN)
│   ├── __init__.py
│   ├── env.py                   # Gym-style RL environment
│   ├── dqn.py                   # Deep Q-Network, replay buffer, agent
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Performance evaluation & baselines
│   └── agent.py                 # Inference interface
│
├── models/                      # Trained model weights (M4)
│   └── model.pt
│
├── app/                         # M5: Streamlit dashboard & pipeline
│   ├── dashboard.py             # Streamlit web app entry point
│   ├── smith_plot.py            # Smith chart visualization & geometry
│   └── pipeline.py              # IO -> Physics -> RL Agent integration loop
│
├── tests/                       # Unit & integration tests
│   ├── test_serial_parser.py    # M2 tests
│   ├── test_physics.py          # M3 tests
│   ├── test_env.py              # M4 tests
│   └── test_end_to_end.py       # M5 end-to-end tests
│
└── docs/                        # Documentation & reports
    ├── physics_notes.md         # M3 notes
    ├── training_report.md       # M4 notes
    ├── demo_script.md           # M5 demo script & presentation guide
    └── figures/
```
