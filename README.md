# STM32-Driven Autonomous RL Impedance Matching Network

Autonomous Reinforcement Learning (DQN) impedance matching network with hardware-in-the-loop (STM32 microcontroller + 4 potentiometers) and real-time Streamlit dashboard.

---

## Project Structure

```
rl-impedance-matching/
├── README.md                    # Project overview & running instructions (M5)
├── Project.md                   # Complete specification & 5-person plan
├── requirements.txt             # Python dependencies
├── config.py                    # Shared constants & physical parameters
│
├── firmware/                    # M1: STM32CubeIDE project
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
├── models/                      # Trained model weights
│   └── model.pt
│
├── app/                         # M5: Streamlit dashboard & pipeline
│   ├── dashboard.py             # Streamlit web app
│   ├── smith_plot.py            # Smith chart visualization
│   └── pipeline.py              # IO -> Physics -> RL Agent -> UI loop
│
├── tests/                       # Unit & integration tests
│   ├── test_serial_parser.py    # M2 tests
│   ├── test_physics.py          # M3 tests
│   ├── test_env.py              # M4 tests
│   └── test_end_to_end.py       # M5 tests
│
├── docs/                        # Documentation & reports
│   ├── physics_notes.md         # M3 notes
│   ├── training_report.md       # M4 notes
│   ├── demo_script.md           # M5 notes
│   └── figures/
│
└── data/                        # Logged captures and test inputs
```

---

## Installation & Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run tests:**
   ```bash
   pytest tests/
   ```

3. **Launch the dashboard:**
   ```bash
   streamlit run app/dashboard.py
   ```
