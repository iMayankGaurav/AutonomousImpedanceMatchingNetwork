# Autonomous RL-Driven Impedance Matching Network
## 4-Day Project Architecture & Execution Plan

**Hardware:** Raspberry Pi, Arduino Uno R4 WiFi, HC-SR04 Ultrasonic Sensor  
**Software:** Python, PyTorch (DQN), Streamlit, Matplotlib, Arduino C++  
**Timeframe:** 4 Days  
**Team Size:** 3-5 Members  

---

## 1. Project Significance: What Problem Are We Solving?

### The Real-World Problem ("The Hand Effect")
In modern wireless communications (5G, IoT, smartphones), antennas are designed to match a standard $50\,\Omega$ characteristic impedance ($Z_0$). However, when a physical object (like a human hand, a metal case, or a table) moves near the antenna, it changes the antenna's electrical properties. This is called **environmental detuning**. 

When the antenna detunes:
1. **Signal Reflection:** Radio Frequency (RF) energy reflects back into the transmitter instead of radiating outward, causing a high Voltage Standing Wave Ratio (VSWR).
2. **Battery Drain:** The transmitter works harder to push the signal, wasting power.
3. **Dropped Connections:** Data rates plummet and calls drop.

### The Solution
To solve this, advanced communication systems use **Dynamic Impedance Matching Networks**. By placing a tunable circuit (usually a $\pi$-network with voltage-controlled varactor diodes) between the transmitter and the antenna, the system can continuously adjust its internal capacitance to counteract the external detuning.

### Project Goal
Building a 100 MHz physical RF circuit requires expensive vector network analyzers and custom PCBs. Instead, **this project simulates the RF physics in software while using physical hardware (Arduino + Ultrasonic Sensor) to drive the dynamic changes.** A Reinforcement Learning (RL) agent on a Raspberry Pi will autonomously navigate a Smith Chart to re-tune the virtual network in real-time as objects move closer to the physical sensor.

---

## 2. Step-by-Step System Architecture

The architecture relies on a **Hardware-in-the-Loop (HIL)** concept.

### Stage 1: Environmental Sensing (Arduino Uno R4 WiFi)
*   **The Sensor:** The HC-SR04 ultrasonic sensor constantly measures the distance to a nearby physical object (representing a hand approaching an antenna).
*   **The Processor:** The Arduino Uno R4 WiFi reads the raw sensor data, applies an Exponential Moving Average (EMA) filter to remove noise, and maps the distance (e.g., 5 cm to 30 cm) to a specific payload.
*   **The Telemetry:** Leveraging the Uno R4's built-in ESP32 module, the Arduino sends the distance data wirelessly over your local WiFi network (via UDP or MQTT) to the Raspberry Pi.

### Stage 2: The RF Physics Engine (Raspberry Pi - Python)
*   **Impedance Mapping:** The Raspberry Pi receives the physical distance data. It maps this distance to a complex antenna load impedance, $Z_L$. (e.g., $30\text{ cm} \rightarrow 50 + j0\,\Omega$ [perfect match], $5\text{ cm} \rightarrow 15 + j35\,\Omega$ [heavy mismatch]).
*   **The $\pi$-Network Math:** The Python script models a virtual circuit with two tunable capacitors ($C_1, C_2$) and an inductor ($L$). 
*   **Reflection Calculation:** It calculates the complex input impedance $Z_{in}$ and the resulting reflection coefficient $\Gamma = \frac{Z_{in} - Z_0}{Z_{in} + Z_0}$.

### Stage 3: The Deep Q-Network (DQN) Agent (Raspberry Pi - PyTorch)
*   **Observation:** The RL agent looks at the current mismatch coordinates on the Smith Chart ($\Gamma_{re}, \Gamma_{im}$) and the current virtual tuning voltages ($V_1, V_2$).
*   **Action:** It outputs a decision to increase, decrease, or hold the voltage applied to $C_1$ or $C_2$ (simulating changing the varactor reverse-bias).
*   **Goal:** The agent’s reward function forces it to find the quickest path to drive $|\Gamma|$ down to near $0$, bringing the dot back to the center of the Smith Chart.

### Stage 4: Visualization (Streamlit UI)
*   A local web dashboard hosted on the Pi displays a live **Smith Chart**.
*   As you move your hand near the Arduino, the UI shows the impedance dot flying off to the edge of the circle (detuning).
*   Millisecond later, the RL agent kicks in, and the UI traces a path back to the center target (re-matching).

---

## 3. The 4-Day Execution Plan

### Day 1: Hardware Telemetry & Foundation
**Goal:** Get the Arduino and Raspberry Pi talking wirelessly.
*   **Task 1 (Hardware):** Wire the HC-SR04 to the Arduino Uno R4 WiFi.
*   **Task 2 (Firmware):** Write the Arduino C++ code to ping the sensor, filter the data, and connect to a local WiFi hotspot.
*   **Task 3 (Networking):** Write a Python UDP/TCP server on the Raspberry Pi to receive the distance data stream at ~20 Hz.
*   **Task 4 (Mapping):** Write a simple Python function to translate physical distance (cm) into a virtual load impedance $Z_L$ ($R + jX$).

### Day 2: RF Physics Engine & Mathematical Environment
**Goal:** Build the virtual RF circuit that the AI will interact with.
*   **Task 1 (Component Models):** Create the $C(V)$ function mapping control voltage ($0\text{-}3.3\text{ V}$) to varactor capacitance.
*   **Task 2 (Network Equations):** Code the complex math to calculate $Z_{in}$ from $Z_L$, $C_1$, $C_2$, and $L$ at a simulated $100\text{ MHz}$.
*   **Task 3 (Smith Chart Logic):** Calculate $\Gamma_{re}$, $\Gamma_{im}$, $|\Gamma|$, and VSWR.
*   **Task 4 (Testing):** Manually plug in dummy voltages and distances to ensure the physics math outputs correct values.

### Day 3: Reinforcement Learning (DQN)
**Goal:** Train the AI to autonomously fix the mismatches.
*   **Task 1 (Environment Setup):** Wrap the RF Engine in a standard OpenAI Gym-style environment (`reset`, `step`, `reward`).
*   **Task 2 (Reward Design):** Implement a reward function: $+10$ for reducing $|\Gamma|$, $-0.1$ for taking a step, $+50$ for achieving VSWR $< 1.2$.
*   **Task 3 (Training):** Run an offline PyTorch DQN training loop. Train the agent on random starting impedances (so it learns how to navigate from *anywhere* on the Smith Chart).
*   **Task 4 (Inference):** Export the trained model weights and write the real-time inference loop on the Pi. *Optional: Write a simple coordinate-search fallback algorithm in case the AI oscillates.*

### Day 4: UI Dashboard & System Integration
**Goal:** Bring it all together into a visually stunning presentation.
*   **Task 1 (Streamlit):** Build a Python Streamlit app. Use Matplotlib to draw the complex Smith Chart grid (circles of constant resistance and reactance).
*   **Task 2 (Integration):** Combine the UDP receiver, physics engine, RL inference, and UI into one main asynchronous loop.
*   **Task 3 (Data Logging):** Add a live readout of VSWR, Return Loss (dB), and current Agent actions. 
*   **Task 4 (Demo Prep):** Practice the physical interaction. Move a hand over the Arduino, watch the UI detune, and watch the AI dynamically walk the impedance back to the center of the Smith chart. Record a video of this working.

---

## 4. Why This Approach Wins

1. **No Expensive RF Gear:** You avoid needing a $5,000 VNA and custom microwave PCB design.
2. **True IoT Integration:** Utilizing the Arduino Uno R4 WiFi showcases modern distributed telemetry.
3. **Applied AI:** You are solving a non-linear control problem using Reinforcement Learning, which is highly impressive for a college-level project.
4. **Visual Impact:** Real-time Smith Charts reacting to physical hand movements make for a captivating, easy-to-understand demonstration.