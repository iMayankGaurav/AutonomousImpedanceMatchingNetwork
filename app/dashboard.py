"""M5: Streamlit Web Dashboard for Autonomous RL Impedance Matching Network.

Hardware-in-the-Loop RF Impedance Matching Interface:
    - Real-time Smith chart with target center (50 Ohm), VSWR boundaries, and trajectory trail
    - Professional engineering aesthetic with zero emojis
    - Interactive 'Manual (Pots)' vs 'Auto (RL)' mode arbitrator
    - Telemetry source selection: STM32 UART Serial stream vs Mock telemetry
    - Live metric telemetry: VSWR, |Gamma|, Return Loss, Zin, ZL, V1/V2, C1/C2
    - Ground-truth physics benchmark comparison and suboptimality tracking
    - Real-time strip charts and system diagnostics
"""

from __future__ import annotations

import collections
import sys
import time
from pathlib import Path
from typing import Any, Deque, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Configure project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IO_DIR = PROJECT_ROOT / "io"
for p in (str(PROJECT_ROOT), str(IO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.pipeline import PipelineController, PipelineSnapshot
from app.smith_plot import draw_smith_chart
import config

# Attempt serial tool port discovery
try:
    import serial.tools.list_ports

    def get_available_serial_ports() -> List[str]:
        return [port.device for port in serial.tools.list_ports.comports()]
except Exception:

    def get_available_serial_ports() -> List[str]:
        return []


# Set clean page layout
st.set_page_config(
    page_title="Autonomous RL Impedance Matching",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_pipeline() -> PipelineController:
    """Retrieve or initialize the singleton pipeline controller in session state."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = PipelineController(
            source_type="Mock",
            mode="Manual (Pots)",
            model_path="models/model.pt",
            trail_max_len=35,
            max_rollout_steps=140,
        )
        st.session_state.history_vswr = collections.deque(maxlen=60)
        st.session_state.history_v1 = collections.deque(maxlen=60)
        st.session_state.history_v2 = collections.deque(maxlen=60)
        st.session_state.history_t = collections.deque(maxlen=60)
        st.session_state.stream_active = True
    return st.session_state.pipeline


def main() -> None:
    pipeline = get_pipeline()

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.title("System Controls")
        st.markdown("---")

        # 1. Operating Mode Arbitrator
        st.subheader("1. Operating Mode")
        selected_mode = st.radio(
            "Control Source:",
            ["Manual (Pots)", "Auto (RL)"],
            index=0 if pipeline.mode == "Manual (Pots)" else 1,
            help="Manual: V1/V2 driven by physical pots 3 & 4. Auto: DQN agent tunes V1/V2 to 50 Ohms.",
        )
        if selected_mode != pipeline.mode:
            pipeline.set_mode(selected_mode)

        if pipeline.mode == "Auto (RL)":
            if pipeline.agent_loaded:
                st.success("DQN Agent: ACTIVE (models/model.pt)")
            else:
                st.warning("Status: Grid-Search Fallback")
        else:
            st.info("Mode: Manual Potentiometer Control")

        st.markdown("---")

        # 2. Telemetry Data Source
        st.subheader("2. Telemetry Source")
        source_options = ["Mock Simulation", "Real Hardware (STM32 UART)"]
        selected_source_label = st.radio(
            "Input Stream:",
            source_options,
            index=0 if pipeline.source_type == "Mock" else 1,
        )
        selected_source = "Mock" if selected_source_label.startswith("Mock") else "Serial"

        serial_port = None
        if selected_source == "Serial":
            ports = get_available_serial_ports()
            if ports:
                serial_port = st.selectbox("Serial COM Port:", ports)
            else:
                serial_port = st.text_input("COM Port Name:", value="COM3")
            if st.button("Connect / Reconnect"):
                pipeline.set_source("Serial", serial_port=serial_port)
        else:
            if pipeline.source_type != "Mock":
                pipeline.set_source("Mock")

        st.markdown("---")

        # 3. Visualization & Performance Settings
        st.subheader("3. Visualization Settings")
        refresh_rate_hz = st.slider("Refresh Rate (Hz):", min_value=4, max_value=20, value=10, step=1)
        trail_len = st.slider("Trajectory Trail Length:", min_value=5, max_value=60, value=30, step=5)
        pipeline.trail_max_len = trail_len
        pipeline._trail = collections.deque(pipeline._trail, maxlen=trail_len)

        rollout_steps = st.slider("RL Max Rollout Steps:", min_value=35, max_value=200, value=140, step=5)
        pipeline.max_rollout_steps = rollout_steps

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Clear Trail"):
                pipeline.reset_trail()
        with col_btn2:
            st.session_state.stream_active = st.checkbox("Live Stream", value=True)

        st.markdown("---")

        # 4. Optional Manual Knob Overrides
        use_manual_override = st.checkbox("Manual Disturbance Override", value=False)
        override_data = None
        if use_manual_override:
            st.caption("Direct telemetry injection:")
            sl_rl = st.slider("RL (Load Resistance, Ohm):", 10.0, 150.0, 13.5, 0.5)
            sl_xl = st.slider("XL (Load Reactance, Ohm):", -100.0, 100.0, 69.5, 0.5)
            sl_v1 = st.slider("V1 Manual (V):", 0.0, 3.3, 1.65, 0.05)
            sl_v2 = st.slider("V2 Manual (V):", 0.0, 3.3, 1.65, 0.05)
            override_data = {
                "RL": sl_rl,
                "XL": sl_xl,
                "V1_man": sl_v1,
                "V2_man": sl_v2,
                "t": time.time(),
            }

    # --- MAIN PAGE HEADER ---
    st.markdown(
        """
        <div style="background: #1e293b; padding: 18px 24px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #0284c7;">
            <h1 style="color: #f8fafc; margin: 0; font-size: 24px; font-weight: 600; letter-spacing: -0.5px;">
                Autonomous RL RF Impedance Matching Network
            </h1>
            <p style="color: #94a3b8; margin: 6px 0 0 0; font-size: 13.5px; font-family: monospace;">
                Hardware-in-the-Loop RF Simulation & Smith Chart Interface | 100 MHz Pi-Network | Double DQN Control
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Placeholder containers for smooth updating
    kpi_container = st.empty()
    center_container = st.empty()
    diagnostics_container = st.empty()

    # Step the pipeline once per render cycle
    snapshot: PipelineSnapshot = pipeline.step(
        override_data=override_data,
        auto_full_rollout=True,
    )

    # Record trend history
    st.session_state.history_vswr.append(snapshot.vswr)
    st.session_state.history_v1.append(snapshot.V1_active)
    st.session_state.history_v2.append(snapshot.V2_active)
    st.session_state.history_t.append(snapshot.timestamp)

    # --- 1. TOP KPI METRIC CARDS ---
    with kpi_container.container():
        k1, k2, k3, k4, k5 = st.columns(5)

        # VSWR card
        vswr_val = snapshot.vswr
        if vswr_val <= 1.20:
            vswr_status = "Optimal Match (VSWR < 1.20)"
            delta_color = "normal"
        elif vswr_val <= 1.50:
            vswr_status = "Acceptable (VSWR < 1.50)"
            delta_color = "off"
        else:
            vswr_status = "Mismatched (VSWR >= 1.50)"
            delta_color = "inverse"

        k1.metric(
            label="Live VSWR",
            value=f"{vswr_val:.2f}" if vswr_val < 100 else "Inf",
            delta=vswr_status,
            delta_color=delta_color,
        )

        # Reflection Magnitude & Return Loss
        k2.metric(
            label="Reflection |Gamma|",
            value=f"{snapshot.gamma_abs:.3f}",
            delta=f"{snapshot.return_loss_db:.1f} dB RL",
        )

        # Input Impedance Zin
        sign_str = "+" if snapshot.Z_in.imag >= 0 else "-"
        k3.metric(
            label="Input Impedance Zin",
            value=f"{snapshot.Z_in.real:.1f} {sign_str} j{abs(snapshot.Z_in.imag):.1f} Ohm",
            delta="Target: 50.0 + j0 Ohm",
        )

        # Disturbance Load ZL
        sign_l_str = "+" if snapshot.XL >= 0 else "-"
        k4.metric(
            label="Antenna Load ZL",
            value=f"{snapshot.RL:.1f} {sign_l_str} j{abs(snapshot.XL):.1f} Ohm",
            delta="Detuning Disturbance",
        )

        # Active Varactors
        k5.metric(
            label="Varactors (V1 | V2)",
            value=f"{snapshot.V1_active:.2f}V | {snapshot.V2_active:.2f}V",
            delta=f"C1: {snapshot.C1_pF:.1f}pF | C2: {snapshot.C2_pF:.1f}pF",
        )

    # --- 2. CENTER CONTENT (SMITH CHART & TELEMETRY CHARTS) ---
    with center_container.container():
        left_col, right_col = st.columns([1.1, 0.9])

        with left_col:
            st.markdown("#### Live Smith Chart")
            fig = draw_smith_chart(
                current_gamma=snapshot.gamma_in,
                target_gamma=snapshot.gamma_opt,
                trail=snapshot.trail,
                gamma_load=snapshot.gamma_load,
                figsize=(5.5, 5.5),
                dark_mode=False,
                show_labels=True,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right_col:
            st.markdown("#### Real-Time Telemetry & Convergence")

            # Benchmarking metrics
            b1, b2, b3 = st.columns(3)
            b1.metric("Match Status", snapshot.match_status)
            b2.metric("Optimal VSWR*", f"{snapshot.vswr_opt:.2f}")
            b3.metric("Gap Delta-VSWR", f"{snapshot.delta_vswr:.3f}")

            # Rolling VSWR chart
            if len(st.session_state.history_vswr) > 1:
                df_vswr = pd.DataFrame({
                    "VSWR": list(st.session_state.history_vswr),
                    "Target (1.20)": [1.20] * len(st.session_state.history_vswr),
                })
                st.line_chart(df_vswr, height=170)

            # Rolling Varactor Voltage chart
            if len(st.session_state.history_v1) > 1:
                df_v = pd.DataFrame({
                    "V1 (Source)": list(st.session_state.history_v1),
                    "V2 (Load)": list(st.session_state.history_v2),
                })
                st.line_chart(df_v, height=170)

            # Agent Action Banner
            if snapshot.mode == "Auto (RL)":
                st.text(f"DQN Decision: {snapshot.last_action_name} | Cycle Escaped: {snapshot.cycle_escaped} | Boundary Recovered: {snapshot.boundary_recovered}")
            else:
                st.text(f"Manual Mode: Direct Potentiometer Pass-Through (V1={snapshot.V1_active:.2f}V, V2={snapshot.V2_active:.2f}V)")

    # --- 3. SYSTEM DIAGNOSTICS & STATUS PANEL ---
    with diagnostics_container.container():
        st.markdown("---")
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.text(f"Stream: {snapshot.connection_status}")
        d2.text(f"Latency Age: {snapshot.last_update_age:.3f} s")
        d3.text(f"Loop Rate: {snapshot.fps:.1f} Hz")
        d4.text(f"Packets: {snapshot.packet_count}")
        d5.text(f"State: {snapshot.match_status}")

        # Demo Script Walkthrough Expander
        with st.expander("Demonstration Walkthrough Guide (Section 7)"):
            st.markdown(
                """
                1. **Initial Calibration:** Launch in **Manual (Pots)** mode. Verify the dot sits near the center with nominal load values.
                2. **Environmental Detuning:** Rotate **Potentiometer 1 (RL)** down to approximately 12 Ohm and **Potentiometer 2 (XL)** up to +75 Ohm. Observe the VSWR jump above 8.0 with the red mismatch indicator.
                3. **Manual Human Tuning Challenge:** Attempt to steer the dot back to the 50 Ohm center using **Potentiometers 3 and 4**. Note how difficult simultaneous non-linear coupled varactor adjustment is.
                4. **Autonomous AI Engagement:** Switch the control mode to **Auto (RL)**. The Double DQN agent evaluates reflection state and converges inside the VSWR < 1.20 target ring within moments.
                5. **Dynamic Continuous Tracking:** Rotate the disturbance pots while in Auto mode to demonstrate how the agent continuously tracks and re-acquires optimal match autonomously.
                """
            )

    # Refresh loop for live animation
    if st.session_state.stream_active:
        time.sleep(1.0 / max(1.0, float(refresh_rate_hz)))
        st.rerun()


if __name__ == "__main__":
    main()
