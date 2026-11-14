"""M5: Streamlit Web Dashboard for Autonomous RL Impedance Matching Network.

Hardware-in-the-Loop RF Impedance Matching Interface:
    - Real-time Smith chart with target center (50 Ohm), VSWR boundaries, and trajectory trail
    - Professional engineering aesthetic with zero emojis and pure native Streamlit components
    - High-performance, flicker-free rendering via Streamlit fragments (st.fragment)
    - Interactive 'Manual (Pots)' vs 'Auto (RL)' mode arbitrator
    - Telemetry source selection: STM32 UART Serial stream vs Mock telemetry
    - Live metric telemetry: VSWR, |Gamma|, Return Loss, Zin, ZL, V1/V2, C1/C2
    - Ground-truth physics benchmark comparison and suboptimality tracking
    - Real-time strip charts and system diagnostics
"""

from __future__ import annotations

import collections
import inspect
import sys
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

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


def _stretch_kw() -> Dict[str, Any]:
    """Compatibility helper for button/pyplot full width across Streamlit versions."""
    if "width" in inspect.signature(st.button).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


# Set clean page configuration
st.set_page_config(
    page_title="RF Impedance Matching Network",
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

    # --- SIDEBAR: INSTRUMENT CONTROLS ---
    with st.sidebar:
        st.header("Instrument Controls")

        # 1. Control Mode Arbitrator
        selected_mode = st.segmented_control(
            "Control Mode",
            options=["Manual (Pots)", "Auto (RL)"],
            default=pipeline.mode,
            help="Manual: V1/V2 driven by physical potentiometers 3 & 4. Auto: DQN agent tunes V1/V2 to 50 Ohms.",
        )
        if selected_mode and selected_mode != pipeline.mode:
            pipeline.set_mode(selected_mode)

        if pipeline.mode == "Auto (RL)":
            if pipeline.agent_loaded:
                st.caption("Active Controller: Double DQN Policy (models/model.pt)")
            else:
                st.caption("Active Controller: Physics Grid-Search Fallback")
        else:
            st.caption("Active Controller: Manual Potentiometer Pass-Through")

        st.divider()

        # 2. Telemetry Source Selection
        selected_source_label = st.segmented_control(
            "Telemetry Source",
            options=["Mock Simulation", "STM32 UART"],
            default="Mock Simulation" if pipeline.source_type == "Mock" else "STM32 UART",
        )
        selected_source = "Mock" if (selected_source_label or "").startswith("Mock") else "Serial"

        if selected_source == "Serial":
            ports = get_available_serial_ports()
            if ports:
                serial_port = st.selectbox("COM Port", ports)
            else:
                serial_port = st.text_input("COM Port", value="COM3")
            if st.button("Connect Serial Port", **_stretch_kw()):
                pipeline.set_source("Serial", serial_port=serial_port)
        else:
            if pipeline.source_type != "Mock":
                pipeline.set_source("Mock")

        st.divider()

        # 3. Stream Controls & Trail Reset
        st.subheader("Data Stream")
        c_stream, c_clear = st.columns(2)
        with c_stream:
            st.session_state.stream_active = st.toggle("Acquisition", value=st.session_state.stream_active)
        with c_clear:
            if st.button("Clear Trail", **_stretch_kw()):
                pipeline.reset_trail()

        # 4. Settings Expander
        with st.expander("Acquisition & RL Parameters", expanded=False):
            refresh_rate_hz = st.slider("Sampling Rate (Hz)", min_value=1, max_value=10, value=4, step=1)
            trail_len = st.slider("Trajectory Trail Length", min_value=5, max_value=60, value=30, step=5)
            pipeline.trail_max_len = trail_len
            pipeline._trail = collections.deque(pipeline._trail, maxlen=trail_len)

            rollout_steps = st.slider("RL Max Rollout Steps", min_value=35, max_value=200, value=140, step=5)
            pipeline.max_rollout_steps = rollout_steps

        # 5. Disturbance Override Expander
        with st.expander("Manual Disturbance Injection", expanded=False):
            use_manual_override = st.toggle("Enable Disturbance Injection", value=False)
            override_data = None
            if use_manual_override:
                st.caption("Inject synthetic load impedance detuning and potentiometer bias:")
                sl_rl = st.slider("Load Resistance RL (Ohm)", 10.0, 150.0, 13.5, 0.5)
                sl_xl = st.slider("Load Reactance XL (Ohm)", -100.0, 100.0, 69.5, 0.5)
                sl_v1 = st.slider("Manual Pot V1 (V)", 0.0, 3.3, 1.65, 0.05)
                sl_v2 = st.slider("Manual Pot V2 (V)", 0.0, 3.3, 1.65, 0.05)
                override_data = {
                    "RL": sl_rl,
                    "XL": sl_xl,
                    "V1_man": sl_v1,
                    "V2_man": sl_v2,
                    "t": time.time(),
                }
            else:
                override_data = None

    # --- MAIN PAGE HEADER (Native Streamlit) ---
    st.title("RF Impedance Matching Network")
    st.caption("Hardware-in-the-Loop RF Telemetry & Double DQN Autonomous Tuning | 100 MHz Pi-Network")

    # Refresh interval in seconds for fragment execution
    refresh_rate = refresh_rate_hz if "refresh_rate_hz" in locals() else 4
    refresh_interval_sec = (1.0 / float(refresh_rate)) if st.session_state.stream_active else None

    # --- DYNAMIC TELEMETRY FRAGMENT (Only this section updates, eliminating flicker) ---
    @st.fragment(run_every=refresh_interval_sec)
    def render_live_telemetry_dashboard() -> None:
        # Step the pipeline once per render cycle
        snapshot: PipelineSnapshot = pipeline.step(
            override_data=override_data,
            auto_full_rollout=True,
        )

        # Record history for trend charts
        st.session_state.history_vswr.append(snapshot.vswr)
        st.session_state.history_v1.append(snapshot.V1_active)
        st.session_state.history_v2.append(snapshot.V2_active)
        st.session_state.history_t.append(snapshot.timestamp)

        # 1. Primary Metrics Row
        with st.container(border=True):
            m1, m2, m3, m4, m5 = st.columns(5)

            vswr_val = snapshot.vswr
            if vswr_val <= 1.20:
                vswr_delta_label = "Optimal Match"
                vswr_delta_color = "normal"
            elif vswr_val <= 1.50:
                vswr_delta_label = "Acceptable Match"
                vswr_delta_color = "off"
            else:
                vswr_delta_label = "Mismatched"
                vswr_delta_color = "inverse"

            m1.metric(
                label="Voltage Standing Wave Ratio (VSWR)",
                value=f"{vswr_val:.2f}" if vswr_val < 100 else "Inf",
                delta=vswr_delta_label,
                delta_color=vswr_delta_color,
            )

            m2.metric(
                label="Reflection Coefficient |Gamma|",
                value=f"{snapshot.gamma_abs:.3f}",
                delta=f"{snapshot.return_loss_db:.1f} dB Return Loss",
            )

            sign_zin = "+" if snapshot.Z_in.imag >= 0 else "-"
            m3.metric(
                label="Input Impedance Zin",
                value=f"{snapshot.Z_in.real:.1f} {sign_zin} j{abs(snapshot.Z_in.imag):.1f} Ohm",
                delta="Target: 50.0 + j0.0 Ohm",
            )

            sign_zl = "+" if snapshot.XL >= 0 else "-"
            m4.metric(
                label="Antenna Load ZL",
                value=f"{snapshot.RL:.1f} {sign_zl} j{abs(snapshot.XL):.1f} Ohm",
                delta="Detuned Load",
            )

            m5.metric(
                label="Varactor Bias (V1 | V2)",
                value=f"{snapshot.V1_active:.2f} V | {snapshot.V2_active:.2f} V",
                delta=f"C1: {snapshot.C1_pF:.1f} pF | C2: {snapshot.C2_pF:.1f} pF",
            )

        # 2. Main Visualization & Telemetry Section
        col_smith, col_telemetry = st.columns([1.1, 0.9])

        with col_smith:
            with st.container(border=True):
                st.subheader("Smith Chart (100 MHz)")
                fig = draw_smith_chart(
                    current_gamma=snapshot.gamma_in,
                    target_gamma=snapshot.gamma_opt,
                    trail=snapshot.trail,
                    gamma_load=snapshot.gamma_load,
                    figsize=(5.2, 5.2),
                    dark_mode=False,
                    show_labels=True,
                )
                st.pyplot(fig, **_stretch_kw())
                plt.close(fig)

        with col_telemetry:
            with st.container(border=True):
                st.subheader("Telemetry & Convergence")

                # Benchmarking readouts
                b1, b2, b3 = st.columns(3)
                b1.metric("Status", snapshot.match_status)
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
                        "V1 Source": list(st.session_state.history_v1),
                        "V2 Load": list(st.session_state.history_v2),
                    })
                    st.line_chart(df_v, height=170)

                # Agent Action Readout
                if snapshot.mode == "Auto (RL)":
                    st.caption(
                        f"RL Action: {snapshot.last_action_name} | "
                        f"Cycle Escaped: {snapshot.cycle_escaped} | "
                        f"Boundary Recovered: {snapshot.boundary_recovered}"
                    )
                else:
                    st.caption(
                        f"Manual Mode: Direct Potentiometer Pass-Through "
                        f"(V1={snapshot.V1_active:.2f} V, V2={snapshot.V2_active:.2f} V)"
                    )

        # 3. System Diagnostics Footer
        with st.container(border=True):
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.caption(f"Stream: **{snapshot.connection_status}**")
            d2.caption(f"Latency: **{snapshot.last_update_age:.3f} s**")
            d3.caption(f"Rate: **{snapshot.fps:.1f} Hz**")
            d4.caption(f"Packets: **{snapshot.packet_count}**")
            d5.caption(f"Match: **{snapshot.match_status}**")

    # Render fragment
    render_live_telemetry_dashboard()

    # --- TEST PROCEDURE & DEMONSTRATION GUIDE ---
    with st.expander("Demonstration Walkthrough Guide", expanded=False):
        st.markdown(
            """
            1. **Initial Calibration:** Launch in **Manual (Pots)** mode. Verify the dot sits near the center with nominal load values.
            2. **Environmental Detuning:** Rotate **Potentiometer 1 (RL)** down to approximately 12 Ohm and **Potentiometer 2 (XL)** up to +75 Ohm. Observe the VSWR jump above 8.0 with the mismatched indicator.
            3. **Manual Human Tuning Challenge:** Attempt to steer the dot back to the 50 Ohm center using **Potentiometers 3 and 4**. Note how difficult simultaneous non-linear coupled varactor adjustment is.
            4. **Autonomous AI Engagement:** Switch the control mode to **Auto (RL)**. The Double DQN agent evaluates reflection state and converges inside the VSWR < 1.20 target ring within moments.
            5. **Dynamic Continuous Tracking:** Rotate the disturbance pots while in Auto mode to demonstrate how the agent continuously tracks and re-acquires optimal match autonomously.
            """
        )


if __name__ == "__main__":
    main()
