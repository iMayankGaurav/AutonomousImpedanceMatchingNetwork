"""M5 end-to-end integration tests.

Verifies:
1. End-to-end pipeline initialization (MockSource -> M3 Physics -> M4 Agent).
2. Mode arbitration: Manual mode pot pass-through vs. Auto (RL) autonomous tuning.
3. Smith chart generation: Figure creation, trajectory trail, and visual elements.
4. Failure handling: missing model file, disconnected serial port, extreme loads.
5. Real-time loop throughput: pipeline cycle execution speed exceeding 20 Hz.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Ensure project root and io directory are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IO_DIR = PROJECT_ROOT / "io"
for p in (str(PROJECT_ROOT), str(IO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.pipeline import PipelineController, PipelineSnapshot
from app.smith_plot import create_smith_chart
from physics.smith import impedance_to_gamma, gamma_to_impedance


# ============================================================================
# 1. Pipeline Lifecycle & Manual Mode Tests
# ============================================================================

def test_pipeline_initialization_mock_mode():
    """Verify PipelineController initializes in Mock mode and produces valid snapshots."""
    pipe = PipelineController(source_type="Mock", mode="Manual (Pots)", trail_max_len=20)
    try:
        snap = pipe.step()
        assert isinstance(snap, PipelineSnapshot)
        assert snap.mode == "Manual (Pots)"
        assert snap.source_type == "Mock"
        assert 10.0 <= snap.RL <= 150.0
        assert -100.0 <= snap.XL <= 100.0
        assert 0.0 <= snap.V1_active <= 3.3
        assert 0.0 <= snap.V2_active <= 3.3
        assert isinstance(snap.Z_in, complex)
        assert isinstance(snap.gamma_in, complex)
        assert snap.gamma_abs <= 1.0 + 1e-9
        assert snap.vswr >= 1.0
        assert len(snap.trail) == 1
    finally:
        pipe.close()


def test_pipeline_manual_override_passthrough():
    """Verify manual override voltages pass straight through to varactors."""
    pipe = PipelineController(source_type="Mock", mode="Manual (Pots)")
    try:
        override = {
            "RL": 50.0,
            "XL": 0.0,
            "V1_man": 1.25,
            "V2_man": 2.45,
            "t": time.time(),
        }
        snap = pipe.step(override_data=override)
        assert snap.RL == 50.0
        assert snap.XL == 0.0
        assert snap.V1_active == 1.25
        assert snap.V2_active == 2.45
        assert snap.mode == "Manual (Pots)"
    finally:
        pipe.close()


# ============================================================================
# 2. Autonomous RL Mode Tuning Tests
# ============================================================================

def test_pipeline_auto_rl_tuning_convergence():
    """Verify that switching to Auto (RL) mode uses DQN agent to drive VSWR down."""
    pipe = PipelineController(source_type="Mock", mode="Manual (Pots)")
    try:
        # Inject detuned antenna load (RL=13.5, XL=69.5) with initial mid-scale voltages
        override = {
            "RL": 13.5,
            "XL": 69.5,
            "V1_man": 1.65,
            "V2_man": 1.65,
            "t": time.time(),
        }
        # Step 1: In Manual mode, VSWR is high (~9.77)
        initial_snap = pipe.step(override_data=override)
        initial_vswr = initial_snap.vswr
        assert initial_vswr > 5.0, f"Expected high mismatch VSWR in manual mode; got {initial_vswr:.2f}"

        # Step 2: Switch to Auto (RL) mode and step
        pipe.set_mode("Auto (RL)")
        final_snap = pipe.step(override_data=override, auto_full_rollout=True)

        # Agent should have taken tuning steps and improved VSWR significantly
        assert final_snap.steps_in_auto > 0
        assert final_snap.vswr < initial_vswr
        assert final_snap.vswr < 1.5, f"Expected VSWR < 1.5 after RL tuning; got {final_snap.vswr:.2f}"
        assert final_snap.is_matched is True
    finally:
        pipe.close()


# ============================================================================
# 3. Trajectory Trail & History Tests
# ============================================================================

def test_pipeline_trajectory_trail_accumulation_and_reset():
    """Verify trajectory breadcrumbs accumulate up to max_len and reset properly."""
    max_len = 10
    pipe = PipelineController(source_type="Mock", trail_max_len=max_len)
    try:
        for _ in range(15):
            snap = pipe.step()
        assert len(snap.trail) == max_len

        pipe.reset_trail()
        snap_after_reset = pipe.step()
        assert len(snap_after_reset.trail) == 1
    finally:
        pipe.close()


# ============================================================================
# 4. Smith Chart Visualization Tests
# ============================================================================

def test_smith_chart_figure_generation():
    """Verify create_smith_chart generates valid Matplotlib figures."""
    import matplotlib.pyplot as plt

    trail_points = [complex(0.5, 0.2), complex(0.3, 0.1), complex(0.05, -0.02)]
    fig = create_smith_chart(
        gamma_in=complex(0.05, -0.02),
        gamma_load=complex(0.6, 0.4),
        trail=trail_points,
        vswr=1.12,
        figsize=(5.0, 5.0),
        dark_mode=False,
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
    plt.close(fig)

    # Test dark mode variant
    fig_dark = create_smith_chart(
        gamma_in=complex(0.8, -0.5),
        gamma_load=complex(0.8, -0.5),
        trail=[],
        vswr=9.5,
        dark_mode=True,
    )
    assert isinstance(fig_dark, plt.Figure)
    plt.close(fig_dark)


# ============================================================================
# 5. Robust Failure Handling Tests
# ============================================================================

def test_pipeline_missing_model_fallback():
    """Verify pipeline handles a missing model file gracefully without crashing."""
    pipe = PipelineController(
        source_type="Mock",
        mode="Auto (RL)",
        model_path="non_existent_model_dir/model_missing.pt",
    )
    try:
        assert pipe.agent_loaded is False
        snap = pipe.step()
        # Fallback grid search should still provide valid active voltages and physics
        assert isinstance(snap, PipelineSnapshot)
        assert 0.0 <= snap.V1_active <= 3.3
        assert 0.0 <= snap.V2_active <= 3.3
        assert snap.vswr >= 1.0
    finally:
        pipe.close()


def test_pipeline_disconnected_serial_fallback():
    """Verify selecting a nonexistent serial port handles errors gracefully."""
    pipe = PipelineController(source_type="Serial", serial_port="NON_EXISTENT_COM99")
    try:
        snap = pipe.step()
        assert isinstance(snap, PipelineSnapshot)
        # Should report disconnected without crashing
        assert snap.connection_status in ("Disconnected", "Error", "Fallback")
        assert snap.vswr >= 1.0
    finally:
        pipe.close()


# ============================================================================
# 6. Real-Time Performance Benchmark
# ============================================================================

def test_pipeline_loop_throughput():
    """Verify that the integration pipeline execution easily satisfies the 20 Hz loop."""
    pipe = PipelineController(source_type="Mock", mode="Manual (Pots)")
    try:
        iterations = 500
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = pipe.step()
        elapsed = time.perf_counter() - t0
        rate_hz = iterations / elapsed

        # Must comfortably exceed 20 Hz
        assert rate_hz > 50.0, f"Expected pipeline throughput > 50 Hz; got {rate_hz:.1f} Hz"
    finally:
        pipe.close()
