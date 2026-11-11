"""Unit tests for Module 3 (RF Physics Engine).

Verifies:
1. Varactor diode C(V) model, clamping, and boundary conditions (0V vs 3.3V).
2. Pi-matching network computation, API dictionary format, matched load tuning,
   and VSWR near 1.0.
3. Passive network physical invariant: reflection coefficient |Gamma| <= 1.0.
4. Smith chart helpers: impedance_to_gamma(50) == 0j, gamma_to_impedance roundtrips,
   and edge case behaviors.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is present in sys.path for direct test runs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physics.varactor import capacitance
from physics.pi_network import compute, grid_search_match, F, Z0, L
from physics.smith import (
    impedance_to_gamma,
    gamma_to_impedance,
    resistance_circle,
    reactance_arc,
    vswr_circle,
)


# ============================================================================
# 1. Varactor Diode Tests
# ============================================================================

def test_capacitance_boundary_conditions():
    """Verify varactor diode capacitance at boundary conditions (0V and 3.3V)."""
    # Boundary 1: At 0.0V bias, C(0) = C0 / ((1 + 0) ** gamma) = C0 = 100 pF
    c_zero = capacitance(0.0, C0=100e-12, Vj=0.7, gamma=0.5)
    assert np.isclose(c_zero, 100e-12, rtol=1e-6)
    assert isinstance(c_zero, float)

    # Boundary 2: At 3.3V maximum reverse bias:
    # C(3.3) = 100e-12 / ((1 + 3.3 / 0.7) ** 0.5) = 100e-12 / sqrt(5.7142857)
    # C(3.3) approx 41.833 pF
    c_max = capacitance(3.3, C0=100e-12, Vj=0.7, gamma=0.5)
    expected_c_max = 100e-12 / np.sqrt(1.0 + 3.3 / 0.7)
    assert np.isclose(c_max, expected_c_max, rtol=1e-6)
    assert np.isclose(c_max, 41.833001e-12, rtol=1e-5)
    assert isinstance(c_max, float)


def test_capacitance_voltage_clamping():
    """Ensure bias voltage is strictly clamped between 0.0V and 3.3V."""
    # Negative voltage should clamp to 0.0V
    c_neg = capacitance(-1.5)
    c_zero = capacitance(0.0)
    assert np.isclose(c_neg, c_zero, rtol=1e-9)

    # Voltage exceeding 3.3V should clamp to 3.3V
    c_over = capacitance(5.0)
    c_3v3 = capacitance(3.3)
    assert np.isclose(c_over, c_3v3, rtol=1e-9)


def test_capacitance_monotonic_decrease():
    """Verify capacitance decreases monotonically with increasing reverse bias."""
    voltages = np.linspace(0.0, 3.3, 20)
    caps = [capacitance(v) for v in voltages]
    assert all(caps[i] > caps[i + 1] for i in range(len(caps) - 1))


def test_capacitance_array_input():
    """Verify capacitance supports NumPy array inputs."""
    v_arr = np.array([0.0, 1.0, 2.0, 3.3])
    c_arr = capacitance(v_arr)
    assert isinstance(c_arr, np.ndarray)
    assert len(c_arr) == 4
    assert np.isclose(c_arr[0], 100e-12)


# ============================================================================
# 2. Pi-Network Simulation & Matched Load Tests
# ============================================================================

def test_compute_api_contract():
    """Verify compute returns the exact required dictionary keys and types."""
    res = compute(RL=50.0, XL=0.0, V1=1.5, V2=1.5)
    assert isinstance(res, dict)
    assert set(res.keys()) == {"Zin", "gamma", "gamma_abs", "vswr"}
    assert isinstance(res["Zin"], complex)
    assert isinstance(res["gamma"], complex)
    assert isinstance(res["gamma_abs"], float)
    assert isinstance(res["vswr"], float)


def test_compute_matched_load_low_gamma_and_vswr():
    """
    Test compute() with a matched load (RL=50, XL=0).
    
    Tuning V1 and V2 to specific values should yield a low gamma (|Gamma| < 0.05)
    and VSWR very close to 1.0 (VSWR < 1.1).
    """
    # Case A: Tuned symmetric Pi-network at 100 MHz with L = 70 nH
    # Symmetric Pi-network with Z0=50, RL=50 has an exact resonance match at V1=V2=1.755V
    res_70nh = compute(RL=50.0, XL=0.0, V1=1.755, V2=1.755, L=70e-9, f=100e6)
    assert res_70nh["gamma_abs"] < 0.01, f"Expected |Gamma| < 0.01, got {res_70nh['gamma_abs']}"
    assert 1.0 <= res_70nh["vswr"] < 1.02, f"Expected VSWR near 1.0, got {res_70nh['vswr']}"

    # Case B: Tuned symmetric Pi-network with default L = 150 nH at f = 50 MHz
    res_50mhz = compute(RL=50.0, XL=0.0, V1=2.762, V2=2.762, L=150e-9, f=50e6)
    assert res_50mhz["gamma_abs"] < 0.01, f"Expected |Gamma| < 0.01, got {res_50mhz['gamma_abs']}"
    assert 1.0 <= res_50mhz["vswr"] < 1.02, f"Expected VSWR near 1.0, got {res_50mhz['vswr']}"

    # Case C: Reactive load matching at 100 MHz with default L = 150 nH
    # For a detuned antenna RL=10 Ohm, XL=+50 Ohm, tuning V1=3.3V, V2=1.90V matches well
    res_detuned = compute(RL=10.0, XL=50.0, V1=3.3, V2=1.904, L=150e-9, f=100e6)
    assert res_detuned["gamma_abs"] < 0.06, f"Expected |Gamma| < 0.06, got {res_detuned['gamma_abs']}"
    assert 1.0 <= res_detuned["vswr"] < 1.15, f"Expected VSWR near 1.0, got {res_detuned['vswr']}"


# ============================================================================
# 3. Passive Network Rules (Physical Invariant: |Gamma| <= 1.0)
# ============================================================================

def test_compute_gamma_abs_never_exceeds_unity():
    """
    Test that compute() never returns gamma_abs > 1.0 for any passive load.

    In linear passive RF network theory, lossless or lossy passive components
    cannot create energy, so the reflection coefficient magnitude referenced
    to a positive real characteristic impedance (Z0 > 0) satisfies |Gamma| <= 1.
    """
    # Test across the entire project load space
    rl_values = [10.0, 25.0, 50.0, 75.0, 100.0, 150.0]
    xl_values = [-100.0, -50.0, 0.0, 50.0, 100.0]
    v_values = [0.0, 0.8, 1.65, 2.5, 3.3]

    for rl in rl_values:
        for xl in xl_values:
            for v1 in v_values:
                for v2 in v_values:
                    res = compute(RL=rl, XL=xl, V1=v1, V2=v2)
                    gamma_abs = res["gamma_abs"]
                    assert gamma_abs <= 1.0 + 1e-9, (
                        f"Passive network violation: |Gamma| = {gamma_abs} > 1.0 "
                        f"for RL={rl}, XL={xl}, V1={v1}, V2={v2}"
                    )
                    assert res["vswr"] >= 1.0, f"VSWR must be >= 1.0, got {res['vswr']}"


def test_compute_pure_reactive_boundary():
    """Verify that pure reactive load (RL=0) maintains |Gamma| == 1.0."""
    res = compute(RL=0.0, XL=50.0, V1=1.5, V2=1.5)
    assert np.isclose(res["gamma_abs"], 1.0, atol=1e-6)
    assert res["vswr"] == float("inf")


# ============================================================================
# 4. Smith Chart Helper Tests
# ============================================================================

def test_impedance_to_gamma_matched_50_returns_zero_j():
    """Test impedance_to_gamma(50) returns exact 0j."""
    gamma_int = impedance_to_gamma(50)
    assert gamma_int == 0j
    assert isinstance(gamma_int, complex)

    gamma_float = impedance_to_gamma(50.0)
    assert gamma_float == 0j

    gamma_complex = impedance_to_gamma(complex(50.0, 0.0))
    assert gamma_complex == 0j


def test_smith_edge_cases():
    """Test Smith transform edge cases: short circuit, open circuit, and matched."""
    # Short circuit: Z = 0 -> Gamma = -1.0 + 0j
    gamma_short = impedance_to_gamma(0.0)
    assert np.isclose(gamma_short.real, -1.0)
    assert np.isclose(gamma_short.imag, 0.0)

    # Gamma = -1 -> Z = 0
    z_short = gamma_to_impedance(-1.0 + 0j)
    assert np.isclose(abs(z_short), 0.0, atol=1e-9)

    # Gamma = 0 -> Z = 50 + 0j
    z_matched = gamma_to_impedance(0j)
    assert np.isclose(z_matched.real, 50.0)
    assert np.isclose(z_matched.imag, 0.0)

    # Open circuit: Gamma = 1.0 -> Z has infinite magnitude
    z_open = gamma_to_impedance(1.0 + 0j)
    assert np.isinf(z_open.real)


def test_smith_roundtrip_consistency():
    """Verify that converting Z -> Gamma -> Z preserves the original impedance."""
    test_impedances = [
        complex(25.0, 10.0),
        complex(50.0, -30.0),
        complex(120.0, 45.0),
        complex(10.0, 0.0),
        complex(75.0, -80.0),
    ]
    for z in test_impedances:
        gamma = impedance_to_gamma(z, Z0=50.0)
        z_recov = gamma_to_impedance(gamma, Z0=50.0)
        assert np.isclose(z.real, z_recov.real, rtol=1e-7)
        assert np.isclose(z.imag, z_recov.imag, rtol=1e-7)


def test_smith_geometry_generators():
    """Verify geometric circle generators return valid numerical coordinates."""
    # Resistance circle for r = 1.0 (center at (0.5, 0), radius 0.5)
    u_r, v_r = resistance_circle(1.0, num_points=100)
    assert len(u_r) == 100
    assert len(v_r) == 100
    assert np.all(np.isfinite(u_r))
    assert np.all(np.isfinite(v_r))
    # Check circle radius
    dist_from_center = np.sqrt((u_r - 0.5) ** 2 + v_r ** 2)
    assert np.allclose(dist_from_center, 0.5)

    # Reactance arc for x = 1.0
    u_x, v_x = reactance_arc(1.0, num_points=100)
    assert len(u_x) == 100
    assert len(v_x) == 100
    # Inside the unit circle
    assert np.all(u_x ** 2 + v_x ** 2 <= 1.0 + 1e-6)

    # VSWR circle for VSWR = 2.0 (radius = (2-1)/(2+1) = 1/3)
    u_v, v_v = vswr_circle(2.0, num_points=100)
    radii = np.sqrt(u_v ** 2 + v_v ** 2)
    assert np.allclose(radii, 1.0 / 3.0)


# ============================================================================
# 5. Grid Search Optimization Baseline Tests
# ============================================================================

def test_grid_search_match_baseline():
    """Verify grid search finds optimal match parameters for arbitrary loads."""
    # Test on an inductive antenna disturbance load
    match_res = grid_search_match(RL=10.0, XL=50.0, grid_size=34, L=150e-9, f=100e6)
    assert match_res["best_gamma_abs"] < 0.06
    assert 1.0 <= match_res["best_vswr"] < 1.15
    assert 0.0 <= match_res["best_V1"] <= 3.3
    assert 0.0 <= match_res["best_V2"] <= 3.3
    assert isinstance(match_res["Zin"], complex)
    assert isinstance(match_res["gamma"], complex)

