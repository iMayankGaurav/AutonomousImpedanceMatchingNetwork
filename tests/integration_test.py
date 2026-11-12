#!/usr/bin/env python3
"""M1–M4 Cross-Module Integration Test Suite.

Verifies correct communication and data flow across the full pipeline:
    M1 (STM32 Firmware) → M2 (UART/Input Processing) →
    M3 (RF Physics) → M4 (RL Agent)

before M5 dashboard development begins.

This script is independently executable and does not modify any existing
module implementation, model weights, physics parameters, or config.

Usage:
    python tests/integration_test.py                      # simulated tests only
    python tests/integration_test.py --live-uart COM3     # include live UART test
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Path bootstrapping — ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IO_DIR = PROJECT_ROOT / "io"
if str(IO_DIR) not in sys.path:
    sys.path.insert(0, str(IO_DIR))

# Augment stdlib 'io' with our project's io package
import io as stdlib_io

if hasattr(stdlib_io, "__path__"):
    if str(IO_DIR) not in stdlib_io.__path__:
        stdlib_io.__path__.append(str(IO_DIR))
else:
    stdlib_io.__path__ = [str(IO_DIR)]

# ---------------------------------------------------------------------------
# Module imports (actual implementation — no mocks, no invention)
# ---------------------------------------------------------------------------
import numpy as np

import config
from filters import (
    ADC_MAX,
    ADC_VREF,
    RL_MAX,
    RL_MIN,
    V_MAX,
    V_MIN,
    XL_MAX,
    XL_MIN,
    ChannelFilter,
    adc_to_physical,
    adc_to_rl,
    adc_to_v,
    adc_to_xl,
)
from serial_reader import parse_adc_line
from physics.pi_network import compute as physics_compute
from physics.varactor import capacitance
from physics.smith import impedance_to_gamma, gamma_to_impedance
from rl.agent import load_agent, InferenceAgent
from rl.env import ImpedanceMatchingEnv, ACTION_NAMES, _clamp_voltage


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------
class TestResult:
    """Container for a single test outcome."""

    def __init__(
        self,
        name: str,
        passed: bool,
        details: str = "",
        expected: str = "",
        actual: str = "",
        file_func: str = "",
        owner: str = "",
    ) -> None:
        self.name = name
        self.passed = passed
        self.details = details
        self.expected = expected
        self.actual = actual
        self.file_func = file_func
        self.owner = owner


class TestSuite:
    """Collects and reports test results for one interface."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.results: list[TestResult] = []

    def add(self, result: TestResult) -> None:
        self.results.append(result)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)


def _assert(
    condition: bool,
    msg: str,
    expected: str = "",
    actual: str = "",
    file_func: str = "",
    owner: str = "",
) -> TestResult:
    """Create a TestResult from an assertion."""
    if condition:
        return TestResult(name=msg, passed=True)
    return TestResult(
        name=msg,
        passed=False,
        details=msg,
        expected=expected,
        actual=actual,
        file_func=file_func,
        owner=owner,
    )


# ===========================================================================
# Test A — M1/M2 Data Interface
# ===========================================================================
def test_a_m1_m2_interface() -> TestSuite:
    """Feed representative UART packets into M2 parser and verify mapping."""
    suite = TestSuite("M1/M2 Interface")

    # --- A1: Valid packet parsing (boundary values) ---
    valid_cases = [
        ("0,0,0,0", (0, 0, 0, 0), "minimum ADC"),
        ("2048,2048,2048,2048", (2048, 2048, 2048, 2048), "midpoint ADC"),
        ("4095,4095,4095,4095", (4095, 4095, 4095, 4095), "maximum ADC"),
        ("100,200,300,400", (100, 200, 300, 400), "typical values"),
    ]
    for line, expected, label in valid_cases:
        result = parse_adc_line(line)
        suite.add(_assert(
            result == expected,
            f"A1 parse valid packet ({label}): '{line}'",
            expected=str(expected),
            actual=str(result),
            file_func="io/serial_reader.py :: parse_adc_line()",
            owner="M2",
        ))

    # --- A1b: Byte input and line-ending variants ---
    byte_cases = [
        (b"2048,1900,512,3000\r\n", (2048, 1900, 512, 3000), "bytes CRLF"),
        (b"2048,1900,512,3000\n", (2048, 1900, 512, 3000), "bytes LF"),
        ("2048,1900,512,3000\r\n", (2048, 1900, 512, 3000), "string CRLF"),
        (" 2048 , 1900 , 512 , 3000 ", (2048, 1900, 512, 3000), "spaces"),
    ]
    for line, expected, label in byte_cases:
        result = parse_adc_line(line)
        suite.add(_assert(
            result == expected,
            f"A1b parse with {label}",
            expected=str(expected),
            actual=str(result),
            file_func="io/serial_reader.py :: parse_adc_line()",
            owner="M2",
        ))

    # --- A2: ADC-to-physical-unit mapping at boundaries ---
    # Minimum: ADC=0 → RL=10.0, XL=62.0, V1=0.0, V2=0.0
    rl, xl, v1, v2 = adc_to_physical(0, 0, 0, 0)
    suite.add(_assert(
        math.isclose(rl, 10.0, rel_tol=1e-6) and math.isclose(xl, 62.0, rel_tol=1e-6)
        and math.isclose(v1, 0.0, abs_tol=1e-9) and math.isclose(v2, 0.0, abs_tol=1e-9),
        "A2 ADC→physical minimum (ADC=0)",
        expected="RL=10.0, XL=62.0, V1=0.0, V2=0.0",
        actual=f"RL={rl:.6f}, XL={xl:.6f}, V1={v1:.6f}, V2={v2:.6f}",
        file_func="io/filters.py :: adc_to_physical()",
        owner="M2",
    ))

    # Midpoint: ADC=2048
    rl, xl, v1, v2 = adc_to_physical(2048, 2048, 2048, 2048)
    rl_exp = 10.0 + (2048.0 / 4095.0) * 7.0
    xl_exp = 62.0 + (2048.0 / 4095.0) * 15.0
    v_exp = (2048.0 / 4095.0) * 3.3
    suite.add(_assert(
        math.isclose(rl, rl_exp, rel_tol=1e-6) and math.isclose(xl, xl_exp, rel_tol=1e-6)
        and math.isclose(v1, v_exp, rel_tol=1e-6) and math.isclose(v2, v_exp, rel_tol=1e-6),
        "A2 ADC→physical midpoint (ADC=2048)",
        expected=f"RL={rl_exp:.6f}, XL={xl_exp:.6f}, V={v_exp:.6f}",
        actual=f"RL={rl:.6f}, XL={xl:.6f}, V1={v1:.6f}, V2={v2:.6f}",
        file_func="io/filters.py :: adc_to_physical()",
        owner="M2",
    ))

    # Maximum: ADC=4095 → RL=17.0, XL=77.0, V1=3.3, V2=3.3
    rl, xl, v1, v2 = adc_to_physical(4095, 4095, 4095, 4095)
    suite.add(_assert(
        math.isclose(rl, 17.0, rel_tol=1e-6) and math.isclose(xl, 77.0, rel_tol=1e-6)
        and math.isclose(v1, 3.3, rel_tol=1e-6) and math.isclose(v2, 3.3, rel_tol=1e-6),
        "A2 ADC→physical maximum (ADC=4095)",
        expected="RL=17.0, XL=77.0, V1=3.3, V2=3.3",
        actual=f"RL={rl:.6f}, XL={xl:.6f}, V1={v1:.6f}, V2={v2:.6f}",
        file_func="io/filters.py :: adc_to_physical()",
        owner="M2",
    ))

    # --- A3: Malformed packet rejection ---
    malformed_cases = [
        ("", "empty string"),
        (None, "None input"),
        ("2048,1900,512", "3 channels"),
        ("2048,1900,512,3000,100", "5 channels"),
        ("abc,1900,512,3000", "non-numeric token"),
        ("2048.0,1900,512,3000", "decimal token"),
        ("-1,1900,512,3000", "negative value"),
        ("4096,1900,512,3000", "exceeds 4095"),
        ("2048,,512,3000", "empty field"),
        ("2048;1900;512;3000", "wrong delimiter"),
        (b"\xff\xfe\x00\x01", "corrupt bytes"),
    ]
    for line, label in malformed_cases:
        result = parse_adc_line(line)
        suite.add(_assert(
            result is None,
            f"A3 reject malformed ({label})",
            expected="None",
            actual=str(result),
            file_func="io/serial_reader.py :: parse_adc_line()",
            owner="M2",
        ))

    # --- A4: ChannelFilter processes converted values correctly ---
    cf = ChannelFilter(alpha=1.0, use_median=False)  # alpha=1.0 = pass-through
    raw_phys = adc_to_physical(2048, 2048, 2048, 2048)
    filtered = cf.update(raw_phys)
    suite.add(_assert(
        all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(filtered, raw_phys)),
        "A4 ChannelFilter pass-through (alpha=1.0)",
        expected=str(raw_phys),
        actual=str(filtered),
        file_func="io/filters.py :: ChannelFilter.update()",
        owner="M2",
    ))

    return suite


# ===========================================================================
# Test B — M2/M3 Interface
# ===========================================================================
def test_b_m2_m3_interface() -> TestSuite:
    """Pass mapped RL/XL/V1/V2 into M3 physics and verify outputs."""
    suite = TestSuite("M2/M3 Interface")

    # --- B1: Physics accepts demonstration-range inputs and produces finite results ---
    test_loads = [
        (10.0, 62.0, 1.65, 1.65, "min RL, min XL"),
        (17.0, 77.0, 1.65, 1.65, "max RL, max XL"),
        (13.5, 69.5, 1.65, 1.65, "midpoint load"),
        (13.5, 69.5, 0.0, 0.0, "midpoint load, V=0"),
        (13.5, 69.5, 3.3, 3.3, "midpoint load, V=3.3"),
        (10.0, 77.0, 0.5, 2.8, "cross corner 1"),
        (17.0, 62.0, 2.0, 0.3, "cross corner 2"),
    ]
    for rl, xl, v1, v2, label in test_loads:
        try:
            result = physics_compute(rl, xl, v1, v2)
            has_keys = all(
                k in result for k in ("Zin", "gamma", "gamma_abs", "vswr")
            )
            is_finite = (
                math.isfinite(result["gamma_abs"])
                and np.isfinite(result["gamma"].real)
                and np.isfinite(result["gamma"].imag)
                and np.isfinite(result["Zin"].real)
                and np.isfinite(result["Zin"].imag)
            )
            vswr_valid = result["vswr"] >= 1.0 or result["vswr"] == float("inf")
            suite.add(_assert(
                has_keys and is_finite and vswr_valid,
                f"B1 physics.compute({label})",
                expected="finite Zin, gamma, gamma_abs; vswr >= 1.0",
                actual=(
                    f"Zin={result['Zin']:.4f}, γ={result['gamma']:.4f}, "
                    f"|γ|={result['gamma_abs']:.6f}, VSWR={result['vswr']:.4f}"
                ),
                file_func="physics/pi_network.py :: compute()",
                owner="M3",
            ))
        except Exception as exc:
            suite.add(TestResult(
                name=f"B1 physics.compute({label})",
                passed=False,
                details=f"Exception: {exc}",
                file_func="physics/pi_network.py :: compute()",
                owner="M3",
            ))

    # --- B2: Independent Gamma verification ---
    # Verify: gamma = (Zin - Z0) / (Zin + Z0) matches the value returned by compute()
    rl, xl, v1, v2 = 13.5, 69.5, 1.65, 1.65
    result = physics_compute(rl, xl, v1, v2)
    z_in = result["Zin"]
    gamma_manual = (z_in - config.Z0) / (z_in + config.Z0)
    suite.add(_assert(
        math.isclose(abs(gamma_manual - result["gamma"]), 0.0, abs_tol=1e-12),
        "B2 independent Gamma formula verification",
        expected=f"gamma={gamma_manual}",
        actual=f"gamma={result['gamma']}",
        file_func="physics/pi_network.py :: compute()",
        owner="M3",
    ))

    # --- B3: Independent VSWR verification ---
    # VSWR = (1 + |Gamma|) / (1 - |Gamma|)
    gamma_abs = result["gamma_abs"]
    if gamma_abs < 1.0:
        vswr_manual = (1.0 + gamma_abs) / (1.0 - gamma_abs)
        suite.add(_assert(
            math.isclose(vswr_manual, result["vswr"], rel_tol=1e-6),
            "B3 independent VSWR formula verification",
            expected=f"VSWR={vswr_manual:.6f}",
            actual=f"VSWR={result['vswr']:.6f}",
            file_func="physics/pi_network.py :: compute()",
            owner="M3",
        ))
    else:
        suite.add(_assert(
            result["vswr"] == float("inf"),
            "B3 VSWR should be inf when |Gamma| >= 1",
            expected="inf",
            actual=str(result["vswr"]),
            file_func="physics/pi_network.py :: compute()",
            owner="M3",
        ))

    # --- B4: Smith chart impedance_to_gamma round-trip ---
    z_test = complex(30.0, 45.0)
    gamma_smith = impedance_to_gamma(z_test, Z0=config.Z0)
    z_back = gamma_to_impedance(gamma_smith, Z0=config.Z0)
    suite.add(_assert(
        math.isclose(z_back.real, z_test.real, rel_tol=1e-6)
        and math.isclose(z_back.imag, z_test.imag, rel_tol=1e-6),
        "B4 Smith chart Z→Γ→Z round-trip",
        expected=str(z_test),
        actual=str(z_back),
        file_func="physics/smith.py :: impedance_to_gamma(), gamma_to_impedance()",
        owner="M3",
    ))

    # --- B5: Varactor capacitance is positive and finite across voltage range ---
    v_test_points = [0.0, 0.5, 1.0, 1.65, 2.5, 3.3]
    for v in v_test_points:
        c = capacitance(v)
        suite.add(_assert(
            math.isfinite(c) and c > 0,
            f"B5 varactor C({v}V) > 0 and finite",
            expected="positive finite capacitance",
            actual=f"C={c:.4e} F",
            file_func="physics/varactor.py :: capacitance()",
            owner="M3",
        ))

    return suite


# ===========================================================================
# Test C — M3/M4 Interface
# ===========================================================================
def test_c_m3_m4_interface() -> TestSuite:
    """Load model, construct observation, call agent.act()."""
    suite = TestSuite("M3/M4 Interface")

    # --- C1: Load the trained model ---
    model_path = PROJECT_ROOT / "models" / "model.pt"
    agent: Optional[InferenceAgent] = None
    try:
        agent = load_agent(str(model_path), device="cpu")
        suite.add(_assert(
            agent is not None,
            "C1 load_agent() succeeds",
            expected="InferenceAgent instance",
            actual=str(type(agent).__name__),
            file_func="rl/agent.py :: load_agent()",
            owner="M4",
        ))
    except Exception as exc:
        suite.add(TestResult(
            name="C1 load_agent() succeeds",
            passed=False,
            details=f"Exception: {exc}",
            file_func="rl/agent.py :: load_agent()",
            owner="M4",
        ))
        return suite  # Cannot proceed without agent

    # --- C2: Verify model dimensions ---
    suite.add(_assert(
        agent.state_dim == 4,
        "C2 model state_dim == 4",
        expected="4",
        actual=str(agent.state_dim),
        file_func="rl/agent.py :: InferenceAgent",
        owner="M4",
    ))
    suite.add(_assert(
        agent.action_dim == 5,
        "C2 model action_dim == 5",
        expected="5",
        actual=str(agent.action_dim),
        file_func="rl/agent.py :: InferenceAgent",
        owner="M4",
    ))

    # --- C3: Construct observation from M3 output and feed to M4 ---
    # Use midpoint demonstration values
    rl, xl, v1, v2 = 13.5, 69.5, 1.65, 1.65
    phys_result = physics_compute(rl, xl, v1, v2)
    gamma = phys_result["gamma"]
    observation = np.array(
        [gamma.real, gamma.imag, v1, v2], dtype=np.float32
    )

    suite.add(_assert(
        observation.shape == (4,) and np.all(np.isfinite(observation)),
        "C3 observation shape (4,) and all finite",
        expected="shape=(4,), finite values",
        actual=f"shape={observation.shape}, values={observation}",
        file_func="physics/pi_network.py + rl/agent.py",
        owner="M3/M4",
    ))

    # --- C4: agent.act() returns valid action ---
    try:
        # Set deterministic seed for reproducibility
        np.random.seed(42)
        import torch
        torch.manual_seed(42)

        action = agent.act(observation)
        suite.add(_assert(
            isinstance(action, int) and 0 <= action <= 4,
            "C4 agent.act() returns int in {0..4}",
            expected="int in {0,1,2,3,4}",
            actual=f"action={action} ({ACTION_NAMES.get(action, '?')})",
            file_func="rl/agent.py :: InferenceAgent.act()",
            owner="M4",
        ))
    except Exception as exc:
        suite.add(TestResult(
            name="C4 agent.act() returns int in {0..4}",
            passed=False,
            details=f"Exception: {exc}",
            file_func="rl/agent.py :: InferenceAgent.act()",
            owner="M4",
        ))

    # --- C5: agent.q_values() returns 5 finite Q-values ---
    try:
        q = agent.q_values(observation)
        suite.add(_assert(
            q.shape == (5,) and np.all(np.isfinite(q)),
            "C5 q_values() returns shape (5,) with all finite",
            expected="shape=(5,), finite values",
            actual=f"shape={q.shape}, q={q}",
            file_func="rl/agent.py :: InferenceAgent.q_values()",
            owner="M4",
        ))
    except Exception as exc:
        suite.add(TestResult(
            name="C5 q_values() returns shape (5,) with all finite",
            passed=False,
            details=f"Exception: {exc}",
            file_func="rl/agent.py :: InferenceAgent.q_values()",
            owner="M4",
        ))

    # --- C6: Multiple observations from different loads ---
    test_observations = [
        (10.0, 62.0, 0.5, 2.0),
        (17.0, 77.0, 3.0, 0.5),
        (13.5, 69.5, 1.65, 1.65),
        (12.0, 70.0, 1.0, 1.0),
        (15.0, 65.0, 2.5, 2.5),
    ]
    for rl, xl, v1, v2 in test_observations:
        phys = physics_compute(rl, xl, v1, v2)
        obs = np.array(
            [phys["gamma"].real, phys["gamma"].imag, v1, v2], dtype=np.float32
        )
        try:
            a = agent.act(obs)
            suite.add(_assert(
                isinstance(a, int) and 0 <= a <= 4,
                f"C6 act(RL={rl}, XL={xl}, V1={v1}, V2={v2}) → valid action",
                expected="int in {0..4}",
                actual=f"{a} ({ACTION_NAMES.get(a, '?')})",
                file_func="rl/agent.py :: InferenceAgent.act()",
                owner="M4",
            ))
        except Exception as exc:
            suite.add(TestResult(
                name=f"C6 act(RL={rl}, XL={xl}, V1={v1}, V2={v2})",
                passed=False,
                details=f"Exception: {exc}",
                file_func="rl/agent.py :: InferenceAgent.act()",
                owner="M4",
            ))

    return suite


# ===========================================================================
# Test D — Action Execution
# ===========================================================================
def test_d_action_execution() -> TestSuite:
    """Apply actions through the environment and verify voltage behavior."""
    suite = TestSuite("Action Execution")

    # --- D1: Create environment with a fixed load and seed ---
    np.random.seed(42)
    env = ImpedanceMatchingEnv(
        load_mode="full",
        max_steps=120,
        random_initial_v=False,
        seed=42,
    )

    # Reset with a known demonstration-range load
    obs, info = env.reset(options={"RL": 13.5, "XL": 69.5})
    v1_before = env.V1
    v2_before = env.V2

    suite.add(_assert(
        obs.shape == (4,) and np.all(np.isfinite(obs)),
        "D1 env.reset() produces valid observation",
        expected="shape=(4,), finite",
        actual=f"shape={obs.shape}, obs={obs}",
        file_func="rl/env.py :: ImpedanceMatchingEnv.reset()",
        owner="M4",
    ))

    # --- D2: Each action changes voltages by exactly V_STEP or holds ---
    v_step = config.V_STEP
    action_tests = [
        (0, "V1+", lambda v1, v2, v1b, v2b: (
            math.isclose(v1, min(v1b + v_step, config.V_MAX), abs_tol=1e-4)
            and math.isclose(v2, v2b, abs_tol=1e-9)
        )),
        (1, "V1-", lambda v1, v2, v1b, v2b: (
            math.isclose(v1, max(v1b - v_step, config.V_MIN), abs_tol=1e-4)
            and math.isclose(v2, v2b, abs_tol=1e-9)
        )),
        (2, "V2+", lambda v1, v2, v1b, v2b: (
            math.isclose(v1, v1b, abs_tol=1e-9)
            and math.isclose(v2, min(v2b + v_step, config.V_MAX), abs_tol=1e-4)
        )),
        (3, "V2-", lambda v1, v2, v1b, v2b: (
            math.isclose(v1, v1b, abs_tol=1e-9)
            and math.isclose(v2, max(v2b - v_step, config.V_MIN), abs_tol=1e-4)
        )),
        (4, "Hold", lambda v1, v2, v1b, v2b: (
            math.isclose(v1, v1b, abs_tol=1e-9)
            and math.isclose(v2, v2b, abs_tol=1e-9)
        )),
    ]

    for action_id, action_name, check_fn in action_tests:
        # Reset environment before each action test
        env.reset(options={"RL": 13.5, "XL": 69.5, "V1": 1.65, "V2": 1.65})
        v1_before = env.V1
        v2_before = env.V2

        obs, reward, term, trunc, info = env.step(action_id)
        v1_after = env.V1
        v2_after = env.V2

        suite.add(_assert(
            check_fn(v1_after, v2_after, v1_before, v2_before),
            f"D2 action {action_id} ({action_name}) voltage change correct",
            expected=f"V1: {v1_before} → step by {action_name}, V2 unchanged" if action_id < 2
                else f"V1 unchanged, V2: {v2_before} → step by {action_name}" if action_id < 4
                else "V1 and V2 unchanged",
            actual=f"V1: {v1_before:.4f}→{v1_after:.4f}, V2: {v2_before:.4f}→{v2_after:.4f}",
            file_func="rl/env.py :: ImpedanceMatchingEnv.step()",
            owner="M4",
        ))

    # --- D3: Voltage boundary clamping ---
    # Test V1 at upper boundary (3.3V + step should stay at 3.3V)
    env.reset(options={"RL": 13.5, "XL": 69.5, "V1": 3.3, "V2": 0.0})
    obs, _, _, _, _ = env.step(0)  # V1+ at max
    suite.add(_assert(
        math.isclose(env.V1, config.V_MAX, abs_tol=1e-4),
        "D3 V1 clamped at V_MAX=3.3V (V1+ at boundary)",
        expected=f"{config.V_MAX}",
        actual=f"{env.V1:.4f}",
        file_func="rl/env.py :: _clamp_voltage()",
        owner="M4",
    ))

    # Test V2 at lower boundary (0.0V - step should stay at 0.0V)
    env.reset(options={"RL": 13.5, "XL": 69.5, "V1": 1.65, "V2": 0.0})
    obs, _, _, _, _ = env.step(3)  # V2- at min
    suite.add(_assert(
        math.isclose(env.V2, config.V_MIN, abs_tol=1e-4),
        "D3 V2 clamped at V_MIN=0.0V (V2- at boundary)",
        expected=f"{config.V_MIN}",
        actual=f"{env.V2:.4f}",
        file_func="rl/env.py :: _clamp_voltage()",
        owner="M4",
    ))

    # --- D4: Voltages always stay within [0, 3.3] after multiple steps ---
    env.reset(options={"RL": 13.5, "XL": 69.5, "V1": 1.65, "V2": 1.65})
    all_in_range = True
    for _ in range(200):
        action = np.random.randint(0, 5)
        obs, reward, term, trunc, info = env.step(action)
        if not (config.V_MIN <= env.V1 <= config.V_MAX
                and config.V_MIN <= env.V2 <= config.V_MAX):
            all_in_range = False
            break
        if term or trunc:
            env.reset(options={"RL": 13.5, "XL": 69.5})

    suite.add(_assert(
        all_in_range,
        "D4 V1 and V2 stay within [0.0, 3.3]V over 200 random steps",
        expected="all V1,V2 in [0.0, 3.3]",
        actual=f"V1={env.V1:.4f}, V2={env.V2:.4f}",
        file_func="rl/env.py :: ImpedanceMatchingEnv.step()",
        owner="M4",
    ))

    # --- D5: Reward and observation are finite ---
    env.reset(options={"RL": 13.5, "XL": 69.5, "V1": 1.65, "V2": 1.65})
    obs, reward, term, trunc, info = env.step(0)
    suite.add(_assert(
        np.all(np.isfinite(obs)) and math.isfinite(reward),
        "D5 observation and reward are finite after step",
        expected="finite obs and reward",
        actual=f"obs={obs}, reward={reward:.4f}",
        file_func="rl/env.py :: ImpedanceMatchingEnv.step()",
        owner="M4",
    ))

    return suite


# ===========================================================================
# Test E — End-to-End Integration
# ===========================================================================
def test_e_end_to_end() -> TestSuite:
    """Full pipeline: simulated ADC → M2 → M3 → M4 → action → recalculate."""
    suite = TestSuite("End-to-End Pipeline")

    # Set deterministic seeds
    np.random.seed(42)
    import torch
    torch.manual_seed(42)

    # --- E1: Load agent ---
    model_path = PROJECT_ROOT / "models" / "model.pt"
    try:
        agent = load_agent(str(model_path), device="cpu")
    except Exception as exc:
        suite.add(TestResult(
            name="E1 load agent for end-to-end",
            passed=False,
            details=f"Cannot load agent: {exc}",
            file_func="rl/agent.py :: load_agent()",
            owner="M4",
        ))
        return suite

    suite.add(TestResult(name="E1 agent loaded for end-to-end", passed=True))

    # --- E2: Simulate ADC inputs representing demo-range values ---
    # RL=13.5 Ω → ADC = (13.5-10)/7 * 4095 = 2047.5 ≈ 2048
    # XL=69.5 Ω → ADC = (69.5-62)/15 * 4095 = 2047.5 ≈ 2048
    # V1=1.65V → ADC = (1.65/3.3) * 4095 = 2047.5 ≈ 2048
    # V2=1.65V → ADC = (1.65/3.3) * 4095 = 2047.5 ≈ 2048
    adc_line = "2048,2048,2048,2048"

    # Step 1: M1 → M2: Parse UART line
    parsed = parse_adc_line(adc_line)
    suite.add(_assert(
        parsed is not None,
        "E2 Step1: M1→M2 parse UART packet",
        expected="tuple of 4 ints",
        actual=str(parsed),
        file_func="io/serial_reader.py :: parse_adc_line()",
        owner="M1/M2",
    ))

    # Step 2: M2: Convert ADC to physical
    rl, xl, v1, v2 = adc_to_physical(*parsed)
    suite.add(_assert(
        RL_MIN <= rl <= RL_MAX and XL_MIN <= xl <= XL_MAX
        and V_MIN <= v1 <= V_MAX and V_MIN <= v2 <= V_MAX,
        "E2 Step2: M2 ADC→physical values in range",
        expected=f"RL∈[{RL_MIN},{RL_MAX}], XL∈[{XL_MIN},{XL_MAX}], V∈[{V_MIN},{V_MAX}]",
        actual=f"RL={rl:.4f}, XL={xl:.4f}, V1={v1:.4f}, V2={v2:.4f}",
        file_func="io/filters.py :: adc_to_physical()",
        owner="M2",
    ))

    # Step 3: M3: Compute RF physics
    phys = physics_compute(rl, xl, v1, v2)
    gamma = phys["gamma"]
    gamma_abs = phys["gamma_abs"]
    vswr = phys["vswr"]
    suite.add(_assert(
        math.isfinite(gamma_abs) and gamma_abs >= 0,
        "E2 Step3: M3 physics produces valid reflection coefficient",
        expected="finite |Γ| >= 0",
        actual=f"|Γ|={gamma_abs:.6f}, Γ={gamma}, VSWR={vswr:.4f}",
        file_func="physics/pi_network.py :: compute()",
        owner="M3",
    ))

    # Step 4: M4: Construct observation and get action
    observation = np.array(
        [gamma.real, gamma.imag, v1, v2], dtype=np.float32
    )
    action = agent.act(observation)
    suite.add(_assert(
        isinstance(action, int) and 0 <= action <= 4,
        "E2 Step4: M4 agent selects valid action",
        expected="int in {0..4}",
        actual=f"action={action} ({ACTION_NAMES.get(action, '?')})",
        file_func="rl/agent.py :: InferenceAgent.act()",
        owner="M4",
    ))

    # Step 5: Execute action (voltage adjustment)
    v1_new, v2_new = v1, v2
    if action == 0:
        v1_new = _clamp_voltage(v1 + config.V_STEP)
    elif action == 1:
        v1_new = _clamp_voltage(v1 - config.V_STEP)
    elif action == 2:
        v2_new = _clamp_voltage(v2 + config.V_STEP)
    elif action == 3:
        v2_new = _clamp_voltage(v2 - config.V_STEP)
    # action == 4: Hold

    suite.add(_assert(
        config.V_MIN <= v1_new <= config.V_MAX and config.V_MIN <= v2_new <= config.V_MAX,
        "E2 Step5: post-action voltages within [0.0, 3.3]V",
        expected=f"V∈[{config.V_MIN},{config.V_MAX}]",
        actual=f"V1={v1_new:.4f}, V2={v2_new:.4f}",
        file_func="rl/env.py :: _clamp_voltage()",
        owner="M4",
    ))

    # Step 6: Recalculate RF state after action
    phys_new = physics_compute(rl, xl, v1_new, v2_new)
    gamma_new = phys_new["gamma"]
    gamma_abs_new = phys_new["gamma_abs"]
    vswr_new = phys_new["vswr"]
    suite.add(_assert(
        math.isfinite(gamma_abs_new),
        "E2 Step6: M3 recalculation produces finite results",
        expected="finite |Γ|",
        actual=f"|Γ|={gamma_abs_new:.6f}, VSWR={vswr_new:.4f}",
        file_func="physics/pi_network.py :: compute()",
        owner="M3",
    ))

    # --- E3: Print detailed initial and final state ---
    print("\n  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │              End-to-End Pipeline Trace                         │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │  UART input:   \"{adc_line}\"                                   │")
    print(f"  │  ADC parsed:   {parsed}                              │")
    print(f"  │                                                               │")
    print(f"  │  INITIAL STATE                                                │")
    print(f"  │    RL  = {rl:8.4f} Ω    XL  = {xl:8.4f} Ω                     │")
    print(f"  │    V1  = {v1:8.4f} V    V2  = {v2:8.4f} V                     │")
    print(f"  │    Γ   = {gamma.real:+.6f} {gamma.imag:+.6f}j                           │")
    print(f"  │    |Γ| = {gamma_abs:.6f}    VSWR = {vswr:8.4f}                   │")
    print(f"  │                                                               │")
    print(f"  │  ACTION: {action} ({ACTION_NAMES.get(action, '?')})                                       │")
    print(f"  │                                                               │")
    print(f"  │  FINAL STATE                                                  │")
    print(f"  │    RL  = {rl:8.4f} Ω    XL  = {xl:8.4f} Ω                     │")
    print(f"  │    V1  = {v1_new:8.4f} V    V2  = {v2_new:8.4f} V                     │")
    print(f"  │    Γ   = {gamma_new.real:+.6f} {gamma_new.imag:+.6f}j                           │")
    print(f"  │    |Γ| = {gamma_abs_new:.6f}    VSWR = {vswr_new:8.4f}                   │")
    print("  └─────────────────────────────────────────────────────────────────┘")

    return suite


# ===========================================================================
# Phase 3 — Demonstration-Range Testing
# ===========================================================================
def test_phase3_demo_range() -> TestSuite:
    """Test minimum, midpoint, maximum and intermediate demo-range loads."""
    suite = TestSuite("Demonstration Range")

    np.random.seed(42)
    import torch
    torch.manual_seed(42)

    model_path = PROJECT_ROOT / "models" / "model.pt"
    try:
        agent = load_agent(str(model_path), device="cpu")
    except Exception as exc:
        suite.add(TestResult(
            name="Demo range: load agent",
            passed=False,
            details=f"Cannot load agent: {exc}",
            file_func="rl/agent.py :: load_agent()",
            owner="M4",
        ))
        return suite

    # Demo range: RL = 10–17 Ohms, XL = 62–77 Ohms
    demo_loads = [
        (10.0, 62.0, "min RL, min XL"),
        (10.0, 77.0, "min RL, max XL"),
        (17.0, 62.0, "max RL, min XL"),
        (17.0, 77.0, "max RL, max XL"),
        (13.5, 69.5, "midpoint"),
        (11.0, 65.0, "low RL, low-mid XL"),
        (15.0, 73.0, "high-mid RL, high-mid XL"),
        (12.0, 70.0, "intermediate 1"),
        (14.5, 67.0, "intermediate 2"),
        (16.0, 75.0, "intermediate 3"),
    ]

    print("\n  ┌────────────────────────────────────────────────────────────────────────────┐")
    print("  │                   Demonstration Range Results                              │")
    print("  ├──────┬───────┬───────┬───────┬────────┬────────┬──────────┬────────────────┤")
    print("  │  RL  │  XL   │  V1   │  V2   │  |Γ|   │  VSWR  │ Action   │ Label          │")
    print("  ├──────┼───────┼───────┼───────┼────────┼────────┼──────────┼────────────────┤")

    for rl, xl, label in demo_loads:
        v1, v2 = 1.65, 1.65
        try:
            # Compute physics
            phys = physics_compute(rl, xl, v1, v2)
            gamma = phys["gamma"]
            gamma_abs = phys["gamma_abs"]
            vswr_val = phys["vswr"]

            # Get agent action
            obs = np.array([gamma.real, gamma.imag, v1, v2], dtype=np.float32)
            action = agent.act(obs)
            action_name = ACTION_NAMES.get(action, "?")

            # Verify finite and valid
            is_valid = (
                math.isfinite(gamma_abs)
                and (vswr_val >= 1.0 or vswr_val == float("inf"))
                and isinstance(action, int)
                and 0 <= action <= 4
            )

            suite.add(_assert(
                is_valid,
                f"Demo range RL={rl}, XL={xl} ({label})",
                expected="finite gamma, valid VSWR, action in {0..4}",
                actual=f"|Γ|={gamma_abs:.6f}, VSWR={vswr_val:.4f}, action={action}",
                file_func="physics/pi_network.py + rl/agent.py",
                owner="M3/M4",
            ))

            vswr_str = f"{vswr_val:6.2f}" if vswr_val < 1e6 else "  inf "
            print(
                f"  │{rl:5.1f} │{xl:6.1f} │{v1:6.2f} │{v2:6.2f} │"
                f"{gamma_abs:7.4f} │{vswr_str} │ {action}({action_name:4s})  │ {label:<14s} │"
            )

        except Exception as exc:
            suite.add(TestResult(
                name=f"Demo range RL={rl}, XL={xl} ({label})",
                passed=False,
                details=f"Exception: {exc}",
                file_func="physics/pi_network.py + rl/agent.py",
                owner="M3/M4",
            ))
            print(f"  │{rl:5.1f} │{xl:6.1f} │  ERR  │  ERR  │  ERR   │  ERR   │  ERROR   │ {label:<14s} │")

    print("  └──────┴───────┴───────┴───────┴────────┴────────┴──────────┴────────────────┘")

    return suite


# ===========================================================================
# Phase 4 — Optional Live UART Test
# ===========================================================================
def test_live_uart(port: str) -> TestSuite:
    """Read actual STM32 UART data from a user-supplied serial port."""
    suite = TestSuite("Live UART")

    try:
        import serial
    except ImportError:
        suite.add(TestResult(
            name="Live UART: pyserial import",
            passed=False,
            details="pyserial is not installed. Install with: pip install pyserial",
            file_func="(dependency)",
            owner="M2",
        ))
        return suite

    print(f"\n  Attempting live UART read from port: {port}")
    print(f"  Baud rate: {config.BAUD} (8-N-1)")
    print(f"  Timeout: 5 seconds")

    try:
        ser = serial.Serial(port=port, baudrate=config.BAUD, timeout=5.0)
    except Exception as exc:
        suite.add(TestResult(
            name=f"Live UART: open port {port}",
            passed=False,
            details=f"Failed to open serial port: {exc}",
            file_func="io/serial_reader.py",
            owner="M1/M2",
        ))
        return suite

    suite.add(TestResult(name=f"Live UART: port {port} opened", passed=True))

    # Try to read up to 10 lines with a 5-second overall timeout
    valid_frames = 0
    rejected_frames = 0
    start = time.time()

    try:
        for i in range(10):
            if time.time() - start > 5.0:
                break
            raw_line = ser.readline()
            if not raw_line:
                continue  # timeout

            parsed = parse_adc_line(raw_line)
            if parsed is not None:
                valid_frames += 1
                rl, xl, v1, v2 = adc_to_physical(*parsed)
                print(
                    f"    Frame {i+1}: ADC={parsed} → "
                    f"RL={rl:.2f}Ω, XL={xl:.2f}Ω, V1={v1:.3f}V, V2={v2:.3f}V"
                )
            else:
                rejected_frames += 1
                print(f"    Frame {i+1}: rejected: {raw_line!r}")
    except Exception as exc:
        suite.add(TestResult(
            name="Live UART: read frames",
            passed=False,
            details=f"Read error: {exc}",
            file_func="io/serial_reader.py",
            owner="M1/M2",
        ))
    finally:
        ser.close()

    suite.add(_assert(
        valid_frames > 0,
        f"Live UART: received {valid_frames} valid frames ({rejected_frames} rejected)",
        expected="at least 1 valid frame",
        actual=f"{valid_frames} valid, {rejected_frames} rejected",
        file_func="io/serial_reader.py :: parse_adc_line()",
        owner="M1/M2",
    ))

    return suite


# ===========================================================================
# Report Generation
# ===========================================================================
def print_report(suites: list[TestSuite], live_uart_status: str) -> bool:
    """Print the final test report. Returns True if all tests passed."""
    print("\n")
    print("=" * 78)
    print("         M1–M4 INTEGRATION TEST REPORT")
    print("=" * 78)

    all_passed = True
    for s in suites:
        status = "PASS" if s.passed else "FAIL"
        if not s.passed:
            all_passed = False
        icon = "✓" if s.passed else "✗"
        print(f"\n  {icon} {s.name + ':':.<50s} {status}  ({s.pass_count}/{len(s.results)} passed)")

        # Print failures in detail
        for r in s.results:
            if not r.passed:
                print(f"    ├── FAIL: {r.name}")
                if r.expected:
                    print(f"    │   Expected: {r.expected}")
                if r.actual:
                    print(f"    │   Actual:   {r.actual}")
                if r.details:
                    print(f"    │   Details:  {r.details}")
                if r.file_func:
                    print(f"    │   File:     {r.file_func}")
                if r.owner:
                    print(f"    │   Owner:    {r.owner}")

    # Live UART line
    print(f"\n  {'✓' if live_uart_status == 'PASS' else '–' if live_uart_status == 'NOT RUN' else '✗'}"
          f" {'Live UART:':.<50s} {live_uart_status}")

    # Summary
    total_tests = sum(len(s.results) for s in suites)
    total_passed = sum(s.pass_count for s in suites)
    total_failed = sum(s.fail_count for s in suites)

    print("\n" + "─" * 78)
    print(f"  TOTAL: {total_passed} passed, {total_failed} failed, {total_tests} tests")
    print(f"  OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 78)

    return all_passed


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="M1–M4 Cross-Module Integration Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--live-uart",
        type=str,
        default=None,
        metavar="PORT",
        help="Serial port for live UART test (e.g. COM3 or /dev/ttyACM0). "
             "User must explicitly supply this — no auto-detection.",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("  M1–M4 CROSS-MODULE INTEGRATION TEST")
    print("  Autonomous RL Impedance Matching Network")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    suites: list[TestSuite] = []

    # Test A — M1/M2 Interface
    print("\n▶ Test A: M1/M2 Data Interface")
    print("─" * 50)
    try:
        suite_a = test_a_m1_m2_interface()
        suites.append(suite_a)
        print(f"  Result: {suite_a.pass_count}/{len(suite_a.results)} passed")
    except Exception as exc:
        print(f"  CRITICAL ERROR: {exc}")
        traceback.print_exc()
        suites.append(TestSuite("M1/M2 Interface"))
        suites[-1].add(TestResult("M1/M2 suite execution", False, details=str(exc)))

    # Test B — M2/M3 Interface
    print("\n▶ Test B: M2/M3 Interface")
    print("─" * 50)
    try:
        suite_b = test_b_m2_m3_interface()
        suites.append(suite_b)
        print(f"  Result: {suite_b.pass_count}/{len(suite_b.results)} passed")
    except Exception as exc:
        print(f"  CRITICAL ERROR: {exc}")
        traceback.print_exc()
        suites.append(TestSuite("M2/M3 Interface"))
        suites[-1].add(TestResult("M2/M3 suite execution", False, details=str(exc)))

    # Test C — M3/M4 Interface
    print("\n▶ Test C: M3/M4 Interface")
    print("─" * 50)
    try:
        suite_c = test_c_m3_m4_interface()
        suites.append(suite_c)
        print(f"  Result: {suite_c.pass_count}/{len(suite_c.results)} passed")
    except Exception as exc:
        print(f"  CRITICAL ERROR: {exc}")
        traceback.print_exc()
        suites.append(TestSuite("M3/M4 Interface"))
        suites[-1].add(TestResult("M3/M4 suite execution", False, details=str(exc)))

    # Test D — Action Execution
    print("\n▶ Test D: Action Execution")
    print("─" * 50)
    try:
        suite_d = test_d_action_execution()
        suites.append(suite_d)
        print(f"  Result: {suite_d.pass_count}/{len(suite_d.results)} passed")
    except Exception as exc:
        print(f"  CRITICAL ERROR: {exc}")
        traceback.print_exc()
        suites.append(TestSuite("Action Execution"))
        suites[-1].add(TestResult("Action execution suite", False, details=str(exc)))

    # Test E — End-to-End Pipeline
    print("\n▶ Test E: End-to-End Integration")
    print("─" * 50)
    try:
        suite_e = test_e_end_to_end()
        suites.append(suite_e)
        print(f"  Result: {suite_e.pass_count}/{len(suite_e.results)} passed")
    except Exception as exc:
        print(f"  CRITICAL ERROR: {exc}")
        traceback.print_exc()
        suites.append(TestSuite("End-to-End Pipeline"))
        suites[-1].add(TestResult("E2E suite execution", False, details=str(exc)))

    # Phase 3 — Demonstration Range
    print("\n▶ Phase 3: Demonstration Range Testing")
    print("─" * 50)
    try:
        suite_demo = test_phase3_demo_range()
        suites.append(suite_demo)
        print(f"  Result: {suite_demo.pass_count}/{len(suite_demo.results)} passed")
    except Exception as exc:
        print(f"  CRITICAL ERROR: {exc}")
        traceback.print_exc()
        suites.append(TestSuite("Demonstration Range"))
        suites[-1].add(TestResult("Demo range suite execution", False, details=str(exc)))

    # Phase 4 — Optional Live UART
    live_uart_status = "NOT RUN"
    if args.live_uart:
        print(f"\n▶ Phase 4: Live UART Test (port: {args.live_uart})")
        print("─" * 50)
        try:
            suite_live = test_live_uart(args.live_uart)
            suites.append(suite_live)
            live_uart_status = "PASS" if suite_live.passed else "FAIL"
            print(f"  Result: {suite_live.pass_count}/{len(suite_live.results)} passed")
        except Exception as exc:
            print(f"  CRITICAL ERROR: {exc}")
            traceback.print_exc()
            live_uart_status = "FAIL"
            suites.append(TestSuite("Live UART"))
            suites[-1].add(TestResult("Live UART suite execution", False, details=str(exc)))

    # Final Report
    all_passed = print_report(suites, live_uart_status)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
