"""Unit tests for Module 2: Serial Link & Signal Processing.

Verifies:
1. Correct parsing of valid ADC lines.
2. Parsing with spaces.
3. Parsing with Windows line endings (CRLF).
4. Rejection of empty strings.
5. Rejection of malformed CSV.
6. Rejection of missing channels.
7. Rejection of extra channels.
8. Rejection of non-numeric values.
9. Rejection of decimal values.
10. Rejection of negative ADC counts.
11. Rejection of ADC values greater than 4095.
12. ADC-to-physical-unit mapping.
13. EMA initialization.
14. EMA filtering behavior.
15. EMA reset behavior.
16. Mock source returns the correct dictionary structure.
17. Mock source values remain within valid ranges.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

# Ensure project root and io folder are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IO_DIR = PROJECT_ROOT / "io"
if str(IO_DIR) not in sys.path:
    sys.path.insert(0, str(IO_DIR))

# Ensure python's stdlib 'io' is augmented with our project's 'io' package directory
import io as stdlib_io

if hasattr(stdlib_io, "__path__"):
    if str(IO_DIR) not in stdlib_io.__path__:
        stdlib_io.__path__.append(str(IO_DIR))
else:
    stdlib_io.__path__ = [str(IO_DIR)]

# Imports from M2
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
    EMAFilter,
    MedianFilter,
    adc_to_physical,
    adc_to_rl,
    adc_to_v,
    adc_to_xl,
    clamp_adc,
)
from mock_source import MockSource
from serial_reader import (
    BAUD_RATE,
    EXPECTED_CHANNELS,
    SerialReader,
    get_latest,
    parse_adc_line,
    set_default_source,
)


# =============================================================================
# 1. Correct Parsing of Valid ADC Lines
# =============================================================================
def test_valid_adc_line_parsing() -> None:
    """Verify parsing of valid 4-channel CSV strings and byte streams."""
    result = parse_adc_line("2048,1900,512,3000")
    assert result == (2048, 1900, 512, 3000)
    assert isinstance(result, tuple)
    assert len(result) == 4
    assert all(isinstance(v, int) for v in result)

    # Boundary lower count
    assert parse_adc_line("0,0,0,0") == (0, 0, 0, 0)

    # Boundary upper count
    assert parse_adc_line("4095,4095,4095,4095") == (4095, 4095, 4095, 4095)

    # Byte input
    assert parse_adc_line(b"100,200,300,400") == (100, 200, 300, 400)


# =============================================================================
# 2. Parsing with Spaces
# =============================================================================
def test_parsing_with_spaces() -> None:
    """Verify whitespace around commas and tokens is handled correctly."""
    assert parse_adc_line(" 2048 , 1900 , 512 , 3000 ") == (2048, 1900, 512, 3000)
    assert parse_adc_line("   2048,  1900,  512,  3000   ") == (2048, 1900, 512, 3000)
    assert parse_adc_line("\t2048\t,\t1900\t,\t512\t,\t3000\t") == (2048, 1900, 512, 3000)


# =============================================================================
# 3. Parsing with Windows Line Endings
# =============================================================================
def test_parsing_with_windows_line_endings() -> None:
    """Verify Windows (CRLF), Unix (LF), and legacy Mac (CR) line endings."""
    assert parse_adc_line("2048,1900,512,3000\r\n") == (2048, 1900, 512, 3000)
    assert parse_adc_line("2048,1900,512,3000\n") == (2048, 1900, 512, 3000)
    assert parse_adc_line("2048,1900,512,3000\r") == (2048, 1900, 512, 3000)
    assert parse_adc_line(b"2048,1900,512,3000\r\n") == (2048, 1900, 512, 3000)


# =============================================================================
# 4. Rejection of Empty Strings
# =============================================================================
def test_rejection_of_empty_strings() -> None:
    """Verify empty lines, whitespace-only lines, and None return None."""
    assert parse_adc_line("") is None
    assert parse_adc_line("   ") is None
    assert parse_adc_line("\r\n") is None
    assert parse_adc_line("\n") is None
    assert parse_adc_line(None) is None
    assert parse_adc_line(b"") is None
    assert parse_adc_line(b"\r\n") is None


# =============================================================================
# 5. Rejection of Malformed CSV
# =============================================================================
def test_rejection_of_malformed_csv() -> None:
    """Verify malformed CSV structures return None without crashing."""
    assert parse_adc_line("2048,,512,3000") is None
    assert parse_adc_line(",,,") is None
    assert parse_adc_line(",,,,") is None
    assert parse_adc_line("2048,1900,512,3000,") is None
    assert parse_adc_line(",2048,1900,512,3000") is None
    assert parse_adc_line("2048 1900 512 3000") is None  # missing commas
    assert parse_adc_line("2048;1900;512;3000") is None  # wrong delimiter


# =============================================================================
# 6. Rejection of Missing Channels
# =============================================================================
def test_rejection_of_missing_channels() -> None:
    """Verify lines with fewer than 4 channels are safely rejected."""
    assert parse_adc_line("2048,1900,512") is None  # 3 channels
    assert parse_adc_line("2048,1900") is None      # 2 channels
    assert parse_adc_line("2048") is None           # 1 channel


# =============================================================================
# 7. Rejection of Extra Channels
# =============================================================================
def test_rejection_of_extra_channels() -> None:
    """Verify lines with more than 4 channels are safely rejected."""
    assert parse_adc_line("2048,1900,512,3000,100") is None   # 5 channels
    assert parse_adc_line("1,2,3,4,5,6") is None              # 6 channels


# =============================================================================
# 8. Rejection of Non-Numeric Values
# =============================================================================
def test_rejection_of_non_numeric_values() -> None:
    """Verify alphabetic and symbolic tokens return None."""
    assert parse_adc_line("2048,abc,512,3000") is None
    assert parse_adc_line("word,1900,512,3000") is None
    assert parse_adc_line("2048,1900,512,NaN") is None
    assert parse_adc_line("2048,1900,512,inf") is None
    assert parse_adc_line("2048,0x800,512,3000") is None  # hex not allowed


# =============================================================================
# 9. Rejection of Decimal Values
# =============================================================================
def test_rejection_of_decimal_values() -> None:
    """Verify decimal floats and scientific notation are rejected."""
    assert parse_adc_line("2048.0,1900,512,3000") is None
    assert parse_adc_line("2048,1900.5,512,3000") is None
    assert parse_adc_line("2048,1900,512.2,3000") is None
    assert parse_adc_line("2e3,1900,512,3000") is None


# =============================================================================
# 10. Rejection of Negative ADC Counts
# =============================================================================
def test_rejection_of_negative_adc_counts() -> None:
    """Verify negative values in any channel are rejected."""
    assert parse_adc_line("-1,1900,512,3000") is None
    assert parse_adc_line("2048,-50,512,3000") is None
    assert parse_adc_line("2048,1900,-1,3000") is None
    assert parse_adc_line("2048,1900,512,-100") is None
    assert parse_adc_line("-2048,-1900,-512,-3000") is None


# =============================================================================
# 11. Rejection of ADC Values Greater Than 4095
# =============================================================================
def test_rejection_of_adc_values_greater_than_4095() -> None:
    """Verify counts exceeding 12-bit maximum (4095) are rejected."""
    assert parse_adc_line("4096,1900,512,3000") is None
    assert parse_adc_line("2048,5000,512,3000") is None
    assert parse_adc_line("2048,1900,65535,3000") is None
    assert parse_adc_line("2048,1900,512,99999") is None


# =============================================================================
# 12. ADC-to-Physical-Unit Mapping
# =============================================================================
def test_adc_to_physical_mapping() -> None:
    """Verify exact formula conversions and boundary conditions."""
    # Boundary 1: Minimum ADC count (0)
    rl_min, xl_min, v1_min, v2_min = adc_to_physical(0, 0, 0, 0)
    assert math.isclose(rl_min, 10.0, rel_tol=1e-6)
    assert math.isclose(xl_min, 62.0, rel_tol=1e-6)
    assert math.isclose(v1_min, 0.0, rel_tol=1e-6)
    assert math.isclose(v2_min, 0.0, rel_tol=1e-6)

    # Boundary 2: Maximum ADC count (4095)
    rl_max, xl_max, v1_max, v2_max = adc_to_physical(4095, 4095, 4095, 4095)
    assert math.isclose(rl_max, 17.0, rel_tol=1e-6)
    assert math.isclose(xl_max, 77.0, rel_tol=1e-6)
    assert math.isclose(v1_max, 3.3, rel_tol=1e-6)
    assert math.isclose(v2_max, 3.3, rel_tol=1e-6)

    # Midpoint: ADC count 2048
    # RL = 10 + (2048/4095) * 7.0
    # XL = 62 + (2048/4095) * 15.0
    # V  = (2048/4095) * 3.3
    rl_mid = adc_to_rl(2048)
    xl_mid = adc_to_xl(2048)
    v_mid = adc_to_v(2048)

    assert math.isclose(rl_mid, 10.0 + (2048.0 / 4095.0) * 7.0, rel_tol=1e-6)
    assert math.isclose(xl_mid, 62.0 + (2048.0 / 4095.0) * 15.0, rel_tol=1e-6)
    assert math.isclose(v_mid, (2048.0 / 4095.0) * 3.3, rel_tol=1e-6)

    # Clamping behavior for out-of-range counts
    assert math.isclose(adc_to_rl(-50), 10.0, rel_tol=1e-6)
    assert math.isclose(adc_to_rl(9999), 17.0, rel_tol=1e-6)
    assert math.isclose(adc_to_xl(-10), 62.0, rel_tol=1e-6)
    assert math.isclose(adc_to_xl(5000), 77.0, rel_tol=1e-6)
    assert math.isclose(adc_to_v(-5), 0.0, rel_tol=1e-6)
    assert math.isclose(adc_to_v(8000), 3.3, rel_tol=1e-6)


# =============================================================================
# 13. EMA Initialization
# =============================================================================
def test_ema_initialization() -> None:
    """Verify EMA initializes to first received sample without delay or lag."""
    ema = EMAFilter(alpha=0.25)
    assert ema.value is None

    first_val = 80.0
    out = ema.update(first_val)
    assert math.isclose(out, first_val, rel_tol=1e-6)
    assert math.isclose(ema.value, first_val, rel_tol=1e-6)


# =============================================================================
# 14. EMA Filtering Behavior
# =============================================================================
def test_ema_filtering_behavior() -> None:
    """Verify EMA mathematical formula: alpha * new + (1 - alpha) * prev."""
    alpha = 0.25
    ema = EMAFilter(alpha=alpha)

    # Sample 1: initializes to 100.0
    s1 = ema.update(100.0)
    assert math.isclose(s1, 100.0, rel_tol=1e-6)

    # Sample 2: 0.25 * 200.0 + 0.75 * 100.0 = 50.0 + 75.0 = 125.0
    s2 = ema.update(200.0)
    assert math.isclose(s2, 125.0, rel_tol=1e-6)

    # Sample 3: 0.25 * 200.0 + 0.75 * 125.0 = 50.0 + 93.75 = 143.75
    s3 = ema.update(200.0)
    assert math.isclose(s3, 143.75, rel_tol=1e-6)

    # Pass-through test with alpha = 1.0
    ema_pt = EMAFilter(alpha=1.0)
    assert math.isclose(ema_pt.update(10.0), 10.0, rel_tol=1e-6)
    assert math.isclose(ema_pt.update(50.0), 50.0, rel_tol=1e-6)


# =============================================================================
# 15. EMA Reset Behavior & Validation
# =============================================================================
def test_ema_reset_behavior() -> None:
    """Verify reset() clears internal state and alpha validation."""
    ema = EMAFilter(alpha=0.25)
    ema.update(100.0)
    ema.update(200.0)
    assert ema.value is not None

    ema.reset()
    assert ema.value is None

    # Next sample must re-initialize without remembering old state
    new_init = ema.update(45.0)
    assert math.isclose(new_init, 45.0, rel_tol=1e-6)

    # Test alpha validation
    try:
        EMAFilter(alpha=0.0)
        assert False, "Should have raised ValueError for alpha=0.0"
    except ValueError:
        pass

    try:
        EMAFilter(alpha=-0.1)
        assert False, "Should have raised ValueError for negative alpha"
    except ValueError:
        pass

    try:
        EMAFilter(alpha=1.01)
        assert False, "Should have raised ValueError for alpha > 1.0"
    except ValueError:
        pass


# =============================================================================
# 16. Mock Source Returns Correct Dictionary Structure
# =============================================================================
def test_mock_source_dictionary_structure() -> None:
    """Verify MockSource matches the public API contract dictionary specification."""
    source = MockSource(update_rate=20.0, seed=42)
    data = source.get_latest()

    required_keys = {"RL", "XL", "V1_man", "V2_man", "t"}
    assert set(data.keys()) == required_keys

    # Check data types: all must be float
    for k in required_keys:
        assert isinstance(data[k], float), f"Key '{k}' must be float, got {type(data[k])}"

    # Timestamp must be positive epoch timestamp
    assert data["t"] > 1700000000.0


# =============================================================================
# 17. Mock Source Values Remain Within Valid Ranges
# =============================================================================
def test_mock_source_value_ranges() -> None:
    """Verify 100 generated mock frames strictly stay inside physical boundaries."""
    source = MockSource(
        update_rate=20.0,
        enable_noise=True,
        enable_disturbances=True,
        seed=123,
    )

    t_base = time.time()
    for i in range(100):
        frame = source.generate_step(timestamp=t_base + i * 0.05)
        rl = frame["RL"]
        xl = frame["XL"]
        v1 = frame["V1_man"]
        v2 = frame["V2_man"]

        assert 10.0 <= rl <= 17.0, f"RL out of bounds: {rl}"
        assert 62.0 <= xl <= 77.0, f"XL out of bounds: {xl}"
        assert 0.0 <= v1 <= 3.3, f"V1 out of bounds: {v1}"
        assert 0.0 <= v2 <= 3.3, f"V2 out of bounds: {v2}"


# =============================================================================
# 18. Additional Component Tests (MedianFilter, ChannelFilter, Lifecycle)
# =============================================================================
def test_median_filter_spike_rejection() -> None:
    """Verify median filter suppresses single-sample impulsive glitches."""
    med = MedianFilter(window_size=3, enabled=True)
    assert med.update(10.0) == 10.0
    assert med.update(10.0) == 10.0
    # Impulsive spike from 10.0 to 999.0
    spike_filtered = med.update(999.0)
    assert spike_filtered == 10.0  # Median of [10.0, 10.0, 999.0] is 10.0


def test_channel_filter_independence() -> None:
    """Verify ChannelFilter maintains strict independence across all 4 channels."""
    cf = ChannelFilter(alpha=0.5, use_median=False)

    f1 = cf.update((10.0, 20.0, 30.0, 40.0))
    assert f1 == (10.0, 20.0, 30.0, 40.0)

    # Change only channel 1 (RL); other channels must remain steady
    f2 = cf.update((20.0, 20.0, 30.0, 40.0))
    assert math.isclose(f2[0], 15.0, rel_tol=1e-6)
    assert math.isclose(f2[1], 20.0, rel_tol=1e-6)
    assert math.isclose(f2[2], 30.0, rel_tol=1e-6)
    assert math.isclose(f2[3], 40.0, rel_tol=1e-6)


def test_mock_source_lifecycle() -> None:
    """Verify MockSource starts, runs background thread, and stops cleanly."""
    mock = MockSource(update_rate=50.0, seed=99)
    assert not mock.is_connected()

    mock.start()
    assert mock.is_connected()
    time.sleep(0.08)

    d = mock.get_latest()
    assert "RL" in d
    assert "t" in d

    mock.stop()
    assert not mock.is_connected()


def test_public_get_latest_api_contract() -> None:
    """Verify module-level get_latest() function returns valid contract dictionary."""
    mock = MockSource(update_rate=20.0, seed=55)
    set_default_source(mock)

    d = get_latest()
    assert isinstance(d, dict)
    assert set(d.keys()) == {"RL", "XL", "V1_man", "V2_man", "t"}
    assert 10.0 <= d["RL"] <= 17.0
    assert 62.0 <= d["XL"] <= 77.0
    assert 0.0 <= d["V1_man"] <= 3.3
    assert 0.0 <= d["V2_man"] <= 3.3


# =============================================================================
# Unittest TestCase wrapper for standard runner compatibility
# =============================================================================
import unittest


class TestSerialParserSuite(unittest.TestCase):
    """Unittest wrapper ensuring compatibility with `python -m unittest`."""

    def test_01_valid_adc_line_parsing(self) -> None:
        test_valid_adc_line_parsing()

    def test_02_parsing_with_spaces(self) -> None:
        test_parsing_with_spaces()

    def test_03_parsing_with_windows_line_endings(self) -> None:
        test_parsing_with_windows_line_endings()

    def test_04_rejection_of_empty_strings(self) -> None:
        test_rejection_of_empty_strings()

    def test_05_rejection_of_malformed_csv(self) -> None:
        test_rejection_of_malformed_csv()

    def test_06_rejection_of_missing_channels(self) -> None:
        test_rejection_of_missing_channels()

    def test_07_rejection_of_extra_channels(self) -> None:
        test_rejection_of_extra_channels()

    def test_08_rejection_of_non_numeric_values(self) -> None:
        test_rejection_of_non_numeric_values()

    def test_09_rejection_of_decimal_values(self) -> None:
        test_rejection_of_decimal_values()

    def test_10_rejection_of_negative_adc_counts(self) -> None:
        test_rejection_of_negative_adc_counts()

    def test_11_rejection_of_adc_values_greater_than_4095(self) -> None:
        test_rejection_of_adc_values_greater_than_4095()

    def test_12_adc_to_physical_mapping(self) -> None:
        test_adc_to_physical_mapping()

    def test_13_ema_initialization(self) -> None:
        test_ema_initialization()

    def test_14_ema_filtering_behavior(self) -> None:
        test_ema_filtering_behavior()

    def test_15_ema_reset_behavior(self) -> None:
        test_ema_reset_behavior()

    def test_16_mock_source_dictionary_structure(self) -> None:
        test_mock_source_dictionary_structure()

    def test_17_mock_source_value_ranges(self) -> None:
        test_mock_source_value_ranges()

    def test_18_median_filter_spike_rejection(self) -> None:
        test_median_filter_spike_rejection()

    def test_19_channel_filter_independence(self) -> None:
        test_channel_filter_independence()

    def test_20_mock_source_lifecycle(self) -> None:
        test_mock_source_lifecycle()

    def test_21_public_get_latest_api_contract(self) -> None:
        test_public_get_latest_api_contract()


# =============================================================================
# Standalone Test Runner (Enables execution with or without pytest)
# =============================================================================
def run_all_tests() -> bool:
    """Execute all test functions in this module and report results."""
    test_functions = [
        obj
        for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj) and not isinstance(obj, type)
    ]

    passed = 0
    failed = 0
    print(f"\nRunning {len(test_functions)} Module 2 unit tests...")
    print("=" * 70)

    for fn in test_functions:
        test_name = fn.__name__
        try:
            fn()
            print(f"  PASS: {test_name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL: {test_name} -> {exc}")
            failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_functions)} tests.")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
