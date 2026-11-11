"""M2: Serial link and signal processing package.

STM32-Driven Autonomous RL Impedance Matching Network.

This package provides:
    - SerialReader: UART driver for STM32F446RE telemetry streaming at 115200 baud.
    - MockSource: Simulated telemetry generator matching SerialReader API for offline dev.
    - filters: ADC count to physical unit conversions, EMA and median filters.
    - get_latest(): Public API contract returning the latest valid measurement:
          {"RL": float, "XL": float, "V1_man": float, "V2_man": float, "t": float}
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import submodules with relative / package fallback
try:
    from .serial_reader import (
        SerialReader,
        get_latest,
        get_latest_raw,
        parse_adc_line,
        set_default_source,
    )
    from .mock_source import MockSource
    from .filters import (
        ADC_MAX,
        ADC_VREF,
        ChannelFilter,
        EMAFilter,
        MedianFilter,
        adc_to_physical,
        adc_to_rl,
        adc_to_v,
        adc_to_xl,
    )
except (ImportError, AttributeError):
    try:
        from io.serial_reader import (
            SerialReader,
            get_latest,
            get_latest_raw,
            parse_adc_line,
            set_default_source,
        )
        from io.mock_source import MockSource
        from io.filters import (
            ADC_MAX,
            ADC_VREF,
            ChannelFilter,
            EMAFilter,
            MedianFilter,
            adc_to_physical,
            adc_to_rl,
            adc_to_v,
            adc_to_xl,
        )
    except (ImportError, AttributeError):
        from serial_reader import (  # type: ignore[no-redef]
            SerialReader,
            get_latest,
            get_latest_raw,
            parse_adc_line,
            set_default_source,
        )
        from mock_source import MockSource  # type: ignore[no-redef]
        from filters import (  # type: ignore[no-redef]
            ADC_MAX,
            ADC_VREF,
            ChannelFilter,
            EMAFilter,
            MedianFilter,
            adc_to_physical,
            adc_to_rl,
            adc_to_v,
            adc_to_xl,
        )

# Coexistence with Python's built-in 'io' standard library module:
# If 'io' is already in sys.modules, attach our package directory to its __path__
# and attach M2 public symbols so that both `from io import get_latest` and
# `import io.serial_reader` work seamlessly without breaking stdlib io.StringIO/BytesIO.
if "io" in sys.modules:
    _stdlib_io = sys.modules["io"]
    _io_dir = str(Path(__file__).resolve().parent)
    if hasattr(_stdlib_io, "__path__"):
        if _io_dir not in _stdlib_io.__path__:
            _stdlib_io.__path__.append(_io_dir)
    else:
        try:
            _stdlib_io.__path__ = [_io_dir]
        except Exception:
            pass

    # Expose public M2 symbols on sys.modules['io']
    _m2_exports = {
        "get_latest": get_latest,
        "get_latest_raw": get_latest_raw,
        "SerialReader": SerialReader,
        "MockSource": MockSource,
        "parse_adc_line": parse_adc_line,
        "set_default_source": set_default_source,
        "adc_to_physical": adc_to_physical,
        "adc_to_rl": adc_to_rl,
        "adc_to_xl": adc_to_xl,
        "adc_to_v": adc_to_v,
        "EMAFilter": EMAFilter,
        "MedianFilter": MedianFilter,
        "ChannelFilter": ChannelFilter,
        "ADC_MAX": ADC_MAX,
        "ADC_VREF": ADC_VREF,
    }
    for _k, _v in _m2_exports.items():
        if not hasattr(_stdlib_io, _k):
            try:
                setattr(_stdlib_io, _k, _v)
            except Exception:
                pass

__all__ = [
    "SerialReader",
    "MockSource",
    "get_latest",
    "get_latest_raw",
    "parse_adc_line",
    "set_default_source",
    "ADC_MAX",
    "ADC_VREF",
    "adc_to_physical",
    "adc_to_rl",
    "adc_to_xl",
    "adc_to_v",
    "EMAFilter",
    "MedianFilter",
    "ChannelFilter",
]
