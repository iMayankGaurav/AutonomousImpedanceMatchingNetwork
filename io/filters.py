"""M2: Signal filtering (EMA/median) and ADC count to physical unit mapping.

Converts 12-bit ADC raw counts (0-4095) from STM32 potentiometer channels into
physical units:
    - ADC 1 (RL): Load resistance (10 to 17 Ohms)
    - ADC 2 (XL): Load reactance (62 to 77 Ohms)
    - ADC 3 (V1): Manual varactor bias voltage 1 (0 to 3.3 V)
    - ADC 4 (V2): Manual varactor bias voltage 2 (0 to 3.3 V)

Provides Exponential Moving Average (EMA) and Median filters for jitter reduction
and noise rejection.
"""

from __future__ import annotations

import collections
from typing import Sequence

# -----------------------------------------------------------------------------
# ADC Hardware & Conversion Constants (STM32 12-bit ADC)
# -----------------------------------------------------------------------------
ADC_MAX: int = 4095
ADC_VREF: float = 3.3

RL_MIN: float = 10.0
RL_MAX: float = 17.0
XL_MIN: float = 62.0
XL_MAX: float = 77.0
V_MIN: float = 0.0
V_MAX: float = 3.3


# -----------------------------------------------------------------------------
# ADC to Physical Unit Conversion Functions
# -----------------------------------------------------------------------------
def clamp_adc(val: float | int) -> float:
    """Clamp ADC counts to the valid 12-bit range [0, 4095].

    Args:
        val: Raw numeric ADC count.

    Returns:
        Clamped float in [0.0, 4095.0].
    """
    return max(0.0, min(float(ADC_MAX), float(val)))


def adc_to_rl(adc_rl: float | int) -> float:
    """Convert raw ADC count to load resistance RL in Ohms.

    Formula:
        RL = 10 + (adc_RL / 4095) * (17 - 10)

    Args:
        adc_rl: ADC count for channel 1 (0..4095).

    Returns:
        Resistance in Ohms in range [10.0, 17.0].
    """
    clamped = clamp_adc(adc_rl)
    return RL_MIN + (clamped / ADC_MAX) * (RL_MAX - RL_MIN)


def adc_to_xl(adc_xl: float | int) -> float:
    """Convert raw ADC count to load reactance XL in Ohms.

    Formula:
        XL = 62 + (adc_XL / 4095) * (77 - 62)

    Args:
        adc_xl: ADC count for channel 2 (0..4095).

    Returns:
        Reactance in Ohms in range [62.0, 77.0].
    """
    clamped = clamp_adc(adc_xl)
    return XL_MIN + (clamped / ADC_MAX) * (XL_MAX - XL_MIN)


def adc_to_v(adc_v: float | int) -> float:
    """Convert raw ADC count to tuning voltage in Volts.

    Formula:
        V = (adc_V / 4095) * 3.3

    Args:
        adc_v: ADC count for channel 3 or 4 (0..4095).

    Returns:
        Voltage in Volts in range [0.0, 3.3].
    """
    clamped = clamp_adc(adc_v)
    return (clamped / ADC_MAX) * ADC_VREF


def adc_to_physical(
    adc_rl: float | int,
    adc_xl: float | int,
    adc_v1: float | int,
    adc_v2: float | int,
) -> tuple[float, float, float, float]:
    """Convert all 4 ADC channels to their corresponding physical values.

    Args:
        adc_rl: Raw ADC counts for load resistance.
        adc_xl: Raw ADC counts for load reactance.
        adc_v1: Raw ADC counts for varactor V1.
        adc_v2: Raw ADC counts for varactor V2.

    Returns:
        Tuple of (RL, XL, V1, V2) in (Ohms, Ohms, Volts, Volts).
    """
    return (
        adc_to_rl(adc_rl),
        adc_to_xl(adc_xl),
        adc_to_v(adc_v1),
        adc_to_v(adc_v2),
    )


# -----------------------------------------------------------------------------
# Signal Filters
# -----------------------------------------------------------------------------
class EMAFilter:
    """Exponential Moving Average (EMA) low-pass filter.

    Formula:
        filtered = alpha * new_value + (1 - alpha) * previous_value

    Attributes:
        alpha: Smoothing factor in range (0.0, 1.0].
            Higher alpha discounts older observations faster.
    """

    def __init__(self, alpha: float = 0.25) -> None:
        """Initialize the EMA filter.

        Args:
            alpha: Weight of the new measurement (0.0 < alpha <= 1.0).
                Default is 0.25 (~20 Hz pot jitter smoothing).

        Raises:
            ValueError: If alpha is outside the valid (0.0, 1.0] range.
        """
        self._alpha = 0.25
        self.alpha = alpha
        self._value: float | None = None

    @property
    def alpha(self) -> float:
        """Get the current smoothing coefficient."""
        return self._alpha

    @alpha.setter
    def alpha(self, val: float) -> None:
        """Set and validate smoothing coefficient."""
        if not (0.0 < val <= 1.0):
            raise ValueError(f"EMA alpha must be in (0.0, 1.0], got {val}")
        self._alpha = float(val)

    @property
    def value(self) -> float | None:
        """Get the latest filtered value, or None if uninitialized."""
        return self._value

    def update(self, new_value: float | int) -> float:
        """Update the filter with a new sample.

        Initializes with the first sample on first update or after reset.

        Args:
            new_value: New sample value.

        Returns:
            Current filtered output.
        """
        val = float(new_value)
        if self._value is None:
            self._value = val
        else:
            self._value = self._alpha * val + (1.0 - self._alpha) * self._value
        return self._value

    def reset(self, initial_value: float | int | None = None) -> None:
        """Reset the internal filter state.

        Args:
            initial_value: Optional value to seed the filter with.
        """
        if initial_value is not None:
            self._value = float(initial_value)
        else:
            self._value = None


class MedianFilter:
    """Sliding-window median filter for impulsive spike rejection.

    Maintains a rolling window of recent samples and returns the median.
    Useful for removing single-sample ADC bit flips or potentiometer glitches.
    """

    def __init__(self, window_size: int = 3, enabled: bool = True) -> None:
        """Initialize median filter.

        Args:
            window_size: Number of samples in the rolling window (positive integer).
            enabled: Whether median filtering is active. If False, updates
                pass values through unchanged.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = int(window_size)
        self.enabled = bool(enabled)
        self._buffer: collections.deque[float] = collections.deque(maxlen=self.window_size)

    def update(self, new_value: float | int) -> float:
        """Process a new sample through the median window.

        Args:
            new_value: New sample value.

        Returns:
            Median of the current window, or new_value if disabled.
        """
        val = float(new_value)
        if not self.enabled or self.window_size <= 1:
            return val

        self._buffer.append(val)
        sorted_samples = sorted(self._buffer)
        n = len(sorted_samples)
        mid = n // 2
        if n % 2 == 1:
            return sorted_samples[mid]
        return 0.5 * (sorted_samples[mid - 1] + sorted_samples[mid])

    def reset(self) -> None:
        """Clear the rolling buffer."""
        self._buffer.clear()


class ChannelFilter:
    """Four-channel independent signal filter.

    Applies optional median spike rejection followed by Exponential Moving
    Average (EMA) low-pass filtering independently to each of the 4 ADC / physical
    channels (RL, XL, V1, V2).

    Filtering on one channel never affects another channel.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        use_median: bool = False,
        median_window: int = 3,
    ) -> None:
        """Initialize independent filters for 4 channels.

        Args:
            alpha: EMA smoothing coefficient for each channel.
            use_median: Whether to enable median pre-filtering.
            median_window: Window size for median filter if enabled.
        """
        self.use_median = use_median
        self._median_filters = [
            MedianFilter(window_size=median_window, enabled=use_median)
            for _ in range(4)
        ]
        self._ema_filters = [EMAFilter(alpha=alpha) for _ in range(4)]

    def update(
        self, values: Sequence[float | int]
    ) -> tuple[float, float, float, float]:
        """Filter a 4-channel measurement tuple.

        Args:
            values: Sequence containing exactly 4 numeric values.

        Returns:
            Tuple of 4 filtered values.

        Raises:
            ValueError: If input length is not 4.
        """
        if len(values) != 4:
            raise ValueError(f"ChannelFilter expects 4 values, got {len(values)}")

        filtered: list[float] = []
        for i, val in enumerate(values):
            # Step 1: Median filter (if enabled)
            m_val = self._median_filters[i].update(val)
            # Step 2: EMA filter
            e_val = self._ema_filters[i].update(m_val)
            filtered.append(e_val)

        return (filtered[0], filtered[1], filtered[2], filtered[3])

    def reset(self) -> None:
        """Reset all channel filters."""
        for m_filt in self._median_filters:
            m_filt.reset()
        for e_filt in self._ema_filters:
            e_filt.reset()
