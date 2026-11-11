"""M2: Serial reader with auto-detection, line parsing, filtering, and reconnection.

Reads 4 raw ADC counts streamed over UART from STM32F446RE:
    Format: "<adc_RL>,<adc_XL>,<adc_V1>,<adc_V2>\\r\\n"
    Default baud: 115200, 8-N-1, ~20 Hz streaming rate

Performs:
    1. Automatic detection of ST-Link / STM32 virtual COM port.
    2. Resilient CSV line parsing with strict validation.
    3. Conversion from 12-bit ADC counts (0-4095) to physical units.
    4. EMA low-pass filtering to eliminate potentiometer jitter.
    5. Automatic background reconnection when the board is unplugged/replugged.
    6. Thread-safe cached access to the latest valid telemetry frame.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Tuple

# Try importing pyserial; handle missing package gracefully
try:
    import serial
    import serial.tools.list_ports

    HAS_PYSERIAL = True
except ImportError:
    serial = None  # type: ignore[assignment]
    HAS_PYSERIAL = False

# Import filters with fallback strategies
try:
    from io.filters import (
        ADC_MAX,
        ADC_VREF,
        ChannelFilter,
        adc_to_physical,
    )
except (ImportError, AttributeError):
    try:
        from .filters import (
            ADC_MAX,
            ADC_VREF,
            ChannelFilter,
            adc_to_physical,
        )
    except (ImportError, AttributeError):
        from filters import (  # type: ignore[no-redef]
            ADC_MAX,
            ADC_VREF,
            ChannelFilter,
            adc_to_physical,
        )

logger = logging.getLogger(__name__)

# Protocol Constants
BAUD_RATE: int = 115200
EXPECTED_CHANNELS: int = 4
DEFAULT_FILTER_ALPHA: float = 0.25
DEFAULT_RECONNECT_INTERVAL: float = 2.0


def parse_adc_line(line: str | bytes | None) -> tuple[int, int, int, int] | None:
    """Parse and validate a single CSV line from the STM32 UART stream.

    Expected protocol format:
        "<adc_RL>,<adc_XL>,<adc_V1>,<adc_V2>"
    Example:
        "2048,1900,512,3000"

    Validation rules:
        - Accepts valid 4-channel CSV strings.
        - Tolerates leading/trailing whitespace and Windows CRLF (\\r\\n) line endings.
        - Rejects lines with fewer or more than 4 channels.
        - Rejects empty lines or missing fields.
        - Rejects non-numeric tokens.
        - Rejects floating-point or scientific notation tokens (requires integer ADC counts).
        - Rejects negative numbers or values exceeding ADC_MAX (4095).
        - Never crashes or raises unhandled exceptions on corrupt/partial serial bytes.

    Args:
        line: Raw line as string, bytes, or None.

    Returns:
        tuple[int, int, int, int] containing valid ADC counts (0..4095),
        or None if parsing fails.
    """
    if line is None:
        return None

    try:
        if isinstance(line, (bytes, bytearray)):
            line_str = line.decode("utf-8", errors="replace")
        elif isinstance(line, str):
            line_str = line
        else:
            return None

        stripped = line_str.strip()
        if not stripped:
            return None

        parts = stripped.split(",")
        if len(parts) != EXPECTED_CHANNELS:
            return None

        parsed_values: list[int] = []
        for part in parts:
            token = part.strip()
            # Must be non-empty and composed strictly of digits (disallowing floats, signs, NaN)
            if not token or not token.isdigit():
                return None

            val = int(token)
            if not (0 <= val <= ADC_MAX):
                return None

            parsed_values.append(val)

        return (
            parsed_values[0],
            parsed_values[1],
            parsed_values[2],
            parsed_values[3],
        )
    except Exception:
        return None


def detect_serial_port() -> str | None:
    """Auto-detect connected STMicroelectronics / STM32 virtual COM port.

    Searches available system serial ports using USB VID/PID (0x0483 for ST)
    and port description strings (STLink, STM32, Nucleo). Falls back to
    generic USB serial adapters if specific STM32 hardware is not found.

    Returns:
        Device port name (e.g. 'COM3' or '/dev/ttyACM0') or None if not found.
    """
    if not HAS_PYSERIAL:
        return None

    try:
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            return None

        # Tier 1: Look for STMicroelectronics VID (0x0483) or STLink description
        for p in ports:
            vid = getattr(p, "vid", None)
            desc = (getattr(p, "description", "") or "").upper()
            mfg = (getattr(p, "manufacturer", "") or "").upper()
            hwid = (getattr(p, "hwid", "") or "").upper()

            if vid == 0x0483 or "0483" in hwid:
                return p.device
            if any(k in desc for k in ("STLINK", "ST-LINK", "STM32", "NUCLEO")):
                return p.device
            if any(k in mfg for k in ("STLINK", "ST-LINK", "STMICROELECTRONICS")):
                return p.device

        # Tier 2: Look for common USB-UART bridges (CP210x, FTDI, CH340)
        for p in ports:
            desc = (getattr(p, "description", "") or "").upper()
            if any(k in desc for k in ("USB SERIAL", "CP210", "FTDI", "CH340")):
                return p.device

        # Tier 3: Return the first available serial device
        return ports[0].device
    except Exception as exc:
        logger.debug("Port detection failed: %s", exc)
        return None


class SerialReader:
    """Thread-safe background UART telemetry reader for STM32 board.

    Maintains a robust background thread that continuously reads raw ADC counts,
    rejects malformed lines, converts counts to physical units, filters pot jitter,
    and caches the latest measurement for non-blocking consumption.
    """

    def __init__(
        self,
        port: str | None = None,
        baudrate: int = BAUD_RATE,
        filter_alpha: float = DEFAULT_FILTER_ALPHA,
        reconnect_interval: float = DEFAULT_RECONNECT_INTERVAL,
        use_median: bool = False,
        median_window: int = 3,
    ) -> None:
        """Initialize SerialReader.

        Args:
            port: Serial port name (e.g. 'COM3' or '/dev/ttyACM0'). If None,
                auto-detection will be performed.
            baudrate: UART baud rate (default 115200).
            filter_alpha: Exponential Moving Average alpha (0.0 < alpha <= 1.0).
            reconnect_interval: Seconds to wait between reconnection attempts.
            use_median: Whether to enable small-window median filter for spike removal.
            median_window: Median window length (odd integer).
        """
        self.port = port
        self.baudrate = int(baudrate)
        self.reconnect_interval = float(reconnect_interval)

        self._filter = ChannelFilter(
            alpha=filter_alpha,
            use_median=use_median,
            median_window=median_window,
        )

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._serial_conn: Any = None
        self._connected = False
        self._active_port: str | None = None

        # Statistics & diagnostics
        self._lines_read = 0
        self._lines_rejected = 0
        self._last_error: str | None = None

        # Latest valid measurement (nominal fallback matching contract)
        self._latest_data: dict[str, float] = {
            "RL": 50.0,
            "XL": 0.0,
            "V1_man": 1.65,
            "V2_man": 1.65,
            "t": time.time(),
        }

        # Latest raw measurement (unfiltered ADC counts and converted physical values)
        self._latest_raw: dict[str, Any] = {
            "adc_RL": 2048,
            "adc_XL": 2048,
            "adc_V1": 2048,
            "adc_V2": 2048,
            "RL_raw": 80.0,
            "XL_raw": 0.0,
            "V1_raw": 1.65,
            "V2_raw": 1.65,
            "line": "",
            "t": time.time(),
        }

    def start(self) -> None:
        """Start the background serial reader thread.

        If already running, this call is a no-op.

        Raises:
            ImportError: If pyserial is not installed when attempting to start.
        """
        if not HAS_PYSERIAL:
            raise ImportError(
                "pyserial is not installed. Please install pyserial ('pip install pyserial') "
                "to connect to STM32 serial hardware, or use io.MockSource for offline simulation."
            )

        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="SerialReaderThread",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the background reader thread and close the serial port cleanly.

        Args:
            timeout: Maximum seconds to wait for worker thread termination.
        """
        self._stop_event.set()

        # Close serial port to unblock any pending blocking readline()
        with self._lock:
            if self._serial_conn is not None:
                try:
                    self._serial_conn.close()
                except Exception:
                    pass
                self._serial_conn = None
            self._connected = False

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    def is_connected(self) -> bool:
        """Thread-safe check if currently connected to a serial port."""
        with self._lock:
            return self._connected

    def get_latest(self) -> dict[str, float]:
        """Return the latest valid filtered physical measurement.

        Thread-safe and non-blocking. Downstream modules (M3, M4, M5) consume
        this API.

        Returns:
            Dictionary with exact required keys:
            {
                "RL": float,       # Load resistance (Ohms)
                "XL": float,       # Load reactance (Ohms)
                "V1_man": float,   # Manual varactor voltage 1 (Volts)
                "V2_man": float,   # Manual varactor voltage 2 (Volts)
                "t": float         # Unix epoch timestamp (seconds)
            }
        """
        with self._lock:
            return {
                "RL": float(self._latest_data["RL"]),
                "XL": float(self._latest_data["XL"]),
                "V1_man": float(self._latest_data["V1_man"]),
                "V2_man": float(self._latest_data["V2_man"]),
                "t": float(self._latest_data["t"]),
            }

    def get_latest_raw(self) -> dict[str, Any]:
        """Return the latest raw ADC counts and unfiltered physical values.

        Useful for diagnostics, calibration, and comparing raw vs filtered data.

        Returns:
            Dictionary with keys:
                adc_RL, adc_XL, adc_V1, adc_V2 (raw integer counts 0-4095)
                RL_raw, XL_raw, V1_raw, V2_raw (unfiltered floats in physical units)
                line (raw string received over UART)
                t (timestamp)
        """
        with self._lock:
            return dict(self._latest_raw)

    def get_connection_status(self) -> dict[str, Any]:
        """Return diagnostic and connection status information."""
        with self._lock:
            return {
                "connected": self._connected,
                "port": self._active_port or self.port,
                "baudrate": self.baudrate,
                "lines_read": self._lines_read,
                "lines_rejected": self._lines_rejected,
                "last_error": self._last_error,
            }

    def _worker_loop(self) -> None:
        """Continuous background thread handling connection and data reading."""
        while not self._stop_event.is_set():
            # Handle disconnected state
            if not self._connected or self._serial_conn is None:
                target_port = self.port or detect_serial_port()
                if not target_port:
                    with self._lock:
                        self._last_error = "No serial port detected"
                    self._stop_event.wait(timeout=self.reconnect_interval)
                    continue

                try:
                    conn = serial.Serial(
                        port=target_port,
                        baudrate=self.baudrate,
                        timeout=0.5,
                    )
                    with self._lock:
                        self._serial_conn = conn
                        self._connected = True
                        self._active_port = target_port
                        self._last_error = None
                    logger.info("Connected to serial port %s at %d baud", target_port, self.baudrate)
                except Exception as exc:
                    with self._lock:
                        self._connected = False
                        self._serial_conn = None
                        self._last_error = str(exc)
                    self._stop_event.wait(timeout=self.reconnect_interval)
                    continue

            # Connected: read UART stream
            try:
                line_bytes = self._serial_conn.readline()
                if not line_bytes:
                    # Timeout occurred with no data received; check stop and continue
                    continue

                parsed = parse_adc_line(line_bytes)
                if parsed is None:
                    # Malformed or partial line: increment error count and keep prior valid data
                    with self._lock:
                        self._lines_rejected += 1
                    continue

                # Valid line: convert to physical units and filter
                adc_rl, adc_xl, adc_v1, adc_v2 = parsed
                raw_phys = adc_to_physical(adc_rl, adc_xl, adc_v1, adc_v2)
                rl_f, xl_f, v1_f, v2_f = self._filter.update(raw_phys)
                now = time.time()

                with self._lock:
                    self._latest_raw = {
                        "adc_RL": int(adc_rl),
                        "adc_XL": int(adc_xl),
                        "adc_V1": int(adc_v1),
                        "adc_V2": int(adc_v2),
                        "RL_raw": float(raw_phys[0]),
                        "XL_raw": float(raw_phys[1]),
                        "V1_raw": float(raw_phys[2]),
                        "V2_raw": float(raw_phys[3]),
                        "line": line_bytes.decode("utf-8", errors="replace").strip(),
                        "t": float(now),
                    }
                    self._latest_data = {
                        "RL": float(rl_f),
                        "XL": float(xl_f),
                        "V1_man": float(v1_f),
                        "V2_man": float(v2_f),
                        "t": float(now),
                    }
                    self._lines_read += 1

            except (serial.SerialException, OSError) as exc:
                # Hardware disconnected or read failure
                logger.warning("Serial connection lost on %s: %s", self._active_port, exc)
                with self._lock:
                    self._connected = False
                    self._last_error = str(exc)
                    if self._serial_conn is not None:
                        try:
                            self._serial_conn.close()
                        except Exception:
                            pass
                        self._serial_conn = None

                # Wait before attempting reconnection
                self._stop_event.wait(timeout=self.reconnect_interval)
            except Exception as exc:
                # Catch-all to ensure background thread never terminates unexpectedly
                logger.debug("Unexpected error in serial worker: %s", exc)
                self._stop_event.wait(timeout=0.05)


# -----------------------------------------------------------------------------
# Module-level default reader & Public API Contract
# -----------------------------------------------------------------------------
_active_source: Any = None
_source_lock = threading.Lock()


def get_default_reader() -> SerialReader:
    """Get or create the singleton default SerialReader instance."""
    global _active_source
    with _source_lock:
        if _active_source is None:
            _active_source = SerialReader()
        return _active_source


def set_default_source(source: Any) -> None:
    """Set the active telemetry source (SerialReader or MockSource) for get_latest().

    Args:
        source: Any object implementing get_latest() -> dict.
    """
    global _active_source
    with _source_lock:
        _active_source = source


def get_latest() -> dict[str, float]:
    """Public API Contract: Return the latest measurement frame.

    Exposes the contract required by downstream modules M3, M4, and M5.
    If no source has been explicitly configured, returns the latest cached
    reading from the default SerialReader.

    Returns:
        dict:
        {
            "RL": float,
            "XL": float,
            "V1_man": float,
            "V2_man": float,
            "t": float
        }
    """
    return get_default_reader().get_latest()


def get_latest_raw() -> dict[str, Any]:
    """Return the latest raw ADC counts and unfiltered physical values from active reader."""
    reader = get_default_reader()
    if hasattr(reader, "get_latest_raw"):
        return reader.get_latest_raw()
    return {}


# -----------------------------------------------------------------------------
# Standalone CLI Monitor for Live Hardware Testing
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Live STM32 UART Telemetry Monitor (Module 2)")
    parser.add_argument("--port", "-p", type=str, default=None, help="COM port (e.g. COM3, COM4, COM6)")
    parser.add_argument("--baud", "-b", type=int, default=BAUD_RATE, help=f"Baud rate (default {BAUD_RATE})")
    parser.add_argument("--alpha", "-a", type=float, default=DEFAULT_FILTER_ALPHA, help="EMA alpha (default 0.25)")
    args = parser.parse_args()

    print("=" * 86)
    print("STM32 Autonomous RL Impedance Matching - Live Telemetry Monitor (Module 2)")
    print("=" * 86)

    if not HAS_PYSERIAL:
        print("\n[ERROR] 'pyserial' is not installed in the current Python environment.")
        print("To connect to the physical STM32 board, please install pyserial:")
        print("    pip install pyserial\n")
        sys.exit(1)

    target_port = args.port or detect_serial_port()
    print(f"Target Port : {target_port or 'AUTO-DETECTING...'}")
    print(f"Baud Rate   : {args.baud} 8-N-1")
    print(f"EMA Alpha   : {args.alpha}")
    print("Starting background UART reader...")

    reader = SerialReader(port=args.port, baudrate=args.baud, filter_alpha=args.alpha)
    try:
        reader.start()
    except Exception as exc:
        print(f"\n[ERROR] Failed to start serial reader: {exc}")
        sys.exit(1)

    print("\nListening for STM32 telemetry stream... (Press Ctrl+C to exit)\n")
    print(f"{'TIME':<10} | {'RAW ADC (RL, XL, V1, V2)':<26} | {'FILTERED PHYSICAL VALUES':<42}")
    print("-" * 86)

    last_timestamp = 0.0
    try:
        while True:
            raw = reader.get_latest_raw()
            filt = reader.get_latest()
            status = reader.get_connection_status()

            if status["connected"] and raw.get("t", 0.0) != last_timestamp:
                last_timestamp = raw["t"]
                raw_counts = f"{raw['adc_RL']:4d}, {raw['adc_XL']:4d}, {raw['adc_V1']:4d}, {raw['adc_V2']:4d}"
                filt_str = (
                    f"RL={filt['RL']:5.1f}Ω | XL={filt['XL']:6.1f}Ω | "
                    f"V1={filt['V1_man']:4.2f}V | V2={filt['V2_man']:4.2f}V"
                )
                t_str = time.strftime("%H:%M:%S", time.localtime(filt["t"]))
                print(f"{t_str:<10} | {raw_counts:<26} | {filt_str:<42}")
            elif not status["connected"]:
                err = status.get("last_error") or "Searching for STM32..."
                print(f"[STATUS] Waiting for STM32... ({err})", end="\r", flush=True)

            time.sleep(0.04)
    except KeyboardInterrupt:
        print("\n\nStopping serial reader...")
    finally:
        reader.stop()
        print("Reader stopped cleanly.")
