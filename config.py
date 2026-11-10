"""Shared constants for Autonomous RL Impedance Matching Network.

Refer to Project.md Section 4.5.
"""

# RF & Physics parameters
Z0 = 50.0  # Characteristic impedance (Ohms)
FREQ = 100e6  # Operating frequency (100 MHz)

# Voltage boundaries & action resolution
V_MIN = 0.0  # Minimum varactor bias voltage (V)
V_MAX = 3.3  # Maximum varactor bias voltage (V)
V_STEP = 0.05  # Step size for voltage control (V)

# Disturbance load ranges
RL_RANGE = (10.0, 150.0)  # Load resistance disturbance range (Ohms)
XL_RANGE = (-100.0, 100.0)  # Load reactance disturbance range (Ohms)

# Hardware communication
BAUD = 115200  # Serial baud rate
SAMPLE_RATE_HZ = 20  # Telemetry streaming rate (~20 Hz)
