"""Varactor Diode Capacitance-Voltage C(V) Model.

This module models a reverse-biased varactor diode as a voltage-controlled
variable capacitor using the standard semiconductor junction capacitance formula:
    C(V) = C0 / ((1 + V / Vj) ** gamma)

Hardware & Circuit Context:
- Applied bias voltage V is restricted to the 0.0V to 3.3V range (DAC/Potentiometer range).
- C0 is the zero-bias junction capacitance (~100 pF default).
- Vj is the built-in junction potential (~0.7V for silicon PN junction).
- gamma is the junction grading coefficient (~0.5 for abrupt junction).
"""

import numpy as np


def capacitance(V, C0=100e-12, Vj=0.7, gamma=0.5):
    """
    Calculate the varactor diode capacitance for a given reverse-bias voltage.

    The voltage V is clamped to the hardware operating limits [0.0V, 3.3V]
    before evaluating the capacitance formula:
        C(V) = C0 / ((1 + V / Vj) ** gamma)

    Parameters
    ----------
    V : float or array-like
        Applied reverse bias voltage in Volts. Values outside [0.0, 3.3]
        are automatically clamped to this range.
    C0 : float, optional
        Zero-bias junction capacitance in Farads (default: 100e-12 F = 100 pF).
    Vj : float, optional
        Built-in junction potential in Volts (default: 0.7 V).
    gamma : float, optional
        Junction grading coefficient, dimensionless (default: 0.5 for abrupt junction).

    Returns
    -------
    float or np.ndarray
        Junction capacitance in Farads. Returns a Python float for scalar inputs,
        or a NumPy ndarray for array-like inputs.
    """
    # Clamp bias voltage to hardware operational boundaries [0.0V, 3.3V]
    v_clamped = np.clip(V, 0.0, 3.3)

    # Calculate capacitance using the diode junction capacitance equation
    c = C0 / ((1.0 + v_clamped / Vj) ** gamma)

    # Return native float for scalar inputs for compatibility and clean serialization
    if np.isscalar(V):
        return float(c)
    return c
