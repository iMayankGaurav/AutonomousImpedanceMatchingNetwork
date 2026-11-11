"""Smith Chart Mathematical Helper Utilities.

Provides bilinear transform conversions between complex reflection coefficient
(Gamma) and complex impedance (Z), as well as geometric coordinate generators
for rendering constant resistance and reactance circles on Smith charts.

Note:
This module contains pure mathematical calculations and NumPy array manipulations.
No plotting libraries (such as Matplotlib) are imported or used here.
"""

import numpy as np


def impedance_to_gamma(Z, Z0=50.0):
    """
    Convert a complex impedance Z to the complex reflection coefficient Gamma.

    Formula:
        Gamma = (Z - Z0) / (Z + Z0)

    Parameters
    ----------
    Z : complex, float, or int
        Impedance in Ohms (Z = R + j*X).
    Z0 : float, optional
        System characteristic reference impedance in Ohms (default: 50.0).

    Returns
    -------
    complex
        Complex reflection coefficient Gamma in the complex plane.
        When Z == Z0, returns exact 0j.
    """
    z_c = complex(Z)
    denom = z_c + Z0
    if denom == 0.0:
        return complex(np.inf, np.inf)

    gamma = (z_c - Z0) / denom

    # Return clean 0j for exact match condition
    if gamma == 0.0:
        return 0j

    return complex(gamma)


def gamma_to_impedance(gamma, Z0=50.0):
    """
    Convert a complex reflection coefficient Gamma to complex impedance Z.

    Formula:
        Z = Z0 * (1 + Gamma) / (1 - Gamma)

    Parameters
    ----------
    gamma : complex, float, or int
        Complex reflection coefficient (Gamma = u + j*v).
    Z0 : float, optional
        System characteristic reference impedance in Ohms (default: 50.0).

    Returns
    -------
    complex
        Complex impedance Z in Ohms (Z = R + j*X).
        Returns infinite impedance if Gamma == 1 (open circuit).
    """
    g_c = complex(gamma)
    denom = 1.0 - g_c
    if denom == 0.0:
        return complex(float("inf"), 0.0)

    z = Z0 * (1.0 + g_c) / denom
    return complex(z)


def resistance_circle(r, num_points=200):
    """
    Generate coordinates for a constant normalized resistance circle.

    In the Gamma plane (Gamma = u + j*v), circles of constant normalized
    resistance r = R / Z0 satisfy:
        (u - r / (r + 1))**2 + v**2 = (1 / (r + 1))**2

    Parameters
    ----------
    r : float
        Normalized resistance (r >= 0).
    num_points : int, optional
        Number of points along the circumference (default: 200).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (u, v) real and imaginary coordinates tracing the circle.
    """
    center_u = r / (r + 1.0)
    radius = 1.0 / (r + 1.0)
    theta = np.linspace(0.0, 2.0 * np.pi, num_points)
    u = center_u + radius * np.cos(theta)
    v = radius * np.sin(theta)
    return u, v


def reactance_arc(x, num_points=200):
    """
    Generate coordinates for a constant normalized reactance arc inside the unit circle.

    In the Smith Chart, curves of constant normalized reactance x = X / Z0
    correspond to normalized impedances z(r) = r + j*x for r in [0, inf).
    Transforming z to Gamma via Gamma = (z - 1) / (z + 1) traces the arc
    starting at the chart boundary (|Gamma| = 1 at r = 0) and terminating
    at the open-circuit point (Gamma = 1 + 0j as r -> inf).

    Parameters
    ----------
    x : float
        Normalized reactance (positive for inductive, negative for capacitive).
    num_points : int, optional
        Number of points along the arc (default: 200).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (u, v) real and imaginary coordinates bounded within |Gamma| <= 1.
    """
    if x == 0.0:
        # Real axis (pure resistance line, x=0)
        u = np.linspace(-1.0, 1.0, num_points)
        v = np.zeros_like(u)
        return u, v

    # Parameterize normalized resistance r from 0 to 100 on a smooth scale
    r = np.geomspace(1e-4, 100.0, num_points) - 1e-4
    z = r + 1j * x
    gamma = (z - 1.0) / (z + 1.0)
    return gamma.real, gamma.imag


def vswr_circle(vswr, num_points=200):
    """
    Generate coordinates for a constant VSWR circle centered at the origin.

    A circle of constant VSWR has radius |Gamma| = (VSWR - 1) / (VSWR + 1).

    Parameters
    ----------
    vswr : float
        Voltage Standing Wave Ratio (vswr >= 1.0).
    num_points : int, optional
        Number of points along the circumference (default: 200).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (u, v) real and imaginary coordinates tracing the circle.
    """
    if vswr < 1.0:
        vswr = 1.0
    gamma_radius = (vswr - 1.0) / (vswr + 1.0)
    theta = np.linspace(0.0, 2.0 * np.pi, num_points)
    u = gamma_radius * np.cos(theta)
    v = gamma_radius * np.sin(theta)
    return u, v
