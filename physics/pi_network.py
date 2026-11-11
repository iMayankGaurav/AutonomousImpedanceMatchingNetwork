"""Pi-Network Impedance Matching Simulation Engine.

This module models an RF Pi-matching network placed between a 50 Ohm source
(transmitter) and an arbitrary complex load (antenna / disturbance).

Topology:
    Load (ZL = RL + j*XL)
         │
         ├───[Shunt Varactor C2 (V2)]─── GND
         │
        [Series Inductor L]
         │
         ├───[Shunt Varactor C1 (V1)]─── GND
         │
    Source (Z0 = 50 Ohms, Zin looking in)

Mathematical Chain of Transformations:
    1. Angular frequency:   w = 2 * pi * f
    2. Varactor values:     C1 = capacitance(V1), C2 = capacitance(V2)
    3. Load impedance:      Z_L = RL + j*XL
    4. Shunt C2 admittance: Y2 = (1 / Z_L) + j * w * C2
    5. Series L impedance:  Z3 = (1 / Y2) + j * w * L
    6. Shunt C1 admittance: Y_in = (1 / Z3) + j * w * C1
    7. Input impedance:     Z_in = 1 / Y_in
    8. Reflection coeff:    gamma = (Z_in - Z0) / (Z_in + Z0)
    9. Gamma magnitude:     gamma_abs = |gamma|
   10. Standing wave ratio: VSWR = (1 + gamma_abs) / (1 - gamma_abs)
"""

import numpy as np

try:
    from physics.varactor import capacitance
except ImportError:
    try:
        from .varactor import capacitance
    except ImportError:
        from varactor import capacitance

# Default system constants
F = 100e6       # Operating frequency (100 MHz)
Z0 = 50.0       # System reference / characteristic impedance (50 Ohms)
L = 150e-9      # Default series inductor value (150 nH)


def compute(RL, XL, V1, V2, L=150e-9, f=100e6):
    """
    Compute input impedance, reflection coefficient, and VSWR of the Pi-network.

    Simulates the cascading transformations across the Pi-network from the
    antenna load (RL + j*XL) back to the transmitter interface (Z_in).

    Parameters
    ----------
    RL : float
        Load resistance in Ohms.
    XL : float
        Load reactance in Ohms.
    V1 : float
        Control voltage for shunt varactor C1 (at source side) in Volts [0.0, 3.3].
    V2 : float
        Control voltage for shunt varactor C2 (at load side) in Volts [0.0, 3.3].
    L : float, optional
        Series inductance in Henries (default: 150e-9 H = 150 nH).
    f : float, optional
        Operating frequency in Hertz (default: 100e6 Hz = 100 MHz).

    Returns
    -------
    dict
        Dictionary containing:
        - "Zin": complex
            Input impedance looking into the Pi-network (Ohms).
        - "gamma": complex
            Complex reflection coefficient referenced to Z0 (50 Ohms).
        - "gamma_abs": float
            Magnitude of the reflection coefficient (|gamma|).
        - "vswr": float
            Voltage Standing Wave Ratio (VSWR >= 1.0).
    """
    # Step 1: Calculate angular frequency (omega)
    w = 2.0 * np.pi * f

    # Step 2: Convert varactor control voltages to capacitances in Farads
    C1 = capacitance(V1)
    C2 = capacitance(V2)

    # Step 3: Define complex load impedance (ZL = RL + j*XL)
    # Guard against complete short circuit to avoid ZeroDivisionError
    if RL == 0.0 and XL == 0.0:
        Z_L = complex(1e-12, 0.0)
    else:
        Z_L = complex(float(RL), float(XL))

    # Step 4: Shunt C2 admittance (Load in parallel with C2)
    # Y2 = Y_L + j * w * C2
    Y2 = (1.0 / Z_L) + complex(0.0, w * C2)
    if Y2 == 0.0:
        Y2 = complex(1e-12, 0.0)

    # Step 5: Series L impedance (Parallel combo of Load || C2 in series with L)
    # Z3 = Z2 + j * w * L
    Z3 = (1.0 / Y2) + complex(0.0, w * L)
    if Z3 == 0.0:
        Z3 = complex(1e-12, 0.0)

    # Step 6: Shunt C1 admittance (Upstream parallel combo in parallel with C1)
    # Y_in = 1 / Z3 + j * w * C1
    Y_in = (1.0 / Z3) + complex(0.0, w * C1)
    if Y_in == 0.0:
        Y_in = complex(1e-12, 0.0)

    # Step 7: Input impedance looking into the Pi-network from the source
    Z_in = 1.0 / Y_in

    # Step 8: Complex reflection coefficient (Gamma) relative to Z0 (50 Ohms)
    denom = Z_in + Z0
    if denom == 0.0:
        gamma = complex(1.0, 0.0)
    else:
        gamma = (Z_in - Z0) / denom

    # Step 9: Magnitude of the reflection coefficient (|Gamma|)
    gamma_abs = float(abs(gamma))

    # Step 10: Voltage Standing Wave Ratio (VSWR)
    # Handle division by zero when |Gamma| == 1 (total reflection / open / short)
    if gamma_abs >= 1.0 or np.isclose(gamma_abs, 1.0):
        vswr = float("inf")
    else:
        vswr = float((1.0 + gamma_abs) / (1.0 - gamma_abs))

    return {
        "Zin": Z_in,
        "gamma": gamma,
        "gamma_abs": gamma_abs,
        "vswr": vswr,
    }


def grid_search_match(RL, XL, grid_size=34, L=150e-9, f=100e6):
    """
    Perform a brute-force 2D grid search over (V1, V2) to find the best match.

    Useful as a baseline for Reinforcement Learning (M4) and for computing the
    theoretical matchable region of the network across disturbance loads.

    Parameters
    ----------
    RL : float
        Load resistance in Ohms.
    XL : float
        Load reactance in Ohms.
    grid_size : int, optional
        Number of voltage steps per axis across [0.0, 3.3]V (default: 34,
        giving ~0.1V resolution over ~1,156 evaluations).
    L : float, optional
        Series inductance in Henries (default: 150e-9 H).
    f : float, optional
        Operating frequency in Hertz (default: 100e6 Hz).

    Returns
    -------
    dict
        Dictionary containing:
        - "best_V1": float, optimal bias voltage for C1 (V)
        - "best_V2": float, optimal bias voltage for C2 (V)
        - "best_gamma_abs": float, lowest achieved reflection magnitude
        - "best_vswr": float, corresponding VSWR
        - "Zin": complex, input impedance at best match
        - "gamma": complex, reflection coefficient at best match
    """
    v_vals = np.linspace(0.0, 3.3, grid_size)
    V1_grid, V2_grid = np.meshgrid(v_vals, v_vals)
    C1_grid = capacitance(V1_grid)
    C2_grid = capacitance(V2_grid)
    w = 2.0 * np.pi * f

    if RL == 0.0 and XL == 0.0:
        ZL = complex(1e-12, 0.0)
    else:
        ZL = complex(float(RL), float(XL))

    Y2 = (1.0 / ZL) + 1j * w * C2_grid
    Z3 = (1.0 / Y2) + 1j * w * L
    Y_in = (1.0 / Z3) + 1j * w * C1_grid
    Z_in = 1.0 / Y_in
    gamma = (Z_in - Z0) / (Z_in + Z0)
    gamma_abs = np.abs(gamma)

    min_idx = np.unravel_index(np.argmin(gamma_abs), gamma_abs.shape)
    best_V1 = float(V1_grid[min_idx])
    best_V2 = float(V2_grid[min_idx])
    best_gamma_abs = float(gamma_abs[min_idx])

    if best_gamma_abs >= 1.0 or np.isclose(best_gamma_abs, 1.0):
        best_vswr = float("inf")
    else:
        best_vswr = float((1.0 + best_gamma_abs) / (1.0 - best_gamma_abs))

    return {
        "best_V1": best_V1,
        "best_V2": best_V2,
        "best_gamma_abs": best_gamma_abs,
        "best_vswr": best_vswr,
        "Zin": complex(Z_in[min_idx]),
        "gamma": complex(gamma[min_idx]),
    }
