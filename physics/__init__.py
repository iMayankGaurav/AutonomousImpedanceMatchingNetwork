"""M3: RF Physics Engine Package.

Provides core simulation components:
- varactor: Varactor diode capacitance-voltage C(V) model.
- pi_network: Pi-matching network impedance, Gamma, and VSWR calculations.
- smith: Smith chart transformations (Gamma <-> Impedance) and geometry.
"""

from .varactor import capacitance
from .pi_network import compute, grid_search_match, F, Z0, L
from .smith import (
    impedance_to_gamma,
    gamma_to_impedance,
    resistance_circle,
    reactance_arc,
    vswr_circle,
)

__all__ = [
    "capacitance",
    "compute",
    "grid_search_match",
    "F",
    "Z0",
    "L",
    "impedance_to_gamma",
    "gamma_to_impedance",
    "resistance_circle",
    "reactance_arc",
    "vswr_circle",
]
