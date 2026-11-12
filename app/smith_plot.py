"""M5: Matplotlib-based Smith chart visualization engine.

Standardized RF impedance Smith chart renderer:
    - Standard Smith chart impedance grid: constant-resistance circles and constant-reactance arcs
    - Reference boundaries: Unit circle (|Gamma| = 1.0) and Target Center (50 Ohm / Gamma = 0)
    - Target match circle: VSWR = 1.20 threshold (primary goal) and VSWR = 1.50 (secondary)
    - Ground Truth Physics Optimum target (Gamma* from grid search)
    - Antenna load disturbance marker (Z_L)
    - Current network input reflection marker (Z_in / Gamma_in) with color-coded match status
    - Trajectory trail visualizing human vs. AI convergence dynamics
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for thread safety
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is accessible
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physics.smith import (
    reactance_arc,
    resistance_circle,
    vswr_circle,
)

# Standard grid parameter sets
DEFAULT_R_CIRCLES = [0.2, 0.5, 1.0, 2.0, 5.0]
DEFAULT_X_ARCS = [0.2, 0.5, 1.0, 2.0, -0.2, -0.5, -1.0, -2.0]


def draw_smith_chart(
    current_gamma: complex | tuple[float, float],
    target_gamma: Optional[complex | tuple[float, float]] = None,
    trail: Optional[Sequence[complex | tuple[float, float]]] = None,
    gamma_load: Optional[complex | tuple[float, float]] = None,
    figsize: tuple[float, float] = (5.5, 5.5),
    dark_mode: bool = False,
    show_labels: bool = True,
) -> plt.Figure:
    """Render a professional engineering Smith chart figure.

    Parameters
    ----------
    current_gamma : complex or tuple of (float, float)
        Current input reflection coefficient Gamma_in = (Z_in - Z0) / (Z_in + Z0).
    target_gamma : complex or tuple of (float, float), optional
        Theoretical optimal reflection coefficient Gamma* from brute-force physics search.
    trail : sequence of complex or (float, float), optional
        Recent trajectory breadcrumbs showing the impedance tuning path.
    gamma_load : complex or tuple of (float, float), optional
        Antenna disturbance load reflection coefficient Gamma_load.
    figsize : tuple of (float, float), optional
        Figure dimensions in inches (default: (5.5, 5.5)).
    dark_mode : bool, optional
        Whether to style the chart with a dark slate background (default: False).
    show_labels : bool, optional
        Whether to display resistance and reactance grid text labels (default: True).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered Smith chart figure.
    """
    if dark_mode:
        bg_color = "#0e1117"
        grid_color = "#2a3447"
        r1_color = "#415a77"
        text_color = "#8b9bb4"
        vswr_12_color = "#22c55e"
        vswr_15_color = "#eab308"
        unit_circle_color = "#e0e1dd"
        center_color = "#00f5d4"
    else:
        bg_color = "#ffffff"
        grid_color = "#e2e8f0"
        r1_color = "#94a3b8"
        text_color = "#64748b"
        vswr_12_color = "#16a34a"
        vswr_15_color = "#ca8a04"
        unit_circle_color = "#1e293b"
        center_color = "#0284c7"

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"aspect": "equal"}, facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(-1.14, 1.14)
    ax.set_ylim(-1.14, 1.14)
    ax.axis("off")

    # 1. Background Grid: Constant Resistance Circles (r = R / Z0)
    for r in DEFAULT_R_CIRCLES:
        u_r, v_r = resistance_circle(r, num_points=180)
        line_col = r1_color if r == 1.0 else grid_color
        line_w = 1.3 if r == 1.0 else 0.7
        line_style = "-" if r == 1.0 else "--"
        ax.plot(u_r, v_r, color=line_col, linewidth=line_w, linestyle=line_style, zorder=2)

        if show_labels and r != 5.0:
            u_label = (r - 1.0) / (r + 1.0)
            ax.text(
                u_label,
                0.02,
                f"r={r:g}",
                color=text_color,
                fontsize=7,
                ha="center",
                va="bottom",
                zorder=3,
            )

    # 2. Background Grid: Constant Reactance Arcs (x = X / Z0)
    for x in DEFAULT_X_ARCS:
        u_x, v_x = reactance_arc(x, num_points=120)
        ax.plot(u_x, v_x, color=grid_color, linewidth=0.7, linestyle=":", zorder=2)

    # 3. Real Axis (Pure resistance line x = 0)
    ax.axhline(0, color=grid_color, linewidth=0.9, zorder=2)

    # 4. Outer Unit Boundary (|Gamma| = 1.0)
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color=unit_circle_color, linewidth=1.5, zorder=3)

    # 5. Target Match Circles: VSWR = 1.20 and VSWR = 1.50
    # Primary target circle: VSWR = 1.20 -> radius = (1.20 - 1) / (1.20 + 1) ~ 0.0909
    u_12, v_12 = vswr_circle(1.20, num_points=180)
    ax.plot(
        u_12,
        v_12,
        color=vswr_12_color,
        linewidth=1.3,
        linestyle="--",
        label="VSWR < 1.20 Target",
        zorder=4,
    )

    # Secondary acceptable circle: VSWR = 1.50 -> radius = (1.50 - 1) / (1.50 + 1) = 0.20
    u_15, v_15 = vswr_circle(1.50, num_points=180)
    ax.plot(
        u_15,
        v_15,
        color=vswr_15_color,
        linewidth=0.9,
        linestyle=":",
        label="VSWR < 1.50 Acceptable",
        zorder=4,
    )

    # 6. Target Center Point (Gamma = 0, Z0 = 50 Ohm)
    ax.scatter(
        [0.0],
        [0.0],
        marker="+",
        s=100,
        color=center_color,
        linewidth=1.8,
        label="Target (50 Ohm)",
        zorder=5,
    )

    # 7. Historical Breadcrumb Trail
    if trail and len(trail) > 1:
        pts = [complex(p) if isinstance(p, (complex, float, int)) else complex(p[0], p[1]) for p in trail]
        u_tr = [p.real for p in pts]
        v_tr = [p.imag for p in pts]
        ax.plot(
            u_tr,
            v_tr,
            color="#2563eb",
            linewidth=1.1,
            alpha=0.5,
            linestyle=":",
            zorder=4,
        )
        n_pts = len(pts)
        for idx, (tu, tv) in enumerate(zip(u_tr, v_tr)):
            alpha = 0.15 + 0.65 * (idx / max(1, n_pts - 1))
            size = 10 + 26 * (idx / max(1, n_pts - 1))
            ax.scatter([tu], [tv], color="#2563eb", s=size, alpha=alpha, zorder=4)

    # 8. Antenna Load Disturbance Marker (Z_L)
    if gamma_load is not None:
        g_l = complex(gamma_load) if isinstance(gamma_load, (complex, float, int)) else complex(gamma_load[0], gamma_load[1])
        ax.scatter(
            [g_l.real],
            [g_l.imag],
            marker="s",
            s=80,
            facecolor="#f97316",
            edgecolor="#1e293b",
            linewidth=1.2,
            label="Load Z_L",
            zorder=6,
        )

    # 9. Physics Optimal Target (Gamma*)
    if target_gamma is not None:
        g_t = complex(target_gamma) if isinstance(target_gamma, (complex, float, int)) else complex(target_gamma[0], target_gamma[1])
        ax.scatter(
            [g_t.real],
            [g_t.imag],
            marker="*",
            s=130,
            facecolor="#eab308",
            edgecolor="#1e293b",
            linewidth=1.0,
            label="Physics Optimum",
            zorder=6,
        )

    # 10. Active Input Reflection Dot (Z_in / Gamma_in)
    g_c = complex(current_gamma) if isinstance(current_gamma, (complex, float, int)) else complex(current_gamma[0], current_gamma[1])
    g_abs = abs(g_c)
    vswr = (1.0 + g_abs) / max(1e-9, 1.0 - g_abs) if g_abs < 0.9999 else 999.0

    if vswr <= 1.20:
        dot_color = "#16a34a"  # Green
        status_label = f"Z_in (Matched: {vswr:.2f})"
    elif vswr <= 1.50:
        dot_color = "#ca8a04"  # Amber
        status_label = f"Z_in (Near-Opt: {vswr:.2f})"
    else:
        dot_color = "#dc2626"  # Red
        status_label = f"Z_in (Mismatched: {vswr:.2f})"

    ax.scatter(
        [g_c.real],
        [g_c.imag],
        marker="o",
        s=190,
        facecolor=dot_color,
        alpha=0.3,
        zorder=7,
    )
    ax.scatter(
        [g_c.real],
        [g_c.imag],
        marker="o",
        s=85,
        facecolor=dot_color,
        edgecolor="#1e293b",
        linewidth=1.5,
        label=status_label,
        zorder=8,
    )

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=7.5,
        frameon=True,
        facecolor=bg_color,
        edgecolor=grid_color,
    )
    for text in leg.get_texts():
        text.set_color(text_color)

    fig.tight_layout()
    return fig


def create_smith_chart(
    gamma_in: complex | tuple[float, float],
    gamma_load: Optional[complex | tuple[float, float]] = None,
    trail: Optional[Sequence[complex | tuple[float, float]]] = None,
    vswr: Optional[float] = None,
    target_gamma: Optional[complex | tuple[float, float]] = None,
    figsize: tuple[float, float] = (5.5, 5.5),
    dark_mode: bool = False,
    show_labels: bool = True,
) -> plt.Figure:
    """Backward-compatible wrapper mapping to draw_smith_chart."""
    return draw_smith_chart(
        current_gamma=gamma_in,
        target_gamma=target_gamma,
        trail=trail,
        gamma_load=gamma_load,
        figsize=figsize,
        dark_mode=dark_mode,
        show_labels=show_labels,
    )
