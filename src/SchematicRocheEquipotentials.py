"""
================================================================================
SCHEMATIC — ROCHE EQUIPOTENTIALS IN THE ORBITAL PLANE (NOT A PIPELINE STAGE)
================================================================================
Summary:
    Draws the classical Roche-geometry diagram for Section 2.1: dimensionless
    equipotential contours of the co-rotating potential in the orbital plane,
    the inner critical surface through L1 (the Roche lobes), the outer critical
    surface through L2, and the three collinear Lagrangian points. Purely
    analytic — no data, no fitted model, and deliberately no PHOEBE or Mayavi
    dependency, so it is cheap to regenerate.

    Complements SchematicRocheGeometry.py, which renders the same concepts as
    3D meshes (detached vs. contact). This script supplies the *definitions*
    (lobe, L1/L2/L3, critical equipotentials); that one supplies the
    morphologies.

    Mass ratio defaults to q = 0.5, matching the value held fixed for contact
    systems throughout this thesis, so the figure is consistent with the
    results rather than a generic textbook value.

Outputs:
    <PLOT_THESIS_DIR>/roche_schematic_equipotentials.png

Dependencies:
    pip install numpy scipy matplotlib

References:
    Kopal, Z. (1959). Close Binary Systems. Chapman & Hall.
    Kallrath, J., & Milone, E. F. (2009). Eclipsing Binary Stars, 2nd ed.
    Eggleton, P. P. (1983). ApJ 268, 368 — Roche-lobe radius approximation.
================================================================================
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import paths
paths.ensure_dirs()

MASS_RATIO: float = 0.5           # q = M2/M1, the value fixed for contact fits
GRID_N: int = 1200                # contour resolution
FIGURE_SIZE = (9.0, 5.4)


def roche_potential(x, y, q: float):
    """
    Dimensionless Kopal potential in the co-rotating frame, separation a = 1,
    M1 at the origin and M2 at (1, 0). Larger values lie closer to a star.
    """
    r1 = np.hypot(x, y)
    r2 = np.hypot(x - 1.0, y)
    x_com = q / (1.0 + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1.0 / r1 + q / r2
                + 0.5 * (1.0 + q) * ((x - x_com) ** 2 + y ** 2))


def _dpotential_dx(x: float, q: float) -> float:
    """d(potential)/dx along y = 0, used to locate the collinear Lagrange points."""
    r1c = abs(x) ** 3
    r2c = abs(x - 1.0) ** 3
    return (-x / r1c - q * (x - 1.0) / r2c
            + (1.0 + q) * (x - q / (1.0 + q)))


def lagrange_points(q: float, eps: float = 1e-7) -> dict:
    """L1 between the stars, L2 beyond the secondary, L3 beyond the primary."""
    return {
        "L1": brentq(_dpotential_dx, eps, 1.0 - eps, args=(q,)),
        "L2": brentq(_dpotential_dx, 1.0 + eps, 4.0, args=(q,)),
        "L3": brentq(_dpotential_dx, -4.0, -eps, args=(q,)),
    }


def draw(q: float = MASS_RATIO, out_name: str = "roche_schematic_equipotentials.png") -> str:
    lp = lagrange_points(q)
    omega_l1 = roche_potential(lp["L1"], 0.0, q)
    omega_l2 = roche_potential(lp["L2"], 0.0, q)
    x_com = q / (1.0 + q)

    # Kept tight around the system: at large radius the centrifugal term brings
    # the same two critical levels back as far-field branches, which are correct
    # but read as stray lines in a figure meant to define the lobes.
    x = np.linspace(-0.95, 1.80, GRID_N)
    y = np.linspace(-0.82, 0.82, GRID_N)
    X, Y = np.meshgrid(x, y)
    Z = roche_potential(X, Y, q)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Ordinary equipotentials: a few inside the lobes, a few enclosing both.
    inner = omega_l1 * np.array([1.45, 1.22, 1.08])
    outer = np.array([omega_l2 - 0.18, omega_l2 - 0.40])
    ax.contour(X, Y, Z, levels=np.sort(np.concatenate([outer, inner])),
               colors="0.62", linewidths=0.8, linestyles="-")

    # The two critical surfaces carry the physics, so they are emphasised.
    ax.contour(X, Y, Z, levels=[omega_l1], colors="crimson", linewidths=2.0)
    ax.contour(X, Y, Z, levels=[omega_l2], colors="steelblue",
               linewidths=1.6, linestyles="--")

    for xl, label, va in ((lp["L3"], r"$L_3$", "bottom"),
                          (lp["L1"], r"$L_1$", "bottom"),
                          (lp["L2"], r"$L_2$", "bottom")):
        ax.plot(xl, 0.0, "k+", ms=9, mew=1.6)
        ax.annotate(label, (xl, 0.0), textcoords="offset points",
                    xytext=(0, 7), ha="center", va=va, fontsize=11)

    ax.plot(0.0, 0.0, "o", color="0.15", ms=7)
    ax.plot(1.0, 0.0, "o", color="0.15", ms=6)
    ax.annotate(r"$M_1$", (0.0, 0.0), textcoords="offset points",
                xytext=(-4, -16), ha="center", fontsize=11)
    ax.annotate(r"$M_2$", (1.0, 0.0), textcoords="offset points",
                xytext=(4, -16), ha="center", fontsize=11)
    ax.plot(x_com, 0.0, "x", color="0.35", ms=7, mew=1.4)
    ax.annotate("centre of mass", (x_com, 0.0), textcoords="offset points",
                xytext=(0, -20), ha="center", fontsize=8.5, color="0.35")

    handles = [
        plt.Line2D([], [], color="crimson", lw=2.0,
                   label=r"inner critical surface (Roche lobes, through $L_1$)"),
        plt.Line2D([], [], color="steelblue", lw=1.6, ls="--",
                   label=r"outer critical surface (through $L_2$)"),
        plt.Line2D([], [], color="0.62", lw=0.8, label="other equipotentials"),
    ]
    ax.legend(handles=handles, loc="lower center", fontsize=8.0, framealpha=0.92, ncol=1)

    ax.set_title(f"Roche equipotentials in the orbital plane  ($q = M_2/M_1 = {q}$)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("$x$  (orbital separation $a = 1$)")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(paths.PLOT_THESIS_DIR, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path, lp, omega_l1, omega_l2


if __name__ == "__main__":
    # Run timing → Runs/run_stats/run_summary.txt (written at exit). Inside the
    # __main__ guard so that importing this module never starts a tracker.
    import sys as _rs_sys, os as _rs_os
    _rs_sys.path.insert(0, _rs_os.path.dirname(_rs_os.path.abspath(__file__)))
    import runstats
    _run_tracker = runstats.RunTracker("SchematicRocheEquipotentials", catalogue=_rs_sys.argv[1:],
                                       per_star=False)
    path, lp, o1, o2 = draw()
    print(f"q = {MASS_RATIO}")
    for k in ("L3", "L1", "L2"):
        print(f"  {k} at x = {lp[k]:+.5f}")
    print(f"  Omega(L1) = {o1:.5f}")
    print(f"  Omega(L2) = {o2:.5f}")
    print(f"wrote {path}")
