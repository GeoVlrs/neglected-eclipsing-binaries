"""
================================================================================
AUDIT — DOES THE PIPELINE PERIOD FOLD THE TESS DATA? (PHASE-DISPERSION CHECK)
================================================================================
Summary:
    Every bundle is built by folding the cleaned TESS light curve with Script 03's
    refined period P_new (04Modeling.py). This audit asks, model-free, whether
    that period actually folds the data, by comparing three periods per star:

        P_new      — Script 03's refined period (what 04 / 05 use)
        P_kreiner  — Kreiner (2004) catalogue period
        P_best     — the period that minimises the phase dispersion of this
                     star's own TESS data (scan below)

    A wrong period makes successive cycles land at drifting phases. Across one
    ~27 d sector the drift is  |P_new − P_best| × span / P_best²  cycles; with
    150 phase bins one bin is 0.0067 cycles, so a drift of a few hundredths
    already smears the eclipses and can split a minimum into two offset copies
    (the "bumps" / "shoulders" seen in phoebe_final_*.png, Pipeline_Log Entry 76).

    Phase dispersion: fold at trial P into 100 bins, sum the within-bin variance
    (pooled, ddof=1 per bin), divide by the total variance. ~0 for a perfect
    fold, → 1 for a fold that carries no signal. The scan covers both P_new and
    P_kreiner with a 0.3 % margin, at a step that moves the last cycle by 0.005
    cycles, then refines around the best grid point.

    Per-segment check: the light curve is split at gaps > 1 d (TESS orbits /
    downlinks); each segment is folded at P_new with Script 03's T0_new and its
    primary-minimum phase is measured (parabola through the deepest 150-bin
    fold bins). With a correct period every segment agrees; with a wrong one
    they walk apart. The spread is taken modulo 0.5 (equal-depth minima can
    swap between segments).

Inputs:
    data/lightcurves/lc_<star>_clean.fits   — Script 02 output.
    data/timing/ephem_<star>.txt            — Script 03 output.

Outputs:
    Audits/period_fold_audit.csv             — one row per star.
    data/plots/audit_period/period_fold_<star>.png — fold at P_new vs P_best
        (150 bins, same binning as 04) for the stars named on the command line.
    Table printed to stdout, worst drift first.

Usage (from Code/, light: single process, ~30 s for the full sample):
    python Audits/audit_period_fold.py                    # table + CSV, all stars
    python Audits/audit_period_fold.py "V899 Her" "DN Cam" # + figures for these

Dependencies:
    pip install astropy numpy matplotlib
================================================================================
"""

import csv
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

# Audits/ sits one level below Code/, where paths.py lives.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths                                              # noqa: E402

# Kept local, not imported: 04Modeling.py runs on import (no __main__ guard).
TESS_BTJD_TO_BJD_OFFSET: float = 2_457_000.0
PHASE_BIN_COUNT: int = 150          # 04Modeling.PHASE_BIN_COUNT
DISPERSION_BINS: int = 100
SEGMENT_GAP_DAYS: float = 1.0
MIN_SEGMENT_SPAN_DAYS: float = 2.0  # a segment must cover a few cycles to fold

AUDIT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "period_fold_audit.csv")
PLOT_AUDIT_DIR = os.path.join(paths.PLOT_DIR, "audit_period")


def read_ephemeris(safe_name: str) -> dict:
    """Parse ephem_<star>.txt into floats (PERIOD_REVERTED kept as text)."""
    text = open(os.path.join(paths.TIMING_DIR, f"ephem_{safe_name}.txt"),
                encoding="utf-8").read()
    out = {}
    for key in ("T0_new", "P_new", "P_kreiner"):
        m = re.search(rf"^{key}\s*=\s*([-+0-9.eE]+)", text, re.MULTILINE)
        out[key] = float(m.group(1)) if m else np.nan
    m = re.search(r"^PERIOD_REVERTED\s*=\s*(\w+)", text, re.MULTILINE)
    out["reverted"] = m.group(1) if m else "n/a"
    return out


def read_light_curve(safe_name: str) -> tuple[np.ndarray, np.ndarray]:
    """BTJD time and median-normalised flux, finite points only."""
    with fits.open(os.path.join(paths.LIGHTCURVE_DIR, f"lc_{safe_name}_clean.fits")) as h:
        data = h[1].data
        t = np.asarray(data["TIME"], float)
        f = np.asarray(data["FLUX"], float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    return t, f / np.nanmedian(f)


def dispersion(t: np.ndarray, f: np.ndarray, period: float,
               n_bins: int = DISPERSION_BINS) -> float:
    """Pooled within-bin variance / total variance of the fold at `period`."""
    b = ((t / period) % 1.0 * n_bins).astype(int)
    n = np.bincount(b, minlength=n_bins)
    s = np.bincount(b, f, n_bins)
    s2 = np.bincount(b, f * f, n_bins)
    ok = n > 1
    within = np.sum(s2[ok] - s[ok] ** 2 / n[ok]) / np.sum(n[ok] - 1)
    return float(within / np.var(f))


def best_period(t: np.ndarray, f: np.ndarray, p_lo: float, p_hi: float) -> float:
    """Grid scan + two refinements; the step keeps the end-of-span phase error ≤ 0.005."""
    span = t.max() - t.min()
    p_mid = 0.5 * (p_lo + p_hi)
    step = 0.005 * p_mid ** 2 / span
    grid = np.arange(p_lo, p_hi + step, step)
    d = np.array([dispersion(t, f, p) for p in grid])
    p_best = grid[np.argmin(d)]
    for _ in range(2):
        grid = np.linspace(p_best - 2 * step, p_best + 2 * step, 81)
        step = grid[1] - grid[0]
        d = np.array([dispersion(t, f, p) for p in grid])
        p_best = grid[np.argmin(d)]
    return float(p_best)


def bin_fold(phase: np.ndarray, f: np.ndarray, n_bins: int = PHASE_BIN_COUNT):
    """Median per bin over [-0.5, 0.5), like 04's bin_phase_curve (no error floor)."""
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    med = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = idx == i
        if np.any(m):
            med[i] = np.median(f[m])
    ok = np.isfinite(med)
    return centres[ok], med[ok]


def fold(t_btjd: np.ndarray, period: float, t0_bjd: float) -> np.ndarray:
    """Phase in [-0.5, 0.5) with phase 0 at t0 (04's convention)."""
    return ((t_btjd + TESS_BTJD_TO_BJD_OFFSET - t0_bjd) / period + 0.5) % 1.0 - 0.5


def min_phase(phase: np.ndarray, f: np.ndarray) -> float:
    """Primary-minimum phase of a fold: parabola through the 7 bins around the deepest bin."""
    c, m = bin_fold(phase, f)
    if len(c) < 20:
        return np.nan
    k = int(np.argmin(m))
    sel = [(k + j) % len(c) for j in range(-3, 4)]
    x = c[sel] - c[k]
    x = (x + 0.5) % 1.0 - 0.5                     # unwrap across ±0.5
    a, b, _ = np.polyfit(x, m[sel], 2)
    if a <= 0:
        return float(c[k])
    return float(((c[k] - b / (2 * a)) + 0.5) % 1.0 - 0.5)


def amplitude_loss_pct(t, f, p_new, p_best, t0) -> float:
    """
    Light-curve amplitude (max − min of the 150-bin median fold) lost by folding
    at p_new instead of p_best, in percent. This is what the smeared fold cost
    the fit's input; it is the criterion of the Entry-78 keep / re-run decision
    (< 3 % kept, documented).
    """
    def amp(period):
        c, m = bin_fold(fold(t, period, t0), f)
        return float(m.max() - m.min())
    return 100.0 * (1.0 - amp(p_new) / amp(p_best))


def segments(t: np.ndarray) -> list[np.ndarray]:
    gaps = np.where(np.diff(t) > SEGMENT_GAP_DAYS)[0]
    return [s for s in np.split(np.arange(len(t)), gaps + 1)
            if len(s) > 1 and t[s[-1]] - t[s[0]] >= MIN_SEGMENT_SPAN_DAYS]


def audit_star(safe_name: str) -> dict:
    eph = read_ephemeris(safe_name)
    t, f = read_light_curve(safe_name)
    p_new, p_k = eph["P_new"], eph["P_kreiner"]
    span = float(t.max() - t.min())
    p_lo = min(p_new, p_k) * (1 - 0.003)
    p_hi = max(p_new, p_k) * (1 + 0.003)
    p_best = best_period(t, f, p_lo, p_hi)

    seg_phases = []
    for s in segments(t):
        seg_phases.append(min_phase(fold(t[s], p_new, eph["T0_new"]), f[s]))
    seg_phases = np.array([p for p in seg_phases if np.isfinite(p)])
    if len(seg_phases) > 1:
        # Spread of the per-segment minimum phases, taken modulo 0.5: in a
        # near-equal-minima system the deepest bin can be the primary in one
        # segment and the secondary in the next, which is not period drift.
        d = (seg_phases[:, None] - seg_phases[None, :] + 0.25) % 0.5 - 0.25
        seg_spread = float(np.max(np.abs(d)))
    else:
        seg_spread = np.nan

    return {
        "star": safe_name.replace("_", " "),
        "P_new": p_new, "P_kreiner": p_k, "P_best": p_best,
        "dP_new_s": (p_new - p_best) * 86400.0,
        "dP_kreiner_s": (p_k - p_best) * 86400.0,
        "span_d": span,
        "drift_new_cyc": abs(p_new - p_best) * span / p_best ** 2,
        "drift_kreiner_cyc": abs(p_k - p_best) * span / p_best ** 2,
        "disp_new": dispersion(t, f, p_new),
        "disp_kreiner": dispersion(t, f, p_k),
        "disp_best": dispersion(t, f, p_best),
        "n_segments": len(seg_phases),
        "segment_min_phase_spread": seg_spread,
        "segment_min_phases": ";".join(f"{p:+.4f}" for p in seg_phases),
        "period_reverted": eph["reverted"],
        "amp_loss_pct": amplitude_loss_pct(t, f, p_new, p_best, eph["T0_new"]),
        "_t": t, "_f": f, "_T0": eph["T0_new"],
    }


def plot_star(r: dict) -> str:
    """Two folds, 150 bins each: at P_new (as 04 builds it) and at P_best."""
    t, f = r["_t"], r["_f"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, period, label in ((axes[0], r["P_new"], "P_new (Script 03, used by 04/05)"),
                              (axes[1], r["P_best"], "P_best (phase-dispersion minimum)")):
        ph = fold(t, period, r["_T0"])
        # re-centre on the deepest bin, as 04's t0 anchor does
        c, m = bin_fold(ph, f)
        ph = (ph - c[np.argmin(m)] + 0.5) % 1.0 - 0.5
        c, m = bin_fold(ph, f)
        ax.plot(ph, f, ".", ms=1, color="0.75", alpha=0.4, rasterized=True)
        ax.plot(c, m, "o", ms=3, color="k")
        ax.set_title(f"{label}\nP = {period:.7f} d   dispersion = {dispersion(t, f, period):.3f}",
                     fontsize=10)
        ax.set_xlabel("Orbital phase")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Normalised flux")
    fig.suptitle(f"{r['star']} — fold check (drift at P_new: {r['drift_new_cyc']:.3f} cycles "
                 f"over {r['span_d']:.1f} d; one 150-bin bin = 0.0067)", fontsize=11)
    fig.tight_layout()
    os.makedirs(PLOT_AUDIT_DIR, exist_ok=True)
    out = os.path.join(PLOT_AUDIT_DIR, f"period_fold_{r['star'].replace(' ', '_')}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


COLUMNS = ["star", "P_new", "P_kreiner", "P_best", "dP_new_s", "dP_kreiner_s", "span_d",
           "drift_new_cyc", "drift_kreiner_cyc", "disp_new", "disp_kreiner", "disp_best",
           "n_segments", "segment_min_phase_spread", "segment_min_phases", "period_reverted",
           "amp_loss_pct"]


if __name__ == "__main__":
    # Run timing → Runs/run_stats/run_summary.txt (written at exit). Inside the
    # __main__ guard so that importing this module never starts a tracker.
    import runstats
    _run_tracker = runstats.RunTracker("audit_period_fold", catalogue=sys.argv[1:], per_star=False)

    plot_stars = {s.replace(" ", "_") for s in sys.argv[1:]}
    rows = []
    for ephem_path in sorted(glob.glob(os.path.join(paths.TIMING_DIR, "ephem_*.txt"))):
        safe = os.path.basename(ephem_path)[len("ephem_"):-len(".txt")]
        if not os.path.exists(os.path.join(paths.LIGHTCURVE_DIR, f"lc_{safe}_clean.fits")):
            continue
        try:
            r = audit_star(safe)
        except Exception as err:                          # one bad file must not stop the audit
            print(f"  {safe}: ERROR {err}")
            continue
        if safe in plot_stars:
            print(f"  plot: {plot_star(r)}")
        rows.append(r)

    rows.sort(key=lambda r: -r["drift_new_cyc"])
    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([f"{r[c]:.10g}" if isinstance(r[c], float) else r[c] for c in COLUMNS])

    print(f"\n{'star':<11} {'dP_new s':>9} {'dP_kr s':>8} {'drift':>7} {'disp_new':>9} "
          f"{'disp_best':>9} {'seg spread':>10} reverted")
    for r in rows:
        print(f"{r['star']:<11} {r['dP_new_s']:>+9.1f} {r['dP_kreiner_s']:>+8.1f} "
              f"{r['drift_new_cyc']:>7.3f} {r['disp_new']:>9.3f} {r['disp_best']:>9.3f} "
              f"{r['segment_min_phase_spread']:>10.3f} {r['period_reverted']}")
    n_bad = sum(r["drift_new_cyc"] > 0.02 for r in rows)
    n_kr_ok = sum(abs(r["dP_kreiner_s"]) < 5 for r in rows)
    print(f"\n{len(rows)} stars | drift at P_new > 0.02 cycles (3 bins): {n_bad} | "
          f"|P_kreiner - P_best| < 5 s: {n_kr_ok}")
    print(f"CSV: {AUDIT_CSV}")
