"""
================================================================================
SCRIPT 03 — ORBITAL TIMING ANALYSIS & O–C DIAGRAM GENERATION
================================================================================
Summary:
    Extracts precise modern Times of Minimum (ToM) from TESS light curves and
    quantifies the accumulated orbital period drift relative to the historical
    Kreiner (2004) ephemeris. Two ToM algorithms are implemented in a cascade:

        1. Kwee–van Woerden (KvW): model-independent geometric method that
           identifies the centre of eclipse symmetry by minimising the
           residual sum of squares between the ascending and descending
           branches, reflected over a FIXED symmetric range about each trial
           centre. Preferred for well-sampled eclipses.

        2. Polynomial fallback: fits a parabola to the eclipse bottom (centred
           times) and reports its vertex. Used when the KvW window has a gap
           or the KvW minimum is not bracketed.

    A refined linear ephemeris (T0_new, P_new) is derived by weighted
    least-squares fitting of all accepted ToM measurements against epoch
    numbers counted from the Kreiner (2004) baseline at HALF-cycle resolution
    (primary and secondary minima both used, with a separate zero-point for
    the secondaries). A phase-dispersion check then compares the fold at
    P_new with the fold at the Kreiner period and reverts to Kreiner if the
    refined period folds the data worse.

    History (Pipeline_Log Entry 76): before 2026-10 the KvW residual sum was
    unnormalised over a shrinking mirrored overlap (minimum pulled to the
    window edge), its σ parabola was fitted on absolute BJD (numerically
    unresolved), and every ToM was rounded to an INTEGER epoch. P_new was
    therefore off by 10–510 s for 57 of 60 stars. The replica of that code in
    Audits/audit_tom_estimator.py documents it.

    A three-panel O–C diagram is produced for each target:
        Panel (a) — Global drift over the ~20-year baseline since Kreiner (2004).
        Panel (b) — Residuals against the newly fitted ephemeris.
        Panel (c) — TESS-era zoom, revealing all individual eclipse measurements.

Inputs:
    neglected_targets.csv  — Target list from Script 01.
    lc_<star>_clean.fits   — Detrended light curves from Script 02.

Outputs:
    tominima_<star>.csv    — All accepted ToM measurements with uncertainties.
    ephem_<star>.txt       — Refined T0 (BJD) and period for Script 04.
    plots/oc_<star>.png    — Three-panel O–C diagram.

Dependencies:
    pip install lightkurve numpy scipy matplotlib astropy astroquery

References:
    Kreiner (2004), Acta Astronomica 54, 207 — historical ephemerides.
    Kwee & van Woerden (1956), BAN 12, 327 — ToM extraction algorithm.
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import csv
import os
import sys
import warnings
from typing import Optional

# ── Scientific libraries ──────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal  import argrelmin
from scipy.ndimage import uniform_filter1d
import lightkurve as lk
from astroquery.vizier import Vizier

warnings.filterwarnings("ignore")

import paths
paths.ensure_dirs()

# Run timing / statistics → Runs/run_stats/ (summary printed and saved at exit,
# one CSV row per star). See runstats.py.
import runstats
run_tracker = runstats.RunTracker("03OCAnalysis", catalogue=sys.argv[1:],
                                  columns=["n_minima", "n_toms", "period_refined", "period_err"])

# TESS mission time offset: BTJD + offset = BJD_TDB.
# Historical ephemerides (Kreiner 2004) are in HJD(UTC). HJD(UTC) and BJD(TDB)
# differ by ~65–69 s (mostly the TDB–UTC clock offset). Kreiner's T0 only sets
# the epoch COUNT here (an absolute O–C at the minute level is not quoted), so
# the offset is noted rather than corrected.
TESS_BTJD_TO_BJD_OFFSET: float = 2_457_000.0

# Maximum acceptable formal uncertainty for a ToM measurement (days).
# 0.015 d ≈ 21 min. With the corrected KvW, TESS ToM errors are ~10–60 s, so
# this only rejects pathological minima (partial eclipses, gaps).
MAX_TOM_UNCERTAINTY_DAYS: float = 0.015

# KvW reflection: trial centres within ±KVW_REFLECT_HALF_FRAC·P of the candidate
# and branches reflected over a fixed |dt| ≤ KVW_REFLECT_HALF_FRAC·P. 0.06 P
# covers the steep part of a contact-binary minimum without reaching the maxima.
KVW_REFLECT_HALF_FRAC: float = 0.06
KVW_MIN_POINTS_PER_BRANCH: int = 8

# Smoothing before candidate detection, in TIME (not samples): 30 min, whatever
# the cadence (120 s, 200 s or 600 s).
SMOOTH_MINUTES: float = 30.0

# Phase-dispersion self-check (same statistic as Audits/audit_period_fold.py).
DISPERSION_BINS: int = 100
# Revert to Kreiner if the refined period's fold is more than 1 % more dispersed.
# With a working ToM estimator the two periods agree to a few seconds on clean
# stars, so reverting costs nothing there; a looser 5 % let BW Dra (timings
# pulled by its blended neighbour BV Dra, Entry 76) keep a period 2.4 % worse.
DISPERSION_REVERT_TOLERANCE: float = 1.01


# =============================================================================
# SECTION 1 — KWEE–VAN WOERDEN ALGORITHM
# =============================================================================

def compute_tom_kwee_van_woerden(
    time_bjd: np.ndarray,
    flux: np.ndarray,
    t_trial: float,
    reflect_half: float,
    cadence_days: float,
) -> tuple[float, float]:
    """
    Determine the Time of Minimum (ToM) via the Kwee–van Woerden (1956) method.

    For each trial centre t_c within ±reflect_half of t_trial, the light curve
    is interpolated at t_c + dt and t_c − dt over a FIXED grid of offsets
    0 < dt ≤ reflect_half, and the residual S(t_c) = Σ [f(t_c+dt) − f(t_c−dt)]²
    is formed. Because the reflection range is the same for every t_c, S is
    comparable across trial centres and its minimum marks the centre of
    symmetry. A parabola S(x) = A x² + B x + C is fitted around the minimum on
    CENTRED times x = t − t_k (on absolute BJD the quadratic term is not
    numerically resolvable), and the ToM is its vertex.

    Uncertainty (Kwee & van Woerden 1956):
        σ² = (4AC − B²) / (4A² (Z − 1)),
    with Z the number of independent reflected pairs (≈ points per branch).

    Raises
    ------
    ValueError:
        If the window has a gap (> 3 cadences) or is not fully covered, if too
        few points fall on each branch, or if the S minimum is not bracketed
        inside the trial range (eclipse centre further than reflect_half from
        the candidate) or the parabola is not convex.
    """
    w = 2.0 * reflect_half                   # data needed: t_c ± reflect_half, t_c ∈ t_trial ± reflect_half
    sel = np.abs(time_bjd - t_trial) <= w
    t, f = time_bjd[sel], flux[sel]
    n_ref = max(KVW_MIN_POINTS_PER_BRANCH, int(reflect_half / cadence_days))
    if len(t) < 2 * n_ref:
        raise ValueError("too few points in the KvW window")
    if np.max(np.diff(t)) > 3.0 * cadence_days:
        raise ValueError("gap inside the KvW window")
    if t.min() > t_trial - w + 2 * cadence_days or t.max() < t_trial + w - 2 * cadence_days:
        raise ValueError("KvW window not fully covered")

    offsets = np.linspace(cadence_days, reflect_half, n_ref)
    centres = np.linspace(t_trial - reflect_half, t_trial + reflect_half, 241)
    s = np.array([np.sum((np.interp(tc + offsets, t, f) - np.interp(tc - offsets, t, f)) ** 2)
                  for tc in centres])
    k = int(np.argmin(s))
    if k < 5 or k > len(centres) - 6:
        raise ValueError("KvW minimum not bracketed inside the trial range")
    x = centres[k - 5:k + 6] - centres[k]
    a, b, c = np.polyfit(x, s[k - 5:k + 6], 2)
    if a <= 0:
        raise ValueError("KvW residual parabola not convex")
    t_minimum = centres[k] - b / (2.0 * a)
    variance = (4.0 * a * c - b * b) / (4.0 * a * a * (n_ref - 1))
    t_uncertainty = float(np.sqrt(variance)) if variance > 0 else float(cadence_days)
    return float(t_minimum), t_uncertainty


# =============================================================================
# SECTION 2 — POLYNOMIAL FALLBACK
# =============================================================================

def compute_tom_polynomial_fallback(
    time_bjd: np.ndarray,
    flux: np.ndarray,
    t_trial: float,
    half_window: float,
) -> tuple[float, float]:
    """
    Determine the ToM from a parabola fitted to the eclipse bottom.

    Used when KvW raises (gap in its window, minimum not bracketed). The
    parabola is fitted on times centred on t_trial within ±half_window; the
    ToM is its vertex −b/2a and σ comes from propagating the fit covariance
    (scaled by the residual scatter) through the vertex formula.

    Raises
    ------
    ValueError:
        If fewer than 8 points fall in the window, the parabola is not convex,
        or the vertex lies outside the window.
    """
    sel = np.abs(time_bjd - t_trial) <= half_window
    x, f = time_bjd[sel] - t_trial, flux[sel]
    if len(x) < 8:
        raise ValueError("insufficient points for the parabola fallback")
    (a, b, c), cov = np.polyfit(x, f, 2, cov=True)
    if a <= 0:
        raise ValueError("fallback parabola not convex")
    x_min = -b / (2.0 * a)
    if abs(x_min) > half_window:
        raise ValueError("fallback vertex outside the window")
    # σ(x_min)² = J Σ Jᵀ with J = ∂x_min/∂(a, b) = (b/(2a²), −1/(2a))
    jac = np.array([b / (2.0 * a * a), -1.0 / (2.0 * a)])
    var = float(jac @ cov[:2, :2] @ jac)
    return float(t_trial + x_min), float(np.sqrt(var)) if var > 0 else float(np.nan)


# =============================================================================
# SECTION 3 — O–C RESIDUAL CALCULATION
# =============================================================================

def compute_oc_residuals(
    t_observed_bjd: np.ndarray,
    t_observed_err: np.ndarray,
    t0_kreiner: float,
    period_kreiner: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute O–C (Observed minus Calculated) residuals against Kreiner (2004).

    Epochs are assigned at HALF-cycle resolution,
        E = round(2 (T_obs − T0_kreiner) / P_kreiner) / 2,
    because both primary (integer E) and secondary (half-integer E) minima are
    timed. (Rounding every ToM to an integer — the pre-2026-10 behaviour —
    put each secondary ±P/2 off and biased the refined period; Entry 76.)

    Returns
    -------
    epoch_numbers:
        Epoch E (float; integer for primaries, half-integer for secondaries).
    oc_days:
        O–C residuals in days (T_obs − T0_kreiner − P_kreiner·E).
    oc_uncertainties:
        Per-ToM uncertainties (days).
    is_secondary:
        True where E is half-integer.
    """
    # Kreiner's ephemeris extrapolated ~20 000 cycles can land the TESS minima
    # anywhere up to ±P/4 off; rounding about the MEAN offset (circular mean in
    # half-cycle units) keeps every ToM on the same side of the rounding boundary.
    frac2 = 2.0 * (t_observed_bjd - t0_kreiner) / period_kreiner
    angle = 2.0 * np.pi * frac2
    centre = np.arctan2(np.mean(np.sin(angle)), np.mean(np.cos(angle))) / (2.0 * np.pi)
    epoch_numbers = np.round(frac2 - centre) / 2.0
    is_secondary  = (np.round(2.0 * epoch_numbers).astype(int) % 2) == 1
    t_calculated  = t0_kreiner + period_kreiner * epoch_numbers
    oc_days       = t_observed_bjd - t_calculated
    return epoch_numbers, oc_days, t_observed_err, is_secondary


# =============================================================================
# SECTION 4 — WEIGHTED LEAST-SQUARES EPHEMERIS REFINEMENT
# =============================================================================

def refine_linear_ephemeris(
    epoch_numbers: np.ndarray,
    t_observed_bjd: np.ndarray,
    t_observed_err: np.ndarray,
    is_secondary: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Fit a refined linear ephemeris by weighted least squares:
        T_min(E) = T0_new + P_new × E + δ·[secondary]

    δ is a separate zero-point for the secondaries (non-zero for eccentric
    orbits or asymmetric minima), so secondaries help pin P without being
    forced to sit exactly at phase 0.5. When there are no secondaries δ is
    dropped. Epochs are centred before the fit (E can exceed 20 000 cycles),
    and the covariance is scaled by the reduced χ² of the fit, so the errors
    reflect the actual scatter of the ToMs, not only their formal σ.

    Returns
    -------
    t0_new, period_new, t0_err, period_err, secondary_offset_days
        T0_new refers to E = 0 (the Kreiner epoch count), as before.
    """
    e_ref = float(np.round(np.mean(epoch_numbers)))
    x = epoch_numbers - e_ref
    use_sec = bool(np.any(is_secondary)) and bool(np.any(~is_secondary))
    cols = [np.ones_like(x), x] + ([is_secondary.astype(float)] if use_sec else [])
    design = np.vstack(cols).T
    w = 1.0 / np.square(np.where(t_observed_err > 0, t_observed_err, np.nanmedian(t_observed_err)))
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(design * sw[:, None], t_observed_bjd * sw, rcond=None)
    resid = t_observed_bjd - design @ coef
    dof = max(len(x) - design.shape[1], 1)
    chi2_red = float(np.sum(w * resid ** 2) / dof)
    cov = np.linalg.inv(design.T @ (design * w[:, None])) * max(chi2_red, 1.0)
    t_ref, period_new = coef[0], coef[1]
    t0_new = t_ref - period_new * e_ref                     # back to E = 0
    period_err = float(np.sqrt(cov[1, 1]))
    t0_err = float(np.sqrt(cov[0, 0] + (e_ref ** 2) * cov[1, 1] - 2 * e_ref * cov[0, 1]))
    sec_offset = float(coef[2]) if use_sec else 0.0
    return float(t0_new), float(period_new), t0_err, period_err, sec_offset


def phase_dispersion(
    time_bjd: np.ndarray,
    flux: np.ndarray,
    period: float,
    n_bins: int = DISPERSION_BINS,
) -> float:
    """
    Pooled within-bin variance of the fold at `period` divided by the total
    variance (~0 for a perfect fold, → 1 for a fold carrying no signal).

    The correct period folds every cycle onto the same phase; a wrong one
    smears successive cycles. Unlike a peak-to-trough depth, which barely
    changes when a contact binary's smooth curve is smeared by ~0.1 cycle,
    the dispersion responds to ANY misalignment. Same statistic as
    Audits/audit_period_fold.py, used by the Section-6D self-check.
    """
    b = ((time_bjd / period) % 1.0 * n_bins).astype(int)
    n = np.bincount(b, minlength=n_bins)
    s = np.bincount(b, flux, n_bins)
    s2 = np.bincount(b, flux * flux, n_bins)
    ok = n > 1
    within = np.sum(s2[ok] - s[ok] ** 2 / n[ok]) / np.sum(n[ok] - 1)
    return float(within / np.var(flux))


def best_dispersion_period(
    time_bjd: np.ndarray,
    flux: np.ndarray,
    p_lo: float,
    p_hi: float,
) -> float:
    """Phase-dispersion minimum over [p_lo, p_hi] (cross-check, written to the ephem file)."""
    span = time_bjd.max() - time_bjd.min()
    step = 0.005 * (0.5 * (p_lo + p_hi)) ** 2 / span        # ≤ 0.005-cycle end-of-span error
    grid = np.arange(p_lo, p_hi + step, step)
    p_best = grid[int(np.argmin([phase_dispersion(time_bjd, flux, p) for p in grid]))]
    for _ in range(2):
        grid = np.linspace(p_best - 2 * step, p_best + 2 * step, 81)
        step = grid[1] - grid[0]
        p_best = grid[int(np.argmin([phase_dispersion(time_bjd, flux, p) for p in grid]))]
    return float(p_best)


# =============================================================================
# SECTION 5 — KREINER (2004) HISTORICAL BASELINE
# =============================================================================

print("=" * 70)
print("STEP 1: Loading Kreiner (2004) historical ephemerides from VizieR...")

historical_ephemerides: dict[str, tuple[float, float]] = {}
try:
    Vizier.ROW_LIMIT = 5000
    kreiner_result   = Vizier.get_catalogs("J/AcA/54/207")
    kreiner_table    = kreiner_result[0]
    for row in kreiner_table:
        star_name_clean = " ".join(str(row["Name"]).split())  # normalise whitespace
        historical_ephemerides[star_name_clean] = (float(row["M0"]), float(row["Per"]))
    print(f"  Loaded {len(historical_ephemerides)} historical ephemerides.")
except Exception as fetch_error:
    print(f"  ERROR fetching Kreiner database: {fetch_error}")


# =============================================================================
# SECTION 6 — MAIN PROCESSING LOOP
# =============================================================================

# Optional positional argument selects the target catalogue, so a single batch
# (e.g. Targets/neglected_targets_batch1.csv) can be processed in isolation.
INPUT_CATALOG: str = sys.argv[1] if len(sys.argv) > 1 else "neglected_targets.csv"

target_list: list[dict] = []
try:
    with open(INPUT_CATALOG, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            target_list.append({"name": row["name"], "period": float(row["period"])})
except FileNotFoundError:
    print(f"ERROR: {INPUT_CATALOG} not found. Run Script 01 first or check the path.")
    raise SystemExit(1)

for target in target_list:
    star_name      = target["name"]
    orbital_period = target["period"]
    safe_name      = star_name.replace(" ", "_")
    run_tracker.star_start(star_name)

    print("\n" + "=" * 70)
    print(f"O–C ANALYSIS: {star_name}")
    print("=" * 70)

    lc_fits_path = os.path.join(paths.LIGHTCURVE_DIR, f"lc_{safe_name}_clean.fits")
    if not os.path.exists(lc_fits_path):
        print(f"  WARNING: {lc_fits_path} not found. Skipping.")
        run_tracker.star_end(star_name, "skipped", reason="clean light curve not found")
        continue

    lc        = lk.read(lc_fits_path)
    time_btjd = lc.time.value
    flux      = lc.flux.value
    # Convert BTJD to BJD for alignment with historical HJD-based ephemerides.
    # The BJD–HJD offset (<4 s for solar-system targets) is absorbed into T0_new.
    time_bjd  = time_btjd + TESS_BTJD_TO_BJD_OFFSET

    print(f"  Loaded {len(lc)} data points.")
    print(f"  Time range: BTJD {time_btjd.min():.2f}–{time_btjd.max():.2f}  "
          f"(BJD {time_bjd.min():.2f}–{time_bjd.max():.2f})")

    # ── 6A: ECLIPSE MINIMUM DETECTION ─────────────────────────────────────────
    print("\n  [6A] Identifying eclipse minima candidates...")

    # Smooth over SMOOTH_MINUTES of TIME before local-minimum detection (a fixed
    # sample count would mean 30 min at 120 s cadence but 2.5 h at 600 s).
    cadence_days  = float(np.median(np.diff(time_btjd)))
    smooth_points = max(3, int(round(SMOOTH_MINUTES / 1440.0 / cadence_days)))
    flux_smoothed = uniform_filter1d(flux, size=smooth_points)

    # Set the argrelmin order so that two candidate minima must be separated
    # by at least 20% of one orbital period — preventing double-detection of
    # the same eclipse bottom in noisy data.
    points_per_period  = orbital_period / cadence_days
    min_separation_pts = max(5, int(points_per_period * 0.2))

    candidate_min_indices = argrelmin(flux_smoothed, order=min_separation_pts)[0]
    print(f"  Found {len(candidate_min_indices)} candidate minima "
          f"(separation order = {min_separation_pts} pts, smoothing {smooth_points} pts).")

    if len(candidate_min_indices) == 0:
        print("  ERROR: No minima detected. Check light curve quality.")
        run_tracker.star_end(star_name, "failed", reason="no minima detected")
        continue

    # ── 6B: PRECISE ToM EXTRACTION ────────────────────────────────────────────
    print("\n  [6B] Computing precise Times of Minimum...")

    reflect_half = orbital_period * KVW_REFLECT_HALF_FRAC
    accepted_toms: list[tuple[float, float]] = []
    n_fallback = 0

    for min_idx in candidate_min_indices:
        t_trial_bjd = time_bjd[min_idx]
        try:
            t_min, t_err = compute_tom_kwee_van_woerden(
                time_bjd, flux, t_trial_bjd, reflect_half=reflect_half, cadence_days=cadence_days,
            )
            method = "KvW"
        except ValueError as kvw_err:
            try:
                t_min, t_err = compute_tom_polynomial_fallback(
                    time_bjd, flux, t_trial_bjd, half_window=reflect_half,
                )
                method = f"poly ({kvw_err})"
                n_fallback += 1
            except ValueError as poly_err:
                print(f"    Skipped minimum at BJD {t_trial_bjd:.4f}: KvW: {kvw_err}; poly: {poly_err}")
                continue

        if np.isfinite(t_err) and t_err <= MAX_TOM_UNCERTAINTY_DAYS:
            accepted_toms.append((t_min, t_err))
            print(f"    ToM = {t_min:.6f} BJD  ±{t_err:.6f} d  ({method})")
        else:
            print(f"    Rejected: ToM at {t_min:.4f} BJD  "
                  f"uncertainty ±{t_err:.4f} d > {MAX_TOM_UNCERTAINTY_DAYS} d")

    if len(accepted_toms) < 3:
        print("  ERROR: fewer than 3 ToM measurements passed the quality filter. Skipping.")
        run_tracker.star_end(star_name, "failed", reason="fewer than 3 ToMs passed quality filter",
                             n_minima=len(candidate_min_indices))
        continue

    t_obs_array = np.array([t for t, _ in accepted_toms])
    t_err_array = np.array([e for _, e in accepted_toms])

    # ── 6C: O–C RESIDUALS ─────────────────────────────────────────────────────
    print("\n  [6C] Computing O–C residuals vs. Kreiner (2004) baseline...")

    has_kreiner = star_name in historical_ephemerides
    if has_kreiner:
        t0_kreiner, period_kreiner = historical_ephemerides[star_name]
        print(f"    Kreiner baseline: T0 = {t0_kreiner},  P = {period_kreiner}")
    else:
        # No catalogue ephemeris: count epochs from the first measured ToM at the
        # catalogue period. The ephem file labels this honestly (not "Kreiner").
        print(f"  WARNING: No Kreiner entry for {star_name}. Using first ToM as T0 "
              f"and the catalogue period.")
        t0_kreiner     = t_obs_array[0]
        period_kreiner = orbital_period

    epoch_numbers, oc_days, oc_uncertainties, is_secondary = compute_oc_residuals(
        t_obs_array, t_err_array, t0_kreiner, period_kreiner,
    )
    print(f"    {int(np.sum(~is_secondary))} primary + {int(np.sum(is_secondary))} secondary minima")

    # ── 6D: EPHEMERIS REFINEMENT ──────────────────────────────────────────────
    print("\n  [6D] Fitting refined linear ephemeris...")

    t0_new, period_new, t0_err, period_err, secondary_offset = refine_linear_ephemeris(
        epoch_numbers, t_obs_array, t_err_array, is_secondary,
    )
    print(f"  Refined ephemeris (BJD):")
    print(f"    T_min = {t0_new:.6f} (±{t0_err:.6f}) "
          f"+ {period_new:.8f} (±{period_err:.8f}) × E   "
          f"[secondary offset {secondary_offset * 86400:+.1f} s]")

    # ── 6D-check: PERIOD FOLD-QUALITY SELF-CHECK ──────────────────────────────
    # A refined period that folds the light curve WORSE than the Kreiner baseline
    # means the timing fit went wrong (aliased / mis-counted epochs, distorted
    # minima). The check compares PHASE DISPERSION at the two periods: the
    # former peak-to-trough depth test passed contact-binary folds smeared by
    # up to ~0.1 cycle (Pipeline_Log Entry 76). The phase-dispersion minimum
    # near both periods is also found and written as a cross-check.
    disp_new     = phase_dispersion(time_bjd, flux, period_new)
    disp_kreiner = phase_dispersion(time_bjd, flux, period_kreiner)
    period_best_pdm = best_dispersion_period(
        time_bjd, flux,
        min(period_new, period_kreiner) * (1 - 0.003),
        max(period_new, period_kreiner) * (1 + 0.003),
    )
    span_days = float(time_btjd.max() - time_btjd.min())
    drift_vs_pdm = abs(period_new - period_best_pdm) * span_days / period_best_pdm ** 2
    period_reverted = False
    if disp_new > DISPERSION_REVERT_TOLERANCE * disp_kreiner:
        period_reverted = True
        # Re-derive T0 at the fixed Kreiner period from the PRIMARY minima (the
        # secondaries carry their own zero-point).
        primaries  = ~is_secondary if np.any(~is_secondary) else np.ones_like(is_secondary)
        weights    = 1.0 / np.square(t_err_array[primaries])
        t0_new     = float(np.average(t_obs_array[primaries]
                                      - epoch_numbers[primaries] * period_kreiner, weights=weights))
        t0_err     = float(np.sqrt(1.0 / np.sum(weights)))
        period_new = period_kreiner
        period_err = np.nan
        if np.any(is_secondary):
            secondary_offset = float(np.median(t_obs_array[is_secondary]
                                               - (t0_new + period_new * epoch_numbers[is_secondary])))
        drift_vs_pdm = abs(period_new - period_best_pdm) * span_days / period_best_pdm ** 2
        print(f"  [WARNING] Refined period folds worse than Kreiner "
              f"(dispersion {disp_new:.4f} > {DISPERSION_REVERT_TOLERANCE} × {disp_kreiner:.4f}). "
              f"Reverting to Kreiner P = {period_kreiner:.8f} d.")
    else:
        print(f"  [fold-check] dispersion refined {disp_new:.4f} vs Kreiner {disp_kreiner:.4f} "
              f"— refined period retained.")
    print(f"  [pdm cross-check] P_best_pdm = {period_best_pdm:.8f} d; "
          f"|P_new − P_best_pdm| = {abs(period_new - period_best_pdm) * 86400:.2f} s "
          f"→ drift {drift_vs_pdm:.4f} cycles over {span_days:.1f} d")
    if drift_vs_pdm > 0.01:
        print(f"  [WARNING] P_new drifts {drift_vs_pdm:.3f} cycles from the phase-dispersion "
              f"period across the data — inspect this star before modelling.")

    # ── 6E: WRITE OUTPUT FILES ────────────────────────────────────────────────
    ephem_path = os.path.join(paths.TIMING_DIR, f"ephem_{safe_name}.txt")
    t0_label = ("HJD(UTC) (Kreiner 2004)" if has_kreiner
                else "BJD — first measured ToM (no Kreiner entry)")
    p_label = "days (Kreiner 2004)" if has_kreiner else "days (catalogue period; no Kreiner entry)"
    with open(ephem_path, "w") as ephem_file:
        ephem_file.write(f"# Refined linear ephemeris for {star_name}\n")
        ephem_file.write("# T_min (BJD_TDB) = T0_new + P_new * E   (E half-integer for secondaries;\n"
                         "#                   secondaries also add SEC_OFFSET_days)\n")
        ephem_file.write("# NOTE: Kreiner T0 is HJD(UTC), ToMs are BJD(TDB) (~65-69 s apart). Kreiner\n"
                         "#       only sets the epoch count; T0_new is fitted to the TESS ToMs.\n")
        ephem_file.write(f"T0_new     = {t0_new:.8f}    # BJD_TDB at E = 0\n")
        ephem_file.write(f"T0_err     = {t0_err:.8f}    # days\n")
        ephem_file.write(f"P_new      = {period_new:.10f}  # days\n")
        ephem_file.write(f"P_err      = {period_err:.10f}  # days (fit covariance scaled by reduced chi2)\n")
        ephem_file.write(f"T0_kreiner = {t0_kreiner:.8f}  # {t0_label}\n")
        ephem_file.write(f"P_kreiner  = {period_kreiner:.10f}  # {p_label}\n")
        ephem_file.write(f"PERIOD_REVERTED = {period_reverted}  "
                         "# True if the refined period was rejected (folded worse than Kreiner)\n")
        ephem_file.write(f"SEC_OFFSET_days = {secondary_offset:.8f}  # secondary minima: T - (T0 + P*E)\n")
        ephem_file.write(f"N_PRIMARY  = {int(np.sum(~is_secondary))}\n")
        ephem_file.write(f"N_SECONDARY = {int(np.sum(is_secondary))}\n")
        ephem_file.write(f"N_FALLBACK = {n_fallback}  # minima timed by the parabola fallback\n")
        ephem_file.write(f"DISP_NEW   = {disp_new:.6f}  # phase dispersion at P_new\n")
        ephem_file.write(f"DISP_KREINER = {disp_kreiner:.6f}  # phase dispersion at P_kreiner\n")
        ephem_file.write(f"P_best_pdm = {period_best_pdm:.10f}  # phase-dispersion minimum (cross-check)\n")
        ephem_file.write(f"DRIFT_VS_PDM_cycles = {drift_vs_pdm:.5f}  # |P_new - P_best_pdm| x span / P^2\n")

    tominima_path = os.path.join(paths.TIMING_DIR, f"tominima_{safe_name}.csv")
    with open(tominima_path, "w", newline="") as tominima_file:
        writer = csv.writer(tominima_file)
        writer.writerow(["Star", "T_min_BJD", "T_min_err_days", "Epoch_E", "OC_days", "Type"])
        for i in range(len(t_obs_array)):
            writer.writerow([
                star_name,
                f"{t_obs_array[i]:.8f}",
                f"{t_err_array[i]:.8f}",
                f"{epoch_numbers[i]:.1f}",
                f"{oc_days[i]:.8f}",
                "S" if is_secondary[i] else "P",
            ])

    print(f"  Saved: {tominima_path}")
    print(f"  Saved: {ephem_path}")

    # ── 6F: THREE-PANEL O–C DIAGRAM ───────────────────────────────────────────
    print(f"  Generating O–C diagram...")

    oc_vs_new_ephemeris = t_obs_array - (t0_new + period_new * epoch_numbers
                                         + secondary_offset * is_secondary)

    fig, axes = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={"hspace": 0.4})
    fig.suptitle(
        f"{star_name} — O–C Analysis & Ephemeris Refinement",
        fontsize=16, fontweight="bold",
    )

    # Panel (a): Global O–C trend spanning ~20 years since Kreiner (2004).
    # TESS data appear bunched on the right because the epoch counter
    # has accumulated ~10 000–60 000 cycles since the Kreiner reference.
    axes[0].errorbar(
        epoch_numbers, oc_days * 1440, yerr=oc_uncertainties * 1440,
        fmt="ko", ms=5, capsize=3, elinewidth=1.2, label="TESS Measurements",
    )
    axes[0].axhline(0, color="r", lw=1.5, ls="--", label="Kreiner (2004) Prediction")
    axes[0].set_ylabel("O–C (minutes)")
    axes[0].set_title(
        f"(a) Global O–C vs. Kreiner (2004) Baseline  [Epoch 0 to {max(epoch_numbers)}]",
        loc="left", fontsize=12, fontweight="bold",
    )
    axes[0].grid(alpha=0.3)
    axes[0].text(
        0.015, 0.05,
        "• Shows the full drift accumulated over ~20 years.\n"
        "• TESS points compressed right due to the large epoch range.",
        transform=axes[0].transAxes, fontsize=10, linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.85),
    )
    axes[0].legend(loc="upper left")

    # Panel (b): Residuals after subtracting the newly fitted linear ephemeris.
    # A flat scatter around zero confirms that a constant period model is adequate.
    axes[1].errorbar(
        epoch_numbers, oc_vs_new_ephemeris * 1440, yerr=oc_uncertainties * 1440,
        fmt="bs", ms=5, capsize=3, elinewidth=1.2, label="Residuals vs. Refined Ephemeris",
    )
    axes[1].axhline(0, color="r", lw=1.5, ls="--", label="Refined Ephemeris")
    axes[1].set_ylabel("O–C (minutes)")
    axes[1].set_title(
        f"(b) Residuals vs. Refined Linear Ephemeris  (P = {period_new:.7f} d)",
        loc="left", fontsize=12, fontweight="bold",
    )
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper left")

    # Panel (c): TESS-era zoom — reveals all individual eclipse measurements
    # and their internal consistency, proving the dense TESS timing baseline.
    epoch_range = max(epoch_numbers) - min(epoch_numbers)
    x_margin    = max(0.1 * epoch_range, 1)

    axes[2].errorbar(
        epoch_numbers, oc_vs_new_ephemeris * 1440, yerr=oc_uncertainties * 1440,
        fmt="ro", ms=6, capsize=4, elinewidth=1.5, mfc="white", mew=1.5,
        label="Individual TESS Eclipses",
    )
    axes[2].axhline(0, color="k", lw=1.0)
    axes[2].set_xlim(min(epoch_numbers) - x_margin, max(epoch_numbers) + x_margin)
    axes[2].set_xlabel("Orbital Epoch (E)")
    axes[2].set_ylabel("O–C (minutes)")
    axes[2].set_title(
        f"(c) TESS-Era Zoom — {len(epoch_numbers)} Individual Eclipse Measurements",
        loc="left", fontsize=12, fontweight="bold",
    )
    axes[2].grid(alpha=0.4, linestyle="--")
    axes[2].text(
        0.015, 0.05,
        f"• All {len(epoch_numbers)} ToM measurements from the TESS sector.\n"
        "• Error bars: formal KvW σ. The scatter is larger (spots, noise);\n"
        "  P_err is scaled to the observed scatter (reduced χ²).",
        transform=axes[2].transAxes, fontsize=10, linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="honeydew",
                  edgecolor="gray", alpha=0.85),
    )
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plot_path = os.path.join(paths.PLOT_OC_DIR, f"oc_{safe_name}.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  O–C diagram saved: {plot_path}")
    run_tracker.star_end(star_name, "done", n_minima=len(candidate_min_indices),
                         n_toms=len(accepted_toms), period_refined=period_new,
                         period_err=period_err)

print("\n" + "=" * 70)
print("SCRIPT 03 COMPLETE — O–C analyses done for all targets.")
print("=" * 70)
