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
           residual sum of squares (RSS) between the ascending and descending
           branches. Preferred for well-sampled eclipses.

        2. Polynomial fallback: fits a degree-4 polynomial to the eclipse
           bottom and reports its analytic minimum. Used when the KvW window
           contains too few data points for reliable branch-interleaving.

    A refined linear ephemeris (T0_new, P_new) is derived by weighted
    least-squares fitting of all accepted ToM measurements against the
    orbital epoch numbers computed from the Kreiner (2004) baseline.

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

# TESS mission time offset: BTJD + offset = BJD.
# Historical ephemerides (Kreiner 2004) are in HJD; the BJD–HJD difference
# is <4 s for solar-system targets and is absorbed into T0_new during the fit.
TESS_BTJD_TO_BJD_OFFSET: float = 2_457_000.0

# Maximum acceptable formal uncertainty for a ToM measurement (days).
# 0.015 d ≈ 21 min; chosen to accommodate the higher Poisson scatter
# inherent to 2-min PDCSAP cadence for faint targets.
MAX_TOM_UNCERTAINTY_DAYS: float = 0.015


# =============================================================================
# SECTION 1 — KWEE–VAN WOERDEN ALGORITHM
# =============================================================================

def compute_tom_kwee_van_woerden(
    time_bjd: np.ndarray,
    flux: np.ndarray,
    t_trial: float,
    half_window: float = 0.05,
) -> tuple[float, float]:
    """
    Determine the Time of Minimum (ToM) via the Kwee–van Woerden (1956) method.

    The algorithm mirrors the eclipse profile about a trial centre t_test,
    interpolates the descending branch onto the ascending branch's time grid,
    and computes the RSS of the per-point flux differences. The t_test that
    minimises the RSS is adopted as the ToM. This is model-independent: it
    makes no assumption about the eclipse profile shape.

    Uncertainty estimate:
        A parabola is fit to the RSS trough. Its curvature (the coefficient of
        the quadratic term 'a') encodes how sharply the RSS rises away from the
        minimum. The formal error is σ_t = sqrt(RSS_min / (N_pairs × a)).

    Parameters
    ----------
    time_bjd:
        Barycentric Julian Date time array for the full light curve.
    flux:
        Normalised flux array (same length as time_bjd).
    t_trial:
        Approximate centre of the eclipse (initial guess in BJD).
    half_window:
        Half-width of the time window around t_trial (days). Should be
        slightly larger than the eclipse half-duration.

    Returns
    -------
    t_minimum:
        Best-fit time of minimum (BJD).
    t_uncertainty:
        Formal 1σ uncertainty (days); falls back to 1e-4 d if the parabola
        fit fails.

    Raises
    ------
    ValueError:
        If fewer than 6 data points fall within the eclipse window.
    """
    in_window = (time_bjd >= t_trial - half_window) & (time_bjd <= t_trial + half_window)
    t_window  = time_bjd[in_window]
    f_window  = flux[in_window]

    if len(t_window) < 6:
        raise ValueError("Fewer than 6 points in eclipse window — KvW unreliable.")

    # Dense trial grid: 500 points across the window gives sub-minute resolution
    # for typical TESS 2-min cadence data.
    trial_centres = np.linspace(t_window.min(), t_window.max(), 500)
    rss_values    = np.zeros(len(trial_centres))

    for idx, t_test in enumerate(trial_centres):
        dt         = t_window - t_test
        right_mask = dt >= 0   # ascending branch (t > t_test)
        left_mask  = dt <  0   # descending branch (t < t_test)
        dt_right   = dt[right_mask]
        dt_left    = -dt[left_mask]      # mirror so both run 0 → positive
        f_right    = f_window[right_mask]
        f_left     = f_window[left_mask]

        if len(dt_right) < 2 or len(dt_left) < 2:
            rss_values[idx] = np.inf
            continue

        # Interpolate the descending branch onto the ascending branch's δt grid.
        f_left_interp = np.interp(dt_right, dt_left[::-1], f_left[::-1],
                                  left=np.nan, right=np.nan)
        valid_pairs   = np.isfinite(f_left_interp)
        if valid_pairs.sum() < 2:
            rss_values[idx] = np.inf
            continue

        rss_values[idx] = np.sum((f_right[valid_pairs] - f_left_interp[valid_pairs]) ** 2)

    best_idx    = np.argmin(rss_values)
    t_minimum   = trial_centres[best_idx]

    # Fit a parabola to the ±15 points around the RSS minimum to estimate the
    # curvature and derive the formal timing uncertainty.
    lo = max(best_idx - 15, 0)
    hi = min(best_idx + 16, len(trial_centres))
    finite_mask = np.isfinite(rss_values[lo:hi])

    t_uncertainty = 1e-4  # fallback if parabola fit fails (≈ 9 s)
    if finite_mask.sum() >= 3:
        try:
            parabola_coeffs = np.polyfit(
                trial_centres[lo:hi][finite_mask],
                rss_values[lo:hi][finite_mask],
                2,
            )
            curvature = parabola_coeffs[0]  # coefficient of the quadratic term
            if curvature > 0:
                n_pairs       = max(1, finite_mask.sum())
                t_uncertainty = np.sqrt(rss_values[best_idx] / (n_pairs * curvature))
        except np.linalg.LinAlgError:
            pass

    return t_minimum, t_uncertainty


# =============================================================================
# SECTION 2 — POLYNOMIAL FALLBACK
# =============================================================================

def compute_tom_polynomial_fallback(
    time_bjd: np.ndarray,
    flux: np.ndarray,
    t_trial: float,
    half_window: float = 0.05,
    polynomial_degree: int = 4,
) -> tuple[float, float]:
    """
    Determine the ToM by fitting a polynomial to the eclipse bottom.

    This fallback is invoked when the KvW window contains too few data points
    for reliable branch-interleaving (fewer than 6 points). A degree-4
    polynomial is appropriate for typical smooth eclipse profiles without
    significant starspot contamination.

    Uncertainty estimate:
        σ_t = RMS_residuals / curvature, where curvature is the second
        derivative of the fitted polynomial at its minimum. A steeply
        curved, low-scatter eclipse yields a small uncertainty; a shallow
        eclipse with high scatter yields a large one.

    Parameters
    ----------
    time_bjd, flux:
        Full light-curve arrays.
    t_trial:
        Approximate eclipse centre (BJD).
    half_window:
        Half-width of fitting window (days).
    polynomial_degree:
        Degree of the fitted polynomial (default 4).

    Returns
    -------
    t_minimum, t_uncertainty:
        Best-fit ToM (BJD) and formal 1σ uncertainty (days).

    Raises
    ------
    ValueError:
        If the window contains fewer than (degree + 2) data points.
    """
    in_window = (time_bjd >= t_trial - half_window) & (time_bjd <= t_trial + half_window)
    t_window  = time_bjd[in_window]
    f_window  = flux[in_window]

    if len(t_window) < polynomial_degree + 2:
        raise ValueError("Insufficient data points for polynomial fit.")

    # Centre the time axis on t_trial to improve numerical conditioning of the
    # polynomial fit (avoids catastrophic cancellation for large BJD values).
    t_centred    = t_window - t_trial
    poly_coeffs  = np.polyfit(t_centred, f_window, polynomial_degree)
    poly_func    = np.poly1d(poly_coeffs)

    t_fine_centred = np.linspace(t_centred.min(), t_centred.max(), 10_000)
    f_fine         = poly_func(t_fine_centred)
    t_minimum      = t_trial + t_fine_centred[np.argmin(f_fine)]

    residuals     = f_window - poly_func(t_centred)
    rms_residuals = np.std(residuals)
    poly_deriv2   = np.polyder(poly_func, 2)
    curvature     = abs(float(poly_deriv2(t_fine_centred[np.argmin(f_fine)])))
    t_uncertainty = rms_residuals / curvature if curvature > 0 else 1e-4

    return t_minimum, t_uncertainty


# =============================================================================
# SECTION 3 — O–C RESIDUAL CALCULATION
# =============================================================================

def compute_oc_residuals(
    t_observed_bjd: np.ndarray,
    t_observed_err: np.ndarray,
    t0_kreiner: float,
    period_kreiner: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute O–C (Observed minus Calculated) residuals against Kreiner (2004).

    The orbital epoch E for each observed minimum is the integer cycle count
    elapsed since the Kreiner (2004) reference epoch T0_kreiner:
        E = round((T_obs − T0_kreiner) / P_kreiner)

    A cycle-jump guard checks whether the fractional residual exceeds 0.25
    phase units, and if so, reassigns the epoch to the neighbouring integer
    that minimises the residual — preventing ±1-cycle aliasing.

    Parameters
    ----------
    t_observed_bjd:
        Array of observed ToM values (BJD).
    t_observed_err:
        Array of formal ToM uncertainties (days).
    t0_kreiner:
        Kreiner (2004) reference epoch (HJD; BJD–HJD offset absorbed in fit).
    period_kreiner:
        Kreiner (2004) orbital period (days).

    Returns
    -------
    epoch_numbers:
        Integer orbital epoch E for each observed ToM.
    oc_days:
        O–C residuals in days (T_obs − T_calc).
    oc_uncertainties:
        Propagated uncertainties (equal to t_observed_err at this stage).
    """
    fractional_epochs = (t_observed_bjd - t0_kreiner) / period_kreiner
    epoch_numbers     = np.round(fractional_epochs).astype(int)

    # Guard against ±1-cycle aliasing (can occur at the beginning or end of
    # a TESS sector when fewer eclipses constrain the phase solution).
    for i, (frac_e, epoch_e) in enumerate(zip(fractional_epochs, epoch_numbers)):
        if abs(frac_e - epoch_e) > 0.25:
            neighbour_epochs = np.array([epoch_e - 1, epoch_e, epoch_e + 1])
            best_neighbour   = neighbour_epochs[np.argmin(np.abs(frac_e - neighbour_epochs))]
            epoch_numbers[i] = best_neighbour

    t_calculated  = t0_kreiner + period_kreiner * epoch_numbers
    oc_days        = t_observed_bjd - t_calculated
    return epoch_numbers, oc_days, t_observed_err


# =============================================================================
# SECTION 4 — WEIGHTED LEAST-SQUARES EPHEMERIS REFINEMENT
# =============================================================================

def refine_linear_ephemeris(
    epoch_numbers: np.ndarray,
    t_observed_bjd: np.ndarray,
    t_observed_err: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Fit a refined linear ephemeris via weighted least squares:
        T_min(E) = T0_new + P_new × E

    A polyfit-derived initial guess is used regardless of how far the epoch
    counter has grown from the Kreiner reference (E can exceed 10 000 cycles),
    preventing poor initialisation of the scipy curve_fit optimiser.

    Parameters
    ----------
    epoch_numbers:
        Integer orbital epoch array E.
    t_observed_bjd:
        Observed ToM values (BJD).
    t_observed_err:
        Formal 1σ uncertainties on each ToM (days), used as weights.

    Returns
    -------
    t0_new, period_new:
        Refined reference epoch (BJD) and orbital period (days).
    t0_err, period_err:
        Formal 1σ uncertainties derived from the covariance matrix diagonal.
        Returns NaN if scipy.optimize.curve_fit fails to converge.
    """
    def linear_model(epoch: np.ndarray, t0: float, period: float) -> np.ndarray:
        return t0 + period * epoch

    # polyfit gives a numerically stable initial guess for any epoch range.
    slope, intercept = np.polyfit(epoch_numbers, t_observed_bjd, 1)
    initial_guess    = [intercept, slope]

    try:
        optimal_params, covariance = curve_fit(
            linear_model, epoch_numbers, t_observed_bjd,
            p0=initial_guess, sigma=t_observed_err, absolute_sigma=True,
        )
        param_errors = np.sqrt(np.diag(covariance))
        return optimal_params[0], optimal_params[1], param_errors[0], param_errors[1]
    except RuntimeError:
        # curve_fit failed to converge: return the polyfit values with NaN errors.
        return intercept, slope, np.nan, np.nan


def fold_depth(
    time_bjd: np.ndarray,
    flux: np.ndarray,
    period: float,
    n_bins: int = 50,
) -> float:
    """
    Peak-to-trough depth of the phase-folded, median-binned light curve.

    The correct orbital period folds every eclipse onto the same phase, maximising
    the binned eclipse depth; an incorrect period smears the eclipses across phase
    and shrinks the depth. This makes the metric a cheap, model-free sharpness
    score used by the Section-6D self-check to reject a refined period that folds
    worse than the Kreiner baseline.

    Parameters
    ----------
    time_bjd, flux:
        Cleaned light-curve time (BJD) and normalised flux.
    period:
        Trial period in days.
    n_bins:
        Number of phase bins.

    Returns
    -------
    Depth = median(binned flux) − min(binned flux); 0.0 if it cannot be computed.
    """
    phase    = (time_bjd / period) % 1.0
    edges    = np.linspace(0.0, 1.0, n_bins + 1)
    binned   = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (phase >= edges[i]) & (phase < edges[i + 1])
        if np.any(m):
            binned[i] = np.nanmedian(flux[m])
    if np.all(np.isnan(binned)):
        return 0.0
    return float(np.nanmedian(binned) - np.nanmin(binned))


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

    # Smooth with a uniform filter to suppress high-frequency noise before
    # local minimum detection; size 15 ≈ 30 min for 2-min cadence data.
    flux_smoothed = uniform_filter1d(flux, size=15)

    # Set the argrelmin order so that two candidate minima must be separated
    # by at least 20% of one orbital period — preventing double-detection of
    # the same eclipse bottom in noisy data.
    cadence_days       = float(np.median(np.diff(time_btjd)))
    points_per_period  = orbital_period / cadence_days
    min_separation_pts = max(5, int(points_per_period * 0.2))

    candidate_min_indices = argrelmin(flux_smoothed, order=min_separation_pts)[0]
    print(f"  Found {len(candidate_min_indices)} candidate minima "
          f"(separation order = {min_separation_pts} pts).")

    if len(candidate_min_indices) == 0:
        print("  ERROR: No minima detected. Check light curve quality.")
        run_tracker.star_end(star_name, "failed", reason="no minima detected")
        continue

    # ── 6B: PRECISE ToM EXTRACTION ────────────────────────────────────────────
    print("\n  [6B] Computing precise Times of Minimum...")

    eclipse_half_window = orbital_period * 0.15
    accepted_toms: list[tuple[float, float]] = []

    for min_idx in candidate_min_indices:
        t_trial_bjd = time_bjd[min_idx]
        try:
            t_min, t_err = compute_tom_kwee_van_woerden(
                time_bjd, flux, t_trial_bjd, half_window=eclipse_half_window,
            )
            method = "KvW"
        except ValueError:
            try:
                t_min, t_err = compute_tom_polynomial_fallback(
                    time_bjd, flux, t_trial_bjd, half_window=eclipse_half_window,
                )
                method = "poly"
            except ValueError as poly_err:
                print(f"    Skipped minimum at BJD {t_trial_bjd:.4f}: {poly_err}")
                continue

        if t_err <= MAX_TOM_UNCERTAINTY_DAYS:
            accepted_toms.append((t_min, t_err))
            print(f"    ToM = {t_min:.6f} BJD  ±{t_err:.6f} d  ({method})")
        else:
            print(f"    Rejected: ToM at {t_min:.4f} BJD  "
                  f"uncertainty ±{t_err:.4f} d > {MAX_TOM_UNCERTAINTY_DAYS} d")

    if not accepted_toms:
        print("  ERROR: No ToM measurements passed the quality filter. Skipping.")
        run_tracker.star_end(star_name, "failed", reason="no ToM passed quality filter",
                             n_minima=len(candidate_min_indices))
        continue

    t_obs_array = np.array([t for t, _ in accepted_toms])
    t_err_array = np.array([e for _, e in accepted_toms])

    # ── 6C: O–C RESIDUALS ─────────────────────────────────────────────────────
    print("\n  [6C] Computing O–C residuals vs. Kreiner (2004) baseline...")

    if star_name in historical_ephemerides:
        t0_kreiner, period_kreiner = historical_ephemerides[star_name]
        print(f"    Kreiner baseline: T0 = {t0_kreiner},  P = {period_kreiner}")
    else:
        print(f"  WARNING: No Kreiner entry for {star_name}. Using first ToM as T0.")
        t0_kreiner     = t_obs_array[0]
        period_kreiner = orbital_period

    epoch_numbers, oc_days, oc_uncertainties = compute_oc_residuals(
        t_obs_array, t_err_array, t0_kreiner, period_kreiner,
    )

    # ── 6D: EPHEMERIS REFINEMENT ──────────────────────────────────────────────
    print("\n  [6D] Fitting refined linear ephemeris...")

    t0_new, period_new, t0_err, period_err = refine_linear_ephemeris(
        epoch_numbers, t_obs_array, t_err_array,
    )
    print(f"  Refined ephemeris (BJD):")
    print(f"    T_min = {t0_new:.6f} (±{t0_err:.6f}) "
          f"+ {period_new:.8f} (±{period_err:.8f}) × E")

    # ── 6D-check: PERIOD FOLD-QUALITY SELF-CHECK (Flag 2 remediation) ─────────
    # A refined period that folds the light curve WORSE than the Kreiner baseline
    # signals that the O–C linear fit locked onto an aliased / mis-counted set of
    # epochs. This was observed for large-|ΔP| stars (e.g. II UMa, GK Cep), where
    # the refined period smeared the eclipses and corrupted the Script 04 bundle.
    # We compare the median-binned fold depth at both periods and, if the refined
    # fold is materially shallower, reject it and retain the Kreiner period.
    depth_new       = fold_depth(time_bjd, flux, period_new)
    depth_kreiner   = fold_depth(time_bjd, flux, period_kreiner)
    period_reverted = False
    if depth_kreiner > 0.0 and depth_new < 0.85 * depth_kreiner:
        period_reverted = True
        # Re-derive T0 at the fixed Kreiner period from the observed ToMs so the
        # reference epoch sits within the TESS data rather than the 2004 baseline.
        epochs_k   = np.round((t_obs_array - t0_kreiner) / period_kreiner)
        weights    = 1.0 / np.square(t_err_array)
        t0_new     = float(np.average(t_obs_array - epochs_k * period_kreiner, weights=weights))
        t0_err     = float(np.sqrt(1.0 / np.sum(weights)))
        period_new = period_kreiner
        period_err = np.nan
        print(f"  [WARNING] Refined period folds worse than Kreiner "
              f"(depth {depth_new:.3f} < 0.85 × {depth_kreiner:.3f}). "
              f"Reverting to Kreiner P = {period_kreiner:.8f} d.")
    else:
        print(f"  [fold-check] refined depth {depth_new:.3f} vs "
              f"Kreiner {depth_kreiner:.3f} — refined period retained.")

    # ── 6E: WRITE OUTPUT FILES ────────────────────────────────────────────────
    ephem_path = os.path.join(paths.TIMING_DIR, f"ephem_{safe_name}.txt")
    with open(ephem_path, "w") as ephem_file:
        ephem_file.write(f"# Refined linear ephemeris for {star_name}\n")
        ephem_file.write("# T_min (BJD) = T0_new + P_new * E\n")
        ephem_file.write("# NOTE: Kreiner T0 is HJD; ToM are in BJD.\n"
                         "#       The <4 s HJD-to-BJD offset is absorbed into T0_new.\n")
        ephem_file.write(f"T0_new     = {t0_new:.8f}    # BJD\n")
        ephem_file.write(f"T0_err     = {t0_err:.8f}    # BJD\n")
        ephem_file.write(f"P_new      = {period_new:.10f}  # days\n")
        ephem_file.write(f"P_err      = {period_err:.10f}  # days\n")
        ephem_file.write(f"T0_kreiner = {t0_kreiner:.8f}  # HJD (Kreiner 2004)\n")
        ephem_file.write(f"P_kreiner  = {period_kreiner:.10f}  # days (Kreiner 2004)\n")
        ephem_file.write(f"PERIOD_REVERTED = {period_reverted}  "
                         "# True if the refined period was rejected (folded worse than Kreiner)\n")

    tominima_path = os.path.join(paths.TIMING_DIR, f"tominima_{safe_name}.csv")
    with open(tominima_path, "w", newline="") as tominima_file:
        writer = csv.writer(tominima_file)
        writer.writerow(["Star", "T_min_BJD", "T_min_err_days", "Epoch_E", "OC_days"])
        for i in range(len(t_obs_array)):
            writer.writerow([
                star_name,
                f"{t_obs_array[i]:.8f}",
                f"{t_err_array[i]:.8f}",
                epoch_numbers[i],
                f"{oc_days[i]:.8f}",
            ])

    print(f"  Saved: {tominima_path}")
    print(f"  Saved: {ephem_path}")

    # ── 6F: THREE-PANEL O–C DIAGRAM ───────────────────────────────────────────
    print(f"  Generating O–C diagram...")

    oc_vs_new_ephemeris = t_obs_array - (t0_new + period_new * epoch_numbers)

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
        "• Vertical scatter reflects the formal per-eclipse timing uncertainty.\n"
        "• Internal consistency validates the refined orbital solution.",
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
