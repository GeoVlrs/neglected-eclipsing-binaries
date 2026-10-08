"""
================================================================================
AUDIT — SCRIPT 03's TIMES OF MINIMUM AND THE REFINED PERIOD (REPLICA vs FIX)
================================================================================
Summary:
    audit_period_fold.py showed that Script 03's refined period P_new misfolds
    the TESS data for most stars while the Kreiner (2004) period folds them
    cleanly. This audit pins down WHY, by running two estimators on the same
    cleaned light curves:

    (1) REPLICA — a verbatim copy of 03OCAnalysis.py §1/§3/§4/§6A-6D:
        smoothed argrelmin candidates → compute_tom_kwee_van_woerden →
        0.015 d acceptance cut → integer epochs from the Kreiner T0 →
        weighted linear fit. If the replica reproduces the P_new stored in
        ephem_<star>.txt, the mechanism below is confirmed by execution.
        Known defects it carries (Pipeline_Log Entry 76):
          * the KvW residual sum is unnormalised and runs over a mirrored overlap
            that shrinks to ~2 pairs at the window edges, so its minimum sits at
            an edge (t_trial ± ~0.15 P), not at the eclipse centre;
          * the σ parabola is fitted on absolute BJD (~2.46e6 d), so the
            quadratic term is numerically unresolved: σ is either huge (rejected)
            or the 1e-4 d fallback (accepted);
          * every ToM, primary or secondary, is rounded to an INTEGER epoch.

    (2) FIXED — a standard Kwee & van Woerden (1956) estimator:
        trial centres within ±w/2 of the candidate, a FIXED symmetric reflection
        range |dt| ≤ w/2 interpolated on both branches, a normalised residual,
        a parabola on centred times, the vertex as the ToM and the KvW σ;
        windows with gaps are rejected. Epochs at HALF-cycle resolution with
        primaries/secondaries labelled by depth, one common period and a
        separate zero-point for secondaries; σ_P from the residual scatter.

    Both periods are compared with P_best from audit_period_fold.py (phase
    dispersion, model-free) and with Kreiner.

Inputs:
    data/lightcurves/lc_<star>_clean.fits, data/timing/ephem_<star>.txt,
    Targets/neglected_targets*.csv (catalogue period, as 03 uses it),
    Audits/period_fold_audit.csv (P_best; run audit_period_fold.py first).

Outputs:
    Audits/tom_estimator_audit.csv — one row per star.
    Table printed to stdout.

Usage (from Code/; light: single process, ~1-2 min for the full sample):
    python Audits/audit_tom_estimator.py               # all stars
    python Audits/audit_tom_estimator.py "V899 Her"    # selected stars

Dependencies:
    pip install astropy numpy scipy
================================================================================
"""

import csv
import glob
import os
import re
import sys

import warnings

import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelmin

# 03OCAnalysis.py:64 silences all warnings, including the RankWarning its own σ
# parabola raises on every minimum; the replica does the same so it behaves alike.
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths                                              # noqa: E402

TESS_BTJD_TO_BJD_OFFSET: float = 2_457_000.0
MAX_TOM_UNCERTAINTY_DAYS: float = 0.015                   # 03OCAnalysis.py:83
FIXED_WINDOW_FRAC: float = 0.12                           # full KvW reflection range, in P
FIXED_MIN_POINTS_PER_BRANCH: int = 8

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIT_CSV = os.path.join(AUDIT_DIR, "tom_estimator_audit.csv")
FOLD_CSV = os.path.join(AUDIT_DIR, "period_fold_audit.csv")


# ── REPLICA of 03OCAnalysis.py (verbatim logic; do not "improve") ─────────────

def replica_kvw(time_bjd, flux, t_trial, half_window=0.05):
    """03OCAnalysis.compute_tom_kwee_van_woerden, lines 135-194, unchanged.
    Also returns the index of the RSS minimum on the 500-point grid."""
    in_window = (time_bjd >= t_trial - half_window) & (time_bjd <= t_trial + half_window)
    t_window = time_bjd[in_window]
    f_window = flux[in_window]
    if len(t_window) < 6:
        raise ValueError("Fewer than 6 points in eclipse window — KvW unreliable.")
    trial_centres = np.linspace(t_window.min(), t_window.max(), 500)
    rss_values = np.zeros(len(trial_centres))
    for idx, t_test in enumerate(trial_centres):
        dt = t_window - t_test
        right_mask = dt >= 0
        left_mask = dt < 0
        dt_right = dt[right_mask]
        dt_left = -dt[left_mask]
        f_right = f_window[right_mask]
        f_left = f_window[left_mask]
        if len(dt_right) < 2 or len(dt_left) < 2:
            rss_values[idx] = np.inf
            continue
        f_left_interp = np.interp(dt_right, dt_left[::-1], f_left[::-1],
                                  left=np.nan, right=np.nan)
        valid_pairs = np.isfinite(f_left_interp)
        if valid_pairs.sum() < 2:
            rss_values[idx] = np.inf
            continue
        rss_values[idx] = np.sum((f_right[valid_pairs] - f_left_interp[valid_pairs]) ** 2)
    best_idx = int(np.argmin(rss_values))
    t_minimum = trial_centres[best_idx]
    lo = max(best_idx - 15, 0)
    hi = min(best_idx + 16, len(trial_centres))
    finite_mask = np.isfinite(rss_values[lo:hi])
    t_uncertainty = 1e-4
    if finite_mask.sum() >= 3:
        try:
            coeffs = np.polyfit(trial_centres[lo:hi][finite_mask], rss_values[lo:hi][finite_mask], 2)
            curvature = coeffs[0]
            if curvature > 0:
                n_pairs = max(1, finite_mask.sum())
                t_uncertainty = np.sqrt(rss_values[best_idx] / (n_pairs * curvature))
        except np.linalg.LinAlgError:
            pass
    return t_minimum, t_uncertainty, best_idx


def replica_period(time_btjd, flux, orbital_period, t0_k, p_k):
    """03 §6A–6D (without the fold-depth revert), returning P_new and diagnostics."""
    time_bjd = time_btjd + TESS_BTJD_TO_BJD_OFFSET
    flux_smoothed = uniform_filter1d(flux, size=15)
    cadence_days = float(np.median(np.diff(time_btjd)))
    min_sep = max(5, int(orbital_period / cadence_days * 0.2))
    cand = argrelmin(flux_smoothed, order=min_sep)[0]
    hw = orbital_period * 0.15
    toms, errs, edge_hits, n_tried, n_rejected = [], [], 0, 0, 0
    for i in cand:
        try:
            t_min, t_err, k = replica_kvw(time_bjd, flux, time_bjd[i], half_window=hw)
        except ValueError:
            continue                    # 03 then tries the polynomial fallback; never used in logs
        n_tried += 1
        edge_hits += (k < 25 or k > 474)        # within 5 % of either end of the 500-point grid
        if t_err <= MAX_TOM_UNCERTAINTY_DAYS:
            toms.append(t_min)
            errs.append(t_err)
        else:
            n_rejected += 1
    toms, errs = np.array(toms), np.array(errs)
    if len(toms) < 3:
        return dict(P=np.nan, n_cand=len(cand), n_tried=n_tried, n_acc=len(toms),
                    edge_frac=np.nan, frac_err_1e4=np.nan, n_rejected=n_rejected)
    frac = (toms - t0_k) / p_k
    epochs = np.round(frac).astype(int)       # the "cycle-jump guard" leaves these unchanged
    slope, intercept = np.polyfit(epochs, toms, 1)
    try:
        popt, _ = curve_fit(lambda e, a, b: a + b * e, epochs, toms, p0=[intercept, slope],
                            sigma=errs, absolute_sigma=True)
        p_rep = popt[1]
    except RuntimeError:
        p_rep = slope
    return dict(P=float(p_rep), n_cand=len(cand), n_tried=n_tried, n_acc=len(toms),
                edge_frac=edge_hits / max(n_tried, 1),
                frac_err_1e4=float(np.mean(np.isclose(errs, 1e-4))),
                n_rejected=n_rejected)


# ── FIXED estimator ──────────────────────────────────────────────────────────

def fixed_kvw(time_bjd, flux, t_guess, period, cadence):
    """Standard KvW with a fixed symmetric reflection range. Returns (ToM, σ, depth) or None."""
    w = FIXED_WINDOW_FRAC * period                       # full reflection range
    half = 0.5 * w
    sel = np.abs(time_bjd - t_guess) <= w                # data span needed: t_c ± half, t_c ∈ t_guess ± half
    t, f = time_bjd[sel], flux[sel]
    if len(t) < 2 * FIXED_MIN_POINTS_PER_BRANCH or np.max(np.diff(t)) > 3 * cadence:
        return None                                      # gap inside the window
    if t.min() > t_guess - w + 2 * cadence or t.max() < t_guess + w - 2 * cadence:
        return None                                      # window not fully covered
    n_ref = max(FIXED_MIN_POINTS_PER_BRANCH, int(half / cadence))
    dts = np.linspace(cadence, half, n_ref)
    centres = np.linspace(t_guess - half, t_guess + half, 241)
    s = np.empty(len(centres))
    for j, tc in enumerate(centres):
        s[j] = np.sum((np.interp(tc + dts, t, f) - np.interp(tc - dts, t, f)) ** 2)
    k = int(np.argmin(s))
    if k < 5 or k > len(centres) - 6:
        return None                                      # minimum not bracketed
    lo, hi = k - 5, k + 6
    x = centres[lo:hi] - centres[k]                      # centred: the quadratic is resolvable
    a, b, c = np.polyfit(x, s[lo:hi], 2)
    if a <= 0:
        return None
    t_min = centres[k] - b / (2 * a)
    z = n_ref                                            # independent pairs ≈ points per branch
    var = (4 * a * c - b * b) / (4 * a * a * (z - 1))
    sigma = float(np.sqrt(var)) if var > 0 else np.nan
    depth = float(np.interp(t_min, t, f))
    return float(t_min), sigma, depth


def fixed_period(time_btjd, flux, t0_k, p_k):
    """Fixed KvW at every predicted half-cycle, half-cycle epochs, common P."""
    time_bjd = time_btjd + TESS_BTJD_TO_BJD_OFFSET
    cadence = float(np.median(np.diff(time_btjd)))
    # Seed the first primary from the data: deepest point of the first full cycle.
    first = time_bjd < time_bjd[0] + p_k
    t_start = time_bjd[first][np.argmin(uniform_filter1d(flux[first], 9))]
    toms, sig, dep, half_e = [], [], [], []
    n_half = int((time_bjd[-1] - t_start) / (0.5 * p_k)) + 1
    for h in range(n_half):
        r = fixed_kvw(time_bjd, flux, t_start + h * 0.5 * p_k, p_k, cadence)
        if r is None:
            continue
        toms.append(r[0]); sig.append(r[1]); dep.append(r[2]); half_e.append(h)
    toms, dep, half_e = np.array(toms), np.array(dep), np.array(half_e)
    if len(toms) < 6:
        return dict(P=np.nan, P_err=np.nan, n=len(toms), sec_offset=np.nan, oc_rms_s=np.nan)
    # Re-tie half-epochs to the measured times (guards against a drifting seed).
    e2 = np.round(2 * (toms - toms[0]) / p_k).astype(int)
    parity = e2 % 2
    if np.mean(dep[parity == 1]) < np.mean(dep[parity == 0]):   # odd = deeper → primaries
        parity = 1 - parity
    epoch = e2 / 2.0
    # T = T0 + P·E + δ·[secondary]
    design = np.vstack([np.ones_like(epoch), epoch, parity.astype(float)]).T
    coef, *_ = np.linalg.lstsq(design, toms, rcond=None)
    resid = toms - design @ coef
    dof = max(len(toms) - 3, 1)
    cov = np.linalg.inv(design.T @ design) * np.sum(resid ** 2) / dof
    sec_off = coef[2] / p_k                          # secondary phase − 0.5, in cycles
    return dict(P=float(coef[1]), P_err=float(np.sqrt(cov[1, 1])), n=len(toms),
                sec_offset=float(sec_off), oc_rms_s=float(np.std(resid) * 86400))


# ── driver ───────────────────────────────────────────────────────────────────

def catalogue_periods() -> dict:
    out = {}
    for p in glob.glob(os.path.join(paths.TARGETS_DIR, "neglected_targets*.csv")):
        with open(p, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                try:
                    out[row["name"].strip()] = float(row["period"])
                except (KeyError, ValueError):
                    pass
    return out


def read_ephem(safe):
    text = open(os.path.join(paths.TIMING_DIR, f"ephem_{safe}.txt"), encoding="utf-8").read()
    g = lambda k: float(re.search(rf"^{k}\s*=\s*([-+0-9.eE]+)", text, re.MULTILINE).group(1))
    return g("P_new"), g("T0_kreiner"), g("P_kreiner")


def read_lc(safe):
    with fits.open(os.path.join(paths.LIGHTCURVE_DIR, f"lc_{safe}_clean.fits")) as h:
        d = h[1].data
        t = np.asarray(d["TIME"], float)
        f = np.asarray(d["FLUX"], float)
    ok = np.isfinite(t) & np.isfinite(f)
    return t[ok], f[ok]


COLUMNS = ["star", "P_new_file", "P_replica", "replica_minus_file_s", "P_fixed", "P_fixed_err_s",
           "P_best", "P_kreiner", "fixed_minus_best_s", "new_minus_best_s", "kreiner_minus_best_s",
           "rep_candidates", "rep_accepted", "rep_rejected", "rep_edge_frac", "rep_frac_err_1e-4",
           "fixed_n_toms", "fixed_oc_rms_s", "fixed_secondary_offset_cyc"]


if __name__ == "__main__":
    import runstats
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console is cp1252
    _run_tracker = runstats.RunTracker("audit_tom_estimator", catalogue=sys.argv[1:], per_star=False)

    p_best = {}
    if os.path.exists(FOLD_CSV):
        with open(FOLD_CSV, newline="", encoding="utf-8") as fh:
            p_best = {r["star"]: float(r["P_best"]) for r in csv.DictReader(fh)}
    else:
        print("  NOTE: period_fold_audit.csv missing — run audit_period_fold.py first for P_best.")
    cat_p = catalogue_periods()
    wanted = {s.replace(" ", "_") for s in sys.argv[1:]}

    rows = []
    for ephem_path in sorted(glob.glob(os.path.join(paths.TIMING_DIR, "ephem_*.txt"))):
        safe = os.path.basename(ephem_path)[len("ephem_"):-len(".txt")]
        star = safe.replace("_", " ")
        if wanted and safe not in wanted:
            continue
        if not os.path.exists(os.path.join(paths.LIGHTCURVE_DIR, f"lc_{safe}_clean.fits")):
            continue
        try:
            p_file, t0_k, p_k = read_ephem(safe)
            t, f = read_lc(safe)
            rep = replica_period(t, f, cat_p.get(star, p_k), t0_k, p_k)
            fix = fixed_period(t, f, t0_k, p_k)
        except Exception as err:
            print(f"  {star}: ERROR {err}")
            continue
        pb = p_best.get(star, np.nan)
        rows.append({
            "star": star, "P_new_file": p_file, "P_replica": rep["P"],
            "replica_minus_file_s": (rep["P"] - p_file) * 86400,
            "P_fixed": fix["P"], "P_fixed_err_s": fix["P_err"] * 86400,
            "P_best": pb, "P_kreiner": p_k,
            "fixed_minus_best_s": (fix["P"] - pb) * 86400,
            "new_minus_best_s": (p_file - pb) * 86400,
            "kreiner_minus_best_s": (p_k - pb) * 86400,
            "rep_candidates": rep["n_cand"], "rep_accepted": rep["n_acc"],
            "rep_rejected": rep["n_rejected"], "rep_edge_frac": rep["edge_frac"],
            "rep_frac_err_1e-4": rep["frac_err_1e4"],
            "fixed_n_toms": fix["n"], "fixed_oc_rms_s": fix["oc_rms_s"],
            "fixed_secondary_offset_cyc": fix["sec_offset"],
        })
        r = rows[-1]
        print(f"{star:<11} replica−file {r['replica_minus_file_s']:+8.1f} s | "
              f"edge {r['rep_edge_frac']:.2f} | acc {r['rep_accepted']}/{r['rep_candidates']} | "
              f"fixed−best {r['fixed_minus_best_s']:+7.1f} s (±{r['P_fixed_err_s']:.1f}) | "
              f"new−best {r['new_minus_best_s']:+7.1f} s | O−C rms {r['fixed_oc_rms_s']:.0f} s")

    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([f"{r[c]:.10g}" if isinstance(r[c], float) else r[c] for c in COLUMNS])

    def med(key):
        v = np.array([abs(r[key]) for r in rows if np.isfinite(r[key])])
        return np.median(v) if len(v) else np.nan
    print(f"\n{len(rows)} stars | median |replica − file| {med('replica_minus_file_s'):.2f} s | "
          f"median |fixed − best| {med('fixed_minus_best_s'):.2f} s | "
          f"median |new − best| {med('new_minus_best_s'):.1f} s")
    print(f"CSV: {AUDIT_CSV}")
