"""
================================================================================
AUDIT — BUNDLE PHASE ALIGNMENT CHECK (POST-SCRIPT-04 QA GATE)
================================================================================
Summary:
    Verifies that Script 04's t0 correction (04Modeling.py, Bug-B fix) actually
    landed a real eclipse minimum at phase 0 in the SAVED bundle, for a batch
    of rebuilt bundle_<star>.phoebe files. This is the "bundle inspection" check
    Pipeline_Log.txt Entries 25/29 describe as the correct verification method —
    audit_phase_offset.py measures the RAW pre-fix ephemeris drift only, which
    stays large regardless of whether Script 04's correction succeeded, and is
    not a check of the corrected bundle at all.

    Folds the dataset using the ephemeris STORED IN THE BUNDLE
    (period@binary, t0_supconj@binary) — the same values Script 04 anchored —
    never re-derived from ephem_<star>.txt (that pre-fix raw ephemeris is
    exactly the quantity being corrected, so re-deriving from it would just
    reproduce audit_phase_offset.py's blind spot).

    Three phase windows are sampled: phase 0 (primary), phase +-0.5 (secondary),
    and phase +-0.25 (quadrature — the out-of-eclipse baseline). Depth is
    measured relative to the quadrature baseline, not just primary-vs-secondary,
    so a gross misalignment (phase 0 landed off-eclipse entirely) is caught even
    though it would leave both depth@0 and depth@0.5 looking "near-equal."

    Expected outcome: these are EW contact systems, several with high T2/T1
    (near-twin temperatures), so "near-equal — review" is expected to fire on
    several genuinely fine stars (RZ UMi, Entry 25, is the known extreme case)
    and is informational only. The checks that indicate a REAL problem are
    MISALIGNED (phase 0 is not a minimum at all) and SWAPPED (phase 0.5 is
    deeper than phase 0) — both should be rare given argmin()'s demonstrated
    reliability for contact/EW morphology.

Inputs:
    bundle_<star>.phoebe  — rebuilt PHOEBE 2 bundles from Script 04.

Outputs:
    Diagnostic summary table printed to stdout.

Usage:
    Run from Code/ (bundles are located via paths.BUNDLE_DIR):
    python Audits/audit_bundle_phase_alignment.py <catalog.csv> [<catalog2.csv> ...]

Dependencies:
    pip install phoebe numpy
================================================================================
"""

import csv
import os
import sys

import numpy as np
import phoebe

# Audits/ sits one level below Code/, where paths.py lives.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths                                              # noqa: E402

WINDOW_HALF_WIDTH = 0.03      # phase units either side of each target phase
NEAR_EQUAL_REL_TOL = 0.15     # |depth0 - depth5| / max(depth0, depth5) below this -> "near-equal"
MIN_SIGMA_MULTIPLE = 3.0      # a dip must clear this many local sigmas to count as real


def _window_mean(phase: np.ndarray, flux: np.ndarray, sigma: np.ndarray, target: float, half_width: float):
    dist = np.abs(((phase - target + 0.5) % 1.0) - 0.5)
    mask = dist <= half_width
    if not np.any(mask):
        return np.nan, np.nan
    return float(np.mean(flux[mask])), float(np.median(sigma[mask]))


def audit_bundle_phase_alignment(star_name: str, data_dir: str | None = None) -> dict:
    # Defaults to paths.BUNDLE_DIR rather than "." — the Entry 39 reorg moved
    # bundles into data/bundles/, so the old default found nothing from Code/.
    data_dir = data_dir if data_dir is not None else paths.BUNDLE_DIR
    safe_name = star_name.replace(" ", "_")
    bundle_path = os.path.join(data_dir, f"bundle_{safe_name}.phoebe")

    result = {"star": star_name, "verdict": "ERROR", "detail": ""}
    try:
        bundle = phoebe.load(bundle_path)
    except Exception as exc:
        result["detail"] = f"could not load bundle: {exc}"
        return result

    try:
        period = bundle.get_value("period@binary")
        t0 = bundle.get_value("t0_supconj@binary")
        times = bundle.get_value("times", dataset="lc01", context="dataset")
        fluxes = bundle.get_value("fluxes", dataset="lc01", context="dataset")
        sigmas = bundle.get_value("sigmas", dataset="lc01", context="dataset")
    except Exception as exc:
        result["detail"] = f"could not read dataset/ephemeris: {exc}"
        return result

    phase = ((times - t0) / period + 0.5) % 1.0 - 0.5

    flux0, sig0 = _window_mean(phase, fluxes, sigmas, 0.0, WINDOW_HALF_WIDTH)
    flux5, sig5 = _window_mean(phase, fluxes, sigmas, 0.5, WINDOW_HALF_WIDTH)
    fluxq_plus, sigq_plus = _window_mean(phase, fluxes, sigmas, 0.25, WINDOW_HALF_WIDTH)
    fluxq_minus, sigq_minus = _window_mean(phase, fluxes, sigmas, -0.25, WINDOW_HALF_WIDTH)

    if any(np.isnan(v) for v in (flux0, flux5, fluxq_plus, fluxq_minus)):
        result["detail"] = "empty phase window — check dataset/binning"
        return result

    baseline = (fluxq_plus + fluxq_minus) / 2.0
    baseline_sigma = max(sigq_plus, sigq_minus, 1e-6)

    depth0 = baseline - flux0
    depth5 = baseline - flux5

    result.update(
        t0_offset_cycles=None,  # filled in by caller from the rebuild log, if available
        depth0=depth0,
        depth5=depth5,
        baseline=baseline,
    )

    dip0_is_real = depth0 > MIN_SIGMA_MULTIPLE * baseline_sigma
    dip5_is_real = depth5 > MIN_SIGMA_MULTIPLE * baseline_sigma

    if not dip0_is_real:
        result["verdict"] = "MISALIGNED"
        result["detail"] = "phase 0 is not a minimum below the quadrature baseline"
        return result

    if dip5_is_real:
        rel_diff = abs(depth0 - depth5) / max(depth0, depth5)
        if rel_diff < NEAR_EQUAL_REL_TOL:
            result["verdict"] = "near-equal — review"
            result["detail"] = "both eclipses similar depth; likely a genuine near-twin EW system"
            return result
        if depth5 > depth0:
            result["verdict"] = "SWAPPED"
            result["detail"] = "phase 0.5 is robustly deeper than phase 0 — primary/secondary reversed"
            return result

    result["verdict"] = "aligned"
    result["detail"] = ""
    return result


if __name__ == "__main__":
    # Run timing → Runs/run_stats/run_summary.txt (written at exit). Inside the
    # __main__ guard so that importing this module never starts a tracker.
    import sys as _rs_sys, os as _rs_os
    _rs_sys.path.insert(0, _rs_os.path.dirname(_rs_os.path.dirname(_rs_os.path.abspath(__file__))))
    import runstats
    _run_tracker = runstats.RunTracker("audit_bundle_phase_alignment", catalogue=_rs_sys.argv[1:],
                                       per_star=False)
    phoebe.logger(clevel="ERROR")  # suppress PHOEBE's verbose logging during the audit

    if len(sys.argv) < 2:
        print("Usage: python audit_bundle_phase_alignment.py <catalog.csv> [<catalog2.csv> ...]")
        sys.exit(1)

    stars = []
    for catalog_path in sys.argv[1:]:
        with open(catalog_path, newline="", encoding="utf-8") as f:
            stars.extend(row["name"] for row in csv.DictReader(f))

    print(f"{'star':<14} {'verdict':<20} {'depth@0':>9} {'depth@0.5':>10} {'baseline':>9}  detail")
    print("-" * 100)
    counts: dict[str, int] = {}
    for star in stars:
        r = audit_bundle_phase_alignment(star)
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        d0 = f"{r.get('depth0', float('nan')):.4f}" if "depth0" in r else "  n/a"
        d5 = f"{r.get('depth5', float('nan')):.4f}" if "depth5" in r else "  n/a"
        bl = f"{r.get('baseline', float('nan')):.4f}" if "baseline" in r else "  n/a"
        print(f"{star:<14} {r['verdict']:<20} {d0:>9} {d5:>10} {bl:>9}  {r['detail']}")

    print("-" * 100)
    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
