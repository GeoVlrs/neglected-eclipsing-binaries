"""
================================================================================
AUDIT — BUG B PHASE-OFFSET QUANTIFICATION (BEFORE/AFTER METRIC)
================================================================================
Summary:
    Measures the phase offset between each target's data primary eclipse and
    phase 0 (the model's superior conjunction) for all 60 active targets.

    Bug B: t0_supconj in the PHOEBE 2 bundles is set to T0_new from Script 03,
    which is anchored at a ~2002 Kreiner epoch. Folding 2020s TESS data across
    ~20 000 cycles of accumulated period error shifts the data primary minimum
    away from phase 0. The MCMC optimiser then cannot match eclipse depth at the
    wrong phase, collapsing inclination to the prior floor.

    This script quantifies the Bug B offset for every target and provides the
    BEFORE metric. Re-running after the Script 04 fix and rebuild should yield
    |phase_offset| < 0.005 for all stars (< 1 bin width at 150 bins).

    Run from the  02,03,04 Data/  directory (same CWD convention as Script 04).

Inputs:
    lc_<star>_clean.fits   — Detrended light curves from Script 02.
    ephem_<star>.txt       — Refined ephemerides from Script 03.

Outputs:
    Diagnostic table printed to stdout (sorted by |phase_offset|).
    ../Audits/phase_offsets.csv  — Machine-readable results.

Dependencies:
    pip install lightkurve numpy
================================================================================
"""

import csv
import glob
import os

import numpy as np
import lightkurve as lk

TESS_BTJD_TO_BJD_OFFSET: float = 2_457_000.0
PHASE_BIN_COUNT: int = 150
FLAG_THRESHOLD: float = 0.02   # flag stars with |offset| above this
OUTPUT_CSV: str = os.path.join(os.path.dirname(__file__), "phase_offsets.csv")


# ---------------------------------------------------------------------------
# Helpers — copied verbatim from Script 04 so the audit uses identical logic.
# ---------------------------------------------------------------------------

def load_refined_ephemeris(safe_star_name: str) -> dict:
    ephem_path = f"ephem_{safe_star_name}.txt"
    params: dict = {}
    if not os.path.exists(ephem_path):
        return params
    with open(ephem_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, value_str = line.split("=", 1)
            try:
                params[key.strip()] = float(value_str.split("#")[0].strip())
            except ValueError:
                pass
    return params


def bin_phase_curve(
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    n_bins: int = PHASE_BIN_COUNT,
):
    bin_edges   = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_flux    = np.full(n_bins, np.nan)
    bin_err     = np.full(n_bins, 0.002)

    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if np.any(mask):
            bin_flux[i] = np.nanmedian(flux[mask])
            n_in_bin    = int(np.sum(mask))
            if n_in_bin > 1:
                se = np.nanstd(flux[mask]) / np.sqrt(n_in_bin)
                bin_err[i] = se if (se > 0 and np.isfinite(se)) else 0.002

    valid = np.isfinite(bin_flux)
    return bin_centres[valid], bin_flux[valid], bin_err[valid]


# ---------------------------------------------------------------------------
# Main audit loop
# ---------------------------------------------------------------------------

lc_files = sorted(glob.glob("lc_*_clean.fits"))
if not lc_files:
    raise SystemExit(
        "No lc_*_clean.fits files found. "
        "Run this script from the  02,03,04 Data/  directory."
    )

results = []

for lc_path in lc_files:
    safe_name = lc_path[len("lc_"):-len("_clean.fits")]
    star_name = safe_name.replace("_", " ")

    ephem = load_refined_ephemeris(safe_name)
    if not ephem:
        print(f"  SKIP {star_name}: ephem file not found.")
        continue

    period_model = ephem.get("P_new", ephem.get("P_kreiner", None))
    t0_model     = ephem.get("T0_new", TESS_BTJD_TO_BJD_OFFSET)
    if period_model is None:
        print(f"  SKIP {star_name}: no period in ephem file.")
        continue

    try:
        lc_data  = lk.read(lc_path)
    except Exception as exc:
        print(f"  SKIP {star_name}: could not read LC ({exc}).")
        continue

    time_bjd = lc_data.time.value + TESS_BTJD_TO_BJD_OFFSET
    raw_flux = lc_data.flux.value
    raw_ferr = (
        lc_data.flux_err.value
        if hasattr(lc_data, "flux_err")
        else np.full_like(raw_flux, 0.002)
    )

    raw_phase = ((time_bjd - t0_model) / period_model + 0.5) % 1.0 - 0.5
    bin_phases, bin_flux, _ = bin_phase_curve(raw_phase, raw_flux, raw_ferr, PHASE_BIN_COUNT)

    phase_offset = float(bin_phases[np.argmin(bin_flux)])
    results.append({
        "star":         star_name,
        "safe_name":    safe_name,
        "phase_offset": phase_offset,
        "t0_model":     t0_model,
        "period_days":  period_model,
        "shift_sec":    phase_offset * period_model * 86400.0,
    })

# Sort by |phase_offset| descending so worst offenders appear first.
results.sort(key=lambda r: abs(r["phase_offset"]), reverse=True)

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

col_w = 18
header = (
    f"{'Star':<{col_w}}  {'phase_offset':>14}  {'shift_sec':>12}  "
    f"{'period_days':>12}  {'flag':>5}"
)
print("\n" + "=" * len(header))
print("BUG B PHASE-OFFSET AUDIT  (sorted by |phase_offset|)")
print("=" * len(header))
print(header)
print("-" * len(header))

for r in results:
    flag = "***" if abs(r["phase_offset"]) > FLAG_THRESHOLD else ""
    print(
        f"{r['star']:<{col_w}}  {r['phase_offset']:>+14.6f}  "
        f"{r['shift_sec']:>+12.1f}  {r['period_days']:>12.7f}  {flag:>5}"
    )

n_flagged = sum(1 for r in results if abs(r["phase_offset"]) > FLAG_THRESHOLD)
print("-" * len(header))
print(f"\n  {len(results)} targets measured.  {n_flagged} flagged (|offset| > {FLAG_THRESHOLD}).")

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["star", "phase_offset", "shift_sec", "t0_model", "period_days"],
    )
    writer.writeheader()
    for r in results:
        writer.writerow({
            "star":         r["star"],
            "phase_offset": f"{r['phase_offset']:.6f}",
            "shift_sec":    f"{r['shift_sec']:.1f}",
            "t0_model":     f"{r['t0_model']:.6f}",
            "period_days":  f"{r['period_days']:.7f}",
        })

print(f"\n  Results written to: {OUTPUT_CSV}")
