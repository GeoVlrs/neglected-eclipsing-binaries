"""
================================================================================
AUDIT — BENCHMARK LIGHT CURVE QUALITY CHECK
================================================================================
Summary:
    Applies the same light-curve quality audit as audit_lc.py but targets
    the benchmark validation systems (SW Lac, YY Eri) used during pipeline
    development. Separating benchmark and thesis-sample audits keeps the
    working-directory paths clean and the terminal output unambiguous.

Inputs:
    lc_<star>_clean.fits  — Detrended FITS light curves from Script 02.

Outputs:
    Diagnostic report printed to stdout.

Dependencies:
    pip install numpy astropy
================================================================================
"""

import os

import numpy as np
from astropy.io import fits


def audit_light_curve_fits(fits_path: str) -> None:
    """
    Print a quality report for a single FITS light curve file.

    Identical logic to audit_lc.py; extracted here so that benchmark and
    thesis-sample audits can be run independently without modifying path
    configuration in the shared audit module.

    Parameters
    ----------
    fits_path:
        Path to a lightkurve-format FITS file (TIME, FLUX in extension 1).
    """
    print(f"\n{'─' * 60}")
    print(f"AUDITING: {fits_path}")
    print(f"{'─' * 60}")

    if not os.path.exists(fits_path):
        print(f"  ERROR: File not found — {fits_path}")
        return

    with fits.open(fits_path) as hdul:
        table     = hdul[1].data
        time_arr  = table["TIME"]
        flux_arr  = table["FLUX"]

        valid_mask  = np.isfinite(flux_arr)
        clean_flux  = flux_arr[valid_mask]
        median_flux = float(np.nanmedian(clean_flux))
        rms_noise   = float(np.nanstd(clean_flux))
        fill_pct    = valid_mask.sum() / len(time_arr) * 100.0

        print(f"  Total data points : {len(time_arr)}")
        print(f"  Valid (finite) pts : {valid_mask.sum()} ({fill_pct:.2f}%)")
        print(f"  Median flux        : {median_flux:.6f}")
        print(f"  RMS noise (σ)      : {rms_noise:.6f}  "
              f"({rms_noise / median_flux * 100:.4f}% of median)")

        p01, p99  = np.percentile(clean_flux, [1, 99])
        amplitude = float(p99 - p01)
        print(f"  1st percentile     : {p01:.6f}")
        print(f"  99th percentile    : {p99:.6f}")
        print(f"  Amplitude (p99−p01): {amplitude:.6f}  ({amplitude / median_flux * 100:.2f}%)")

        if valid_mask.sum() > 100:
            t_centred = time_arr[valid_mask] - time_arr[valid_mask][0]
            slope, _  = np.polyfit(t_centred, clean_flux, 1)
            print(f"  Linear trend slope : {slope:.2e} flux/day")

        dt_arr    = np.diff(time_arr[valid_mask])
        median_dt = float(np.median(dt_arr))
        max_dt    = float(np.max(dt_arr))
        gap_ratio = max_dt / median_dt if median_dt > 0 else float("inf")
        print(f"  Median cadence     : {median_dt * 1440:.2f} min")
        print(f"  Largest time gap   : {max_dt * 1440:.2f} min  ({gap_ratio:.1f}× cadence)")


if __name__ == "__main__":
    benchmark_stars = ["SW_Lac", "YY_Eri"]
    for star_safe_name in benchmark_stars:
        audit_light_curve_fits(f"lc_{star_safe_name}_clean.fits")
