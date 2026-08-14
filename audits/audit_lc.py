"""
================================================================================
AUDIT — LIGHT CURVE QUALITY VERIFICATION
================================================================================
Summary:
    Loads a FITS light curve produced by Script 02 and prints a suite of
    statistical diagnostics to verify signal quality before the O–C and
    modeling stages. Flags are raised if the amplitude or scatter fall
    outside expected ranges for short-period eclipsing binaries.

    Checks performed:
        1. Basic flux statistics: median, RMS noise level.
        2. Amplitude proxy: 99th − 1st percentile flux span.
        3. Residual linear trend (systematic drift indicator).
        4. Cadence uniformity and largest time gap.
        5. Amplitude and noise sanity-check flags.

Inputs:
    lc_<star>_clean.fits  — Detrended FITS light curves from Script 02.

Outputs:
    Diagnostic report printed to stdout.

Dependencies:
    pip install numpy astropy
================================================================================
"""

import os
from typing import Optional

import numpy as np
from astropy.io import fits


def audit_light_curve_fits(fits_path: str) -> None:
    """
    Print a comprehensive quality report for a single FITS light curve file.

    Parameters
    ----------
    fits_path:
        Absolute or relative path to a FITS file containing TIME and FLUX
        columns in its first binary table extension (the lightkurve convention).
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

        # ── 1. Basic flux statistics ──────────────────────────────────────────
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

        # ── 2. Eclipse amplitude proxy ────────────────────────────────────────
        # The 1st–99th percentile span is a robust amplitude estimate that
        # avoids contamination from isolated cosmic-ray spikes.
        p01, p99   = np.percentile(clean_flux, [1, 99])
        amplitude  = float(p99 - p01)
        print(f"  1st percentile     : {p01:.6f}  (deepest eclipse floor)")
        print(f"  99th percentile    : {p99:.6f}  (out-of-eclipse peak)")
        print(f"  Amplitude (p99−p01): {amplitude:.6f}  ({amplitude / median_flux * 100:.2f}%)")

        # ── 3. Linear trend ───────────────────────────────────────────────────
        # A non-zero slope indicates residual instrumental systematics that
        # the PDCSAP CBV correction did not fully remove.
        if valid_mask.sum() > 100:
            t_centred  = time_arr[valid_mask] - time_arr[valid_mask][0]
            slope, _   = np.polyfit(t_centred, clean_flux, 1)
            print(f"  Linear trend slope : {slope:.2e} flux/day")

        # ── 4. Cadence uniformity ─────────────────────────────────────────────
        # A large gap (>100 cadences) indicates a data anomaly or sector gap
        # that may disrupt the KvW eclipse-detection in Script 03.
        dt_arr        = np.diff(time_arr[valid_mask])
        median_dt     = float(np.median(dt_arr))
        max_dt        = float(np.max(dt_arr))
        gap_ratio     = max_dt / median_dt if median_dt > 0 else np.inf
        print(f"  Median cadence     : {median_dt * 1440:.2f} min")
        print(f"  Largest time gap   : {max_dt * 1440:.2f} min  ({gap_ratio:.1f}× cadence)")

        # ── 5. Sanity-check flags ─────────────────────────────────────────────
        print()
        if amplitude < 0.01:
            print("  ⚠ FLAG: Very low amplitude (<1%). Expected 5–40% for EW/EB systems. "
                  "Check that the correct target was downloaded.")
        if rms_noise > 0.02:
            print("  ⚠ FLAG: High RMS scatter (>2%). The PDCSAP detrending may be "
                  "insufficient; consider inspecting the SAP vs PDCSAP comparison plot.")
        if gap_ratio > 100:
            print(f"  ⚠ FLAG: Large gap detected ({gap_ratio:.0f}× cadence). "
                  "This may split the light curve into disjoint segments.")
        if amplitude >= 0.01 and rms_noise <= 0.02 and gap_ratio <= 100:
            print("  ✓ All quality checks passed.")


if __name__ == "__main__":
    # Default audit targets — adjust paths to match your working directory.
    audit_light_curve_fits("lc_AA_And_clean.fits")
    audit_light_curve_fits("lc_BD_And_clean.fits")
