"""
================================================================================
AUDIT — O–C ANALYSIS QUALITY VERIFICATION (THESIS TARGETS)
================================================================================
Summary:
    Validates the outputs of Script 03 for the primary thesis targets.
    Loads the ToM database (tominima_<star>.csv) and the refined ephemeris
    (ephem_<star>.txt) and reports key quality indicators:

        - Number of successfully extracted Times of Minimum.
        - Mean formal ToM uncertainty (precision of the KvW/polynomial fits).
        - Refined orbital period.
        - RMS scatter of the ToM residuals against the new linear ephemeris.

    Threshold flags are raised when the residual RMS exceeds 5 minutes
    (indicating an unstable period or excessive photometric noise) or when
    fewer than 5 eclipses were detected (risking a poorly constrained ephemeris).

Inputs:
    tominima_<star>.csv  — ToM database from Script 03.
    ephem_<star>.txt     — Refined ephemeris from Script 03.

Outputs:
    Diagnostic summary printed to stdout.

Dependencies:
    pip install pandas numpy
================================================================================
"""

import os

import numpy as np
import pandas as pd


def audit_oc_outputs(star_name: str, data_dir: str = ".") -> None:
    """
    Validate the O–C analysis outputs for a single target system.

    Parameters
    ----------
    star_name:
        Common variable star designation, e.g. "AB And".
    data_dir:
        Directory containing the tominima and ephem files.
        Defaults to the current working directory.
    """
    safe_name   = star_name.replace(" ", "_")
    tominima_path = os.path.join(data_dir, f"tominima_{safe_name}.csv")
    ephem_path    = os.path.join(data_dir, f"ephem_{safe_name}.txt")

    print(f"\n{'─' * 60}")
    print(f"O–C AUDIT: {star_name}")
    print(f"{'─' * 60}")

    if not os.path.exists(tominima_path) or not os.path.exists(ephem_path):
        print(f"  ERROR: Output files not found for {star_name}.")
        print(f"    Expected: {tominima_path}")
        print(f"    Expected: {ephem_path}")
        return

    # ── Load ToM database ─────────────────────────────────────────────────────
    tominima_df    = pd.read_csv(tominima_path)
    n_eclipses     = len(tominima_df)
    mean_err_min   = tominima_df["T_min_err_days"].mean() * 1440.0  # convert days → minutes
    std_err_min    = tominima_df["T_min_err_days"].std()  * 1440.0

    # ── Parse refined ephemeris ───────────────────────────────────────────────
    # The ephem file uses the format "KEY = VALUE  # comment"; we strip the
    # comment portion and convert the numeric value to float.
    with open(ephem_path, "r") as ephem_file:
        lines = ephem_file.readlines()

    def extract_ephem_value(key: str) -> float:
        matching = [l for l in lines if l.strip().startswith(key + " ")]
        if not matching:
            raise ValueError(f"Key '{key}' not found in {ephem_path}")
        return float(matching[0].split("=")[1].split("#")[0].strip())

    t0_refined     = extract_ephem_value("T0_new")
    period_refined = extract_ephem_value("P_new")

    # ── Residuals against the new linear ephemeris ────────────────────────────
    t_observed = tominima_df["T_min_BJD"].values
    epochs     = tominima_df["Epoch_E"].values
    residuals_days = t_observed - (t0_refined + period_refined * epochs)
    rms_oc_min     = float(np.sqrt(np.mean((residuals_days * 1440.0) ** 2)))

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"  Eclipses measured      : {n_eclipses}")
    print(f"  Mean ToM uncertainty   : {mean_err_min:.2f} ± {std_err_min:.2f} min")
    print(f"  Refined period (P_new) : {period_refined:.8f} days")
    print(f"  O–C RMS (vs new fit)   : {rms_oc_min:.2f} min")

    print()
    if n_eclipses < 5:
        print("  ⚠ FLAG: Fewer than 5 eclipses detected. "
              "Ephemeris may be poorly constrained.")
    if rms_oc_min > 5.0:
        print("  ⚠ FLAG: O–C residual RMS exceeds 5 minutes. "
              "Check for period instability or excessive photometric scatter.")
    if n_eclipses >= 5 and rms_oc_min <= 5.0:
        print("  ✓ O–C quality checks passed.")


if __name__ == "__main__":
    for star in ["AA And", "AB And", "BD And"]:
        audit_oc_outputs(star)
