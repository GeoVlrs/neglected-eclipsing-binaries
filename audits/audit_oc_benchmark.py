"""
================================================================================
AUDIT — O–C ANALYSIS QUALITY CHECK (BENCHMARK TARGETS)
================================================================================
Summary:
    Applies the same O–C quality audit logic as audit_oc.py but targets
    the benchmark validation systems (SW Lac, YY Eri). Keeping benchmark
    and thesis-sample audits in separate scripts prevents path confusion
    and makes it straightforward to reproduce only the thesis-relevant checks.

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
    Validate O–C analysis outputs for a single benchmark target.

    Parameters
    ----------
    star_name:
        Common variable star designation, e.g. "SW Lac".
    data_dir:
        Directory containing the tominima and ephem files.
    """
    safe_name     = star_name.replace(" ", "_")
    tominima_path = os.path.join(data_dir, f"tominima_{safe_name}.csv")
    ephem_path    = os.path.join(data_dir, f"ephem_{safe_name}.txt")

    print(f"\n{'─' * 60}")
    print(f"O–C AUDIT (benchmark): {star_name}")
    print(f"{'─' * 60}")

    if not os.path.exists(tominima_path) or not os.path.exists(ephem_path):
        print(f"  ERROR: Files not found for {star_name}.")
        return

    tominima_df  = pd.read_csv(tominima_path)
    n_eclipses   = len(tominima_df)
    mean_err_min = tominima_df["T_min_err_days"].mean() * 1440.0

    with open(ephem_path, "r") as ephem_file:
        lines = ephem_file.readlines()

    def extract_ephem_value(key: str) -> float:
        matching = [l for l in lines if l.strip().startswith(key + " ")]
        return float(matching[0].split("=")[1].split("#")[0].strip())

    t0_refined     = extract_ephem_value("T0_new")
    period_refined = extract_ephem_value("P_new")

    t_observed     = tominima_df["T_min_BJD"].values
    epochs         = tominima_df["Epoch_E"].values
    residuals_min  = (t_observed - (t0_refined + period_refined * epochs)) * 1440.0
    rms_oc_min     = float(np.sqrt(np.mean(residuals_min ** 2)))

    print(f"  Eclipses measured       : {n_eclipses}")
    print(f"  Mean ToM uncertainty    : {mean_err_min:.2f} min")
    print(f"  Refined period (P_new)  : {period_refined:.8f} days")
    print(f"  O–C RMS (vs new fit)    : {rms_oc_min:.2f} min")


if __name__ == "__main__":
    for star in ["SW Lac", "YY Eri"]:
        audit_oc_outputs(star)
