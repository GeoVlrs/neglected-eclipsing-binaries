"""
================================================================================
AUDIT — PHOEBE 2 BUNDLE QUALITY CHECK (BENCHMARK TARGETS)
================================================================================
Summary:
    Applies the same PHOEBE 2 bundle quality audit as audit_modeling.py
    but targets the benchmark validation systems (SW Lac, YY Eri). Benchmark
    audits were used during pipeline development to verify that the
    Double-Flip constraint hierarchy, blackbody atmosphere flags, and mesh
    resolution settings were correctly implemented before applying the
    pipeline to the thesis target sample.

Inputs:
    bundle_<star>.phoebe  — Initialised PHOEBE 2 bundles from Script 04.

Outputs:
    Diagnostic summary printed to stdout.

Dependencies:
    pip install phoebe numpy
================================================================================
"""

import os

import numpy as np
import phoebe


def audit_phoebe_bundle(star_name: str, data_dir: str = ".") -> None:
    """
    Load a PHOEBE 2 bundle and report its initial model quality metrics.

    Parameters
    ----------
    star_name:
        Common variable star designation, e.g. "SW Lac".
    data_dir:
        Directory containing the .phoebe bundle file.
    """
    safe_name   = star_name.replace(" ", "_")
    bundle_path = os.path.join(data_dir, f"bundle_{safe_name}.phoebe")

    print(f"\n{'─' * 60}")
    print(f"MODEL AUDIT (benchmark): {star_name}")
    print(f"{'─' * 60}")

    if not os.path.exists(bundle_path):
        print(f"  ERROR: Bundle not found — {bundle_path}")
        return

    try:
        phoebe.logger(clevel="ERROR")
        bundle = phoebe.load(bundle_path)
    except Exception as load_error:
        print(f"  ERROR: Could not load bundle — {load_error}")
        return

    try:
        obs_flux = bundle.get_value("fluxes@lc01@dataset")
        obs_err  = bundle.get_value("sigmas@lc01@dataset")
        syn_flux = bundle.get_value("fluxes@lc01@init_model")

        obs_norm  = obs_flux / np.nanmedian(obs_flux)
        syn_norm  = syn_flux / np.nanmedian(syn_flux)
        residuals = obs_norm - syn_norm
        dof       = max(1, len(obs_flux) - 4)
        chi2_red  = float(np.sum((residuals / obs_err) ** 2) / dof)

        q    = bundle.get_value("q@binary")
        incl = bundle.get_value("incl@binary")

        print(f"  Initial mass ratio q   : {q:.4f}")
        print(f"  Initial inclination i  : {incl:.2f}°")
        print(f"  Initial reduced χ²     : {chi2_red:.2f}")
        print(f"  Max residual amplitude : {np.max(np.abs(residuals)) * 100:.2f}%")

    except Exception as audit_error:
        print(f"  ERROR during audit: {audit_error}")


if __name__ == "__main__":
    for star in ["SW Lac", "YY Eri"]:
        audit_phoebe_bundle(star)
