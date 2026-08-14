"""
================================================================================
AUDIT — PHOEBE 2 BUNDLE QUALITY CHECK (THESIS TARGETS)
================================================================================
Summary:
    Loads the PHOEBE 2 bundle files produced by Script 04 for the primary
    thesis targets and reports initial model quality metrics. This audit
    verifies that the forward model produces a physically plausible fit
    before the computationally expensive MCMC optimisation begins.

    Checks performed:
        1. Reduced chi-squared of the initial forward model vs. TESS data.
        2. Maximum absolute residual amplitude.
        3. Geometric sanity check: initial mass ratio q, inclination i,
           and (if contact binary) fill-out factor f or (if detached) radii.

    Fit quality interpretation:
        χ²_red < 10   → GOOD  — initial guess is close; MCMC converges quickly.
        χ²_red 10–100 → FAIR  — some refinement needed; typical for EW systems.
        χ²_red > 100  → POOR  — large optimisation required; check initial params.

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
        Common variable star designation, e.g. "AB And".
    data_dir:
        Directory containing the .phoebe bundle file.
    """
    safe_name   = star_name.replace(" ", "_")
    bundle_path = os.path.join(data_dir, f"bundle_{safe_name}.phoebe")

    print(f"\n{'─' * 60}")
    print(f"MODEL AUDIT: {star_name}")
    print(f"{'─' * 60}")

    if not os.path.exists(bundle_path):
        print(f"  ERROR: Bundle not found — {bundle_path}")
        return

    try:
        bundle = phoebe.load(bundle_path)
    except Exception as load_error:
        print(f"  ERROR: Could not load bundle — {load_error}")
        return

    try:
        obs_flux = bundle.get_value("fluxes@lc01@dataset")
        obs_err  = bundle.get_value("sigmas@lc01@dataset")

        # Determine which model is available; prefer the initial forward model
        # but fall back to the solution model if Script 04 was re-run after Script 05.
        available_model = None
        for candidate in ("init_model", "solution_model"):
            if candidate in bundle.models:
                available_model = candidate
                break

        if available_model is None:
            print("  WARNING: No computed model found in bundle. "
                  "Run Script 04 to generate init_model.")
            return

        syn_flux = bundle.get_value(f"fluxes@lc01@{available_model}")

        # Normalise both arrays to median = 1.0 so that the residuals reflect
        # shape differences rather than absolute-flux scale offsets.
        obs_norm = obs_flux / np.nanmedian(obs_flux)
        syn_norm = syn_flux / np.nanmedian(syn_flux)

        residuals = obs_norm - syn_norm
        dof       = max(1, len(obs_flux) - 4)  # 4 = typical number of free parameters
        chi2_red  = float(np.sum((residuals / obs_err) ** 2) / dof)
        max_resid = float(np.max(np.abs(residuals)))

        print(f"  Model used             : {available_model}")
        print(f"  Initial reduced χ²     : {chi2_red:.2f}")
        print(f"  Max residual amplitude : {max_resid * 100:.2f}%")

        if   chi2_red > 100: quality = "POOR  — large optimisation needed."
        elif chi2_red >  10: quality = "FAIR  — refinement needed."
        else:                quality = "GOOD  — initial guess is close."
        print(f"  Fit quality            : {quality}")

        # ── Geometric sanity checks ───────────────────────────────────────────
        q    = bundle.get_value("q@binary")
        incl = bundle.get_value("incl@binary")
        print(f"  Initial mass ratio q   : {q:.4f}")
        print(f"  Initial inclination i  : {incl:.2f}°")

        # Contact systems use fill-out factor; detached systems use individual radii.
        try:
            fillout = bundle.get_value("fillout_factor@contact_envelope")
            print(f"  Fill-out factor (f)    : {fillout:.4f}  (contact binary)")
        except Exception:
            try:
                r1 = bundle.get_value("requiv@primary@component")
                r2 = bundle.get_value("requiv@secondary@component")
                print(f"  Equiv. radii           : R1 = {r1:.4f},  R2 = {r2:.4f}  (detached/SD)")
            except Exception:
                print("  Geometry parameters    : could not retrieve.")

    except Exception as audit_error:
        print(f"  ERROR during audit: {audit_error}")


if __name__ == "__main__":
    phoebe.logger(clevel="ERROR")  # suppress PHOEBE's verbose logging during the audit

    for star in ["AA And", "AB And", "BD And"]:
        audit_phoebe_bundle(star)
