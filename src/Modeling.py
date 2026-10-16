"""
================================================================================
SCRIPT 04 — PHOEBE 2 BUNDLE CONSTRUCTION & FORWARD MODEL
================================================================================
Summary:
    Initialises a PHOEBE 2 three-dimensional stellar mesh (a "bundle") for each
    target system and computes an initial synthetic light curve as a sanity check
    before the MCMC optimisation in Script 05. The bundle encodes the orbital
    geometry, stellar temperatures, atmospheric physics, and the observed TESS
    light curve, forming the complete forward model for the inverse problem.

    Key implementation choices:

    Blackbody atmospheres:
        Contact binaries have heavily distorted photospheres whose equatorial
        bulges can exhibit log(g) values below the lower boundary of standard
        Castelli & Kurucz atmosphere tables. Forcing a blackbody approximation
        avoids grid-boundary crashes while remaining physically consistent with
        the Stefan–Boltzmann treatment in the thesis.

    "Double Flip" constraint hierarchy (EW systems only):
        PHOEBE 2's default constraint tree fixes the Roche potential Ω, which
        in turn sets the equivalent radius. To allow the fill-out factor f to
        be a free parameter in Script 05, two constraints must be inverted:
            (1) free Ω so that it controls requiv (instead of being fixed),
            (2) free f so that it controls Ω.
        This gives the MCMC sampler direct, physically meaningful control
        over the degree of overcontact.

    Dynamic radiative physics:
        β₁ (gravity darkening) and A (reflection/irradiation) are assigned
        per-component based on the initial T_eff: β₁=1.0, A=1.0 for T ≥ 8000 K
        (radiative envelope, von Zeipel 1924); β₁=0.32, A=0.6 for T < 8000 K
        (convective envelope, Lucy 1967).

    Phase binning:
        The raw TESS light curve (~18 000 points per sector) is binned to
        150 median-flux phase points before attaching to the bundle. This
        reduces the per-forward-model compute time by ~50× while preserving
        the eclipse ingress/egress shape.

Inputs:
    neglected_targets.csv  — Target list from Script 01.
    lc_<star>_clean.fits   — Detrended light curves from Script 02.
    ephem_<star>.txt       — Refined ephemerides from Script 03.

Outputs:
    bundle_<star>.phoebe           — Initialised PHOEBE 2 bundle (passed to Script 05).
    plots/phoebe_init_<star>.png   — Initial forward-model vs. data diagnostic plot.

Dependencies:
    pip install phoebe lightkurve numpy matplotlib astropy

References:
    Prša & Zwitter (2005), ApJ 628, 426 — PHOEBE computational framework.
    Conroy et al. (2020), ApJS 250, 34 — PHOEBE 2 inverse-problem interface.
    Lucy (1967), ZA 65, 89 — convective gravity darkening (β₁=0.32).
    von Zeipel (1924), MNRAS 84, 665 — radiative gravity darkening (β₁=1.0).
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import csv
import os
import sys
import warnings
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8")

# ── Scientific libraries ──────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import phoebe
import lightkurve as lk

warnings.filterwarnings("ignore")
phoebe.logger(clevel="WARNING")

import paths
paths.ensure_dirs()

# Run timing / statistics → Runs/run_stats/ (summary printed and saved at exit,
# one CSV row per star). See runstats.py.
import runstats
run_tracker = runstats.RunTracker("04Modeling", catalogue=sys.argv[1:],
                                  columns=["morphology", "teff1_K", "teff2_init_K", "incl_init", "q_init"])

TESS_BTJD_TO_BJD_OFFSET: float = 2_457_000.0

# Temperature boundary between convective (Lucy 1967) and radiative
# (von Zeipel 1924) envelope physics, corresponding to the A-type transition.
RADIATIVE_CONVECTIVE_BOUNDARY_K: float = 8_000.0
import gravdark                                             # noqa: E402

# Number of phase bins used to reduce the light curve before attaching to
# PHOEBE 2. ~150 bins balance ingress/egress resolution against compute cost.
# The count is ODD so that one bin is centred exactly on phase 0 (the primary
# minimum); with 150 bins phase 0 was a bin EDGE and the deepest point was split
# between two half-bins at ±0.0033 (Pipeline_Log Entry 76).
PHASE_BIN_COUNT: int = 151

# Exposure-time integration (PHOEBE fti_method="oversample") for long-cadence
# data. A 600 s TESS-SPOC FFI exposure spans ~0.015 of a 0.47 d orbit (~2 bins)
# and smooths eclipse contacts; a 120-200 s exposure is no longer than one phase
# bin and is left unintegrated (its smoothing is far below the 0.003 σ floor).
FTI_MIN_CADENCE_S: float = 300.0
FTI_OVERSAMPLE: int = 5

# Primary-minimum refinement on the UNBINNED fold: parabola through the points
# within ±T0_FIT_HALF_WIDTH (phase) of a candidate minimum.
T0_FIT_HALF_WIDTH: float = 0.03

# Systematic photometric error floor (in normalised flux), added in quadrature to
# each bin's statistical error. The standard-error-of-the-mean (std/√N) collapses
# to ~0.001 for clean, well-sampled bins, which drives the reduced χ² of an
# otherwise-good fit to 100+ — because real systematics (starspots, the O'Connell
# effect, imperfect model physics) sit at the ~0.3–1% level, not the formal
# statistical level. This floor keeps the per-bin σ physically meaningful.
SYS_ERROR_FLOOR: float = 0.003

# Triangular mesh resolution per stellar surface. 2000 triangles provides
# flux precision of ~10⁻⁴ while remaining stable for distorted contact geometries.
MESH_TRIANGLE_COUNT: int = 2000

# When True, the fixed primary temperature T1 is anchored to the Gaia DR3
# effective temperature (teff_gspphot) where a match is found, falling back to
# the spectral-type / period estimate otherwise. Requires network access and
# astroquery; failures degrade gracefully to the estimate. Set False to run fully
# offline from the catalogue alone.
USE_GAIA_TEFF: bool = True

# Cone-search radius (arcsec) for cross-matching a target against Gaia DR3.
GAIA_MATCH_RADIUS_ARCSEC: float = 5.0


# =============================================================================
# SECTION 1 — SPECTRAL TYPE → EFFECTIVE TEMPERATURE LOOKUP
# Source: Carroll & Ostlie (2017), Table A-1.
# Used to anchor T1 (the primary temperature, which is held fixed during MCMC).
# =============================================================================

SPECTRAL_TYPE_TO_TEFF: dict[str, int] = {
    "O5": 42000, "O6": 39500, "O7": 37000, "O8": 35300, "O9": 33000,
    "B0": 30000, "B1": 25400, "B2": 22000, "B3": 18700, "B5": 15400,
    "B8": 11400, "B9": 10500,
    "A0":  9520, "A1":  9230, "A2":  8970, "A5":  8200, "A7":  7850,
    "F0":  7200, "F2":  6890, "F5":  6440, "F8":  6200,
    "G0":  5930, "G2":  5830, "G5":  5770, "G8":  5570,
    "K0":  5250, "K2":  4780, "K4":  4560, "K5":  4350, "K7":  4060,
    "M0":  3850, "M1":  3720, "M2":  3580, "M3":  3470, "M4":  3370,
    "M5":  3240, "M6":  3050, "M7":  2940, "M8":  2640,
}


# =============================================================================
# SECTION 2 — TARGET-SPECIFIC INITIAL PARAMETERS
# These values seed the PHOEBE 2 bundle and define the MCMC starting region.
# They are based on literature estimates and exploratory forward modeling.
# DEFAULT_PARAMS is applied to any target not explicitly listed here.
# =============================================================================

TARGET_INITIAL_PARAMS: dict[str, dict] = {
    "WZ And": {"spectral_type": "F5", "q_init": 0.40, "incl_init": 82.0, "fillout_init": 0.15},
    "AA And": {"spectral_type": "G2", "q_init": 0.55, "incl_init": 85.0, "fillout_init": 0.20},
    "AB And": {"spectral_type": "G5", "q_init": 0.45, "incl_init": 78.0, "fillout_init": 0.15},
    "AD And": {"spectral_type": "A0", "q_init": 0.80, "incl_init": 88.0, "fillout_init": 0.10},
    "BD And": {"spectral_type": "A5", "q_init": 0.60, "incl_init": 80.0, "fillout_init": 0.10},
}

# Morphology-specific fallback seeds, applied to any target NOT listed in
# TARGET_INITIAL_PARAMS above. Contact (EW) and detached (EB/SD) systems occupy
# different regions of parameter space, so a single blanket default is inadequate
# once the pipeline runs against the full catalogue. For contact systems the
# primary spectral type (and hence the fixed T1) is estimated from the orbital
# period via the W UMa period–spectral-type relation in
# estimate_contact_spectral_type(); the value below is only the fallback when the
# period is out of range.
DEFAULT_CONTACT_PARAMS: dict = {
    "spectral_type": "G5",    # overridden per-period by estimate_contact_spectral_type()
    "q_init":         0.50,
    "incl_init":     82.0,
    "fillout_init":   0.15,
}

DEFAULT_DETACHED_PARAMS: dict = {
    "spectral_type": "F5",    # detached primaries skew hotter than contact systems
    "q_init":         0.80,
    "incl_init":     85.0,    # detached eclipses require higher inclination to be seen
    "fillout_init":   0.15,   # unused for detached geometry; kept for dict uniformity
}

# Pre-resolved Gaia DR3 primary temperatures for the 65-star working sample,
# bypassing the live Gaia query entirely. Values are teff_msc1 — the PRIMARY
# temperature from Gaia DR3's Multiple Star Classifier (MSC), which decomposes the
# unresolved binary into a primary + secondary using BP/RP spectra and parallax.
# MSC is preferable to the blended single-star teff_gspphot for these systems, and
# is often the ONLY Gaia estimate for the brightest targets (GSP-Phot is not run
# on the brightest sources). Source: Gaia DR3 astrophysical_parameters
# (VizieR I/355/paramp), resolved by Gaia source_id (cone-search I/355/gaiadr3 →
# brightest source → paramp by source_id).
#
# VALIDITY CAVEATS (state in the methodology):
#   - MSC is well validated for FGK binaries; for A-type primaries (teff_msc1
#     ≳ 7000 K) it is less reliable.
#   - MSC grid-ceiling correction: a cluster of values pinned near ~7600 K was
#     cross-checked against SIMBAD spectral types and replaced with spectral-type
#     temperatures (Carroll & Ostlie): V1034 Cyg (B9) 10500 K, RU UMi (A2) 8970 K,
#     V921 Her (A5) 8200 K, DN Cam (F0) 7200 K. The remaining A-type values are
#     kept because their MSC and gspphot agree and are NOT ceiling-pinned.
#   - V836 Cyg and GK Cep are MSC-ceiling-pinned at 7600 K while GSP-Phot indicates
#     ~9100 / ~8600 K; for those two the (hotter) teff_gspphot value is used instead.
#
# NOT pre-resolved here (fall through to live query / period estimate) — resolve
# manually before publication:
#   - EM Cep   : GSP-Phot ~23123 K (early-B / β Cep-type) — MSC pinned 7446 K; likely
#                an outlier that does not belong in a cool-binary sample.
#   - IM Cep   : crowded field, GSP-Phot 6150 vs MSC 4951 K — uncertain source ID.
#   - V1305 Cyg: crowded field, GSP-Phot 7377 vs MSC 6161 K — uncertain source ID.
#   - WZ Cep, TV UMi : no MSC and no GSP-Phot for the bright source.
#
# teff_msc2 (secondary) is noted in comments as a reference seed; the secondary
# temperature is a free MCMC parameter in Script 05.
GAIA_TEFF_OVERRIDE_K: dict[str, float] = {
    # ── Batch 1 ──────────────────────────────────────────────────────────────
    "V1044 Her": 4747.0,   # MSC T2 ~ 3998
    "IV Dra":    4916.0,   # MSC T2 ~ 4800
    "VW Cep":    5470.0,   # MSC T2 ~ 4530  (GSP-Phot N/A: G≈7.3)
    "NU Cyg":    5657.0,   # MSC T2 ~ 4629  (crowded field)
    "RT Hyi":    5306.0,   # MSC T2 ~ 4745
    "RW Dor":    5109.0,   # MSC T2 ~ 4900  (GSP-Phot N/A: G≈10.9)
    "BW Dra":    5807.0,   # MSC T2 ~ 5390  (GSP-Phot N/A: G≈8.0)
    "TZ Boo":    6018.0,   # MSC T2 ~ 5271
    # ── Batch 2 ──────────────────────────────────────────────────────────────
    "V1128 Tau": 5399.0,   # MSC T2 ~ 5438
    "FU Dra":    5643.0,   # MSC T2 ~ 4903
    "V1055 Her": 5787.0,   # MSC T2 ~ 4935
    "DF Hya":    5927.0,   # MSC T2 ~ 5290
    "RZ UMi":    5538.0,   # MSC T2 ~ 5203
    "GM Dra":    6207.0,   # MSC T2 ~ 5363
    "BV Dra":    5807.0,   # MSC T2 ~ 5390
    "TY UMa":    6100.0,   # MSC T2 ~ 5205
    # ── Batch 3 ──────────────────────────────────────────────────────────────
    "V608 Cas":  5497.0,   # MSC T2 ~ 4912  (crowded field)
    "V857 Her":  7204.0,   # MSC T2 ~ 5612  (A-type)
    "FI Boo":    6001.0,   # MSC T2 ~ 5110
    "V1918 Cyg": 6909.0,   # MSC T2 ~ 5807
    "V842 Her":  5909.0,   # MSC T2 ~ 5564
    "Y Sex":     6180.0,   # MSC T2 ~ 6499
    "EF Boo":    6422.0,   # MSC T2 ~ 5470
    # WZ Cep : unresolved — see header note
    # ── Batch 4 ──────────────────────────────────────────────────────────────
    "V899 Her":  6557.0,   # MSC T2 ~ 6892
    "GQ Cnc":    5759.0,   # MSC T2 ~ 4864
    "EF Dra":    6097.0,   # MSC T2 ~ 4673
    "HI Pup":    5925.0,   # MSC T2 ~ 5899
    "V776 Cas":  6916.0,   # MSC T2 ~ 5378
    "VY Sex":    6090.0,   # MSC T2 ~ 5779
    "TV Mus":    5780.0,   # MSC T2 ~ 5204  (crowded field)
    "TY Men":    7201.0,   # MSC T2 ~ 5804  (A-type)
    # ── Batch 5 ──────────────────────────────────────────────────────────────
    "DN Cam":    7200.0,   # F0 (SIMBAD) — MSC 7589 was grid-ceiling-pinned; SP-based Teff
    "BO CVn":    7399.0,   # MSC T2 ~ 6102  (A-type)
    "ES UMa":    7237.0,   # MSC T2 ~ 5605  (A-type)
    "EF Cep":    7412.0,   # MSC T2 ~ 5971  (A-type)
    "FP Boo":    7005.0,   # MSC T2 ~ 5912
    "FN Cam":    6566.0,   # MSC T2 ~ 6334
    "II UMa":    6245.0,   # MSC T2 ~ 6645
    # EM Cep : excluded — see header note (hot early-B outlier)
    # ── Batch 6 ──────────────────────────────────────────────────────────────
    "V358 Cep":  5845.0,   # MSC T2 ~ 5225
    "OT Cep":    6060.0,   # MSC T2 ~ 5365
    "V878 Her":  6400.0,   # MSC T2 ~ 5899
    "V576 Lyr":  7336.0,   # MSC T2 ~ 5950  (A-type)
    "EG Cep":    7399.0,   # MSC T2 ~ 5899  (A-type)
    "EP Aur":    6403.0,   # MSC T2 ~ 5625
    "V505 Cyg":  7085.0,   # MSC T2 ~ 5614  (crowded field)
    # TV UMi : unresolved — see header note
    # ── Batch 7 ──────────────────────────────────────────────────────────────
    "PX Car":    7820.0,   # MSC T2 ~ 6367  (A-type)
    "V921 Her":  8200.0,   # A5 (SIMBAD) — MSC 7583 was grid-ceiling-pinned; SP-based Teff
    "SU Cep":    6898.0,   # MSC T2 ~ 6924
    "V2287 Cyg": 6359.0,   # MSC T2 ~ 6552
    "GK Cep":    8621.0,   # GSP-Phot (MSC ceiling-pinned 7600) — hot A-type
    "WZ Cyg":    6896.0,   # MSC T2 ~ 5786  (crowded field)
    # IM Cep, V1305 Cyg : held — see header note (crowded, uncertain ID)
    # ── Batch 8 ──────────────────────────────────────────────────────────────
    "RZ Dra":    7329.0,   # MSC T2 ~ 5901  (A-type)
    "V1034 Cyg": 10500.0,  # B9 (SIMBAD) — MSC 7600 was grid-ceiling-pinned (~2900 K low); SP-based Teff
    "GW Gem":    7204.0,   # MSC T2 ~ 7287  (A-type)
    "V404 Lyr":  7027.0,   # MSC T2 ~ 5686
    "BE Dra":    7137.0,   # MSC T2 ~ 5877
    "RU UMi":    8970.0,   # A2 (SIMBAD) — MSC 7599 was grid-ceiling-pinned; SP-based Teff
    "V836 Cyg":  9101.0,   # GSP-Phot (MSC ceiling-pinned 7600) — hot A-type
    "TZ Lyr":    6393.0,   # MSC T2 ~ 5335
    "V343 Ori":  7407.0,   # MSC T2 ~ 6462  (A-type)
}


# =============================================================================
# SECTION 3 — HELPER FUNCTIONS
# =============================================================================

# Morphology (contact / detached / semi-detached) comes from the shared module
# morphology.py, imported by Scripts 04, 05 and 06 alike, so a bundle is always
# built, optimised and rendered under the same geometry. It replaces the copies of
# MORPHOLOGY_OVERRIDE / is_contact_morphology / resolve_morphology that used to live
# here and in 05 (and a third, override-blind copy in 06) — Pipeline_Log Entry 76.
from morphology import (                                    # noqa: E402
    CONTACT, DETACHED, SEMIDETACHED, MORPHOLOGY_OVERRIDE,
    is_contact_morphology, resolve_morphology, resolve_morphology_kind,
    read_q_adopted,
)


def estimate_contact_spectral_type(period_days: float) -> str:
    """
    Estimate the primary spectral type of a contact binary from its period.

    W UMa contact binaries follow a monotonic period–spectral-type (period–colour)
    relation: shorter-period systems host later, cooler dwarfs. This provides a
    physically-motivated seed for the *fixed* primary temperature T1 when no
    spectroscopic or Gaia value is available, replacing a blanket constant.

    NOTE: T1 is held fixed during the MCMC (Script 05), so this estimate directly
    sets the final primary temperature. It is a seed of last resort — targets with
    literature or Gaia DR3 temperatures should be given explicit entries in
    TARGET_INITIAL_PARAMS, and Gaia-anchored T1 remains a planned refinement.

    Boundaries are a coarse calibration of the W UMa period–type sequence.

    Parameters
    ----------
    period_days:
        Orbital period in days.

    Returns
    -------
    A spectral-type key present in SPECTRAL_TYPE_TO_TEFF.
    """
    if   period_days < 0.27: return "K0"   # ~5250 K
    elif period_days < 0.33: return "G5"   # ~5770 K
    elif period_days < 0.40: return "G0"   # ~5930 K
    elif period_days < 0.50: return "F8"   # ~6200 K
    elif period_days < 0.65: return "F5"   # ~6440 K
    else:                    return "F2"   # ~6890 K


def resolve_initial_params(
    star_name: str,
    gcvs_type: str,
    period_days: float,
) -> dict:
    """
    Return the initial-parameter seed dict for a target.

    Resolution order:
        1. An explicit per-target entry in TARGET_INITIAL_PARAMS (highest trust).
        2. A morphology-specific default; for contact systems the spectral type
           is further refined from the period via the W UMa relation.

    This lets the pipeline run against the full catalogue without hand-coding
    every target, while still honouring literature-calibrated overrides.
    """
    if star_name in TARGET_INITIAL_PARAMS:
        return TARGET_INITIAL_PARAMS[star_name]

    if resolve_morphology(star_name, gcvs_type):
        params = dict(DEFAULT_CONTACT_PARAMS)
        params["spectral_type"] = estimate_contact_spectral_type(period_days)
        # Contact stars: the mass ratio is FIXED in Script 05, so it must be the
        # adopted one (ContactQSearch.py: a published spectroscopic q, else the
        # photometric q-search minimum). q = 0.5 for every star biased the fitted
        # inclination by 5-16 degrees (Entry 90). The q-search's own best i and f
        # at that q seed the bundle.
        adopted = read_q_adopted(paths.QSEARCH_DIR, star_name)
        if adopted:
            params["q_init"] = adopted["q_adopted"]
            if "incl_at_adopted" in adopted:
                params["incl_init"] = adopted["incl_at_adopted"]
            if "fillout_at_adopted" in adopted:
                params["fillout_init"] = adopted["fillout_at_adopted"]
            print(f"  [q] adopted q = {adopted['q_adopted']:.4f} ({adopted.get('q_source', '?')}"
                  f"{', weak' if adopted.get('q_weak') else ''}); seeds i = "
                  f"{params['incl_init']:.1f}, f = {params['fillout_init']:.2f}")
        else:
            print("  [q] WARNING: no adopted q for this contact star — using the placeholder "
                  "q = 0.5. Run ContactQSearch.py, then re-run Script 04 (Script 05 parks it "
                  "until then).")
        return params

    return dict(DEFAULT_DETACHED_PARAMS)


def query_gaia_dr3_teff(
    star_name: str,
    search_radius_arcsec: float = GAIA_MATCH_RADIUS_ARCSEC,
) -> Optional[float]:
    """
    Look up the Gaia DR3 effective temperature for a target, if available.

    The target name is resolved to coordinates via the CDS Sesame service
    (astropy SkyCoord.from_name), then the nearest Gaia DR3 source within
    `search_radius_arcsec` is taken and its `teff_gspphot` returned.

    CAVEAT: Gaia's teff_gspphot is a single-star SED fit and is therefore biased
    for unresolved binaries. For short-period contact systems the primary
    dominates the blended light, so it remains a far better anchor for the FIXED
    T1 than a period–colour estimate — but it is the system temperature, not a
    decomposed primary value, and should be treated as such in the write-up.

    All failure modes (no network, astroquery absent, no match, masked value)
    return None so the caller can fall back to the spectral-type estimate.

    Parameters
    ----------
    star_name:
        Common designation, e.g. "VW Cep".
    search_radius_arcsec:
        Cone-search radius for the Gaia cross-match.

    Returns
    -------
    Effective temperature in Kelvin, or None if unavailable.
    """
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.gaia import Gaia
    except Exception:
        return None

    try:
        coord = SkyCoord.from_name(star_name)
        job   = Gaia.cone_search_async(coord, radius=search_radius_arcsec * u.arcsec)
        table = job.get_results()
        if table is None or len(table) == 0:
            return None

        # cone_search returns rows sorted by angular distance, so row 0 is nearest.
        for column in ("teff_gspphot", "teff_val"):
            if column in table.colnames:
                try:
                    teff = float(table[column][0])
                except Exception:
                    continue
                if np.isfinite(teff) and 2000.0 < teff < 60000.0:
                    return teff
        return None
    except Exception:
        return None


def load_refined_ephemeris(safe_star_name: str) -> dict[str, float]:
    """
    Parse the ephemeris text file written by Script 03.

    Lines beginning with '#' and lines without '=' are skipped. Inline
    comments (text after '#' on a value line) are stripped before float
    conversion so that the annotated format used in Script 03 is handled
    transparently.

    Parameters
    ----------
    safe_star_name:
        Filesystem-safe star name (spaces replaced with underscores).

    Returns
    -------
    A dictionary mapping parameter names to float values, e.g.:
        {"T0_new": 2459001.3, "P_new": 0.40162, ...}
    An empty dict is returned if the file does not exist.
    """
    ephem_path = os.path.join(paths.TIMING_DIR, f"ephem_{safe_star_name}.txt")
    params: dict[str, float] = {}
    if not os.path.exists(ephem_path):
        return params
    with open(ephem_path) as ephem_file:
        for line in ephem_file:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, value_str = line.split("=", 1)
            try:
                params[key.strip()] = float(value_str.split("#")[0].strip())
            except ValueError:
                pass
    return params


def apply_radiative_physics(bundle: object, component: str, teff_k: float,
                            kind: str = "contact") -> None:
    """
    Set gravity-darkening (β₁) and reflection (A) coefficients for one component.

    The 8000 K threshold corresponds to the spectral type A/F boundary where
    stellar envelopes transition from deep convection (Lucy 1967 coefficients)
    to radiative dominance (von Zeipel 1924 coefficients). Using the wrong set
    would bias the synthetic out-of-eclipse maxima and distort the fitted radii.

    Parameters
    ----------
    bundle:
        Active PHOEBE 2 ParameterSet (bundle).
    component:
        "primary" or "secondary".
    teff_k:
        Current effective temperature of the component (K).
    """
    if kind != "contact":
        # Non-contact stars: a ramp over 6600-8000 K at the star's own temperature
        # (gravdark.py, Entry 93). Script 05 re-applies it, and refreshes the
        # secondary's values once its temperature has been fitted.
        beta, albedo = gravdark.coeffs_for(kind, teff_k)
        bundle.set_value(f"gravb_bol@{component}", beta)
        bundle.set_value(f"irrad_frac_refl_bol@{component}", albedo)
        print(f"    {component.capitalize()} ({teff_k:.0f} K): gravity darkening "
              f"{beta:.2f}, albedo {albedo:.2f}  (ramp 6600-8000 K)")
        return
    if teff_k >= RADIATIVE_CONVECTIVE_BOUNDARY_K:
        bundle.set_value(f"gravb_bol@{component}", 1.0)
        bundle.set_value(f"irrad_frac_refl_bol@{component}", 1.0)
        print(f"    {component.capitalize()} ({teff_k:.0f} K): "
              "radiative physics  β₁=1.0, A=1.0  (von Zeipel 1924)")
    else:
        bundle.set_value(f"gravb_bol@{component}", 0.32)
        bundle.set_value(f"irrad_frac_refl_bol@{component}", 0.6)
        print(f"    {component.capitalize()} ({teff_k:.0f} K): "
              "convective physics  β₁=0.32, A=0.6  (Lucy 1967)")


def bin_phase_curve(
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    n_bins: int = PHASE_BIN_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin a phase-folded light curve into n_bins median-flux phase points.

    Binning from ~18 000 raw TESS points down to 150 bins reduces the
    per-forward-model compute time by ~50× while retaining the ingress/egress
    slope shape that constrains fractional radii and inclination (Prša et al. 2016).

    Parameters
    ----------
    phase:
        Orbital phase array in [−0.5, 0.5].
    flux:
        Normalised flux array.
    flux_err:
        Per-point flux uncertainty array.
    n_bins:
        Number of output phase bins (default 150).

    Returns
    -------
    bin_centres:
        Phase at the centre of each non-empty bin.
    bin_flux:
        Median flux within each non-empty bin.
    bin_flux_err:
        Standard error of the MEAN (std/√N) within each non-empty bin (the bin
        value itself is the median; the median's SE is ~1.25× larger, absorbed
        by the systematic floor below).
        Combined in quadrature with SYS_ERROR_FLOOR; never below that floor.
    """
    bin_edges   = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_flux    = np.full(n_bins, np.nan)
    bin_err     = np.full(n_bins, SYS_ERROR_FLOOR)

    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if np.any(mask):
            bin_flux[i] = np.nanmedian(flux[mask])
            n_in_bin    = int(np.sum(mask))
            if n_in_bin > 1:
                se = np.nanstd(flux[mask]) / np.sqrt(n_in_bin)
                se = se if (se > 0 and np.isfinite(se)) else 0.0
                # Statistical error and the systematic floor added in quadrature.
                bin_err[i] = float(np.hypot(se, SYS_ERROR_FLOOR))

    valid       = np.isfinite(bin_flux)
    return bin_centres[valid], bin_flux[valid], bin_err[valid]


# =============================================================================
# SECTION 4 — TARGET LIST
# =============================================================================

# Optional positional argument selects the target catalogue, so a single batch
# (e.g. Targets/neglected_targets_batch1.csv) can be modelled in isolation.
INPUT_CATALOG: str = sys.argv[1] if len(sys.argv) > 1 else "neglected_targets.csv"

target_list: list[dict] = []
try:
    with open(INPUT_CATALOG, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            target_list.append({
                "name":   row["name"],
                "type":   row["type"],
                "period": float(row["period"]),
            })
except FileNotFoundError:
    print(f"ERROR: {INPUT_CATALOG} not found. Run Script 01 first or check the path.")
    raise SystemExit(1)


# =============================================================================
# SECTION 5 — MAIN BUNDLE CONSTRUCTION LOOP
# =============================================================================

for target in target_list:
    star_name      = target["name"]
    variable_type  = target["type"]
    orbital_period = target["period"]
    safe_name      = star_name.replace(" ", "_")
    run_tracker.star_start(star_name)

    print("\n" + "=" * 70)
    print(f"PHOEBE 2 MODELING: {star_name}  (type: {variable_type})")
    print("=" * 70)

    # ── 5A: LOAD LIGHT CURVE & COMPUTE PHASE ──────────────────────────────────
    lc_fits_path = os.path.join(paths.LIGHTCURVE_DIR, f"lc_{safe_name}_clean.fits")
    if not os.path.exists(lc_fits_path):
        print(f"  WARNING: {lc_fits_path} not found. Skipping.")
        run_tracker.star_end(star_name, "skipped", reason="clean light curve not found")
        continue

    lc = lk.read(lc_fits_path)

    ephemeris    = load_refined_ephemeris(safe_name)
    period_model = ephemeris.get("P_new",  orbital_period)
    t0_model     = ephemeris.get("T0_new", 2_457_000.0)

    time_bjd = lc.time.value + TESS_BTJD_TO_BJD_OFFSET
    # Map BJD times to orbital phase in [−0.5, 0.5], centred on superior conjunction.
    raw_phase = ((time_bjd - t0_model) / period_model + 0.5) % 1.0 - 0.5

    raw_flux = lc.flux.value
    raw_ferr = (
        lc.flux_err.value
        if hasattr(lc, "flux_err")
        else np.full_like(raw_flux, 0.002)
    )

    # ── 5B: PHASE BINNING ─────────────────────────────────────────────────────
    print(f"  Raw light curve: {len(raw_flux)} data points — binning to {PHASE_BIN_COUNT}...")
    phase, flux, flux_err = bin_phase_curve(raw_phase, raw_flux, raw_ferr, PHASE_BIN_COUNT)
    print(f"  Binned to {len(flux)} phase points.")

    # ── 5B-t0: PRIMARY MINIMUM AT PHASE 0 ─────────────────────────────────────
    # Script 03's ephemeris is fitted to the TESS times of minimum (since the
    # 2026-10 fix, Entry 76), so phase 0 of the fold above IS a minimum to within
    # seconds — but it is the minimum that Kreiner's epoch count calls "primary",
    # which need not be the deeper one in TESS. Primary = DEEPER minimum: fit a
    # parabola to the UNBINNED fold within ±T0_FIT_HALF_WIDTH of phase 0 and of
    # phase 0.5, take the deeper vertex, and put it at phase 0 (sub-bin precision).
    # This replaces the binned-argmin anchor and the detached-only "|offset| > 0.4
    # → flip by half a period" rule, which assumed T0_new was good to ±0.1 cycle
    # while the old period bug spread the offsets uniformly over ±0.5 (GK Cep was
    # flipped onto its secondary).
    def _local_minimum(centre: float) -> tuple[float, float]:
        """(phase of the vertex, flux at the vertex) near `centre`, unbinned."""
        d = ((raw_phase - centre + 0.5) % 1.0) - 0.5
        sel = np.abs(d) <= T0_FIT_HALF_WIDTH
        if sel.sum() < 10:
            return centre, float("inf")
        a2, b2, c2 = np.polyfit(d[sel], raw_flux[sel], 2)
        if a2 <= 0:
            k = int(np.argmin(raw_flux[sel]))
            return centre + float(d[sel][k]), float(raw_flux[sel][k])
        x_v = float(np.clip(-b2 / (2 * a2), -T0_FIT_HALF_WIDTH, T0_FIT_HALF_WIDTH))
        return centre + x_v, float(a2 * x_v ** 2 + b2 * x_v + c2)

    phase_a, depth_a = _local_minimum(0.0)
    phase_b, depth_b = _local_minimum(0.5)
    primary_phase_offset = phase_a if depth_a <= depth_b else phase_b
    primary_phase_offset = ((primary_phase_offset + 0.5) % 1.0) - 0.5
    t0_tess = t0_model + primary_phase_offset * period_model
    print(f"  Minima on the unbinned fold: φ={phase_a:+.4f} (flux {depth_a:.4f}), "
          f"φ={phase_b:+.4f} (flux {depth_b:.4f})")
    print(f"  t0 → deeper minimum: {primary_phase_offset:+.5f} cycles"
          f"  = {primary_phase_offset * period_model * 86400:+.1f} s from T0_new")

    # Re-fold and re-bin with the corrected t0 so phase 0 = primary eclipse.
    raw_phase_corr = ((time_bjd - t0_tess) / period_model + 0.5) % 1.0 - 0.5
    phase, flux, flux_err = bin_phase_curve(raw_phase_corr, raw_flux, raw_ferr, PHASE_BIN_COUNT)
    residual_offset = float(phase[np.argmin(flux)])
    print(f"  Re-binned: {len(flux)} points  |  t0_tess = {t0_tess:.6f} BJD  |  deepest bin at "
          f"φ = {residual_offset:+.4f}")
    if abs(residual_offset) > 2.0 / PHASE_BIN_COUNT:
        print(f"  [WARNING] deepest bin is {abs(residual_offset) * PHASE_BIN_COUNT:.1f} bins from "
              f"phase 0 — flat-bottomed or near-equal minima; inspect phoebe_init_{safe_name}.png.")

    # ── 5C: PRIMARY TEMPERATURE ANCHOR ────────────────────────────────────────
    # T1 is held fixed throughout the MCMC, so its source matters: a Gaia DR3
    # measurement is preferred over the period/spectral-type estimate when found.
    init_params = resolve_initial_params(star_name, variable_type, orbital_period)
    t1_kelvin   = float(SPECTRAL_TYPE_TO_TEFF.get(init_params["spectral_type"], 5800))
    t1_source   = f"spectral type {init_params['spectral_type']}, Carroll & Ostlie 2017"

    if star_name in GAIA_TEFF_OVERRIDE_K:
        # Pre-resolved value: use it and skip the network query entirely.
        t1_kelvin = GAIA_TEFF_OVERRIDE_K[star_name]
        t1_source = "Gaia DR3 teff_msc1 (pre-resolved override)"
    elif USE_GAIA_TEFF:
        gaia_teff = query_gaia_dr3_teff(star_name)
        if gaia_teff is not None:
            t1_kelvin = gaia_teff
            t1_source = "Gaia DR3 teff_gspphot (live query)"
        else:
            print("  (Gaia DR3 Teff unavailable — falling back to estimate)")

    print(f"  T1 = {t1_kelvin:.0f} K  ({t1_source})")

    # ── 5D: PHOEBE 2 BUNDLE INSTANTIATION ────────────────────────────────────
    morphology_kind   = resolve_morphology_kind(star_name, variable_type)
    is_contact_binary = morphology_kind == CONTACT
    print(f"\n  [5D] Building bundle  (morphology={morphology_kind})...")
    bundle = phoebe.default_binary(contact_binary=is_contact_binary)

    # ── 5E: ORBITAL PARAMETERS ────────────────────────────────────────────────
    bundle.set_value("period@binary",      period_model)
    bundle.set_value("incl@binary",        init_params["incl_init"])
    bundle.set_value("ecc@binary",         0.0)  # e=0 is justified for P < 1 d (tidal circularisation)
    bundle.set_value("t0_supconj@binary",  t0_tess)   # Bug B fix: TESS-epoch minimum

    # ── 5F: STELLAR TEMPERATURES & ATMOSPHERES ────────────────────────────────
    bundle.set_value("teff@primary",   t1_kelvin)
    bundle.set_value("teff@secondary", t1_kelvin * 0.95)  # initial guess: 5% cooler secondary

    # Blackbody atmospheres sidestep Castelli & Kurucz log(g) boundary crashes
    # at the heavily distorted equatorial bulges of contact binaries.
    bundle.set_value("atm@primary",   "blackbody")
    bundle.set_value("atm@secondary", "blackbody")

    # ── 5G: GEOMETRY & MASS RATIO ─────────────────────────────────────────────
    if is_contact_binary:
        # The "Double Flip" restructures the PHOEBE 2 constraint tree so that
        # fillout_factor → controls Ω_pot → controls requiv, giving the MCMC
        # sampler direct, physically interpretable control over overcontact geometry.
        bundle.flip_constraint("pot@contact_envelope", solve_for="requiv@primary")
        bundle.flip_constraint("fillout_factor@contact_envelope", solve_for="pot@contact_envelope")
        bundle.set_value("q@binary",                  init_params["q_init"])
        bundle.set_value("fillout_factor@component",  init_params.get("fillout_init", 0.15))
    else:
        bundle.set_value("q@binary",         init_params["q_init"])
        bundle.set_value("requiv@primary",   0.8)
        bundle.set_value("requiv@secondary", 0.6)
        if morphology_kind == SEMIDETACHED:
            # The secondary fills its Roche lobe: PHOEBE's 'semidetached'
            # constraint sets requiv@secondary = requiv_max@secondary at every q,
            # so r₂ is no longer a free parameter (Script 05 fits q, i, T₂, r₁).
            bundle.add_constraint("semidetached", "secondary")
            print(f"  [5G] Semi-detached: requiv@secondary constrained to its Roche lobe "
                  f"({bundle.get_value('requiv', component='secondary', context='component'):.3f} R⊙ "
                  f"at q={init_params['q_init']:.2f}).")

    # ── 5H: DYNAMIC RADIATIVE PHYSICS ─────────────────────────────────────────
    print("\n  [5H] Setting radiative physics coefficients...")
    for component in ["primary", "secondary"]:
        apply_radiative_physics(bundle, component, bundle.get_value(f"teff@{component}"),
                                kind=morphology_kind)

    # ── 5I: MESH RESOLUTION ───────────────────────────────────────────────────
    # 2000 triangles balances flux accuracy (~10⁻⁴) against compute time and
    # prevents mesh-collapse errors at extreme fillout factors during MCMC.
    bundle.set_value_all("ntriangles", MESH_TRIANGLE_COUNT)
    print(f"\n  [5I] Mesh resolution: {MESH_TRIANGLE_COUNT} triangles per surface element.")

    # ── 5J: ATTACH TESS LIGHT CURVE DATASET ──────────────────────────────────
    print("\n  [5J] Attaching TESS data as PHOEBE 2 dataset...")

    # Convert binned phase back to absolute BJD times for PHOEBE's time axis.
    # Must use t0_tess (not t0_model) so PHOEBE folds these back to the same phases.
    times_bjd_binned = t0_tess + phase * period_model

    bundle.add_dataset(
        "lc",
        times=times_bjd_binned,
        fluxes=flux,
        sigmas=flux_err,
        dataset="lc01",
        passband="TESS:T",
        overwrite=True,
    )

    # "dataset-scaled" automatically matches the synthetic luminosity normalisation
    # to the observed median flux, removing the need to specify absolute pblum.
    bundle.set_value("pblum_mode", dataset="lc01", value="dataset-scaled")

    # Irradiation method is stored in the COMPUTE OPTIONS, so the MCMC samples the
    # same model that Script 05's final compute and Script 06 report. Before
    # 2026-10 it was never set here: the sampler ran PHOEBE's default (Horvat)
    # while the final compute used Wilson for contact stars (Entry 76).
    irrad_method = "wilson" if is_contact_binary else "horvat"
    bundle.set_value("irrad_method", compute="phoebe01", value=irrad_method)
    print(f"  [5J] irrad_method = {irrad_method} (stored in compute options)")

    # Exposure-time integration for long-cadence (FFI) data.
    cadence_s = float(np.nanmedian(np.diff(lc.time.value)) * 86400.0)
    if cadence_s >= FTI_MIN_CADENCE_S:
        bundle.set_value("exptime", dataset="lc01", value=cadence_s)
        bundle.set_value("fti_method", compute="phoebe01", value="oversample")
        bundle.set_value("fti_oversample", compute="phoebe01", value=FTI_OVERSAMPLE)
        print(f"  [5J] Exposure integration: exptime = {cadence_s:.0f} s, "
              f"fti_oversample = {FTI_OVERSAMPLE}")

    # Blackbody atmospheres require manual limb-darkening coefficients because
    # the PHOEBE 2 automatic lookup tables are keyed to Castelli & Kurucz grids.
    bundle.set_value("ld_mode@primary@lc01",   "manual")
    bundle.set_value("ld_mode@secondary@lc01", "manual")

    # ── 5K: INITIAL FORWARD MODEL ─────────────────────────────────────────────
    print("\n  [5K] Running initial PHOEBE 2 forward model...")
    try:
        bundle.run_compute(
            compute="phoebe01",
            model="init_model",
            overwrite=True,
            irrad_method="none",      # irradiation skipped for the initial guess to save time
            distortion_method="roche",
        )
        print("  Forward model computed successfully.")
    except Exception as compute_error:
        print(f"  Forward model error: {compute_error}")
        run_tracker.star_end(star_name, "failed",
                             reason=f"forward model error: {compute_error}")
        continue

    # ── 5L: DIAGNOSTIC PLOT ───────────────────────────────────────────────────
    try:
        synth_times  = bundle.get_value("times",  model="init_model", dataset="lc01")
        synth_fluxes = bundle.get_value("fluxes", model="init_model", dataset="lc01")
        synth_fluxes = synth_fluxes / np.nanmedian(synth_fluxes)  # normalise to data scale

        synth_phase  = ((synth_times - t0_tess) / period_model + 0.5) % 1.0 - 0.5
        sort_idx     = np.argsort(synth_phase)

        fig, (ax_lc, ax_res) = plt.subplots(
            2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]},
        )
        fig.suptitle(
            f"{star_name} — PHOEBE 2 Initial Forward Model vs. TESS Data",
            fontsize=12, fontweight="bold",
        )

        ax_lc.errorbar(phase, flux, yerr=flux_err, fmt="k.", ms=6, alpha=0.6,
                       elinewidth=1.0, label="TESS data (binned)")
        ax_lc.plot(synth_phase[sort_idx], synth_fluxes[sort_idx], "r-", lw=2.5,
                   label="PHOEBE 2 initial mesh")
        ax_lc.set_ylabel("Normalised Flux")
        ax_lc.legend(fontsize=9)
        ax_lc.grid(alpha=0.3)

        synth_interp = np.interp(
            np.sort(phase), synth_phase[sort_idx], synth_fluxes[sort_idx],
        )
        residuals = flux[np.argsort(phase)] - synth_interp

        ax_res.plot(np.sort(phase), residuals, "g.", ms=5, alpha=0.6)
        ax_res.axhline(0, color="r", lw=1.5)
        ax_res.set_xlabel("Orbital Phase φ")
        ax_res.set_ylabel("Residuals")
        ax_res.grid(alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(paths.PLOT_INIT_DIR, f"phoebe_init_{safe_name}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Initial model plot saved: {plot_path}")

    except Exception as plot_error:
        print(f"  Could not generate diagnostic plot: {plot_error}")

    # ── 5M: SAVE BUNDLE ───────────────────────────────────────────────────────
    bundle_path = os.path.join(paths.BUNDLE_DIR, f"bundle_{safe_name}.phoebe")
    bundle.save(bundle_path)
    print(f"\n  Bundle saved: {bundle_path}")
    print(f"    System type  : {morphology_kind}")
    print(f"    T1 (fixed)   = {bundle.get_value('teff@primary'):.0f} K")
    print(f"    T2 (initial) = {bundle.get_value('teff@secondary'):.0f} K")
    print(f"    i  (initial) = {bundle.get_value('incl@binary'):.2f}°")
    print(f"    q  (initial) = {bundle.get_value('q@binary'):.3f}")
    run_tracker.star_end(
        star_name, "done",
        morphology=morphology_kind,
        teff1_K=bundle.get_value('teff@primary'),
        teff2_init_K=bundle.get_value('teff@secondary'),
        incl_init=bundle.get_value('incl@binary'),
        q_init=bundle.get_value('q@binary'),
    )

print("\n" + "=" * 70)
print("SCRIPT 04 COMPLETE — All PHOEBE 2 bundles constructed and saved.")
print("=" * 70)
