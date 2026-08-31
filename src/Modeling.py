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

PLOT_OUTPUT_DIR: str = "plots"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

TESS_BTJD_TO_BJD_OFFSET: float = 2_457_000.0

# Temperature boundary between convective (Lucy 1967) and radiative
# (von Zeipel 1924) envelope physics, corresponding to the A-type transition.
RADIATIVE_CONVECTIVE_BOUNDARY_K: float = 8_000.0

# Number of phase bins used to reduce the light curve before attaching to
# PHOEBE 2. 150 bins balances ingress/egress resolution against compute cost.
PHASE_BIN_COUNT: int = 150

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

def is_contact_morphology(gcvs_type: str) -> bool:
    """
    Decide whether a GCVS variability type denotes a contact (overcontact) binary.

    The GCVS strings in the catalogue carry qualifier suffixes after a slash or
    colon — e.g. "EW/KW", "EW/DW", "EW:", "EW/RS" are all W UMa contact systems,
    while "EB", "EB/SD", "EB/KE" are detached or semi-detached. A naive
    `gcvs_type == "EW"` test (used in earlier drafts) silently misclassified
    every qualified contact system as detached, building it with the wrong Roche
    geometry. We therefore strip qualifiers and test only the leading code.

    Parameters
    ----------
    gcvs_type:
        Raw GCVS type string from the target catalogue (e.g. "EW/KW").

    Returns
    -------
    True for contact (EW-family) systems, False otherwise (EB / SD / detached).
    """
    base = gcvs_type.strip().upper().replace(":", "").split("/")[0]
    return base == "EW"


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

    if is_contact_morphology(gcvs_type):
        params = dict(DEFAULT_CONTACT_PARAMS)
        params["spectral_type"] = estimate_contact_spectral_type(period_days)
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
    ephem_path = f"ephem_{safe_star_name}.txt"
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


def apply_radiative_physics(bundle: object, component: str, teff_k: float) -> None:
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
        Standard error of the median within each non-empty bin.
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

    print("\n" + "=" * 70)
    print(f"PHOEBE 2 MODELING: {star_name}  (type: {variable_type})")
    print("=" * 70)

    # ── 5A: LOAD LIGHT CURVE & COMPUTE PHASE ──────────────────────────────────
    lc_fits_path = f"lc_{safe_name}_clean.fits"
    if not os.path.exists(lc_fits_path):
        print(f"  WARNING: {lc_fits_path} not found. Skipping.")
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

    # ── BUG B FIX: anchor t0 to the TESS-epoch primary eclipse minimum ────────
    # T0_new from Script 03 is anchored at a Kreiner-2002 epoch. Propagating
    # forward ~20 000 cycles of TESS observations with any residual period error
    # shifts the data primary eclipse away from phase 0, preventing the model
    # from matching depth regardless of inclination. We therefore re-anchor t0
    # to the deepest binned minimum of this star's own TESS light curve, which
    # by definition places the primary eclipse at phase 0 in the bundle.
    primary_phase_offset = float(phase[np.argmin(flux)])
    t0_tess = t0_model + primary_phase_offset * period_model
    print(f"  t0 correction: {primary_phase_offset:+.6f} cycles"
          f"  = {primary_phase_offset * period_model * 86400:+.1f} s"
          f"  (T0_new → TESS epoch)")

    # For DETACHED binaries (EB/SD) only: if |offset| > 0.4, a flat-bottomed total
    # secondary eclipse can produce a lower median-flux per 150-bin than the
    # V-shaped primary transit, so argmin(bin_flux) catches the secondary. Flip
    # by half a period to recover the primary. For CONTACT binaries (EW/KW) the
    # two eclipses are similarly shaped so argmin reliably finds the deeper
    # (= primary) eclipse regardless of how large the offset is.
    if abs(primary_phase_offset) > 0.4 and not is_contact_morphology(variable_type):
        primary_phase_offset -= np.sign(primary_phase_offset) * 0.5
        t0_tess = t0_model + primary_phase_offset * period_model
        print(f"  (EB |offset|>0.4 → half-period flip → {primary_phase_offset:+.6f} cycles)")

    # Re-fold and re-bin with the corrected t0 so phase 0 = primary eclipse.
    raw_phase_corr = ((time_bjd - t0_tess) / period_model + 0.5) % 1.0 - 0.5
    phase, flux, flux_err = bin_phase_curve(raw_phase_corr, raw_flux, raw_ferr, PHASE_BIN_COUNT)
    print(f"  Re-binned: {len(flux)} points  |  t0_tess = {t0_tess:.6f} BJD")

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
    is_contact_binary = is_contact_morphology(variable_type)
    print(f"\n  [5D] Building bundle  (contact_binary={is_contact_binary})...")
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

    # ── 5H: DYNAMIC RADIATIVE PHYSICS ─────────────────────────────────────────
    print("\n  [5H] Setting radiative physics coefficients...")
    for component in ["primary", "secondary"]:
        apply_radiative_physics(bundle, component, bundle.get_value(f"teff@{component}"))

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
        continue

    # ── 5L: DIAGNOSTIC PLOT ───────────────────────────────────────────────────
    try:
        synth_times  = bundle.get_value("times",  model="init_model", dataset="lc01")
        synth_fluxes = bundle.get_value("fluxes", model="init_model", dataset="lc01")
        synth_fluxes = synth_fluxes / np.nanmedian(synth_fluxes)  # normalise to data scale

        synth_phase  = ((synth_times - t0_model) / period_model + 0.5) % 1.0 - 0.5
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
        plot_path = os.path.join(PLOT_OUTPUT_DIR, f"phoebe_init_{safe_name}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Initial model plot saved: {plot_path}")

    except Exception as plot_error:
        print(f"  Could not generate diagnostic plot: {plot_error}")

    # ── 5M: SAVE BUNDLE ───────────────────────────────────────────────────────
    bundle_path = f"bundle_{safe_name}.phoebe"
    bundle.save(bundle_path)
    print(f"\n  Bundle saved: {bundle_path}")
    print(f"    System type  : {'Contact (EW)' if is_contact_binary else 'Detached/SD (EB)'}")
    print(f"    T1 (fixed)   = {bundle.get_value('teff@primary'):.0f} K")
    print(f"    T2 (initial) = {bundle.get_value('teff@secondary'):.0f} K")
    print(f"    i  (initial) = {bundle.get_value('incl@binary'):.2f}°")
    print(f"    q  (initial) = {bundle.get_value('q@binary'):.3f}")

print("\n" + "=" * 70)
print("SCRIPT 04 COMPLETE — All PHOEBE 2 bundles constructed and saved.")
print("=" * 70)
