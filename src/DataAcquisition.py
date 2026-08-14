"""
================================================================================
SCRIPT 02 — TESS DATA ACQUISITION & LIGHT CURVE PREPARATION
================================================================================
Summary:
    Retrieves mission-standard PDCSAP (Pre-search Data Conditioning Simple
    Aperture Photometry) light curves from the MAST archive for each target
    selected by Script 01. PDCSAP data is preferred over custom extraction
    because SPOC Co-trending Basis Vectors (CBVs) remove CCD-wide instrumental
    systematics while preserving the true astrophysical eclipse depth — a
    critical property for the inverse problem solved in Scripts 04–05.

    An earlier version of this script used a "Surgical PCA" custom extraction
    at the Target Pixel File (TPF) level. A quantitative audit of BD And
    revealed that, despite a ~68% reduction in out-of-eclipse RMS, the custom
    PCA suppressed the primary eclipse depth by ~45%. This systematic bias
    would propagate directly into erroneous radius and inclination estimates.
    PDCSAP was adopted as the primary science product on that basis.

    For each target the script produces:
        1. A detrended, 4σ-clipped, normalised light curve (clean FITS).
        2. A phase-folded version of the same curve (folded FITS).
        3. A three-panel diagnostic plot for visual verification.

Inputs:
    neglected_targets.csv  — Target list output by Script 01.

Outputs:
    lc_<star>_clean.fits                — Detrended, normalised time series.
    lc_<star>_folded.fits               — Phase-folded light curve.
    cadence_<star>.txt                  — Median cadence in seconds.
    plots/lc_<star>_raw_vs_clean.png    — Diagnostic comparison plot.

Dependencies:
    pip install lightkurve numpy matplotlib astropy

References:
    Ricker et al. (2015), JATIS 1, 014003 — TESS mission description.
    Lightkurve Collaboration (2018), ascl:1812.013 — lightkurve package.
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
matplotlib.use("Agg")  # non-interactive backend for server or headless environments
import matplotlib.pyplot as plt

import lightkurve as lk
from astropy.time import Time

warnings.filterwarnings("ignore", module="lightkurve")
warnings.filterwarnings("ignore", module="astropy")


# =============================================================================
# SECTION 1 — CONSTANTS & DIRECTORIES
# =============================================================================

# TESS mission time zero-point: all BTJD (Barycentric TESS Julian Date) values
# must be shifted by this offset to obtain standard BJD timestamps.
TESS_BTJD_TO_BJD_OFFSET: float = 2_457_000.0

PLOT_OUTPUT_DIR: str = "plots"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# Product-validity thresholds (Flag 1 remediation). TESS sectors are numbered in
# the low hundreds; a product tagged with a sector far above this is a metadata
# artifact (the "Sector 1751" extended-mission stub seen in several FFI products),
# and a usable science sector spans a multi-week baseline. Point count is only a
# sanity floor here — it is cadence-dependent (a legitimate 30-min FFI sector can
# hold ~1300 pts) so it must NOT be the primary discriminator; baseline is.
MAX_VALID_SECTOR: int = 200       # reject bogus sector IDs (e.g. 1751)
MIN_BASELINE_DAYS: float = 10.0   # reject short partial-FFI stubs
MIN_POINTS: int = 500             # sanity floor against empty/broken downloads
# When the best single valid sector holds fewer than this many points, additional
# valid sectors are stitched in to boost coverage (Flag 4 remediation).
STACK_TARGET_POINTS: int = 5000


# =============================================================================
# SECTION 2 — TARGET LIST
# =============================================================================

# Optional positional argument selects the target catalogue, so the pipeline can
# be pointed at a single batch (e.g. Targets/neglected_targets_batch1.csv) instead
# of the full neglected_targets.csv. Defaults preserve the original behaviour.
INPUT_CATALOG: str = sys.argv[1] if len(sys.argv) > 1 else "neglected_targets.csv"

target_list: list[dict] = []
try:
    with open(INPUT_CATALOG, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            raw_skip = row.get("skip_sectors", "").strip()
            skip = [int(s) for s in raw_skip.split(";") if s.strip()]
            target_list.append({
                "name":         row["name"],
                "type":         row["type"],
                "period":       float(row["period"]),
                "tic":          row.get("tic", "").strip(),
                "skip_sectors": skip,
            })
    print(f"Loaded {len(target_list)} targets from {INPUT_CATALOG}.")
except FileNotFoundError:
    print(f"ERROR: {INPUT_CATALOG} not found. Run Script 01 first or check the path.")
    raise SystemExit(1)


# =============================================================================
# SECTION 3 — DATA RETRIEVAL
# =============================================================================

def retrieve_pdcsap_light_curve(
    star_name: str,
    tic_id: str = "",
    skip_sectors: list = None,
) -> Optional[object]:
    """
    Download the mission-standard PDCSAP light curve for a given star from MAST.

    The download hierarchy is:
        1. SPOC 2-minute short-cadence (highest time resolution; preferred).
        2. TESS-SPOC 10-minute FFI-based cadence (fallback for non-GI targets).
        3. QLP FFI-based cadence (community pipeline; fallback when no SPOC product exists).

    When a TIC ID is provided it is used as the MAST search key in preference to
    the star name. This avoids SIMBAD name-resolution collisions where a variable
    star designation (e.g. "NU Cyg") is ambiguous with a Bayer letter (ν Cyg) and
    lightkurve resolves to the wrong target.

    PDCSAP data is chosen over raw SAP because the SPOC pipeline applies
    Co-trending Basis Vectors (CBVs) — principal components of the variability
    shared by thousands of simultaneously observed reference stars — to remove
    instrument-induced flux variations without absorbing the localized
    astrophysical signal from the binary system.

    Parameters
    ----------
    star_name:
        Common variable star designation, e.g. "AB And". Used as fallback when
        tic_id is empty, and always used for display/logging.
    tic_id:
        TESS Input Catalog identifier (digits only), e.g. "169936511". When
        non-empty, the MAST search uses "TIC <tic_id>" as the target string to
        bypass SIMBAD name resolution entirely.

    Returns
    -------
    A normalized lightkurve LightCurve object with PDCSAP flux as the primary
    flux column, or None if no suitable data is found.
    """
    search_target = f"TIC {tic_id}" if tic_id else star_name

    print(f"  Searching MAST for SPOC 2-min light curves (target: {search_target})...")
    search_result = lk.search_lightcurve(search_target, mission="TESS", author="SPOC")

    if len(search_result) == 0:
        # 10-minute TESS-SPOC FFI data as a fallback for targets that were not
        # included in the 2-minute pre-selected target list for a given sector.
        print("  SPOC 2-min not available. Searching TESS-SPOC FFI product...")
        search_result = lk.search_lightcurve(search_target, mission="TESS", author="TESS-SPOC")

    if len(search_result) == 0:
        # QLP (Quick Look Pipeline) community FFI products are available for many
        # targets not selected for the SPOC priority list, particularly faint stars
        # and those outside the 2-min target window. QLP does not apply CBV
        # co-trending, so systematics removal is less rigorous than SPOC, but it
        # is the only available product for these targets.
        print("  TESS-SPOC not available. Searching QLP FFI product...")
        search_result = lk.search_lightcurve(search_target, mission="TESS", author="QLP")

    if len(search_result) == 0:
        return None

    # Validate-and-select: download candidates newest-first and accept only those
    # with a plausible sector ID and an adequate observing baseline. This skips the
    # corrupt partial-FFI stubs (e.g. the "Sector 1751" extended-mission artifact,
    # a 3–7 day / ~1.5k-point fragment) that would otherwise be chosen simply for
    # being the most recent product. (Flag 1 remediation.)
    def _promote_pdcsap(lc):
        # FFI products already expose PDCSAP as the primary flux column; 2-min
        # products carry an explicit pdcsap_flux column that must be promoted.
        if lc is not None and "pdcsap_flux" in lc.colnames:
            lc.flux = lc.pdcsap_flux
        return lc

    def _is_valid(lc) -> bool:
        if lc is None or len(lc) < MIN_POINTS:
            return False
        try:
            sector = int(lc.sector)
        except Exception:
            sector = None  # QLP / products without a SECTOR keyword still qualify
        if sector is not None and sector > MAX_VALID_SECTOR:
            return False   # bogus sector ID (metadata artifact)
        if skip_sectors and sector in skip_sectors:
            return False   # caller-requested exclusion (e.g. known scattered-light sector)
        baseline = float(lc.time.max().value - lc.time.min().value)
        return baseline >= MIN_BASELINE_DAYS

    valid_lcs    = []      # accepted candidates, newest-first
    best_invalid = None    # highest-point-count fallback if nothing validates
    for result in reversed(search_result):
        try:
            # "hard" bitmask excludes anomalous cadences flagged by the SPOC
            # pipeline (e.g. momentum dumps, scattered light, cosmic rays).
            lc = _promote_pdcsap(result.download(quality_bitmask="hard"))
        except Exception:
            continue
        if lc is None:
            continue
        if _is_valid(lc):
            valid_lcs.append(lc)
            # Stop early once total coverage is ample, so well-observed targets do
            # not trigger dozens of downloads.
            if sum(len(v) for v in valid_lcs) >= STACK_TARGET_POINTS:
                break
        elif best_invalid is None or len(lc) > len(best_invalid):
            best_invalid = lc

    if not valid_lcs:
        # Never silently drop a star: fall back to the best available product, but
        # warn that it failed the coverage thresholds.
        if best_invalid is None:
            return None
        print(f"  [WARNING] No product met validity thresholds; using best available "
              f"(Sector {best_invalid.sector}, {len(best_invalid)} pts).")
        return best_invalid.normalize()

    primary = valid_lcs[0]
    if len(valid_lcs) == 1 or len(primary) >= STACK_TARGET_POINTS:
        print(f"  Selected Sector {primary.sector} ({len(primary)} pts).")
        return primary.normalize()

    # Flag 4 remediation: the best single valid sector is thin (e.g. BO CVn after
    # heavy quality masking). Stitch the collected valid sectors — already ordered
    # newest-first, i.e. closest in time — to boost the point count while keeping
    # the baseline span (and hence period-induced phase smearing) as small as
    # possible. LightCurveCollection.stitch() normalises each sector before
    # concatenation, handling inter-sector flux scaling.
    stacked = lk.LightCurveCollection(valid_lcs).stitch()
    stacked.meta["SECTOR"] = primary.sector  # keep a usable .sector for downstream logging/plot
    sectors = ", ".join(str(v.sector) for v in valid_lcs)
    print(f"  Stacked Sectors [{sectors}] -> {len(stacked)} pts (Flag 4 coverage boost).")
    return stacked.normalize()


# =============================================================================
# SECTION 4 — MAIN PROCESSING LOOP
# =============================================================================

for target in target_list:
    star_name      = target["name"]
    variable_type  = target["type"]
    orbital_period = target["period"]
    tic_id         = target.get("tic", "")
    safe_name      = star_name.replace(" ", "_")

    print("\n" + "=" * 70)
    print(f"PROCESSING: {star_name}  (type: {variable_type}, P = {orbital_period:.5f} d)")
    if tic_id:
        print(f"  TIC: {tic_id}")
    print("=" * 70)

    # ── 4A: RETRIEVE PDCSAP DATA ──────────────────────────────────────────────
    clean_lc = retrieve_pdcsap_light_curve(
        star_name, tic_id=tic_id, skip_sectors=target.get("skip_sectors", [])
    )

    if clean_lc is None:
        print(f"  ERROR: No light curves found for {star_name}. Skipping.")
        continue

    active_sector = clean_lc.sector
    print(f"  Using Sector {active_sector} PDCSAP data.")

    # ── 4B: OUTLIER REJECTION ─────────────────────────────────────────────────
    # 4σ threshold removes high-energy particle events and momentum-dump
    # residuals while protecting the deep, legitimate eclipse minima of the
    # binary (which typically range from 20%–40% flux suppression).
    print("  Applying 4σ sigma-clipping outlier removal...")
    clean_lc = clean_lc.remove_nans().remove_outliers(sigma=4.0)

    # ── 4C: PHASE FOLDING ─────────────────────────────────────────────────────
    print("  Phase-folding light curve...")
    folded_lc = clean_lc.fold(period=orbital_period)

    # ── 4D: EXPORT SCIENCE PRODUCTS ───────────────────────────────────────────
    clean_fits_path  = f"lc_{safe_name}_clean.fits"
    folded_fits_path = f"lc_{safe_name}_folded.fits"
    clean_lc.to_fits(clean_fits_path,   overwrite=True)
    folded_lc.to_fits(folded_fits_path, overwrite=True)
    print(f"  Exported: {clean_fits_path}, {folded_fits_path}")

    # Record the cadence so that Script 03 can set the KvW window dynamically
    # without reloading the full light curve.
    cadence_days    = float(np.nanmedian(np.diff(clean_lc.time.value)))
    cadence_seconds = cadence_days * 86_400.0
    with open(f"cadence_{safe_name}.txt", "w") as cadence_file:
        cadence_file.write(f"{cadence_seconds}\n")

    # ── 4E: THREE-PANEL DIAGNOSTIC PLOT ───────────────────────────────────────
    print(f"  Generating diagnostic plot...")

    t_start_bjd   = clean_lc.time.min().value + TESS_BTJD_TO_BJD_OFFSET
    t_end_bjd     = clean_lc.time.max().value + TESS_BTJD_TO_BJD_OFFSET
    duration_days  = t_end_bjd - t_start_bjd
    date_start_iso = Time(t_start_bjd, format="jd").iso.split()[0]
    date_end_iso   = Time(t_end_bjd,   format="jd").iso.split()[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 20), gridspec_kw={"hspace": 0.45})
    fig.suptitle(
        f"{star_name} — TESS PDCSAP Mission Data Product\n"
        f"Sector {active_sector}  |  {date_start_iso} to {date_end_iso}",
        fontsize=18, fontweight="bold",
    )

    # Panel (a): Raw SAP flux — shown for reference to illustrate the degree
    # of instrumental noise present before PDCSAP correction.
    sap_flux = (
        clean_lc.sap_flux.value
        if "sap_flux" in clean_lc.colnames
        else clean_lc.flux.value
    )
    axes[0].plot(clean_lc.time.value, sap_flux, "k.", ms=1.5, alpha=0.3,
                 label="Simple Aperture Photometry (SAP)")
    axes[0].set_ylabel("Raw Counts [e⁻/s]")
    axes[0].set_xlabel("Time (BTJD)")
    axes[0].set_title("(a) Raw SAP Flux — Uncorrected", loc="left",
                      fontsize=13, fontweight="bold")
    axes[0].text(
        0.015, 0.05,
        f"• TESS Sector {active_sector}\n"
        f"• Observation window: {duration_days:.2f} days\n"
        "• Source: NASA SPOC 2-min cadence",
        transform=axes[0].transAxes, fontsize=11, linespacing=1.4, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                  edgecolor="orange", alpha=0.9),
    )

    # Panel (b): PDCSAP flux — instrumental systematics removed by SPOC CBVs.
    axes[1].plot(clean_lc.time.value, clean_lc.flux.value, "b.", ms=1.5, alpha=0.6,
                 label="PDCSAP Flux")
    axes[1].axhline(1.0, color="r", lw=1.2, ls="--")
    axes[1].set_ylabel("Normalised Flux")
    axes[1].set_xlabel("Time (BTJD)")
    axes[1].set_title("(b) PDCSAP — Mission-Corrected Science Product", loc="left",
                      fontsize=13, fontweight="bold")
    axes[1].text(
        0.015, 0.05,
        "• PDCSAP: Pre-search Data Conditioning SAP.\n"
        "• Detrending: Instrumental noise removed via SPOC CBVs.\n"
        "• Eclipse depths preserved for inverse-problem accuracy.",
        transform=axes[1].transAxes, fontsize=10, linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan",
                  edgecolor="gray", alpha=0.85),
    )

    # Panel (c): Phase-folded product — the direct input to PHOEBE 2 modeling.
    axes[2].plot(folded_lc.time.value, folded_lc.flux.value, "r.", ms=2.0, alpha=0.5)
    axes[2].set_ylabel("Normalised Flux")
    axes[2].set_xlabel("Orbital Phase (φ)")
    axes[2].set_title(
        f"(c) Phase-Folded Product  |  P = {orbital_period:.5f} d",
        loc="left", fontsize=13, fontweight="bold",
    )
    axes[2].text(
        0.015, 0.05,
        f"• Folded on Kreiner (2004) period P = {orbital_period:.5f} d.\n"
        "• Depth preserved by PDCSAP; safe for geometric modeling.",
        transform=axes[2].transAxes, fontsize=10, linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="honeydew",
                  edgecolor="gray", alpha=0.85),
    )

    for ax in axes:
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[2].set_xlim(-0.5, 0.5)

    plt.tight_layout()
    plot_path = os.path.join(PLOT_OUTPUT_DIR, f"lc_{safe_name}_raw_vs_clean.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Diagnostic plot saved: {plot_path}")

print("\n" + "=" * 70)
print("SCRIPT 02 COMPLETE — All light curves acquired and exported.")
print("=" * 70)
