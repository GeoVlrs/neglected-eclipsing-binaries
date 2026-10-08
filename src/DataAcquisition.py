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

    PDCSAP's co-trending can, for some large-amplitude stars, absorb part of the
    star's own variation (eclipse depths change from cycle to cycle in PDCSAP
    but not in SAP; Pipeline_Log Entry 76). Every candidate sector is therefore
    scored PDCSAP-vs-SAP and the newest CLEAN sector is used; if no sector is
    clean, crowding-corrected SAP flux is used instead (retrieve_pdcsap_light_curve).
    The chosen flux source, crowding keywords and selection reason are written
    to the clean FITS header (FLUXSRC, CROWDSAP, FLFRCSAP, PDCRATIO, SECTSEL).

    For each target the script produces:
        1. A 4σ-clipped, normalised light curve (clean FITS).
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
from astropy.io import fits
from astropy.time import Time

warnings.filterwarnings("ignore", module="lightkurve")
warnings.filterwarnings("ignore", module="astropy")


# =============================================================================
# SECTION 1 — CONSTANTS & DIRECTORIES
# =============================================================================

# TESS mission time zero-point: all BTJD (Barycentric TESS Julian Date) values
# must be shifted by this offset to obtain standard BJD timestamps.
TESS_BTJD_TO_BJD_OFFSET: float = 2_457_000.0

import paths
paths.ensure_dirs()

# Run timing / statistics → Runs/run_stats/ (summary printed and saved at exit,
# one CSV row per star). See runstats.py.
import runstats
run_tracker = runstats.RunTracker("02DataAcquisition", catalogue=sys.argv[1:],
                                  columns=["sector", "n_points", "cadence_s"])

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

# PDCSAP-distortion check (Pipeline_Log Entry 76). A sector is "clean" when the
# cycle-to-cycle amplitude scatter of PDCSAP is at most PDC_CLEAN_RATIO × that of
# SAP in the same sector, or below PDC_NEGLIGIBLE_SCATTER outright.
PDC_CLEAN_RATIO: float = 1.5
PDC_NEGLIGIBLE_SCATTER: float = 0.005
# Cadence preference: ≤ 200 s first (SPOC 120 s, TESS-SPOC 200 s); 600 s FFI
# sectors only if no clean faster sector exists (04 then integrates exposures).
PREFERRED_MAX_CADENCE_S: float = 210.0
MAX_CADENCE_S: float = 610.0


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

def cycle_amplitude_scatter(time_days: np.ndarray, flux: np.ndarray, period: float) -> float:
    """
    Robust relative scatter of the per-orbit amplitude (p95 − p5)/median.

    Each full orbital cycle is normalised by its own median, so slow SAP trends
    do not count as amplitude changes. Same statistic as
    Audits/audit_crowding_pdc.py (Pipeline_Log Entry 76). NaN if fewer than four
    well-covered cycles.
    """
    ok = np.isfinite(time_days) & np.isfinite(flux)
    t, f = time_days[ok], flux[ok]
    if len(t) < 50:
        return float("nan")
    expected = period / float(np.median(np.diff(t)))
    cyc = np.floor((t - t[0]) / period).astype(int)
    amps = []
    for c in np.unique(cyc):
        m = cyc == c
        if m.sum() < 0.8 * expected:
            continue
        fc = f[m] / np.median(f[m])
        amps.append(np.percentile(fc, 95) - np.percentile(fc, 5))
    amps = np.array(amps)
    if len(amps) < 4:
        return float("nan")
    mad = np.median(np.abs(amps - np.median(amps))) * 1.4826
    return float(mad / np.median(amps))


def pdc_quality(lc, period: float) -> dict:
    """
    PDCSAP-vs-SAP check for one downloaded sector.

    PDC co-trending can absorb part of a large-amplitude star's own variation,
    so the eclipse depth then changes from cycle to cycle in PDCSAP but not in
    SAP (Entry 76: GK Cep ×30, FN Cam ×27, EF Boo ×13, V899 Her ×4). The sector
    is CLEAN when PDCSAP adds no amplitude scatter beyond SAP's (ratio ≤
    PDC_CLEAN_RATIO), or when both are negligible. Products without a SAP column
    (QLP) cannot be scored and are reported as unscored.
    """
    t = lc.time.value
    pdc = cycle_amplitude_scatter(t, np.asarray(lc.flux.value, float), period)
    if "sap_flux" not in lc.colnames:
        return {"pdc_scatter": pdc, "sap_scatter": float("nan"), "ratio": float("nan"),
                "clean": None}
    sap = cycle_amplitude_scatter(t, np.asarray(lc.sap_flux.value, float), period)
    ratio = pdc / sap if (np.isfinite(pdc) and np.isfinite(sap) and sap > 0) else float("nan")
    clean = bool((np.isfinite(ratio) and ratio <= PDC_CLEAN_RATIO)
                 or (np.isfinite(pdc) and pdc <= PDC_NEGLIGIBLE_SCATTER))
    return {"pdc_scatter": pdc, "sap_scatter": sap, "ratio": ratio, "clean": clean}


def sap_crowding_corrected(lc, period: float):
    """
    Build a PDCSAP-equivalent light curve from SAP flux, for sectors where PDC
    distorts the signal (the SAP fallback of the Entry 76 "hybrid" decision).

    1. Crowding and flux-fraction correction exactly as SPOC applies them to
       PDCSAP: f = (SAP − (1 − CROWDSAP)·median(SAP)) / FLFRCSAP. The light curve
       therefore stays on the same footing as PDCSAP stars (third light 0 in 04).
    2. Slow-trend removal that cannot touch the orbital signal: the mean flux of
       every FULL orbital cycle (which averages the binary's variation out) is
       interpolated in time, separately in each spacecraft orbit (segments split
       at gaps > 1 d), and divided out.
    """
    crowd = float(lc.meta.get("CROWDSAP") or 1.0)
    flfr = float(lc.meta.get("FLFRCSAP") or 1.0)
    t = lc.time.value
    sap = np.asarray(lc.sap_flux.value, float)
    corrected = (sap - (1.0 - crowd) * np.nanmedian(sap)) / flfr
    trend = np.full_like(corrected, np.nan)
    gaps = np.where(np.diff(t) > 1.0)[0]
    for seg in np.split(np.arange(len(t)), gaps + 1):
        ts, fs = t[seg], corrected[seg]
        cyc = np.floor((ts - ts[0]) / period).astype(int)
        centres, means = [], []
        expected = period / float(np.median(np.diff(ts))) if len(ts) > 1 else 0
        for c in np.unique(cyc):
            m = (cyc == c) & np.isfinite(fs)
            if m.sum() >= 0.8 * expected and expected > 0:
                centres.append(ts[0] + (c + 0.5) * period)
                means.append(np.mean(fs[m]))
        if len(centres) >= 2:
            trend[seg] = np.interp(ts, centres, means)
        elif len(centres) == 1:
            trend[seg] = means[0]
        else:
            trend[seg] = np.nanmedian(fs)
    scale = np.nanmedian(trend) / trend
    out = lc.copy()
    out.flux = (corrected * scale) * lc.sap_flux.unit
    if "sap_flux_err" in lc.colnames:
        out.flux_err = (np.asarray(lc.sap_flux_err.value, float) / flfr * scale) * lc.sap_flux_err.unit
    else:
        out.flux_err = np.full(len(out), np.nan) * lc.sap_flux.unit
    out.meta["FLUXSRC"] = "SAP_CROWDCORR"
    return out


def retrieve_pdcsap_light_curve(
    star_name: str,
    tic_id: str = "",
    skip_sectors: list = None,
    orbital_period: float = 1.0,
) -> tuple[Optional[object], dict]:
    """
    Download the mission-standard PDCSAP light curve for a given star from MAST,
    choosing a sector in which PDCSAP has not distorted the binary's signal.

    The download hierarchy is:
        1. SPOC 2-minute short-cadence (highest time resolution; preferred).
        2. TESS-SPOC FFI-based cadence (200 s / 600 s / 1800 s; fallback for
           non-GI targets).
        3. QLP FFI-based cadence (community pipeline; fallback when no SPOC
           product exists). QLP flux is SAP-like (no CBV co-trending).

    When a TIC ID is provided it is used as the MAST search key in preference to
    the star name. This avoids SIMBAD name-resolution collisions where a variable
    star designation (e.g. "NU Cyg") is ambiguous with a Bayer letter (ν Cyg) and
    lightkurve resolves to the wrong target.

    PDCSAP is preferred over raw SAP because SPOC's Co-trending Basis Vectors
    remove CCD-wide instrumental systematics. For some large-amplitude stars
    the co-trending ALSO absorbs part of the star's own variation (eclipse
    depths change from cycle to cycle in PDCSAP but not in SAP; Pipeline_Log
    Entry 76). Each candidate sector is therefore scored (pdc_quality), and the
    selection rule ("hybrid", user decision 2026-10-07) is:

        a. the NEWEST valid sector that is clean and has cadence ≤ 200 s;
        b. else the newest clean sector with cadence ≤ 600 s;
        c. else SAP flux of the newest valid sector (cadence ≤ 200 s if any),
           crowding-corrected like PDCSAP and detrended by orbit-averaged
           means (sap_crowding_corrected).

    Sectors are downloaded newest-first and the search stops at the first
    sector satisfying (a), so stars whose newest sector is clean cost exactly
    one download, as before.

    Returns
    -------
    (light_curve, selection_info) — light_curve is normalised (or None if no
    product exists); selection_info records the sector, flux source, cadence,
    PDC/SAP ratio, crowding keywords and the reason for the choice.
    """
    search_target = f"TIC {tic_id}" if tic_id else star_name

    print(f"  Searching MAST for SPOC 2-min light curves (target: {search_target})...")
    search_result = lk.search_lightcurve(search_target, mission="TESS", author="SPOC")
    author = "SPOC"

    if len(search_result) == 0:
        # FFI-based TESS-SPOC data as a fallback for targets that were not
        # included in the 2-minute pre-selected target list for a given sector.
        print("  SPOC 2-min not available. Searching TESS-SPOC FFI product...")
        search_result = lk.search_lightcurve(search_target, mission="TESS", author="TESS-SPOC")
        author = "TESS-SPOC"

    if len(search_result) == 0:
        # QLP (Quick Look Pipeline) community FFI products are available for many
        # targets not selected for the SPOC priority list, particularly faint stars
        # and those outside the 2-min target window. QLP does not apply CBV
        # co-trending, so systematics removal is less rigorous than SPOC, but it
        # is the only available product for these targets.
        print("  TESS-SPOC not available. Searching QLP FFI product...")
        search_result = lk.search_lightcurve(search_target, mission="TESS", author="QLP")
        author = "QLP"

    if len(search_result) == 0:
        return None, {}

    def _promote_pdcsap(lc):
        # FFI products already expose PDCSAP as the primary flux column; 2-min
        # products carry an explicit pdcsap_flux column that must be promoted.
        if lc is not None and "pdcsap_flux" in lc.colnames:
            lc.flux = lc.pdcsap_flux
        return lc

    def _sector(lc):
        try:
            return int(lc.sector)
        except Exception:
            return None   # QLP / products without a SECTOR keyword

    def _hard_reject(lc) -> bool:
        # Never usable, not even as a last-resort fallback: bogus sector IDs (the
        # "Sector 1751" FFI stub) and caller-excluded sectors (skip_sectors).
        sector = _sector(lc)
        if sector is not None and sector > MAX_VALID_SECTOR:
            return True
        return bool(skip_sectors) and sector in skip_sectors

    def _is_valid(lc) -> bool:
        if lc is None or len(lc) < MIN_POINTS:
            return False
        baseline = float(lc.time.max().value - lc.time.min().value)
        return baseline >= MIN_BASELINE_DAYS

    def _cadence_s(lc) -> float:
        return float(np.nanmedian(np.diff(lc.time.value)) * 86400.0)

    scored = []            # (lc, quality) for every valid sector, newest-first
    best_invalid = None    # most points among soft-invalid sectors (short baseline)
    chosen = None
    for result in reversed(search_result):
        try:
            # "hard" bitmask excludes anomalous cadences flagged by the SPOC
            # pipeline (momentum dumps, scattered light, cosmic rays). Checked in
            # Entry 76: on every selected sector it keeps exactly the cadences
            # "default" keeps (the flagged cadences already have NaN PDCSAP).
            lc = _promote_pdcsap(result.download(quality_bitmask="hard"))
        except Exception:
            continue
        if lc is None or _hard_reject(lc):
            continue
        if not _is_valid(lc):
            if best_invalid is None or len(lc) > len(best_invalid):
                best_invalid = lc
            continue
        quality = pdc_quality(lc.remove_nans(), orbital_period)
        cad = _cadence_s(lc)
        scored.append((lc, quality, cad))
        tag = {True: "clean", False: "PDC-DISTORTED", None: "unscored"}[quality["clean"]]
        print(f"    Sector {_sector(lc)} ({cad:.0f} s, {len(lc)} pts): PDC/SAP amplitude "
              f"scatter ratio {quality['ratio']:.1f} → {tag}")
        if chosen is None and quality["clean"] in (True, None) and cad <= PREFERRED_MAX_CADENCE_S:
            chosen = (lc, quality, cad, "newest clean sector (<=200 s)")
            # Stop once coverage is ample (Flag 4: thin sectors are stacked below).
            if len(lc) >= STACK_TARGET_POINTS:
                break
        elif chosen is not None:
            if sum(len(s[0]) for s in scored if s[1]["clean"] in (True, None)) >= STACK_TARGET_POINTS:
                break

    if chosen is None:
        clean_any = [s for s in scored if s[1]["clean"] in (True, None) and s[2] <= MAX_CADENCE_S]
        if clean_any:
            lc, q, cad = clean_any[0]
            chosen = (lc, q, cad, f"newest clean sector (<={MAX_CADENCE_S:.0f} s); none clean <=200 s")
        elif scored:
            # SAP fallback: the sector whose SAP amplitude is most stable from
            # cycle to cycle (not simply the newest — for GK Cep the newest is
            # S79, excluded in Entry 10 for scattered light), cadence ≤ 200 s
            # preferred.
            fast = [s for s in scored if s[2] <= PREFERRED_MAX_CADENCE_S] or scored
            fast = sorted(fast, key=lambda s: (s[1]["sap_scatter"]
                                               if np.isfinite(s[1]["sap_scatter"]) else np.inf))
            lc, q, cad = fast[0]
            print(f"  [WARNING] PDCSAP distorted in every scored sector; using SAP flux of "
                  f"Sector {_sector(lc)}, crowding-corrected and orbit-mean detrended.")
            sap_lc = sap_crowding_corrected(lc.remove_nans(), orbital_period)
            chosen = (sap_lc, q, cad, "SAP fallback: no clean PDCSAP sector")

    if chosen is None:
        # Never silently drop a star: fall back to the best soft-invalid product
        # (short baseline), but warn. Hard-rejected sectors are never used.
        if best_invalid is None:
            return None, {}
        print(f"  [WARNING] No product met validity thresholds; using best available "
              f"(Sector {best_invalid.sector}, {len(best_invalid)} pts).")
        q = pdc_quality(best_invalid.remove_nans(), orbital_period)
        chosen = (best_invalid, q, _cadence_s(best_invalid), "below validity thresholds")

    lc, quality, cad, reason = chosen
    flux_source = lc.meta.get("FLUXSRC") or ("QLP_SAP" if author == "QLP" else "PDCSAP")
    info = {
        "sector": _sector(lc), "author": author, "flux_source": flux_source,
        "cadence_s": cad, "pdc_sap_ratio": quality.get("ratio", float("nan")),
        "crowdsap": lc.meta.get("CROWDSAP"), "flfrcsap": lc.meta.get("FLFRCSAP"),
        "reason": reason, "stacked": "",
    }

    # Flag 4 remediation: the chosen sector is thin (e.g. BO CVn after heavy
    # quality masking). Stitch further CLEAN valid sectors of the same flux
    # source, newest-first (closest in time, so the least period smearing).
    # LightCurveCollection.stitch() normalises each sector before concatenation.
    if flux_source == "PDCSAP" and len(lc) < STACK_TARGET_POINTS:
        extra = [s[0] for s in scored
                 if s[0] is not lc and s[1]["clean"] in (True, None) and s[2] <= MAX_CADENCE_S]
        stack, total = [lc], len(lc)
        for e in extra:
            if total >= STACK_TARGET_POINTS:
                break
            stack.append(e)
            total += len(e)
        if len(stack) > 1:
            stacked = lk.LightCurveCollection(stack).stitch()
            # stitch() concatenates in list order (newest sector first), so time
            # runs backwards at each join; Script 03 interpolates in time and needs
            # it increasing (found in Phase C, Entry 79).
            stacked = stacked[np.argsort(stacked.time.value)]
            stacked.meta.update({k: lc.meta.get(k) for k in ("SECTOR", "CROWDSAP", "FLFRCSAP")})
            info["stacked"] = ";".join(str(_sector(s)) for s in stack)
            print(f"  Stacked Sectors [{info['stacked']}] -> {len(stacked)} pts (Flag 4 coverage boost).")
            return stacked.normalize(), info

    print(f"  Selected Sector {info['sector']} ({len(lc)} pts, {cad:.0f} s, {flux_source}): {reason}.")
    return lc.normalize(), info


# =============================================================================
# SECTION 4 — MAIN PROCESSING LOOP
# =============================================================================

for target in target_list:
    star_name      = target["name"]
    variable_type  = target["type"]
    orbital_period = target["period"]
    tic_id         = target.get("tic", "")
    safe_name      = star_name.replace(" ", "_")
    run_tracker.star_start(star_name)

    print("\n" + "=" * 70)
    print(f"PROCESSING: {star_name}  (type: {variable_type}, P = {orbital_period:.5f} d)")
    if tic_id:
        print(f"  TIC: {tic_id}")
    print("=" * 70)

    # ── 4A: RETRIEVE PDCSAP DATA ──────────────────────────────────────────────
    clean_lc, selection = retrieve_pdcsap_light_curve(
        star_name, tic_id=tic_id, skip_sectors=target.get("skip_sectors", []),
        orbital_period=orbital_period,
    )

    if clean_lc is None:
        print(f"  ERROR: No light curves found for {star_name}. Skipping.")
        run_tracker.star_end(star_name, "failed", reason="no light curves found")
        continue

    active_sector = clean_lc.sector
    flux_source   = selection.get("flux_source", "PDCSAP")
    print(f"  Using Sector {active_sector} {flux_source} data.")

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
    clean_fits_path  = os.path.join(paths.LIGHTCURVE_DIR, f"lc_{safe_name}_clean.fits")
    folded_fits_path = os.path.join(paths.LIGHTCURVE_DIR, f"lc_{safe_name}_folded.fits")
    clean_lc.to_fits(clean_fits_path,   overwrite=True)
    folded_lc.to_fits(folded_fits_path, overwrite=True)
    # Selection provenance in the clean FITS header (lightkurve's to_fits drops
    # CROWDSAP/FLFRCSAP). 04 and the audits read these; FLUXSRC records that
    # constant crowding has already been removed (PDCSAP by SPOC, SAP_CROWDCORR
    # by sap_crowding_corrected), so 04 keeps third light at 0.
    with fits.open(clean_fits_path, mode="update") as hdul:
        for hdu in hdul[:2]:
            hdr = hdu.header
            hdr["FLUXSRC"]  = (str(flux_source), "flux column used (Pipeline_Log Entry 76)")
            hdr["LCAUTHOR"] = (str(selection.get("author", "")), "MAST light-curve author")
            if selection.get("crowdsap") is not None:
                hdr["CROWDSAP"] = (float(selection["crowdsap"]), "SPOC target flux fraction in aperture")
            if selection.get("flfrcsap") is not None:
                hdr["FLFRCSAP"] = (float(selection["flfrcsap"]), "SPOC fraction of target flux captured")
            ratio = selection.get("pdc_sap_ratio")
            if ratio is not None and np.isfinite(ratio):
                hdr["PDCRATIO"] = (float(ratio), "PDC/SAP cycle-amplitude scatter ratio")
            # FITS header values must be printable ASCII.
            reason_ascii = str(selection.get("reason", "")).encode("ascii", "replace").decode()
            hdr["SECTSEL"]  = (reason_ascii[:60], "sector selection reason")
            if selection.get("stacked"):
                hdr["STACKED"] = (selection["stacked"], "stitched sectors")
    print(f"  Exported: {clean_fits_path}, {folded_fits_path}")

    # Record the cadence so that Script 03 can set the KvW window dynamically
    # without reloading the full light curve.
    cadence_days    = float(np.nanmedian(np.diff(clean_lc.time.value)))
    cadence_seconds = cadence_days * 86_400.0
    with open(os.path.join(paths.LIGHTCURVE_DIR, f"cadence_{safe_name}.txt"), "w") as cadence_file:
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
        f"{star_name} — TESS {flux_source} Light Curve\n"
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
        f"• Source: {selection.get('author', '')}, {cadence_seconds:.0f} s cadence",
        transform=axes[0].transAxes, fontsize=11, linespacing=1.4, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                  edgecolor="orange", alpha=0.9),
    )

    # Panel (b): the flux passed downstream — PDCSAP (SPOC CBV-corrected), or
    # crowding-corrected SAP where PDC distorts the signal (Entry 76).
    flux_note = {
        "PDCSAP": "PDCSAP: SPOC co-trended (CBV) SAP, crowding-corrected.",
        "SAP_CROWDCORR": "SAP, crowding-corrected as SPOC does; slow trend removed\n"
                         "  by orbit-averaged means (PDC distorted every sector).",
        "QLP_SAP": "QLP SAP-like flux: no CBV co-trending, no crowding correction.",
    }.get(flux_source, flux_source)
    axes[1].plot(clean_lc.time.value, clean_lc.flux.value, "b.", ms=1.5, alpha=0.6,
                 label=f"{flux_source} flux")
    axes[1].axhline(1.0, color="r", lw=1.2, ls="--")
    axes[1].set_ylabel("Normalised Flux")
    axes[1].set_xlabel("Time (BTJD)")
    axes[1].set_title(f"(b) {flux_source} — Science Flux Used Downstream", loc="left",
                      fontsize=13, fontweight="bold")
    axes[1].text(
        0.015, 0.05,
        f"• {flux_note}\n"
        f"• Selection: {selection.get('reason', '')}.\n"
        f"• PDC/SAP amplitude-scatter ratio: {selection.get('pdc_sap_ratio', float('nan')):.2f}.",
        transform=axes[1].transAxes, fontsize=10, linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan",
                  edgecolor="gray", alpha=0.85),
    )

    # Panel (c): Quick-look fold at the CATALOGUE period. Diagnostic only: 04 folds
    # with Script 03's refined ephemeris, not with this.
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
        "• Diagnostic only; Script 04 folds with Script 03's ephemeris.",
        transform=axes[2].transAxes, fontsize=10, linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="honeydew",
                  edgecolor="gray", alpha=0.85),
    )

    for ax in axes:
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[2].set_xlim(-0.5, 0.5)

    plt.tight_layout()
    plot_path = os.path.join(paths.PLOT_LC_DIR, f"lc_{safe_name}_raw_vs_clean.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Diagnostic plot saved: {plot_path}")
    run_tracker.star_end(star_name, "done", sector=active_sector,
                         n_points=len(clean_lc), cadence_s=cadence_seconds)

print("\n" + "=" * 70)
print("SCRIPT 02 COMPLETE — All light curves acquired and exported.")
print("=" * 70)
