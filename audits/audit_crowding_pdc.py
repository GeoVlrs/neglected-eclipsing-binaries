"""
================================================================================
AUDIT — CONTAMINATION AND PDCSAP DISTORTION, PER STAR AND PER CACHED SECTOR
================================================================================
Summary:
    Two data-quality questions that Script 02 never asks (Pipeline_Log Entry 76):

    (1) CROWDING. SPOC's CROWDSAP is the fraction of the aperture flux that
        belongs to the target (from TIC magnitudes). PDCSAP has already had the
        rest subtracted as a CONSTANT, which is right only if the neighbours do
        not vary. A variable neighbour leaves its own signal in PDCSAP. Test: fold
        the light curve at P_best (audit_period_fold.py), subtract the 150-bin
        median template, and search the residual for a periodic signal
        (Lomb-Scargle, 0.05-10 d). A strong peak that is not a harmonic of the
        orbit is a second variable in the aperture.

    (2) PDCSAP DISTORTION. PDC's co-trending can absorb part of a large-amplitude
        star's own variation, so the eclipse depth changes from cycle to cycle.
        Test: per full orbital cycle, amplitude = (p95 − p5) / cycle median; the
        cycle-to-cycle scatter (MAD-based, relative) is compared between PDCSAP
        and SAP of the SAME sector. SAP has no co-trending, so SAP ≪ PDCSAP flags
        PDC distortion. (Slow SAP trends are removed per cycle by the per-cycle
        normalisation.)

    Every CACHED sector of every star is scored, so a "pick another sector"
    decision can be made from numbers. Sectors that are not in the local
    lightkurve cache are NOT downloaded; the catalogue's sector list is reported
    so the gap is visible.

Inputs:
    ~/.lightkurve/cache/mastDownload/{TESS,HLSP}/**.fits  — raw SPOC / TESS-SPOC /
        QLP light-curve files already downloaded by Script 02.
    Targets/neglected_targets*.csv (name, type, period, sectors, tic).
    data/lightcurves/lc_<star>_clean.fits (header SECTOR = the selected sector).
    Audits/period_fold_audit.csv (P_best; run audit_period_fold.py first).

Outputs:
    Audits/crowding_pdc_audit.csv — one row per (star, cached sector).
    Table printed to stdout.

Usage (from Code/; light, single process, ~1 min):
    python Audits/audit_crowding_pdc.py

Dependencies:
    pip install astropy numpy
================================================================================
"""

import csv
import glob
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.timeseries import LombScargle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths                                              # noqa: E402

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".lightkurve", "cache", "mastDownload")
AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIT_CSV = os.path.join(AUDIT_DIR, "crowding_pdc_audit.csv")
FOLD_CSV = os.path.join(AUDIT_DIR, "period_fold_audit.csv")

TEMPLATE_BINS: int = 150
MIN_CYCLE_COVERAGE: float = 0.8      # fraction of the expected points in a cycle
HARMONIC_TOL: float = 0.01           # |P_peak·k/P_orb − m| tolerance for "harmonic"


def catalogue() -> dict:
    """name → dict(type, period, sectors list, tic) from every neglected_targets*.csv."""
    out = {}
    for p in sorted(glob.glob(os.path.join(paths.TARGETS_DIR, "neglected_targets*.csv"))):
        with open(p, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                name = row.get("name", "").strip()
                if not name or not row.get("tic"):
                    continue
                out[name] = {
                    "type": row.get("type", ""),
                    "sectors": [s for s in (row.get("sectors") or "").split(";") if s],
                    "tic": int(float(row["tic"])),
                }
    return out


def cached_files_for(tic: int) -> list[str]:
    tic16 = f"{tic:016d}"
    hits = []
    for p in glob.glob(os.path.join(CACHE_DIR, "**", "*.fits"), recursive=True):
        b = os.path.basename(p)
        if (tic16 in b or f"-{tic}_" in b or f"_{tic}_" in b or f"{tic:010d}" in b) and \
                ("lc.fits" in b or "llc.fits" in b):
            hits.append(p)
    return sorted(hits)


def read_raw(path: str) -> dict:
    """Time, PDCSAP-like and SAP flux (quality == 0), plus header keywords."""
    with fits.open(path) as h:
        hdr = {}
        for e in h[:2]:
            hdr.update({k: e.header.get(k) for k in ("SECTOR", "CROWDSAP", "FLFRCSAP", "TICID",
                                                       "OBJECT", "TIMEDEL")
                        if e.header.get(k) is not None})
        d = h[1].data
        cols = [c.upper() for c in d.columns.names]
        t = np.asarray(d["TIME"], float)
        q = np.asarray(d["QUALITY"], int) if "QUALITY" in cols else np.zeros(len(t), int)
        pdc_col = "PDCSAP_FLUX" if "PDCSAP_FLUX" in cols else ("KSPSAP_FLUX" if "KSPSAP_FLUX" in cols else None)
        sap_col = "SAP_FLUX" if "SAP_FLUX" in cols else None
        pdc = np.asarray(d[pdc_col], float) if pdc_col else None
        sap = np.asarray(d[sap_col], float) if sap_col else None
    product = ("QLP" if "qlp" in path.lower() else
               "TESS-SPOC FFI" if "tess-spoc" in path.lower() else "SPOC 2-min")
    return dict(t=t, q=q, pdc=pdc, sap=sap, pdc_col=pdc_col, hdr=hdr, product=product)


def clean(t, f, q):
    ok = np.isfinite(t) & np.isfinite(f) & (q == 0)
    t, f = t[ok], f[ok]
    return t, f / np.nanmedian(f)


def cycle_amplitude_scatter(t, f, period) -> tuple[float, int]:
    """Robust relative scatter of the per-cycle (p95 − p5)/median amplitude."""
    if len(t) < 50:
        return np.nan, 0
    cadence = np.median(np.diff(t))
    expected = period / cadence
    cyc = np.floor((t - t[0]) / period).astype(int)
    amps = []
    for c in np.unique(cyc):
        m = cyc == c
        if m.sum() < MIN_CYCLE_COVERAGE * expected:
            continue
        fc = f[m] / np.median(f[m])
        amps.append(np.percentile(fc, 95) - np.percentile(fc, 5))
    amps = np.array(amps)
    if len(amps) < 4:
        return np.nan, len(amps)
    mad = np.median(np.abs(amps - np.median(amps))) * 1.4826
    return float(mad / np.median(amps)), len(amps)


def residual_signal(t, f, period) -> dict:
    """Fold at `period`, remove the binned template, Lomb-Scargle the residual."""
    ph = (t / period) % 1.0
    edges = np.linspace(0, 1, TEMPLATE_BINS + 1)
    idx = np.clip(np.digitize(ph, edges) - 1, 0, TEMPLATE_BINS - 1)
    tmpl = np.array([np.median(f[idx == i]) if np.any(idx == i) else np.nan
                     for i in range(TEMPLATE_BINS)])
    centres = 0.5 * (edges[1:] + edges[:-1])
    good = np.isfinite(tmpl)
    model = np.interp(ph, centres[good], tmpl[good], period=1.0)
    resid = f - model
    freq = np.linspace(1 / 10.0, 1 / 0.05, 20000)
    power = LombScargle(t, resid).power(freq)
    k = int(np.argmax(power))
    p_peak = 1.0 / freq[k]
    # peak-to-peak of the residual folded at the peak period (binned medians)
    ph2 = (t / p_peak) % 1.0
    idx2 = np.clip(np.digitize(ph2, edges) - 1, 0, TEMPLATE_BINS - 1)
    t2 = np.array([np.median(resid[idx2 == i]) for i in range(TEMPLATE_BINS) if np.any(idx2 == i)])
    ptp = float(np.nanmax(t2) - np.nanmin(t2)) if len(t2) else np.nan
    # harmonic of the orbit? p_peak ≈ period × m / k for small integers
    # Harmonic or sub-harmonic of the orbit: P/p_peak or p_peak/P close to 1..4.
    ratio = period / p_peak
    harmonic = any(abs(r - round(r)) < HARMONIC_TOL * round(r) and 1 <= round(r) <= 4
                   for r in (ratio, 1.0 / ratio))
    return dict(resid_rms=float(np.std(resid)), peak_period=float(p_peak),
                peak_power=float(power[k]), peak_ptp=ptp, harmonic=harmonic,
                orbit_ptp=float(np.nanmax(tmpl) - np.nanmin(tmpl)))


def selected_sector(safe: str):
    p = os.path.join(paths.LIGHTCURVE_DIR, f"lc_{safe}_clean.fits")
    if not os.path.exists(p):
        return None
    with fits.open(p) as h:
        for e in h:
            if e.header.get("SECTOR") is not None:
                return int(e.header["SECTOR"])
    return None


def match_sample_star(p_peak: float, own: str, p_best: dict, tol: float = 0.005) -> str:
    """Another sample star whose P or P/2 equals the residual peak period (a W UMa
    neighbour's dominant Lomb-Scargle peak is at P/2)."""
    hits = []
    for other, p in p_best.items():
        if other == own:
            continue
        for mult, tag in ((1.0, "P"), (0.5, "P/2")):
            if abs(p_peak / (p * mult) - 1.0) < tol:
                hits.append(f"{other} {tag}")
    return "; ".join(hits)


COLUMNS = ["star", "tic", "sector", "selected", "product", "flux_used_as_pdc", "crowdsap", "flfrcsap",
           "n_cycles", "amp_scatter_pdc", "amp_scatter_sap", "pdc_over_sap",
           "resid_rms", "orbit_ptp", "resid_peak_period_d", "resid_peak_ptp",
           "resid_peak_power", "resid_peak_harmonic", "resid_peak_matches",
           "catalogue_sectors_total", "cached_sectors"]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console is cp1252
    import runstats
    _run_tracker = runstats.RunTracker("audit_crowding_pdc", per_star=False)

    with open(FOLD_CSV, newline="", encoding="utf-8") as fh:
        p_best = {r["star"]: float(r["P_best"]) for r in csv.DictReader(fh)}
    cat = catalogue()
    rows = []
    for star in sorted(p_best):
        info = cat.get(star)
        if info is None:
            print(f"{star:<11} not in catalogue CSVs — skipped")
            continue
        safe = star.replace(" ", "_")
        sel = selected_sector(safe)
        files = cached_files_for(info["tic"])
        cached = []
        for path in files:
            try:
                raw = read_raw(path)
            except Exception as err:
                print(f"{star:<11} {os.path.basename(path)}: unreadable ({err})")
                continue
            sector = raw["hdr"].get("SECTOR")
            cached.append(sector)
            row = {"star": star, "tic": info["tic"], "sector": sector,
                   "selected": sector == sel, "product": raw["product"],
                   "flux_used_as_pdc": raw["pdc_col"],
                   "crowdsap": raw["hdr"].get("CROWDSAP"), "flfrcsap": raw["hdr"].get("FLFRCSAP"),
                   "catalogue_sectors_total": len(info["sectors"])}
            if raw["pdc"] is not None:
                t, f = clean(raw["t"], raw["pdc"], raw["q"])
                row["amp_scatter_pdc"], row["n_cycles"] = cycle_amplitude_scatter(t, f, p_best[star])
                try:
                    row.update({("resid_" + k if not k.startswith(("orbit", "resid")) else k): v
                                for k, v in residual_signal(t, f, p_best[star]).items()})
                except Exception as err:
                    print(f"{star:<11} S{sector}: residual search failed ({err})")
            if raw["sap"] is not None:
                t, f = clean(raw["t"], raw["sap"], raw["q"])
                row["amp_scatter_sap"], _ = cycle_amplitude_scatter(t, f, p_best[star])
            a, b = row.get("amp_scatter_pdc", np.nan), row.get("amp_scatter_sap", np.nan)
            row["pdc_over_sap"] = a / b if (b and np.isfinite(a) and np.isfinite(b) and b > 0) else np.nan
            rows.append(row)
        for r in rows:
            if r["star"] == star:
                r["cached_sectors"] = ";".join(str(s) for s in sorted(x for x in cached if x is not None))
        if not files:
            rows.append({"star": star, "tic": info["tic"], "sector": None, "selected": False,
                         "product": "not cached", "catalogue_sectors_total": len(info["sectors"]),
                         "cached_sectors": ""})

    # rename keys produced by residual_signal into the CSV schema
    for r in rows:
        if "resid_peak_period" in r:
            r["resid_peak_period_d"] = r.pop("resid_peak_period")
        r["resid_peak_harmonic"] = r.pop("resid_harmonic", r.get("resid_peak_harmonic"))
        r["resid_peak_ptp"] = r.get("resid_peak_ptp")
        if r.get("resid_peak_period_d"):
            r["resid_peak_matches"] = match_sample_star(r["resid_peak_period_d"], r["star"], p_best)
        r["resid_peak_power"] = r.get("resid_peak_power")

    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([(f"{r.get(c):.6g}" if isinstance(r.get(c), float) else r.get(c, ""))
                        for c in COLUMNS])

    def fmt(v, spec):
        return format(v, spec) if isinstance(v, (int, float)) and v is not None and np.isfinite(v) else "-"

    print(f"{'star':<11} {'S':>3} {'sel':>3} {'product':<13} {'crowd':>5} {'ampPDC':>6} {'ampSAP':>6} "
          f"{'PDC/SAP':>7} {'resid':>6} {'peakP':>7} {'peakAmp':>7} harm")
    for r in rows:
        print(f"{r['star']:<11} {str(r.get('sector') or '-'):>3} {'*' if r.get('selected') else ' ':>3} "
              f"{r.get('product',''):<13} {fmt(r.get('crowdsap'), '5.2f'):>5} "
              f"{fmt(r.get('amp_scatter_pdc'), '6.3f'):>6} {fmt(r.get('amp_scatter_sap'), '6.3f'):>6} "
              f"{fmt(r.get('pdc_over_sap'), '7.1f'):>7} {fmt(r.get('resid_rms'), '6.3f'):>6} "
              f"{fmt(r.get('resid_peak_period_d'), '7.4f'):>7} {fmt(r.get('resid_peak_ptp'), '7.3f'):>7} "
              f"{r.get('resid_peak_harmonic', '')} {r.get('resid_peak_matches', '')}")
    print(f"\nCSV: {AUDIT_CSV}")
