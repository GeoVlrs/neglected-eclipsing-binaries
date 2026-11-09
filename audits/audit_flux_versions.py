"""
================================================================================
AUDIT — ECLIPSE AMPLITUDE IN EACH FLUX VERSION, SECTOR BY SECTOR
================================================================================
Summary:
    Written for VW Cep (Pipeline_Log Entry 103): its target-list TIC ID is a
    DUPLICATE entry, its clean light curve is SAP_CROWDCORR with CROWDSAP 0.599,
    and its TESS amplitude (0.58 mag) is well above its V amplitude (0.40 mag).
    The suspicion is that SPOC counted the star as its own neighbour, removed
    about 40 % of its light, and made the eclipses too deep.

    For every SPOC light curve found under each TIC ID given, this audit folds
    the data at the star's period and measures the full amplitude in magnitudes
    (100-bin median fold, max to min) for:
        SAP raw            no crowding correction
        SAP crowd-corr.    (SAP - (1 - CROWDSAP) median) / FLFRCSAP, as SPOC does
        PDCSAP             SPOC's cleaned flux (crowding already removed)
    with the CROWDSAP and FLFRCSAP of that file. Nothing in the pipeline is read
    or written except the period (ephemeris file) and the clean light curve's
    header, for comparison.

    Downloads to Runs/flux_versions/<star>/ (a few MB per sector). No PHOEBE.

Usage (from Code/; Windows or WSL python):
    python Audits/audit_flux_versions.py "VW Cep" --tic 1981621672 333870949 --vmag-amp 0.40
    python Audits/audit_flux_versions.py "TZ Lyr" --tic 76300903 --author TESS-SPOC   # FFI-only stars
Outputs:
    Runs/flux_versions/<star>/flux_versions.csv, printed table
================================================================================
"""
import argparse
import csv
import os
import re
import sys
import warnings

import numpy as np

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(AUDIT_DIR)
sys.path.insert(0, CODE_DIR)
import paths                                              # noqa: E402

NBIN = 100


def read_period(star: str) -> float:
    path = os.path.join(paths.TIMING_DIR, f"ephem_{star.replace(' ', '_')}.txt")
    text = open(path, encoding="utf-8").read()
    return float(re.search(r"^P_new\s*=\s*([0-9.]+)", text, re.M).group(1))


def amplitude_mag(t, f, period):
    """Max-to-min amplitude (mag) of the 100-bin median fold."""
    ok = np.isfinite(t) & np.isfinite(f) & (f > 0)
    t, f = t[ok], f[ok]
    if t.size < 200:
        return float("nan")
    ph = (t / period) % 1.0
    idx = np.clip((ph * NBIN).astype(int), 0, NBIN - 1)
    fb = np.array([np.median(f[idx == k]) if np.any(idx == k) else np.nan for k in range(NBIN)])
    return float(-2.5 * np.log10(np.nanmin(fb) / np.nanmax(fb)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("star")
    ap.add_argument("--tic", nargs="+", required=True)
    ap.add_argument("--vmag-amp", type=float, default=None, help="published V amplitude (mag)")
    ap.add_argument("--author", default="SPOC", help="SPOC (2-min) or TESS-SPOC (FFI)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    import lightkurve as lk
    out_dir = os.path.join(CODE_DIR, "Runs", "flux_versions", a.star.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)
    period = read_period(a.star)
    try:
        from astropy.io import fits
        clean = os.path.join(paths.LC_DIR if hasattr(paths, "LC_DIR") else "data/lightcurves",
                             f"lc_{a.star.replace(' ', '_')}_clean.fits")
        h = fits.getheader(clean, 1) if os.path.exists(clean) else {}
        h0 = fits.getheader(clean, 0) if os.path.exists(clean) else {}
        meta = {k: h.get(k, h0.get(k)) for k in ("FLUXSRC", "CROWDSAP", "FLFRCSAP", "SECTOR", "TICID", "OBJECT")}
        print("pipeline clean light curve:", meta)
    except Exception as e:                                # pragma: no cover - informative only
        print("clean light curve header not read:", e)
    rows = []
    for tic in a.tic:
        res = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author=a.author)
        print(f"TIC {tic}: {len(res)} {a.author} light curves")
        for i in range(len(res)):
            exptime = float(res.table["exptime"][i])
            mission = str(res.table["mission"][i])
            try:
                lc = res[i].download(download_dir=out_dir, quality_bitmask="default")
            except Exception as e:
                print(f"  {mission}: download failed: {e}")
                continue
            t = np.asarray(lc.time.value, float)
            sap = np.asarray(lc.sap_flux.value, float)
            pdc = np.asarray(lc.pdcsap_flux.value, float)
            crowd = float(lc.meta.get("CROWDSAP") or np.nan)
            flfr = float(lc.meta.get("FLFRCSAP") or np.nan)
            sap_cc = (sap - (1.0 - crowd) * np.nanmedian(sap)) / flfr
            row = {"tic": tic, "sector": lc.meta.get("SECTOR"), "exptime_s": exptime,
                   "crowdsap": round(crowd, 4), "flfrcsap": round(flfr, 4),
                   "amp_sap_raw": round(amplitude_mag(t, sap, period), 4),
                   "amp_sap_crowdcorr": round(amplitude_mag(t, sap_cc, period), 4),
                   "amp_pdcsap": round(amplitude_mag(t, pdc, period), 4)}
            rows.append(row)
            print(f"  S{row['sector']:>2} {exptime:5.0f}s  CROWDSAP {crowd:.3f} FLFRCSAP {flfr:.3f}  "
                  f"amp SAP raw {row['amp_sap_raw']:.3f}  SAP corr {row['amp_sap_crowdcorr']:.3f}  "
                  f"PDCSAP {row['amp_pdcsap']:.3f} mag", flush=True)
    if a.vmag_amp:
        print(f"published V amplitude: {a.vmag_amp:.3f} mag")
    with open(os.path.join(out_dir, "flux_versions.csv"), "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["tic", "sector", "exptime_s", "crowdsap", "flfrcsap",
                                          "amp_sap_raw", "amp_sap_crowdcorr", "amp_pdcsap"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
