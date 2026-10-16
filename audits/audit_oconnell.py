"""
================================================================================
AUDIT — UNEQUAL MAXIMA (O'CONNELL EFFECT) AND ORBIT-SHAPE SIGNS, PER STAR
================================================================================
Summary:
    The models in this pipeline are symmetric about each eclipse: they have no
    spots and a circular orbit. This audit measures, model-independently, how
    far each star's TESS light curve departs from that (Pipeline_Log Entry 91):

      * O'CONNELL ASYMMETRY: mean flux in the maximum after the primary minimum
        (phase 0.20-0.30) minus the mean in the other maximum (0.70-0.80), as
        a percentage of the median flux and of the light-curve amplitude. It is
        measured cycle by cycle on the UNBINNED clean light curve, so the
        quoted uncertainty is the standard error over cycles and `oc_scatter`
        shows whether the asymmetry changes during the sector (migrating spots).
      * SECONDARY-MINIMUM PHASE: a parabola through the bins within 0.08 of
        phase 0.5. For a small eccentricity the offset from 0.5 gives
        e cos(omega) = (pi / 2) * offset.
      * ECLIPSE-WIDTH ASYMMETRY: (w_sec - w_pri) / (w_sec + w_pri) from the full
        widths at half depth of the two minima. For two IDENTICAL eclipse shapes
        this would estimate e sin(omega), but a transit and an occultation of
        unequal stars have different shapes even on a circular orbit (GW Gem and
        OT Cep give 0.08-0.10). It is therefore only a comparison number: a star
        is suspicious if it stands out from stars of the same type.

    Read-only and light: clean FITS + the bundle's period and t0; no PHOEBE.

Usage (from Code/, seconds):
    python Audits/audit_oconnell.py Targets/<catalogue>.csv [more catalogues ...]

Outputs:
    Audits/oconnell.csv and a table on stdout (sorted by |asymmetry|).
================================================================================
"""
import csv
import json
import math
import os
import sys

import numpy as np
from astropy.io import fits

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AUDIT_DIR))
import paths                                              # noqa: E402
from morphology import resolve_morphology_kind            # noqa: E402

OUT_CSV = os.path.join(AUDIT_DIR, "oconnell.csv")
MAX1, MAX2 = (0.20, 0.30), (0.70, 0.80)


def bundle_value(d, qualifier, **match):
    for p in d:
        if isinstance(p, dict) and p.get("qualifier") == qualifier and \
                all(p.get(k) == v for k, v in match.items()):
            v = p["value"]
            return v["value"] if isinstance(v, dict) and "value" in v else v
    return None


def half_depth_width(ph, f, centre):
    """Full width at half depth of the minimum at `centre` (phase units)."""
    x = (ph - centre + 0.5) % 1.0 - 0.5
    o = np.argsort(x)
    x, y = x[o], f[o]
    near = np.abs(x) < 0.03
    if not near.any():
        return float("nan")
    fmin = float(np.min(y[near]))
    shoulder = float(np.median(y[(np.abs(x) > 0.18) & (np.abs(x) < 0.30)]))
    level = 0.5 * (fmin + shoulder)
    left = x[(x < 0) & (y >= level)]
    right = x[(x > 0) & (y >= level)]
    if left.size == 0 or right.size == 0:
        return float("nan")
    return float(np.min(right) - np.max(left))


def analyse(star: str, gcvs: str):
    safe = star.replace(" ", "_")
    lc_path = os.path.join(paths.LIGHTCURVE_DIR, f"lc_{safe}_clean.fits")
    b_path = os.path.join(paths.BUNDLE_DIR, f"bundle_{safe}.phoebe")
    if not (os.path.exists(lc_path) and os.path.exists(b_path)):
        return None
    d = json.load(open(b_path, encoding="utf-8"))
    period = float(bundle_value(d, "period", component="binary", context="component"))
    t0 = float(bundle_value(d, "t0_supconj", component="binary", context="component"))
    with fits.open(lc_path) as h:
        t = np.asarray(h[1].data["TIME"], float)
        f = np.asarray(h[1].data["FLUX"], float)
        src = h[0].header.get("FLUXSRC", "")
    if np.nanmedian(t) < 2_400_000:                 # BTJD -> BJD, as Script 04 does
        t = t + 2_457_000.0
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok] / np.nanmedian(f[ok])
    cyc = np.floor((t - t0) / period)
    ph = (t - t0) / period - cyc
    per_cycle = []
    for c in np.unique(cyc):
        m = cyc == c
        a = f[m & (ph >= MAX1[0]) & (ph < MAX1[1])]
        b = f[m & (ph >= MAX2[0]) & (ph < MAX2[1])]
        if a.size >= 5 and b.size >= 5:
            per_cycle.append(float(np.mean(a) - np.mean(b)))
    if len(per_cycle) < 3:
        return None
    oc = np.asarray(per_cycle) * 100.0
    # binned curve (the bundle's own 151 bins) for the shape measurements
    tb = np.asarray(bundle_value(d, "times", dataset="lc01", context="dataset"), float)
    fb = np.asarray(bundle_value(d, "fluxes", dataset="lc01", context="dataset"), float)
    phb = ((tb - t0) / period) % 1.0
    fb = fb / np.median(fb)
    amp = float(fb.max() - fb.min())
    sel = np.abs(phb - 0.5) < 0.08
    c2 = np.polyfit(phb[sel] - 0.5, fb[sel], 2)
    sec_off = float(-c2[1] / (2 * c2[0])) if c2[0] > 0 else float("nan")
    w_pri, w_sec = half_depth_width(phb, fb, 0.0), half_depth_width(phb, fb, 0.5)
    esinw = (w_sec - w_pri) / (w_sec + w_pri) if w_pri == w_pri and w_sec == w_sec else float("nan")
    return {
        "star": star, "routing": resolve_morphology_kind(star, gcvs), "flux_source": src,
        "n_cycles": len(oc), "oconnell_pct": round(float(np.mean(oc)), 3),
        "oconnell_err": round(float(np.std(oc, ddof=1) / math.sqrt(len(oc))), 3),
        "oc_scatter": round(float(np.std(oc, ddof=1)), 3),
        "oconnell_pct_of_amp": round(float(np.mean(oc)) / (amp * 100.0) * 100.0, 2),
        "amplitude": round(amp, 4),
        "sec_min_offset": round(sec_off, 5),
        "e_cos_omega": round(math.pi / 2 * sec_off, 5) if sec_off == sec_off else "",
        "width_pri": round(w_pri, 4), "width_sec": round(w_sec, 4),
        "width_asym": round(esinw, 4) if esinw == esinw else "",
    }


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    seen, rows = set(), []
    for cat in argv[1:]:
        for r in csv.DictReader(open(cat, encoding="utf-8")):
            if r["name"] in seen:
                continue
            seen.add(r["name"])
            out = analyse(r["name"], r.get("type", ""))
            if out:
                rows.append(out)
    rows.sort(key=lambda r: -abs(r["oconnell_pct"]))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cols = ["star", "routing", "n_cycles", "oconnell_pct", "oconnell_err", "oc_scatter",
            "oconnell_pct_of_amp", "e_cos_omega", "width_asym"]
    widths = [max(len(str(r[c])) for r in rows + [dict(zip(cols, cols))]) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))
    a = [abs(r["oconnell_pct"]) for r in rows]
    sig = [r for r in rows if abs(r["oconnell_pct"]) >= 3 * max(r["oconnell_err"], 1e-9)]
    print(f"\n{len(rows)} stars | |asymmetry| >= 1 %: {sum(x >= 1 for x in a)}, >= 2 %: "
          f"{sum(x >= 2 for x in a)}, >= 3 %: {sum(x >= 3 for x in a)} | significant at 3 sigma: "
          f"{len(sig)}")
    print("written:", OUT_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
