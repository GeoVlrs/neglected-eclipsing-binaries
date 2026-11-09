"""
Fourier morphology of the folded TESS light curves (model-type research, 26 Oct 2026).

Rucinski (1997, OGLE paper II, astro-ph/9611158) separates contact from non-contact
light curves with the two even cosine terms of a Fourier fit: the "inner-contact" line
a4 = a2 (0.125 - a2). The paper fits cosine terms 0-4 (plus one sine term for
asymmetric curves) with phase counted from the deeper minimum; the side of the line
that holds contact systems is not stated there, so this script measures it on the
pipeline's own stars of known type (39 contact, 8 semi-detached, 13 detached-routed).

Per star: data/lightcurves/lc_<star>_folded.fits (TIME = days from the epoch), P from
data/timing/ephem_<star>.txt (P_new, or P_kreiner if the refined period was reverted).
Phase -> 200 median bins -> phase 0 moved to the deepest bin -> flux / max(bin flux)
-> least squares l(phi) = sum_{i=0..4} a_i cos(2 pi i phi) + b1 sin(2 pi phi).
Output Audits/fourier_morphology.csv; summary on stdout. Read-only on the pipeline.
"""
import csv
import os
import re
import sys

import numpy as np
from astropy.io import fits

sys.stdout.reconfigure(encoding="utf-8")
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import morphology as m  # noqa: E402

TEN = {"EG Cep", "EP Aur", "PX Car", "RU UMi", "SU Cep", "TZ Lyr", "V2287 Cyg", "V343 Ori", "V505 Cyg", "V576 Lyr"}
NBIN = 200


def period(safe):
    txt = open(os.path.join(CODE, "data", "timing", f"ephem_{safe}.txt"), encoding="utf-8").read()
    get = lambda k: float(re.search(rf"^{k}\s*=\s*([0-9.eE+-]+)", txt, re.M).group(1))
    reverted = re.search(r"^PERIOD_REVERTED\s*=\s*True", txt, re.M) is not None
    return get("P_kreiner") if reverted else get("P_new")


def fourier(safe):
    d = fits.open(os.path.join(CODE, "data", "lightcurves", f"lc_{safe}_folded.fits"))[1].data
    p = period(safe)
    ph = (np.asarray(d["TIME"], float) / p) % 1.0
    fl = np.asarray(d["FLUX"], float)
    ok = np.isfinite(ph) & np.isfinite(fl)
    ph, fl = ph[ok], fl[ok]
    edges = np.linspace(0, 1, NBIN + 1)
    idx = np.digitize(ph, edges) - 1
    cen = 0.5 * (edges[1:] + edges[:-1])
    med = np.array([np.median(fl[idx == k]) if np.any(idx == k) else np.nan for k in range(NBIN)])
    good = np.isfinite(med)
    cen, med = cen[good], med[good]
    # smooth lightly before locating the deepest minimum (3-bin running median)
    sm = np.array([np.median(med[max(0, k - 1):k + 2]) for k in range(len(med))])
    phi = (cen - cen[np.argmin(sm)]) % 1.0
    lum = med / np.max(sm)
    A = np.column_stack([np.cos(2 * np.pi * i * phi) for i in range(5)] + [np.sin(2 * np.pi * phi)])
    coef, *_ = np.linalg.lstsq(A, lum, rcond=None)
    rms = float(np.std(lum - A @ coef))
    return coef, rms, p


def main():
    queue = list(csv.DictReader(open(os.path.join(CODE, "Targets", "phaseF_queue.csv"), encoding="utf-8")))
    rows = []
    for r in queue:
        s, safe = r["name"], r["name"].replace(" ", "_")
        try:
            c, rms, p = fourier(safe)
        except Exception as e:
            print(f"{s}: skipped ({e})")
            continue
        a0, a1, a2, a4 = c[0], c[1], c[2], c[4]
        line = a2 * (0.125 - a2)
        kind = m.resolve_morphology_kind(s, r["type"])
        rows.append({"star": s, "routing": kind, "group": "TEN" if s in TEN else kind, "gcvs": r["type"],
                     "P": round(p, 7), "a0": round(a0, 5), "a1": round(a1, 5), "a2": round(a2, 5),
                     "a3": round(c[3], 5), "a4": round(a4, 5), "b1": round(c[5], 5),
                     "a4_line": round(line, 5), "a4_minus_line": round(a4 - line, 5), "fit_rms": round(rms, 5)})
    out = os.path.join(CODE, "Audits", "fourier_morphology.csv")
    with open(out, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for g in ("contact", "semidetached", "detached", "TEN"):
        grp = [x for x in rows if x["group"] == g]
        dl = np.array([x["a4_minus_line"] for x in grp])
        print(f"\n== {g}: {len(grp)} stars; a4 - line: min {dl.min():+.4f} median {np.median(dl):+.4f} "
              f"max {dl.max():+.4f}; below line {np.sum(dl < 0)}, above {np.sum(dl >= 0)}")
        if g != "contact":
            for x in sorted(grp, key=lambda x: x["a4_minus_line"]):
                print(f"   {x['star']:10s} {x['gcvs']:8s} a2 {x['a2']:+.4f} a4 {x['a4']:+.4f} line {x['a4_line']:+.4f} "
                      f"diff {x['a4_minus_line']:+.4f}  a1 {x['a1']:+.4f} b1 {x['b1']:+.4f} rms {x['fit_rms']:.4f}")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
