"""
================================================================================
AUDIT — LIGHT-CURVE SHAPE SECTOR BY SECTOR (SPOTS THAT MOVE)
================================================================================
Summary:
    The pipeline fits ONE TESS sector per star. A spotted contact binary changes
    its light curve from sector to sector as spots form and migrate, so a fit to
    one sector carries that sector's spot pattern into the geometry. This audit
    downloads every TESS light curve of a star (SPOC 2-min where there is one,
    otherwise TESS-SPOC FFI), folds each sector with the star's refined period,
    re-centres phase 0 on that sector's own deeper minimum, and measures:

      * the two eclipse depths and the amplitude (relative flux);
      * the O'Connell asymmetry (maximum after minimum I minus the other), per
        cycle, with its standard error and cycle-to-cycle scatter, as
        audit_oconnell.py does for the pipeline's sector.

    Written for FI Boo (Pipeline_Log Entry 106) but works for any star with an
    ephemeris file from Script 03.

    Downloads light curves from MAST into Runs/sectors/<star>/ (a few MB each).
    No PHOEBE.

Usage (from Code/, Windows or WSL python):
    python Audits/audit_sector_asymmetry.py "FI Boo" [--tic 193990389] [--sectors 26,54,81]
Outputs:
    Runs/sectors/<star>/sector_shapes.csv, sector_folds.csv (binned curves),
    sector_folds.png
================================================================================
"""
import argparse
import csv
import math
import os
import re
import sys
import warnings

import numpy as np

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(AUDIT_DIR)
sys.path.insert(0, CODE_DIR)
import paths                                              # noqa: E402

MAX1, MAX2 = (0.20, 0.30), (0.70, 0.80)
NBIN = 100
PHASE_BIN_COUNT, SYS_ERROR_FLOOR = 151, 0.003      # as Script 04 (04Modeling.bin_phase_curve)


def bin_like_04(ph, f):
    """Script 04's binning: phase in [-0.5, 0.5), 151 bins, median flux, standard
    error of the mean combined in quadrature with a 0.003 floor."""
    p = ((ph + 0.5) % 1.0) - 0.5
    edges = np.linspace(-0.5, 0.5, PHASE_BIN_COUNT + 1)
    out = []
    for i in range(PHASE_BIN_COUNT):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() >= 3:
            se = float(np.std(f[m], ddof=1) / np.sqrt(m.sum()))
            out.append((0.5 * (edges[i] + edges[i + 1]), float(np.median(f[m])), float(np.hypot(se, SYS_ERROR_FLOOR))))
    return out


def read_ephemeris(star: str) -> float:
    path = os.path.join(paths.TIMING_DIR, f"ephem_{star.replace(' ', '_')}.txt")
    text = open(path, encoding="utf-8").read()
    return float(re.search(r"^P_new\s*=\s*([0-9.]+)", text, re.M).group(1))


def binned(ph, f, n=NBIN):
    edges = np.linspace(0, 1, n + 1)
    idx = np.clip(np.digitize(ph, edges) - 1, 0, n - 1)
    return np.array([np.median(f[idx == k]) if np.any(idx == k) else np.nan for k in range(n)])


def measure(t, f, period):
    """Fold, re-centre on the deeper minimum, and measure one sector."""
    ph = (t / period) % 1.0
    fb = binned(ph, f)
    k = int(np.nanargmin(fb))
    # refine the minimum with a parabola through the 7 bins around it
    sel = [(k + j) % NBIN for j in range(-3, 4)]
    x = np.array([j for j in range(-3, 4)], float) / NBIN
    c = np.polyfit(x, fb[sel], 2)
    shift = (k + 0.5) / NBIN + (-c[1] / (2 * c[0]) if c[0] > 0 else 0.0)
    ph = (ph - shift) % 1.0
    fb = binned(ph, f)
    cyc = np.floor(t / period - shift)
    per = []
    for cc in np.unique(cyc):
        m = cyc == cc
        a = f[m & (ph >= MAX1[0]) & (ph < MAX1[1])]
        b = f[m & (ph >= MAX2[0]) & (ph < MAX2[1])]
        if a.size >= 5 and b.size >= 5:
            per.append(float(np.mean(a) - np.mean(b)) * 100.0)
    near = lambda c0: np.nanmin(fb[[int((c0 * NBIN + j) % NBIN) for j in range(-4, 5)]])  # noqa: E731
    top = np.nanmax(fb)
    per = np.asarray(per)
    return {
        "n_cycles": len(per), "amplitude": round(float(top - np.nanmin(fb)), 4),
        "depth_min1": round(float(top - near(0.0)), 4), "depth_min2": round(float(top - near(0.5)), 4),
        "oconnell_pct": round(float(per.mean()), 3) if per.size else "",
        "oconnell_err": round(float(per.std(ddof=1) / math.sqrt(per.size)), 3) if per.size > 2 else "",
        "oc_scatter": round(float(per.std(ddof=1)), 3) if per.size > 2 else "",
    }, fb, ph


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("star")
    ap.add_argument("--tic", default=None)
    ap.add_argument("--sectors", default="",
                    help="comma list: download and measure only these sectors (Entry 160; default all)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    import lightkurve as lk
    out_dir = os.path.join(CODE_DIR, "Runs", "sectors", a.star.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)
    period = read_ephemeris(a.star)
    res = lk.search_lightcurve(f"TIC {a.tic}" if a.tic else a.star, mission="TESS")
    tab = res.table
    best = {}
    rank = {"SPOC": 0, "TESS-SPOC": 1}
    for i, row in enumerate(tab):
        author = str(row["author"])
        if author not in rank:
            continue
        sec = int(re.search(r"(\d+)", str(row["mission"])).group(1))
        key = (rank[author], float(row["exptime"]))
        if sec not in best or key < best[sec][0]:
            best[sec] = (key, i, author, float(row["exptime"]))
    only = {int(x) for x in a.sectors.split(",") if x.strip()}
    rows, folds = [], {}
    for sec in sorted(best):
        if only and sec not in only:
            continue
        _key, i, author, exptime = best[sec]
        lc = res[i].download(download_dir=out_dir, quality_bitmask="default")
        flux = lc.flux.value if hasattr(lc.flux, "value") else np.asarray(lc.flux)
        t = np.asarray(lc.time.value, float)
        ok = np.isfinite(t) & np.isfinite(flux)
        t, flux = t[ok], flux[ok] / np.nanmedian(flux[ok])
        r = flux - np.median(flux)
        keep = np.abs(r) < 4 * 1.4826 * np.median(np.abs(r - np.median(r))) + 0.5
        m, fb, ph_s = measure(t[keep], flux[keep], period)
        with open(os.path.join(out_dir, f"lc151_S{sec}.csv"), "w", newline="", encoding="utf-8") as h:
            w = csv.writer(h)
            w.writerow(["phase", "flux", "sigma"])
            w.writerows([[round(a_, 6), round(b_, 6), round(c_, 6)] for a_, b_, c_ in bin_like_04(ph_s, flux[keep])])
        m.update({"sector": sec, "author": author, "exptime_s": exptime,
                  "year": int(2018 + (sec + 5) // 13.5) if sec else ""})
        rows.append(m)
        folds[sec] = fb
        print(f"sector {sec:2d} {author:9s} {exptime:6.0f}s  cycles {m['n_cycles']:3d}  amp {m['amplitude']:.4f}  "
              f"depth I {m['depth_min1']:.4f}  II {m['depth_min2']:.4f}  O'Connell {m['oconnell_pct']} "
              f"+- {m['oconnell_err']} (scatter {m['oc_scatter']})", flush=True)
    keys = ["sector", "author", "exptime_s", "n_cycles", "amplitude", "depth_min1", "depth_min2",
            "oconnell_pct", "oconnell_err", "oc_scatter"]
    with open(os.path.join(out_dir, "sector_shapes.csv"), "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    centres = (np.arange(NBIN) + 0.5) / NBIN
    with open(os.path.join(out_dir, "sector_folds.csv"), "w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["phase"] + [f"S{s}" for s in sorted(folds)])
        for j in range(NBIN):
            w.writerow([round(centres[j], 4)] + [round(float(folds[s][j]), 5) for s in sorted(folds)])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for s in sorted(folds):
            ax.plot(centres, folds[s], lw=1.2, label=f"sector {s}")
        ax.set_xlabel("phase (0 = deeper minimum of each sector)")
        ax.set_ylabel("relative flux")
        ax.set_title(f"{a.star}: TESS light curve by sector")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "sector_folds.png"), dpi=130)
    except Exception as plot_error:
        print("plot skipped:", plot_error)
    print("wrote", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
