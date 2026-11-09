"""
================================================================================
AUDIT — HOW MUCH OF EACH STAR'S ERROR BAR IS THE 0.003 FLOOR?
================================================================================
Summary:
    Script 04 bins every light curve into 151 phase bins. Each bin's uncertainty
    is the standard error of the mean combined in quadrature with a systematic
    floor of 0.003 (04Modeling.SYS_ERROR_FLOOR). For bright stars with many
    points per bin the statistical part is far below the floor, so the floor
    alone sets the error bars. The model then fits "too well" (chi2_red << 1:
    V776 Cas 0.09-0.57, FI Boo Sector 78 0.08), the pre-scaling of Script 05
    does nothing (it only widens, max(1, sqrt(chi2_red))), and the posterior is
    wider than the data alone would allow. This is conservative, not wrong, but
    the thesis must say for which stars it holds.

    Reads the binned dataset Script 05 fits (lc01 sigmas in data/bundles/
    bundle_<star>.phoebe, a JSON file; no PHOEBE needed) and reports per star:
      n_bins, median sigma, median statistical part sqrt(sigma^2 - floor^2),
      the share of bins where the floor dominates (statistical part < floor),
      and the ratio floor / median statistical part.

Usage (from Code/, light, seconds):
    python Audits/audit_error_floor.py Targets/phaseF_queue.csv
Output:
    Audits/error_floor.csv, table printed (most floor-dominated first)
================================================================================
"""
import csv
import json
import os
import sys

import numpy as np

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(AUDIT_DIR)
sys.path.insert(0, CODE_DIR)
import paths                                              # noqa: E402

FLOOR = 0.003                                             # 04Modeling.SYS_ERROR_FLOOR


def lc01_sigmas(bundle_path: str) -> np.ndarray:
    with open(bundle_path, encoding="utf-8") as handle:
        params = json.load(handle)
    for p in params:
        if p.get("qualifier") == "sigmas" and p.get("dataset") == "lc01" and p.get("context") == "dataset":
            return np.asarray(p["value"], float)
    return np.array([])


def main() -> int:
    catalogue = sys.argv[1] if len(sys.argv) > 1 else os.path.join(paths.TARGETS_DIR, "phaseF_queue.csv")
    with open(catalogue, encoding="utf-8") as handle:
        stars = [row["name"] for row in csv.DictReader(handle)]
    rows = []
    for star in stars:
        path = os.path.join(paths.BUNDLE_DIR, f"bundle_{star.replace(' ', '_')}.phoebe")
        if not os.path.exists(path):
            continue
        s = lc01_sigmas(path)
        if s.size == 0:
            continue
        stat = np.sqrt(np.clip(s ** 2 - FLOOR ** 2, 0.0, None))
        med_stat = float(np.median(stat))
        rows.append({"star": star, "n_bins": int(s.size),
                     "median_sigma": round(float(np.median(s)), 5),
                     "median_stat": round(med_stat, 5),
                     "frac_floor_dominated": round(float(np.mean(stat < FLOOR)), 3),
                     "floor_over_stat": round(FLOOR / med_stat, 2) if med_stat > 0 else float("inf")})
    rows.sort(key=lambda r: -r["floor_over_stat"])
    out = os.path.join(AUDIT_DIR, "error_floor.csv")
    with open(out, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{'star':<11} {'bins':>4} {'med sigma':>9} {'med stat':>9} {'floor-dom':>9} {'floor/stat':>10}")
    for r in rows:
        print(f"{r['star']:<11} {r['n_bins']:>4} {r['median_sigma']:>9.5f} {r['median_stat']:>9.5f} "
              f"{r['frac_floor_dominated']:>9.2f} {r['floor_over_stat']:>10}")
    n_dom = sum(r["frac_floor_dominated"] >= 0.5 for r in rows)
    print(f"\n{len(rows)} stars; floor dominates at least half the bins in {n_dom}. Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
