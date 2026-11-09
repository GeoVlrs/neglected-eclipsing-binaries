"""
Could the primary be a normal main-sequence star inside its Roche lobe at this period?
(model-type research, 26 Oct 2026)

For each queue star: T1 = the pipeline's own primary temperature (04Modeling.py
GAIA_TEFF_OVERRIDE_K, Gaia DR3 MSC with the documented corrections), parsed from the
file text. Main-sequence mass M1 and radius R1 at T1 interpolated in the dwarf table of
Pecaut & Mamajek (2013), version 2022.04.16 (rows A0V-K3V below, copied from
https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt on 26 Oct 2026).
Semi-major axis from Kepler's third law with M2 = q M1; the primary's volume-equivalent
Roche-lobe radius from Eggleton (1983), r_L = 0.49 x^2 / (0.6 x^2 + ln(1 + x)),
x = (M1/M2)^(1/3). Output: R1_MS / (r_L a) for q = 0.3, 0.5, 1.0.

Values >= 1: a main-sequence primary would overfill its lobe at this period, so the
system cannot be detached unless the primary is undersized. Indicative only: the table
masses are "tentative" and the radii carry ~10 % scatter; T1 itself is uncertain
(MSC for A stars). Output Audits/lobe_plausibility.csv. Read-only on the pipeline.
"""
import csv
import math
import os
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import morphology as m  # noqa: E402

# Teff, R_Rsun, Msun (Pecaut & Mamajek 2013, v2022.04.16), A0V ... K3V
MS = np.array([
    (9700, 2.193, 2.18), (9300, 2.136, 2.05), (8800, 2.117, 1.98), (8600, 1.861, 1.86),
    (8250, 1.794, 1.93), (8100, 1.785, 1.88), (7910, 1.775, 1.83), (7760, 1.750, 1.77),
    (7590, 1.747, 1.81), (7400, 1.747, 1.75), (7220, 1.728, 1.61), (7020, 1.679, 1.50),
    (6820, 1.622, 1.46), (6750, 1.578, 1.44), (6670, 1.533, 1.38), (6550, 1.473, 1.33),
    (6350, 1.359, 1.25), (6280, 1.324, 1.21), (6180, 1.221, 1.18), (6050, 1.167, 1.13),
    (5990, 1.142, 1.08), (5930, 1.100, 1.06), (5860, 1.060, 1.03), (5770, 1.012, 1.00),
    (5720, 1.002, 0.99), (5680, 0.991, 0.985), (5660, 0.977, 0.98), (5600, 0.949, 0.97),
    (5550, 0.927, 0.95), (5480, 0.914, 0.94), (5380, 0.853, 0.90), (5270, 0.813, 0.88),
    (5170, 0.797, 0.86), (5100, 0.783, 0.82), (4830, 0.755, 0.78)])
TEN = {"EG Cep", "EP Aur", "PX Car", "RU UMi", "SU Cep", "TZ Lyr", "V2287 Cyg", "V343 Ori", "V505 Cyg", "V576 Lyr"}
G, MSUN, RSUN, DAY = 6.674e-11, 1.989e30, 6.957e8, 86400.0


def ms(teff):
    t = MS[::-1]  # ascending Teff
    return float(np.interp(teff, t[:, 0], t[:, 1])), float(np.interp(teff, t[:, 0], t[:, 2]))


def eggleton(m1_over_m2):
    x = m1_over_m2 ** (1 / 3)
    return 0.49 * x * x / (0.6 * x * x + math.log(1 + x))


def main():
    src = open(os.path.join(CODE, "04Modeling.py"), encoding="utf-8").read()
    block = src[src.index("GAIA_TEFF_OVERRIDE_K"):]
    block = block[:block.index("\n}")]
    t1 = {k: float(v) for k, v in re.findall(r'"([^"]+)":\s*([0-9.]+)', block)}
    queue = list(csv.DictReader(open(os.path.join(CODE, "Targets", "phaseF_queue.csv"), encoding="utf-8")))
    rows = []
    for r in queue:
        s, p = r["name"], float(r["period"])
        if s not in t1:
            print(f"{s}: no T1 in the override table, skipped")
            continue
        R1, M1 = ms(t1[s])
        row = {"star": s, "routing": m.resolve_morphology_kind(s, r["type"]), "gcvs": r["type"],
               "P": p, "T1": t1[s], "R1_MS": round(R1, 3), "M1_MS": round(M1, 3)}
        for q in (0.3, 0.5, 1.0):
            a = (G * M1 * MSUN * (1 + q) * (p * DAY) ** 2 / (4 * math.pi ** 2)) ** (1 / 3) / RSUN
            row[f"fill_q{q}"] = round(R1 / (eggleton(1 / q) * a), 3)
        row["group"] = "TEN" if s in TEN else row["routing"]
        rows.append(row)
    out = os.path.join(CODE, "Audits", "lobe_plausibility.csv")
    with open(out, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for g in ("contact", "semidetached", "detached", "TEN"):
        grp = [x for x in rows if x["group"] == g]
        f5 = np.array([x["fill_q0.5"] for x in grp])
        print(f"\n== {g}: {len(grp)} stars; R1_MS/lobe at q 0.5: min {f5.min():.2f} median {np.median(f5):.2f} "
              f"max {f5.max():.2f}")
        if g != "contact":
            for x in sorted(grp, key=lambda x: -x["fill_q0.5"]):
                print(f"   {x['star']:10s} P {x['P']:.4f}  T1 {x['T1']:5.0f}  R1_MS {x['R1_MS']:.2f}  "
                      f"fill q0.3 {x['fill_q0.3']:.2f}  q0.5 {x['fill_q0.5']:.2f}  q1.0 {x['fill_q1.0']:.2f}")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
