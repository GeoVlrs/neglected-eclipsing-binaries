"""
================================================================================
AUDIT — ECLIPSE SHAPES: WHICH STAR IS THE LARGER ONE? (model-independent)
================================================================================
Summary:
    For each non-contact star of a catalogue, read the 151-bin phase curve that
    Script 04 stored in bundle_<star>.phoebe and measure both minima
    (Pipeline_Log Entry 87, the q -> 1 investigation):

      * depth (relative to the maximum);
      * BOTTOM CURVATURE: mean flux at |dphase| in [0.015, 0.025] minus the
        minimum, as a fraction of the depth. An occultation (smaller star
        totally hidden) is flat (~0); a total transit is slightly rounded by
        limb darkening (~1-3 %); a partial eclipse is V-shaped (large);
      * FLAT half-width (bins within 1 % of the depth of the minimum) and
        CONTACT half-width (where the eclipse signal, after removing an
        out-of-eclipse cos/sin harmonic model, falls below 2 % of the depth).

    The flatter minimum is taken as the occultation: the star hidden in it is
    the SMALLER star. Phase 0 is the deeper minimum (the hotter star eclipsed),
    so a flatter SECONDARY minimum means the cool star is the smaller one.

    Spherical, i = 90 deg estimates (rough; a Roche-distorted lobe-filling star
    is seen end-on at conjunction):
        r_big + r_small ~ sin(2 pi phi_contact),  r_big - r_small ~ sin(2 pi phi_flat)

    Read-only and light: parses bundle JSON with numpy; no PHOEBE, no model.

Usage (from Code/, seconds):
    python Audits/audit_eclipse_shapes.py Targets/phaseE_queue.csv [--all]
        (default: non-contact stars only; --all includes contact stars)

Outputs:
    Audits/eclipse_shapes.csv and a table on stdout.
================================================================================
"""
import csv
import json
import math
import os
import sys

import numpy as np

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR  = os.path.dirname(AUDIT_DIR)
sys.path.insert(0, CODE_DIR)
import paths                                              # noqa: E402
from morphology import resolve_morphology_kind            # noqa: E402

OUT_CSV = os.path.join(AUDIT_DIR, "eclipse_shapes.csv")


def _value(d: list, qualifier: str, **match):
    for p in d:
        if isinstance(p, dict) and p.get("qualifier") == qualifier and \
                all(p.get(k) == v for k, v in match.items()):
            v = p.get("value")
            return v["value"] if isinstance(v, dict) and "value" in v else v
    return None


def load_curve(bundle_path: str):
    d = json.load(open(bundle_path, encoding="utf-8"))
    t = np.asarray(_value(d, "times", dataset="lc01", context="dataset"), float)
    f = np.asarray(_value(d, "fluxes", dataset="lc01", context="dataset"), float)
    period = float(_value(d, "period", component="binary", context="component"))
    t0 = float(_value(d, "t0_supconj", component="binary", context="component"))
    ph = ((t - t0) / period) % 1.0
    order = np.argsort(ph)
    return ph[order], f[order] / np.max(f)


def out_of_eclipse_model(ph, f, half_width=0.15):
    """Least-squares cos/sin harmonics (1..2) to the bins outside both eclipses."""
    d0 = np.minimum(ph, 1 - ph)
    m = (d0 > half_width) & (np.abs(ph - 0.5) > half_width)
    def design(x):
        w = 2 * np.pi * x
        return np.column_stack([np.ones_like(x), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    coef, *_ = np.linalg.lstsq(design(ph[m]), f[m], rcond=None)
    return design(ph) @ coef


def measure(ph, f, base, centre):
    dphi = (ph - centre + 0.5) % 1.0 - 0.5
    near = np.abs(dphi) < 0.2
    x, y, b = dphi[near], f[near], base[near]
    k = int(np.argmin(np.where(np.abs(x) < 0.03, y, np.inf)))
    fmin, xmin = y[k], x[k]
    depth = float(np.interp(xmin, x, b)) - fmin
    shoulder = y[(np.abs(x - xmin) >= 0.015) & (np.abs(x - xmin) <= 0.025)]
    curv = float((np.mean(shoulder) - fmin) / depth) if shoulder.size and depth > 0 else float("nan")
    flat = np.abs(x[y <= fmin + 0.01 * depth] - xmin)
    phi_flat = float(np.max(flat)) if flat.size else 0.0
    signal = (b - y) / depth
    outside = np.abs(x - xmin)[signal < 0.02]
    phi_contact = float(np.min(outside[outside > phi_flat])) if np.any(outside > phi_flat) else float("nan")
    return {"depth": depth, "curvature": curv, "phi_flat": phi_flat, "phi_contact": phi_contact}


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    rows = list(csv.DictReader(open(argv[1], encoding="utf-8")))
    include_all = "--all" in argv
    out = []
    for r in rows:
        star, safe = r["name"], r["name"].replace(" ", "_")
        kind = resolve_morphology_kind(star, r.get("type", ""))
        if kind == "contact" and not include_all:
            continue
        path = os.path.join(paths.BUNDLE_DIR, f"bundle_{safe}.phoebe")
        if not os.path.exists(path):
            continue
        ph, f = load_curve(path)
        base = out_of_eclipse_model(ph, f)
        p, s = measure(ph, f, base, 0.0), measure(ph, f, base, 0.5)
        # The flatter minimum is the occultation; the star hidden there is smaller.
        if p["curvature"] < s["curvature"]:
            smaller, occ = "hot (primary)", p
        else:
            smaller, occ = "cool (secondary)", s
        phi_c = np.nanmean([p["phi_contact"], s["phi_contact"]])
        r_sum  = math.sin(2 * math.pi * min(phi_c, 0.25)) if phi_c == phi_c else float("nan")
        r_diff = math.sin(2 * math.pi * occ["phi_flat"])
        out.append({
            "star": star, "type": r.get("type", ""), "routing": kind,
            "depth_pri": round(p["depth"], 3), "depth_sec": round(s["depth"], 3),
            "curv_pri": round(p["curvature"], 4), "curv_sec": round(s["curvature"], 4),
            "flat_pri": round(p["phi_flat"], 4), "flat_sec": round(s["phi_flat"], 4),
            "contact_pri": round(p["phi_contact"], 4), "contact_sec": round(s["phi_contact"], 4),
            "smaller_star": smaller,
            "r_big_est": round((r_sum + r_diff) / 2, 3), "r_small_est": round((r_sum - r_diff) / 2, 3),
        })
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    cols = ["star", "routing", "depth_pri", "depth_sec", "curv_pri", "curv_sec",
            "flat_pri", "flat_sec", "contact_pri", "contact_sec", "smaller_star",
            "r_big_est", "r_small_est"]
    widths = [max(len(str(o[c])) for o in out + [dict(zip(cols, cols))]) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for o in out:
        print("  ".join(str(o[c]).ljust(w) for c, w in zip(cols, widths)))
    print(f"\nwritten: {OUT_CSV}")
    print("curv = bottom curvature (fraction of depth): ~0 flat (occultation), "
          "small rounded (transit), large V (partial). r estimates: spherical, i = 90.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
