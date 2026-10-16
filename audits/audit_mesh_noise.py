"""
================================================================================
AUDIT — NUMERICAL NOISE OF THE MODEL: HOW MUCH DOES CHI-SQUARED JUMP?
================================================================================
Summary:
    PHOEBE covers each star with a mesh of triangles. A parameter that moves by
    a hair can change the mesh in a step, and chi-squared jumps with it. Found
    on 13 Oct 2026 (Pipeline_Log Entry 93): at the pipeline's 2000 triangles the
    jumps reach 9 for RZ Dra, 7 for ES UMa and 80 for TY UMa (a shallow W-type
    contact star, whose model also fails just below its solution). A 1-sigma
    error bar corresponds to a chi-squared change of 1, so jumps of this size
    distort error bars and, at the TY UMa level, the fit itself.

    `probe` builds one star's model (as Script 04 builds it, configured as
    Script 05 configures it) at a reference solution and scans two parameters in
    steps far smaller than any error bar:
        inclination              +-0.05 deg, 11 points
        fill-out (contact) or
        primary radius r1/a      +-0.002,    11 points
    A smooth model gives a smooth parabola. Reported per scan: the rms about a
    fitted parabola, the largest step-to-step jump, and the number of failed
    models. `scaled` divides by max(1, reduced chi-squared), which is what the
    sampler sees after Script 05's pre-scaling of the uncertainties.

    Reference solution: contact stars, the adopted q with the q-search's i,
    fill-out and T2; the two semi-detached calibrators, their known fits; other
    non-contact stars, the Script-04 seed (generic, so labelled `seed`).

    COST: 23 forward models per star and mesh size. Ask before a full survey.

Usage (from Code/, WSL python3, one process per star):
    python3 Audits/audit_mesh_noise.py probe --star "TY UMa" --kind contact [--ntriangles 4000]
    python3 Audits/audit_mesh_noise.py collect
Output: Runs/mesh_noise/<star>_n<N>.json, Audits/mesh_noise.csv
================================================================================
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(AUDIT_DIR)
sys.path.insert(0, AUDIT_DIR)
sys.path.insert(0, CODE_DIR)
import paths                                               # noqa: E402
from morphology import read_q_adopted                      # noqa: E402

OUT_DIR = os.path.join(CODE_DIR, "Runs", "mesh_noise")
GEOMETRY = {"contact": "contact", "semidetached": "sd2", "detached": "detached"}
# Known fits of the two semi-detached calibrators (test round B, Entry 93):
# (q, T2 [K], incl [deg], r1/a)
KNOWN = {"RZ Dra": (0.4147, 5171.0, 85.63, 0.452), "GW Gem": (0.4930, 4918.0, 80.19, 0.365)}


def scan_stats(steps, vals):
    vals = np.asarray(vals, float)
    ok = np.isfinite(vals)
    out = {"values": [None if not k else round(float(v), 3) for v, k in zip(vals, ok)],
           "n_failed": int((~ok).sum()), "rms": None, "max_resid": None, "max_jump": None}
    if ok.sum() >= 5:
        # the smooth part (slope and curvature) is removed first: far from a
        # minimum the raw step-to-step differences are slope, not noise
        s, v = np.asarray(steps)[ok], vals[ok]
        resid = v - np.polyval(np.polyfit(s, v, 2), s)
        out["rms"] = float(np.sqrt(np.mean(resid ** 2)))
        out["max_resid"] = float(np.max(np.abs(resid)))
    pairs = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1) if ok[i] and ok[i + 1]]
    if pairs:
        out["max_jump"] = float(max(pairs))
    return out


def mode_probe(a) -> int:
    import audit_model_tests as T
    from audit_q_profile import chi2
    t_start = time.time()
    geom = GEOMETRY[a.kind]
    source = "seed"
    if a.kind == "contact":
        ad = read_q_adopted(paths.QSEARCH_DIR, a.star)
        if not ad:
            print(f"{a.star}: no adopted q — skipped")
            return 1
        q = ad["q_adopted"]
        source = "adopted q, q-search solution"
    elif a.star in KNOWN:
        q = KNOWN[a.star][0]
        source = "known fit"
    else:
        q = None
    b, src = T.build(a.star, geom, q if q is not None else 0.8)
    g = lambda twig, comp: float(src.get_value(twig, component=comp, context="component"))  # noqa: E731
    sma = float(b.get_value("sma", component="binary", context="component"))
    t1 = float(b.get_value("teff", component="primary", context="component"))
    r2 = None
    if a.kind == "contact":
        t2, incl, third = ad["T2_over_T1_at_adopted"] * t1, ad["incl_at_adopted"], ad["fillout_at_adopted"]
    elif a.star in KNOWN:
        _, t2, incl, third = KNOWN[a.star]
    else:
        q = g("q", "binary")
        T.comp_put(b, "q@binary", q)
        sma_src = g("sma", "binary")
        t2, incl = g("teff", "secondary"), g("incl", "binary")
        third = g("requiv", "primary") / sma_src
        if geom == "detached":
            r2 = g("requiv", "secondary") / sma_src
    if a.ntriangles:
        b.set_value_all("ntriangles", a.ntriangles)
    ntri = a.ntriangles or T.MESH_TRIANGLE_COUNT

    def put(incl_, third_):
        T.comp_put(b, "teff@secondary", t2)
        T.comp_put(b, "incl@binary", min(incl_, 90.0))
        if geom == "contact":
            T.comp_put(b, "fillout_factor@contact_envelope", third_)
        else:
            T.comp_put(b, "requiv@primary", third_ * sma)
            if r2 is not None:
                T.comp_put(b, "requiv@secondary", r2 * sma)

    if incl > 89.94:
        incl = 89.94                           # keep the whole scan below 90 deg
    put(incl, third)
    c0 = chi2(b)
    n = len(b.get_value("times", dataset="lc01", context="dataset"))
    d_incl, d_third = np.linspace(-0.05, 0.05, 11), np.linspace(-0.002, 0.002, 11)
    v_incl, v_third = [], []
    for d in d_incl:
        put(incl + d, third)
        v_incl.append(chi2(b))
    for d in d_third:
        put(incl, third + d)
        v_third.append(chi2(b))
    red = c0 / max(1, n - 4) if np.isfinite(c0) else float("nan")
    out = {"star": a.star, "kind": a.kind, "ntriangles": ntri, "source": source, "q": q,
           "T2": t2, "incl": incl, "third": third, "third_name": "f" if geom == "contact" else "r1/a",
           "chi2_0": c0 if np.isfinite(c0) else None, "chi2_red_0": red if np.isfinite(red) else None,
           "n": n, "incl_scan": scan_stats(d_incl, v_incl), "third_scan": scan_stats(d_third, v_third),
           "wall_s": round(time.time() - t_start, 1)}
    jumps = [s["max_jump"] for s in (out["incl_scan"], out["third_scan"]) if s["max_jump"] is not None]
    out["max_jump"] = max(jumps) if jumps else None
    noise = [s["rms"] for s in (out["incl_scan"], out["third_scan"]) if s["rms"] is not None]
    out["noise_rms"] = max(noise) if noise else None
    out["noise_rms_scaled"] = (out["noise_rms"] / max(1.0, red)
                               if noise and np.isfinite(red) else None)
    out["n_failed"] = out["incl_scan"]["n_failed"] + out["third_scan"]["n_failed"] + (0 if np.isfinite(c0) else 1)
    out["max_jump_scaled"] = (out["max_jump"] / max(1.0, red)
                              if out["max_jump"] is not None and np.isfinite(red) else None)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{a.star.replace(' ', '_')}_n{ntri}.json")
    with open(path + ".tmp", "w", encoding="utf-8") as h:
        json.dump(out, h)
    os.replace(path + ".tmp", path)
    fmt = lambda v: "  n/a" if v is None else f"{v:6.2f}"          # noqa: E731
    print(f"{a.star:10s} {a.kind:12s} n{ntri}: chi2r {fmt(out['chi2_red_0'])}  noise rms {fmt(out['noise_rms'])} "
          f"(scaled {fmt(out['noise_rms_scaled'])})  max jump {fmt(out['max_jump'])}  "
          f"rms incl {fmt(out['incl_scan']['rms'])} "
          f"{out['third_name']} {fmt(out['third_scan']['rms'])}  failed {out['n_failed']}/23  "
          f"[{source}] ({out['wall_s']:.0f} s)")
    return 0


def mode_collect(a) -> int:
    rows = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(OUT_DIR, "*.json")))]
    if not rows:
        print("no probes found")
        return 1
    keys = ["star", "kind", "ntriangles", "source", "q", "incl", "third_name", "third", "chi2_red_0",
            "noise_rms", "noise_rms_scaled", "max_jump", "rms_incl", "rms_third", "n_failed",
            "wall_s"]
    out_csv = os.path.join(AUDIT_DIR, "mesh_noise.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["ntriangles"], -(r.get("noise_rms_scaled") or 1e9))):
            r["rms_incl"], r["rms_third"] = r["incl_scan"]["rms"], r["third_scan"]["rms"]
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"{len(rows)} probes -> {out_csv}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["probe", "collect"])
    p.add_argument("--star")
    p.add_argument("--kind", choices=list(GEOMETRY))
    p.add_argument("--ntriangles", type=int, default=0)
    a = p.parse_args()
    return {"probe": mode_probe, "collect": mode_collect}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
