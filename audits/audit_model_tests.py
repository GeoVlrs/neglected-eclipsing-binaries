"""
================================================================================
AUDIT — CONTROLLED MODEL TESTS: ONE FIT, ONE CHANGE AT A TIME
================================================================================
Summary:
    A single Nelder-Mead fit of a star's 151-bin light curve under a chosen
    geometry and a chosen set of model options, so that two fits differing in ONE
    setting can be compared (Pipeline_Log Entry 91). It serves the tests of:

      * limb darkening          --ld legacy (u = 0.6) | table (Claret 2017 by Teff)
      * gravity darkening       --radiative-primary (beta = 1, albedo = 1 for star 1)
      * model type              --geometry contact | sd2 | sd1 | detached
                                (sd2: the cooler star fills its lobe, the pipeline's
                                 semi-detached; sd1: the hotter star fills its lobe)
      * third light             --l3 free | <fixed fraction>
      * unequal maxima          --spot primary|secondary (one cool spot on the
                                equator; longitude, radius, temperature factor free)

    The bundle is rebuilt from the star's Script-04 bundle (data, period, t0, T1,
    exposure settings) in the requested geometry, the way Script 04 builds it, and
    configured as Script 05 configures it (audit_q_profile.configure_like_05,
    verified equal to a 05 checkpoint). `--check` compares the rebuilt bundle's
    chi-squared with the original bundle's at identical parameters.

    q is fixed at --q unless --free-q. The fit minimises chi-squared with
    scipy's Nelder-Mead inside hard bounds; the reported chi-squared is therefore
    an UPPER bound (maxfev-limited).

    HEAVY: ~150 forward models per fit. Not beside a live MCMC without agreement.

Usage (from Code/, WSL python3, no MPI):
    python3 Audits/audit_model_tests.py fit --star "RZ Dra" --geometry sd2 --q 0.4 \
        --r1 0.45 --ld table --label ld_test [--outdir Runs/model_tests]
    python3 Audits/audit_model_tests.py fit --star "RZ Dra" --geometry sd2 --q 0.4 --check
    python3 Audits/audit_model_tests.py collect [--label ld_test]
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
import paths                                              # noqa: E402
from audit_q_profile import chi2, configure_like_05, load  # noqa: E402
from limbdark import LEGACY_LINEAR_LD, tess_linear_ld, tess_quadratic_ld   # noqa: E402
from gravdark import envelope_coeffs                       # noqa: E402

MESH_TRIANGLE_COUNT = 2000
DEFAULT_OUT = os.path.join(CODE_DIR, "Runs", "model_tests")


def build(star: str, geometry: str, q: float):
    """A fresh bundle in `geometry` carrying the star's data and fixed values."""
    import phoebe
    src = load(os.path.join(paths.BUNDLE_DIR, f"bundle_{star.replace(' ', '_')}.phoebe"))
    g = lambda *a, **k: src.get_value(*a, **k)            # noqa: E731
    b = phoebe.default_binary(contact_binary=(geometry == "contact"))
    b.set_value("period@binary", g("period", component="binary", context="component"))
    b.set_value("t0_supconj@binary", g("t0_supconj", component="binary", context="component"))
    b.set_value("ecc@binary", 0.0)
    t1 = float(g("teff", component="primary", context="component"))
    b.set_value("teff@primary", t1)
    b.set_value("teff@secondary", 0.95 * t1)             # Script 04's seed (sets the 5A regime)
    b.set_value("atm@primary", "blackbody")
    b.set_value("atm@secondary", "blackbody")
    if geometry == "contact":
        b.flip_constraint("pot@contact_envelope", solve_for="requiv@primary")
        b.flip_constraint("fillout_factor@contact_envelope", solve_for="pot@contact_envelope")
        b.set_value("q@binary", q)
        b.set_value("fillout_factor@component", 0.15)
    else:
        b.set_value("q@binary", q)
        b.set_value("requiv@primary", 0.8)
        b.set_value("requiv@secondary", 0.6)
        if geometry == "sd2":
            b.add_constraint("semidetached", "secondary")
        elif geometry == "sd1":
            b.add_constraint("semidetached", "primary")
    b.set_value_all("ntriangles", MESH_TRIANGLE_COUNT)
    b.add_dataset("lc", dataset="lc01", passband="TESS:T", overwrite=True,
                  times=g("times", dataset="lc01", context="dataset"),
                  fluxes=g("fluxes", dataset="lc01", context="dataset"),
                  sigmas=g("sigmas", dataset="lc01", context="dataset"))
    exptime = float(g("exptime", dataset="lc01", context="dataset", check_visible=False))
    if exptime > 0:
        b.set_value("exptime", dataset="lc01", context="dataset", value=exptime)
        b.set_value("fti_method", compute="phoebe01", dataset="lc01",
                    value=g("fti_method", compute="phoebe01", dataset="lc01", check_visible=False))
        b.set_value("fti_oversample", compute="phoebe01", dataset="lc01",
                    value=int(g("fti_oversample", compute="phoebe01", dataset="lc01",
                                check_visible=False)))
    configure_like_05(b, kind="contact" if geometry == "contact" else "semidetached")
    return b, src


def comp_put(b, twig, value):
    b.set_value(twig, context="component", value=float(value))


def mode_fit(a) -> int:
    from scipy.optimize import minimize
    t_start = time.time()
    if a.start_from:
        prev = json.load(open(a.start_from, encoding="utf-8"))
        a.t2, a.incl = prev["T2"], prev["incl"]
        a.f = prev.get("f", a.f)
        a.r1, a.r2 = prev["r1_a"], prev["r2_a"]
        if a.free_q:
            a.q = prev["q"]
    name = (f"{a.star.replace(' ', '_')}_{a.geometry}_q{a.q:.3f}{'free' if a.free_q else ''}"
            f"_ld{a.ld}{'_gd' + a.gd if a.gd != 'step8000' else ''}"
            f"{'_rad1' if a.radiative_primary else ''}_l3{a.l3}"
            f"{'_spot' + a.spot if a.spot else ''}_{a.tag}.json")
    if a.skip_existing and not a.check and os.path.exists(os.path.join(a.outdir, a.label, name)):
        print(f"{a.star} [{a.label}] {name}: result exists, skipped")
        return 0
    b, src = build(a.star, a.geometry, a.q)
    t1 = float(b.get_value("teff", component="primary", context="component"))
    sma = float(b.get_value("sma", component="binary", context="component"))
    contact = a.geometry == "contact"

    if a.check:
        # same geometry as the Script-04 bundle required for a like-for-like check
        configure_like_05(src, kind="contact" if contact else "semidetached")
        vals = {"teff@secondary": a.t2 or 0.8 * t1, "incl@binary": a.incl}
        for bb in (src, b):
            comp_put(bb, "q@binary", a.q)
            for k, v in vals.items():
                comp_put(bb, k, v)
            if contact:
                comp_put(bb, "fillout_factor@contact_envelope", a.f)
            else:
                comp_put(bb, "requiv@primary", a.r1 * sma)
        c_src, c_new = chi2(src), chi2(b)
        rel = abs(c_src - c_new) / c_src
        print(json.dumps({"chi2_script04_bundle": c_src, "chi2_rebuilt": c_new,
                          "rel_diff": rel, "pass": rel < 1e-6}))
        return 0 if rel < 1e-6 else 1

    # ── options ───────────────────────────────────────────────────────────────
    if a.radiative_primary:
        b.set_value("gravb_bol@primary", 1.0)
        b.set_value("irrad_frac_refl_bol@primary", 1.0)

    def set_envelope(comp, teff):
        """Gravity darkening and albedo by the star's OWN temperature (--gd ramp |
        step6600). step8000 is the pipeline's rule, already set by
        configure_like_05 from the seed temperatures, and is left alone."""
        if a.gd != "step8000":
            beta, albedo = envelope_coeffs(teff, a.gd)
            b.set_value(f"gravb_bol@{comp}", beta)
            b.set_value(f"irrad_frac_refl_bol@{comp}", albedo)

    def set_ld(comp, teff):
        if a.ld == "table":
            b.set_value(f"ld_coeffs@{comp}@lc01", [tess_linear_ld(teff)])
        elif a.ld == "quad":
            b.set_value(f"ld_coeffs@{comp}@lc01", list(tess_quadratic_ld(teff)))

    if a.ld == "quad":
        for comp in ("primary", "secondary"):
            b.set_value(f"ld_func@{comp}@lc01", "quadratic")
    set_envelope("primary", t1)
    set_ld("primary", t1)
    ld_free = a.ld == "free"
    l3_free = a.l3 == "free"
    if a.l3 != "none":
        b.set_value("l3_mode", dataset="lc01", value="fraction")
        b.set_value("l3_frac", dataset="lc01", context="dataset",
                    value=0.2 if l3_free else float(a.l3))
    if a.spot:
        b.add_feature("spot", component=a.spot, feature="spot01",
                      colat=90.0, long=90.0, radius=20.0, relteff=0.9)

    # ── free parameters: (name, start, lo, hi, simplex scale) ────────────────
    t2_lo = (0.80 if contact else 0.50) * t1
    t2_hi = min((1.25 if contact else 1.10) * t1, 8000.0 if 0.95 * t1 < 8000.0 else 1e9)
    P = [("T2", a.t2 or 0.8 * t1, t2_lo, t2_hi, 200.0),
         ("incl", a.incl, 40.0 if contact else 60.0, 90.0, 1.5)]
    if contact:
        P.append(("f", a.f, 0.03, 0.95, 0.05))
    else:
        if a.geometry != "sd1":
            P.append(("r1", a.r1 * sma, 0.05 * sma, None, 0.02 * sma))
        if a.geometry != "sd2":
            P.append(("r2", a.r2 * sma, 0.05 * sma, None, 0.02 * sma))
    if a.free_q:
        P.append(("q", a.q, 0.05, 12.0, 0.08))
    if l3_free:
        P.append(("l3_fit", 0.2, 0.0, 0.85, 0.05))
    if ld_free:
        # diagnostic: ONE linear coefficient for both stars, fitted — what value
        # of the pipeline's single coefficient do the data ask for?
        P.append(("u_fit", LEGACY_LINEAR_LD, 0.0, 1.0, 0.05))
    if a.spot:
        P += [("spot_long", 90.0, 0.0, 360.0, 30.0), ("spot_radius", 20.0, 5.0, 60.0, 5.0),
              ("spot_relteff", 0.9, 0.6, 1.0, 0.03)]
    names = [p[0] for p in P]
    scale = np.array([p[4] for p in P])
    evals = {"n": 0, "best": (float("inf"), None)}

    def lobe(comp):
        return float(b.get_value("requiv_max", component=comp, context="component"))

    def apply(x):
        """Clip to bounds, set the bundle, return (clipped vector, penalty)."""
        v, pen = [], 0.0
        d = dict(zip(names, x))
        if "q" in d:
            qv = float(np.clip(d["q"], 0.05, 12.0))
            comp_put(b, "q@binary", qv)
        for (name, _s, lo, hi, sc), val in zip(P, x):
            if name in ("r1", "r2"):
                hi = 0.999 * lobe("primary" if name == "r1" else "secondary")
            c = float(np.clip(val, lo, hi))
            pen += 1e3 * abs(val - c) / sc
            v.append(c)
        d = dict(zip(names, v))
        comp_put(b, "teff@secondary", d["T2"])
        comp_put(b, "incl@binary", d["incl"])
        set_ld("secondary", d["T2"])
        set_envelope("secondary", d["T2"])
        if "u_fit" in d:
            for comp in ("primary", "secondary"):
                b.set_value(f"ld_coeffs@{comp}@lc01", [d["u_fit"]])
        if "f" in d:
            comp_put(b, "fillout_factor@contact_envelope", d["f"])
        if "r1" in d:
            comp_put(b, "requiv@primary", d["r1"])
        if "r2" in d:
            comp_put(b, "requiv@secondary", d["r2"])
        if "l3_fit" in d:
            b.set_value("l3_frac", dataset="lc01", context="dataset", value=d["l3_fit"])
        if a.spot:
            b.set_value("long", feature="spot01", value=d["spot_long"])
            b.set_value("radius", feature="spot01", value=d["spot_radius"])
            b.set_value("relteff", feature="spot01", value=d["spot_relteff"])
        return v, pen

    def objective(z):
        evals["n"] += 1
        try:
            v, pen = apply(np.asarray(z) * scale)
            c = chi2(b) + pen
        except Exception as exc:
            evals["failed"] = evals.get("failed", 0) + 1
            evals.setdefault("fail_msg", f"{type(exc).__name__}: {str(exc)[:160]}")
            return 1e30
        if not np.isfinite(c):
            # chi2() hides the reason; fetch it once, for the record
            evals["failed"] = evals.get("failed", 0) + 1
            if "fail_msg" not in evals:
                try:
                    b.run_compute(model="qprof", overwrite=True, progressbar=False)
                    evals["fail_msg"] = "failed once, then computed"
                except Exception as exc:
                    evals["fail_msg"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            return 1e30
        if c < evals["best"][0]:
            evals["best"] = (c, v)
            # snapshot: re-computing the best point afterwards can fail (GW Gem,
            # radiative primary, 13 Oct 2026), so its model is kept as evaluated
            evals["snap"] = {
                "chi2": c - pen,
                "model": list(map(float, b.get_value("fluxes", dataset="lc01", model="qprof"))),
                "q": float(b.get_value("q", component="binary", context="component")),
                "r1": float(b.get_value("requiv", component="primary", context="component")),
                "r2": float(b.get_value("requiv", component="secondary", context="component")),
                "lobe1": None if contact else lobe("primary"),
                "lobe2": None if contact else lobe("secondary")}
        return c

    res = minimize(objective, np.array([p[1] for p in P]) / scale, method="Nelder-Mead",
                   options={"maxfev": a.maxfev, "xatol": 1e-3, "fatol": 1e-2})
    if evals["best"][1] is None:
        print(f"{a.star} {a.geometry}: no valid model")
        return 1
    best = evals["best"][1]
    snap = evals["snap"]
    c_final = snap["chi2"]
    n = len(b.get_value("times", dataset="lc01", context="dataset"))
    phases = ((b.get_value("times", dataset="lc01", context="dataset")
               - b.get_value("t0_supconj", component="binary", context="component"))
              / b.get_value("period", component="binary", context="component")) % 1.0
    r1, r2 = snap["r1"], snap["r2"]
    out = {"star": a.star, "label": a.label, "geometry": a.geometry, "ld": a.ld, "gd": a.gd,
           "converged": bool(res.success), "T1": t1,
           "n_failed": evals.get("failed", 0), "fail_msg": evals.get("fail_msg", ""),
           "beta1": float(b.get_value("gravb_bol", component="primary", context="component")),
           "radiative_primary": bool(a.radiative_primary), "l3": a.l3, "spot": a.spot or "",
           "free_q": bool(a.free_q), "q_start": a.q, "start_tag": a.tag,
           "q": snap["q"],
           "chi2": c_final, "n": n, "n_free": len(P),
           "chi2_red": c_final / max(1, n - len(P)),
           "r1_a": r1 / sma, "r2_a": r2 / sma, "nfev": evals["n"],
           "nm_message": str(res.message), "wall_s": round(time.time() - t_start, 1),
           "phase": list(map(float, phases)),
           "obs": list(map(float, b.get_value("fluxes", dataset="lc01", context="dataset"))),
           "model": snap["model"]}
    out.update({k: float(v) for k, v in zip(names, best) if k not in ("r1", "r2", "q")})
    if not contact:
        out["r1_over_lobe"] = r1 / snap["lobe1"]
        out["r2_over_lobe"] = r2 / snap["lobe2"]
    d = os.path.join(a.outdir, a.label)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".tmp"), "w", encoding="utf-8") as h:
        json.dump(out, h)
    os.replace(os.path.join(d, name + ".tmp"), os.path.join(d, name))
    shown = "  ".join(f"{k} {out[k]:.4g}" for k in names if k in out)
    print(f"{a.star} [{a.label}] {a.geometry} q {out['q']:.3f} ld {a.ld} gd {a.gd}"
          f"{'' if res.success else ' NOT-CONVERGED'}"
          f"{' rad1' if a.radiative_primary else ''}{' l3=' + a.l3 if a.l3 != 'none' else ''}"
          f"{' spot=' + a.spot if a.spot else ''}: chi2r {out['chi2_red']:.2f}  {shown}  "
          f"r1 {out['r1_a']:.3f} r2 {out['r2_a']:.3f} ({evals['n']} models, "
          f"{out['n_failed']} failed{': ' + out['fail_msg'] if out['n_failed'] else ''}, "
          f"{out['wall_s']:.0f} s)")
    return 0


def mode_collect(a) -> int:
    rows = []
    for p in sorted(glob.glob(os.path.join(a.outdir, a.label or "*", "*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        for k in ("phase", "obs", "model"):
            r.pop(k, None)
        rows.append(r)
    if not rows:
        print("no fits found")
        return 1
    keys = ["star", "label", "geometry", "ld", "gd", "radiative_primary", "l3", "spot", "free_q",
            "start_tag", "q", "chi2", "chi2_red", "converged", "T2", "incl", "f", "r1_a", "r2_a",
            "r1_over_lobe", "r2_over_lobe", "u_fit", "beta1", "l3_fit", "spot_long", "spot_radius",
            "spot_relteff", "nfev", "wall_s"]
    out_csv = os.path.join(AUDIT_DIR, "model_tests.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["label"], r["star"], r["geometry"], r["chi2_red"])):
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
            print("  ".join(f"{k}={round(r[k], 4) if isinstance(r[k], float) else r[k]}"
                            for k in keys if k in r and r[k] not in ("", False, "none")))
    print("table:", out_csv)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["fit", "collect"])
    p.add_argument("--star")
    p.add_argument("--geometry", choices=["contact", "sd2", "sd1", "detached"], default="sd2")
    p.add_argument("--q", type=float, default=0.5)
    p.add_argument("--free-q", action="store_true")
    p.add_argument("--t2", type=float, default=None)
    p.add_argument("--incl", type=float, default=80.0)
    p.add_argument("--f", type=float, default=0.3)
    p.add_argument("--r1", type=float, default=0.30, help="start r1/a")
    p.add_argument("--r2", type=float, default=0.25, help="start r2/a")
    p.add_argument("--ld", choices=["legacy", "table", "quad", "free"], default="legacy",
                   help="legacy: linear 0.6; table: linear by Teff; quad: quadratic by Teff "
                        "(Claret 2017); free: one linear coefficient fitted (diagnostic)")
    p.add_argument("--gd", choices=["step8000", "ramp", "step6600"], default="step8000",
                   help="gravity darkening and albedo: the pipeline's 8000 K step, a ramp "
                        "over 6600-8000 K, or a step at 6600 K (gravdark.py)")
    p.add_argument("--start-from", default=None,
                   help="a result .json of an earlier fit: its solution is the start")
    p.add_argument("--skip-existing", action="store_true",
                   help="do nothing if this fit's result file is already there (resume)")
    p.add_argument("--radiative-primary", action="store_true")
    p.add_argument("--l3", default="none", help="none | free | <fixed fraction>")
    p.add_argument("--spot", choices=["primary", "secondary"], default=None)
    p.add_argument("--maxfev", type=int, default=160)
    p.add_argument("--label", default="test")
    p.add_argument("--tag", default="a", help="distinguishes fits with different starts")
    p.add_argument("--check", action="store_true")
    p.add_argument("--outdir", default=DEFAULT_OUT)
    a = p.parse_args()
    return {"fit": mode_fit, "collect": mode_collect}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
