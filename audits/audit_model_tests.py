"""
================================================================================
AUDIT — CONTROLLED MODEL TESTS: ONE FIT, ONE CHANGE AT A TIME
================================================================================
Summary:
    A single Nelder-Mead fit of a star's 151-bin light curve under a chosen
    geometry and a chosen set of model options, so that two fits differing in ONE
    setting can be compared (Pipeline_Log Entry 91). It serves the tests of:

      * limb darkening          --ld legacy (u = 0.6) | table | quad (Claret 2017 by
                                Teff) | free (one linear coefficient fitted)
      * gravity-darkening rule  --gd auto (default, production's rule: contact 8000 K
                                step, others the 6600-8000 K ramp) | step8000 | ramp |
                                step6600. Before Entry 147 the default was step8000.
      * gravity darkening       --radiative-primary (beta = 1, albedo = 1 for star 1)
                                --gd-primary-teff T (star 1 gets the ramp's beta and
                                albedo at T instead of at its own T1; Entry 135)
      * model type              --geometry contact | sd2 | sd1 | detached
                                (sd2: the cooler star fills its lobe, the pipeline's
                                 semi-detached; sd1: the hotter star fills its lobe)
      * third light             --l3 free | <fixed fraction>
                                (raw PHOEBE l3_frac). With --l3 free and --l3-k K
                                (Audits/l3_scaling.csv), --l3-start-share and
                                --l3-max-share give the start and upper bound as
                                SHARES of the mean light, converted with
                                s/(1-s) = K l3_frac/(1-l3_frac) (Entries 136-139)
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

    HEAVY: --maxfev forward models per fit (default 160; batch drivers use 400).
    Ask before running beside a live MCMC.

    Results: one JSON per fit in Runs/model_tests/<label>/ (the name carries the
    geometry, q, options and --tag; give different starts different tags).
    Collect (Entry 147): `collect --label ""` rebuilds the full Audits/model_tests.csv
    (read by audit_error_budget.py and audit_test_registry.py); `collect --label X`
    writes Audits/model_tests_X.csv only. Before Entry 147 a labelled collect
    replaced the full table with label X.
    A conclusion needs a converged fit or a fixed-value profile, not one capped fit
    (Entries 142, 146). Register every batch in Audits/test_registry.csv.

Usage (from Code/, WSL python3, no MPI):
    python3 Audits/audit_model_tests.py fit --star "RZ Dra" --geometry sd2 --q 0.4 \
        --r1 0.45 --ld table --label ld_test [--outdir Runs/model_tests]
    python3 Audits/audit_model_tests.py fit --star "RZ Dra" --geometry sd2 --q 0.4 --check
    python3 Audits/audit_model_tests.py collect --label ""        # full table
    python3 Audits/audit_model_tests.py collect --label ld_test   # that label only
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
from morphology import CONTACT_T2_FLOOR_FRAC, CONTACT_INCL_FLOOR_DEG   # noqa: E402

MESH_TRIANGLE_COUNT = 2000
DEFAULT_OUT = os.path.join(CODE_DIR, "Runs", "model_tests")


def build(star: str, geometry: str, q: float, lc_csv: str = None, ntriangles: int = 0):
    """A fresh bundle in `geometry` carrying the star's data and fixed values.
    lc_csv (Entry 106): a binned light curve of another sector (columns phase,
    flux, sigma; phase 0 = deeper minimum), used instead of the Stage-04 data."""
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
    b.set_value_all("ntriangles", ntriangles or MESH_TRIANGLE_COUNT)
    if lc_csv:
        rows = list(csv.DictReader(open(lc_csv, encoding="utf-8")))
        per = float(g("period", component="binary", context="component"))
        t0 = float(g("t0_supconj", component="binary", context="component"))
        times = np.array([t0 + float(r["phase"]) * per for r in rows])
        fluxes = np.array([float(r["flux"]) for r in rows])
        sigmas = np.array([float(r["sigma"]) for r in rows])
        exptime = 0.0          # 120 s / 200 s sectors only: no exposure integration (as Script 04)
    else:
        times = g("times", dataset="lc01", context="dataset")
        fluxes = g("fluxes", dataset="lc01", context="dataset")
        sigmas = g("sigmas", dataset="lc01", context="dataset")
        exptime = float(g("exptime", dataset="lc01", context="dataset", check_visible=False))
    b.add_dataset("lc", dataset="lc01", passband="TESS:T", overwrite=True,
                  times=times, fluxes=fluxes, sigmas=sigmas)
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
    if a.gd == "auto":      # Entry 147: production's rule unless a test forces another
        a.gd = "step8000" if a.geometry == "contact" else "ramp"
    if a.start_from:
        prev = json.load(open(a.start_from, encoding="utf-8"))
        a.t2, a.incl = prev["T2"], prev["incl"]
        a.f = prev.get("f", a.f)
        a.r1, a.r2 = prev["r1_a"], prev["r2_a"]
        if a.free_q:
            a.q = prev["q"]
        # continue a spotted fit: take its spot unless given on the command line
        if a.spot and prev.get("spot") == a.spot and "spot_long" in prev:
            a.spot_long = prev["spot_long"] if a.spot_long is None else a.spot_long
            a.spot_radius = prev["spot_radius"] if a.spot_radius is None else a.spot_radius
            a.spot_relteff = prev["spot_relteff"] if a.spot_relteff is None else a.spot_relteff
    # spot start (Entry 116): defaults as before (long 90, radius 20, relteff 0.9);
    # relteff kept below its 1.0 bound so the simplex can move both ways
    a.spot_long = 90.0 if a.spot_long is None else float(a.spot_long) % 360.0
    a.spot_radius = 20.0 if a.spot_radius is None else float(a.spot_radius)
    a.spot_relteff = 0.9 if a.spot_relteff is None else min(float(a.spot_relteff), 0.97)
    if a.spot:
        print(f"{a.star} [{a.label}] spot start on {a.spot}: long {a.spot_long:.1f}  "
              f"radius {a.spot_radius:.1f}  relteff {a.spot_relteff:.3f}"
              f"{'  (from ' + os.path.basename(a.start_from) + ')' if a.start_from else ''}", flush=True)
    name = (f"{a.star.replace(' ', '_')}_{a.geometry}_q{a.q:.3f}{'free' if a.free_q else ''}"
            f"_ld{a.ld}{'_gd' + a.gd if a.gd != 'step8000' else ''}"
            f"{'_irr' + a.irrad if a.irrad else ''}"
            f"{'_rad1' if a.radiative_primary else ''}"
            f"{'_gdfree' if a.gd_free else ''}"
            f"{'_atm' + a.atm if a.atm != 'blackbody' else ''}"
            f"{'_ldman' if a.atm != 'blackbody' and a.atm_ld == 'manual' else ''}"
            f"{f'_gd1at{a.gd_primary_teff:.0f}' if a.gd_primary_teff is not None else ''}_l3{a.l3}"
            f"{'_spot' + a.spot if a.spot else ''}"
            f"{f'_i{a.incl:.1f}fix' if a.fix_incl else ''}"
            f"{'_lc' + os.path.splitext(os.path.basename(a.lc_csv))[0] if a.lc_csv else ''}"
            f"{f'_n{a.ntriangles}' if a.ntriangles else ''}"
            f"{f'_s{a.l3_start_share:.3f}' if a.l3 == 'free' and a.l3_k is not None else ''}_{a.tag}.json")
    if a.skip_existing and not a.check and os.path.exists(os.path.join(a.outdir, a.label, name)):
        print(f"{a.star} [{a.label}] {name}: result exists, skipped")
        return 0
    b, src = build(a.star, a.geometry, a.q, lc_csv=a.lc_csv, ntriangles=a.ntriangles)
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

    if a.irrad:
        # Irradiation solver (Entry 106): Script 05 uses Wilson for contact stars and
        # Horvat otherwise; Horvat crashes deterministically for V878 Her.
        b.set_value("irrad_method", compute="phoebe01", value=a.irrad)
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
    if a.gd_primary_teff is not None:
        # Entry 135: hold star 1's gravity darkening and albedo at the ramp's values
        # for another temperature (e.g. RZ Dra at T1 8015 K with its 7329 K values);
        # T1 itself, and so the limb darkening, are unchanged
        beta, albedo = envelope_coeffs(a.gd_primary_teff, "ramp")
        b.set_value("gravb_bol@primary", beta)
        b.set_value("irrad_frac_refl_bol@primary", albedo)
        print(f"{a.star} [{a.label}] primary gravb_bol {beta:.3f}, albedo {albedo:.3f} "
              f"(ramp at {a.gd_primary_teff:.0f} K; T1 {t1:.0f} K)", flush=True)
    set_ld("primary", t1)
    if a.atm != "blackbody":
        # diagnostic (Entry 166): a model atmosphere instead of the blackbody, with the
        # light-curve limb darkening taken from the atmosphere's own angle-dependent
        # intensities (ld_mode "interp"). The passband table must hold that atmosphere:
        # set PHOEBE_PBDIR to a folder with such a table (the installed TESS:T is
        # blackbody-only). Bolometric (irradiation) settings stay as the pipeline's.
        if a.ld != "legacy":
            raise SystemExit("--atm needs --ld legacy (the atmosphere supplies the limb darkening)")
        for comp in ("primary", "secondary"):
            b.set_value(f"atm@{comp}", a.atm)
            if a.atm_ld == "interp":
                b.set_value(f"ld_mode@{comp}@lc01", "interp")
            # --atm-ld manual: keep the pipeline's linear 0.6 (ld_mode manual), so only the
            # intensities change; separates the atmosphere's intensity effect from its
            # limb-darkening effect (Entry 167)
        print(f"{a.star} [{a.label}] atmosphere {a.atm} (LD {a.atm_ld}); PHOEBE_PBDIR="
              f"{os.environ.get('PHOEBE_PBDIR', '(not set)')}", flush=True)
    ld_free = a.ld == "free"
    l3_free = a.l3 == "free"
    # free third light: start, upper bound and simplex step in raw l3_frac
    l3_start, l3_hi, l3_step = 0.2, 0.85, 0.05            # before Entry 142 (raw values)
    if l3_free and a.l3_k is not None:
        def frac_for_share(s):
            x = s / (1.0 - s) / a.l3_k
            return x / (1.0 + x)
        s0, smax = a.l3_start_share, a.l3_max_share
        l3_start, l3_hi = frac_for_share(s0), frac_for_share(smax)
        l3_step = frac_for_share(min(s0 + 0.05, 0.99)) - l3_start
        print(f"{a.star} [{a.label}] third light free: share start {s0:.3f}, max {smax:.3f} "
              f"-> l3_frac start {l3_start:.5f}, max {l3_hi:.5f}, step {l3_step:.5f} (k {a.l3_k})",
              flush=True)
    if a.l3 != "none":
        b.set_value("l3_mode", dataset="lc01", value="fraction")
        b.set_value("l3_frac", dataset="lc01", context="dataset",
                    value=l3_start if l3_free else float(a.l3))
    if a.spot:
        b.add_feature("spot", component=a.spot, feature="spot01",
                      colat=90.0, long=a.spot_long, radius=a.spot_radius,
                      relteff=a.spot_relteff)

    # ── free parameters: (name, start, lo, hi, simplex scale) ────────────────
    t2_lo = (CONTACT_T2_FLOOR_FRAC if contact else 0.50) * t1
    t2_hi = min((1.25 if contact else 1.10) * t1, 8000.0 if 0.95 * t1 < 8000.0 else 1e9)
    P = [("T2", a.t2 or 0.8 * t1, t2_lo, t2_hi, 200.0)]
    if not a.fix_incl:       # --fix-incl (Entry 106): the inclination is held at --incl
        P.append(("incl", a.incl, CONTACT_INCL_FLOOR_DEG if contact else 60.0, 90.0, 1.5))
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
        P.append(("l3_fit", l3_start, 0.0, l3_hi, l3_step))
    if ld_free:
        # diagnostic: ONE linear coefficient for both stars, fitted — what value
        # of the pipeline's single coefficient do the data ask for?
        P.append(("u_fit", LEGACY_LINEAR_LD, 0.0, 1.0, 0.05))
    if a.gd_free:
        # diagnostic (Entry 166): ONE gravity-darkening exponent for both stars, fitted
        # over 0-1 (0.32 convective, 1.0 radiative; von Zeipel 1924, Lucy 1967); the
        # albedo stays at the rule's value
        P.append(("beta_fit", float(b.get_value("gravb_bol", component="primary", context="component")),
                  0.0, 1.0, 0.1))
    if a.spot:
        P += [("spot_long", a.spot_long, 0.0, 360.0, 30.0),
              ("spot_radius", a.spot_radius, 5.0, 60.0, 5.0),
              ("spot_relteff", a.spot_relteff, 0.6, 1.0, 0.03)]
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
        comp_put(b, "incl@binary", d.get("incl", a.incl))
        set_ld("secondary", d["T2"])
        set_envelope("secondary", d["T2"])
        if "u_fit" in d:
            for comp in ("primary", "secondary"):
                b.set_value(f"ld_coeffs@{comp}@lc01", [d["u_fit"]])
        if "beta_fit" in d:          # after set_envelope, so the fitted value wins
            for comp in ("primary", "secondary"):
                b.set_value(f"gravb_bol@{comp}", d["beta_fit"])
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
           "atm": a.atm, "atm_ld": a.atm_ld if a.atm != "blackbody" else "", "pbdir": os.environ.get("PHOEBE_PBDIR", ""),
           "irrad": b.get_value("irrad_method", compute="phoebe01"),
           "converged": bool(res.success), "T1": t1,
           "n_failed": evals.get("failed", 0), "fail_msg": evals.get("fail_msg", ""),
           "beta1": float(b.get_value("gravb_bol", component="primary", context="component")),
           "albedo1": float(b.get_value("irrad_frac_refl_bol", component="primary", context="component")),
           "gd_primary_teff": a.gd_primary_teff if a.gd_primary_teff is not None else "",
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
    out.setdefault("incl", float(a.incl))
    out["fix_incl"] = bool(a.fix_incl)
    out["lc_csv"] = a.lc_csv or ""
    out["ntriangles"] = a.ntriangles or MESH_TRIANGLE_COUNT
    out["start_from"] = os.path.basename(a.start_from) if a.start_from else ""
    out["maxfev"] = a.maxfev                # Entry 147: the cap behind "converged False"
    if l3_free and a.l3_k is not None:      # Entry 142: record the conversion with the fit
        x = a.l3_k * out["l3_fit"] / (1.0 - out["l3_fit"])
        out.update(l3_k=a.l3_k, l3_start_share=a.l3_start_share, l3_max_share=a.l3_max_share,
                   l3_share_fit=x / (1.0 + x))
    if a.spot:
        out["spot_start"] = [a.spot_long, a.spot_radius, a.spot_relteff]
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
    keys = ["star", "label", "geometry", "ld", "gd", "irrad", "radiative_primary", "l3", "spot", "free_q",
            "start_tag", "q", "chi2", "chi2_red", "converged", "T2", "incl", "f", "r1_a", "r2_a",
            "r1_over_lobe", "r2_over_lobe", "u_fit", "beta_fit", "beta1", "l3_fit", "spot_long", "spot_radius",
            "spot_relteff", "nfev", "wall_s", "T1", "albedo1", "gd_primary_teff",
            # appended in Entry 147 (readers key by name, so earlier columns keep their place)
            "fix_incl", "ntriangles", "lc_csv", "start_from", "maxfev", "n_failed", "l3_k", "l3_share_fit"]
    # Entry 147: a labelled collect writes its own file and never replaces the full
    # table (Entry 117: "collect --label X" used to overwrite model_tests.csv with label
    # X only, and audit_error_budget.py reads that table). Only --label "" rebuilds it.
    out_csv = os.path.join(AUDIT_DIR, "model_tests.csv" if not a.label else f"model_tests_{a.label}.csv")
    if os.path.abspath(a.outdir) != os.path.abspath(DEFAULT_OUT):
        # a scratch results folder never writes into Audits/
        out_csv = os.path.join(a.outdir, os.path.basename(out_csv))
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
    p.add_argument("--fix-incl", action="store_true", help="hold the inclination at --incl")
    p.add_argument("--lc-csv", default=None, help="binned light curve (phase, flux, sigma) instead of Stage 04's")
    p.add_argument("--ntriangles", type=int, default=0, help="mesh size (default 2000)")
    p.add_argument("--f", type=float, default=0.3)
    p.add_argument("--r1", type=float, default=0.30, help="start r1/a")
    p.add_argument("--r2", type=float, default=0.25, help="start r2/a")
    p.add_argument("--ld", choices=["legacy", "table", "quad", "free"], default="legacy",
                   help="legacy: linear 0.6; table: linear by Teff; quad: quadratic by Teff "
                        "(Claret 2017); free: one linear coefficient fitted (diagnostic)")
    p.add_argument("--atm", choices=["blackbody", "ck2004"], default="blackbody",
                   help="diagnostic: model atmosphere (needs a passband table with it; PHOEBE_PBDIR)")
    p.add_argument("--atm-ld", choices=["interp", "manual"], default="interp",
                   help="with --atm: limb darkening from the atmosphere (interp) or the pipeline's linear 0.6 (manual)")
    p.add_argument("--gd-free", action="store_true",
                   help="diagnostic: one gravity-darkening exponent for both stars, fitted over 0-1")
    p.add_argument("--gd", choices=["auto", "step8000", "ramp", "step6600"], default="auto",
                   help="gravity darkening and albedo: auto = production's rule (contact: the "
                        "8000 K step; others: the ramp over 6600-8000 K; Entry 147), or force "
                        "the 8000 K step, the ramp, or a step at 6600 K (gravdark.py). Before "
                        "Entry 147 the default was step8000 for every geometry")
    p.add_argument("--start-from", default=None,
                   help="a result .json of an earlier fit: its solution is the start")
    p.add_argument("--irrad", choices=["wilson", "horvat"], default=None,
                   help="irradiation solver (default: as Script 05, Wilson for contact, Horvat otherwise)")
    p.add_argument("--skip-existing", action="store_true",
                   help="do nothing if this fit's result file is already there (resume)")
    p.add_argument("--radiative-primary", action="store_true")
    p.add_argument("--gd-primary-teff", type=float, default=None,
                   help="give the primary the ramp's gravity darkening and albedo at this "
                        "temperature (K) instead of at its own T1; 6600 = convective end, "
                        "8000 = radiative end (Entry 135)")
    p.add_argument("--l3", default="none", help="none | free | <fixed fraction>")
    p.add_argument("--l3-k", type=float, default=None,
                   help="with --l3 free: the star's third-light scaling factor k "
                        "(Audits/l3_scaling.csv); start and bound are then given as shares")
    p.add_argument("--l3-start-share", type=float, default=0.10,
                   help="with --l3-k: start share of the mean light (default 0.10)")
    p.add_argument("--l3-max-share", type=float, default=0.60,
                   help="with --l3-k: upper bound, share of the mean light (default 0.60)")
    p.add_argument("--spot", choices=["primary", "secondary"], default=None)
    p.add_argument("--spot-long", type=float, default=None,
                   help="spot start longitude (deg, the star's own frame; default 90, "
                        "or the --start-from fit's spot)")
    p.add_argument("--spot-radius", type=float, default=None, help="spot start radius (deg; default 20)")
    p.add_argument("--spot-relteff", type=float, default=None, help="spot start relteff (default 0.9, max 0.97)")
    p.add_argument("--maxfev", type=int, default=160)
    p.add_argument("--label", default="test")
    p.add_argument("--tag", default="a", help="distinguishes fits with different starts")
    p.add_argument("--check", action="store_true")
    p.add_argument("--outdir", default=DEFAULT_OUT)
    a = p.parse_args()
    return {"fit": mode_fit, "collect": mode_collect}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
