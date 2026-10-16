"""
================================================================================
AUDIT — CONTACT-MODEL q PROFILE FOR A STAR ROUTED SEMI-DETACHED
================================================================================
Summary:
    Pipeline_Log Entry 87. In the q-profile audit (audit_q_profile.py), V358 Cep's
    better basin (a big hot primary) pins r1 at its Roche lobe at every q while
    the secondary fills its own: both stars at their lobes, i.e. a contact or
    marginal-contact system. This audit fits the same 151-bin light curve with a
    PHOEBE CONTACT model (common envelope, fill-out f free) at fixed q, so the
    two geometries can be compared on equal terms.

    The contact bundle is built the way Script 04 builds contact bundles
    (default_binary(contact_binary=True), the fill-out "double flip", blackbody
    atmospheres, 2000 triangles) and configured exactly as Script 05 configures
    any bundle (audit_q_profile.configure_like_05, verified equal to a 05
    checkpoint). The data, period, t0, exposure time and fti settings are copied
    from the star's Script-04 bundle. Wilson irradiation, as 05 uses for contact.

    Free per job: T2 in [0.65, 1.25] T1 (wider than either 05 prior, so railing is
    visible), i in [60, 90], f in [0.0, 0.8]. q fixed.

    Heavy-ish like audit_q_profile.py: ~100 forward models per job.

Usage (from Code/, WSL python3):
    python3 Audits/audit_contact_profile.py --star "V358 Cep" --bundle <04 bundle> --q 0.5 \
            [--f 0.1 --t2 4400 --incl 78 --maxfev 110 --outdir Runs/q_investigation/jobs_contact]
    python3 Audits/audit_contact_profile.py --star "V358 Cep" --bundle <04 bundle> --q 0.5 --probe
        (one forward model at the start values; prints chi2 and timing)
================================================================================
"""
import argparse
import json
import os
import sys
import time

import numpy as np

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AUDIT_DIR)
sys.path.insert(0, os.path.dirname(AUDIT_DIR))
from audit_q_profile import RADIATIVE_CONVECTIVE_BOUNDARY_K, chi2, configure_like_05, load  # noqa: E402

MESH_TRIANGLE_COUNT = 2000          # Script 04


def build_contact(sd_path: str, q: float, f: float, t2: float, incl: float):
    import phoebe
    sd = load(sd_path)
    b = phoebe.default_binary(contact_binary=True)
    b.set_value("period@binary", sd.get_value("period", component="binary", context="component"))
    b.set_value("t0_supconj@binary", sd.get_value("t0_supconj", component="binary", context="component"))
    b.set_value("ecc@binary", 0.0)
    b.set_value("incl@binary", incl)
    b.set_value("teff@primary", sd.get_value("teff", component="primary", context="component"))
    b.set_value("teff@secondary", t2)
    b.set_value("atm@primary", "blackbody")
    b.set_value("atm@secondary", "blackbody")
    b.flip_constraint("pot@contact_envelope", solve_for="requiv@primary")
    b.flip_constraint("fillout_factor@contact_envelope", solve_for="pot@contact_envelope")
    b.set_value("q@binary", q)
    b.set_value("fillout_factor@component", f)
    b.set_value_all("ntriangles", MESH_TRIANGLE_COUNT)
    b.add_dataset("lc", dataset="lc01", passband="TESS:T", overwrite=True,
                  times=sd.get_value("times", dataset="lc01", context="dataset"),
                  fluxes=sd.get_value("fluxes", dataset="lc01", context="dataset"),
                  sigmas=sd.get_value("sigmas", dataset="lc01", context="dataset"))
    exptime = float(sd.get_value("exptime", dataset="lc01", context="dataset", check_visible=False))
    if exptime > 0:
        b.set_value("exptime", dataset="lc01", context="dataset", value=exptime)
        b.set_value("fti_method", compute="phoebe01", dataset="lc01",
                    value=sd.get_value("fti_method", compute="phoebe01", dataset="lc01",
                                       check_visible=False))
        b.set_value("fti_oversample", compute="phoebe01", dataset="lc01",
                    value=int(sd.get_value("fti_oversample", compute="phoebe01", dataset="lc01",
                                           check_visible=False)))
    configure_like_05(b, kind="contact")
    return b


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--star", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--q", type=float, required=True)
    p.add_argument("--f", type=float, default=0.10)
    p.add_argument("--t2", type=float, default=None)   # default 0.75 T1
    p.add_argument("--incl", type=float, default=78.0)
    p.add_argument("--maxfev", type=int, default=110)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--f-max", type=float, default=0.8)
    p.add_argument("--config-check", action="store_true",
                   help="chi2 of the built contact bundle vs the real Script-04 contact "
                        "bundle (configured like 05) at identical parameters")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(AUDIT_DIR), "Runs",
                                                    "q_investigation", "jobs_contact"))
    a = p.parse_args()
    from scipy.optimize import minimize
    t_start = time.time()
    if a.t2 is None:
        a.t2 = 0.75 * float(load(a.bundle).get_value("teff", component="primary", context="component"))
    b = build_contact(a.bundle, a.q, a.f, a.t2, a.incl)
    t1 = float(b.get_value("teff", component="primary", context="component"))
    t2_lo, t2_hi = 0.65 * t1, min(1.25 * t1, RADIATIVE_CONVECTIVE_BOUNDARY_K)
    if a.config_check:
        real = load(a.bundle)
        configure_like_05(real, kind="contact")
        for bb in (real, b):
            bb.set_value("q@binary", context="component", value=a.q)
            bb.set_value("teff@secondary", context="component", value=a.t2)
            bb.set_value("incl@binary", context="component", value=a.incl)
            bb.set_value("fillout_factor@contact_envelope", context="component", value=a.f)
        c_real, c_built = chi2(real), chi2(b)
        rel = abs(c_real - c_built) / c_real
        print(json.dumps({"chi2_script04_bundle": c_real, "chi2_built": c_built,
                          "rel_diff": rel, "pass": rel < 1e-6}))
        return 0 if rel < 1e-6 else 1
    if a.probe:
        t0 = time.time()
        c = chi2(b)
        print(json.dumps({"star": a.star, "q": a.q, "f": a.f, "chi2": c,
                          "model_s": round(time.time() - t0, 1),
                          "requiv_primary": b.get_value("requiv", component="primary", context="component"),
                          "requiv_secondary": b.get_value("requiv", component="secondary", context="component"),
                          "sma": b.get_value("sma", component="binary", context="component")}))
        return 0

    scale = np.array([200.0, 1.5, 0.05])
    evals = {"n": 0, "best": (float("inf"), None)}

    def objective(z):
        x = np.asarray(z) * scale
        t2 = float(np.clip(x[0], t2_lo, t2_hi))
        inc = float(np.clip(x[1], 60.0, 90.0))
        f = float(np.clip(x[2], 0.0, a.f_max))
        penalty = 1e3 * (abs(x[0] - t2) / 200 + abs(x[1] - inc) / 1.5 + abs(x[2] - f) / 0.05)
        evals["n"] += 1
        b.set_value("teff@secondary", context="component", value=t2)
        b.set_value("incl@binary", context="component", value=inc)
        b.set_value("fillout_factor@contact_envelope", context="component", value=f)
        c = chi2(b) + penalty
        if c < evals["best"][0]:
            evals["best"] = (c, (t2, inc, f))
        return c

    res = minimize(objective, np.array([a.t2, a.incl, a.f]) / scale, method="Nelder-Mead",
                   options={"maxfev": a.maxfev, "xatol": 1e-3, "fatol": 1e-2})
    _, (t2, inc, f) = evals["best"]
    b.set_value("teff@secondary", context="component", value=t2)
    b.set_value("incl@binary", context="component", value=inc)
    b.set_value("fillout_factor@contact_envelope", context="component", value=f)
    c_final = chi2(b)
    sma = float(b.get_value("sma", component="binary", context="component"))
    phases = ((b.get_value("times", dataset="lc01", context="dataset")
               - b.get_value("t0_supconj", component="binary", context="component"))
              / b.get_value("period", component="binary", context="component")) % 1.0
    out = {"star": a.star, "mode": "contact", "q_start": a.q, "r1_start": "contact", "q": a.q,
           "chi2": c_final, "n": len(phases), "chi2_red": c_final / max(1, len(phases) - 3),
           "T2": t2, "T2_over_T1": t2 / t1, "incl": inc, "fillout": f,
           "r1_a": float(b.get_value("requiv", component="primary", context="component")) / sma,
           "r2_a": float(b.get_value("requiv", component="secondary", context="component")) / sma,
           "nm_message": str(res.message), "nfev": evals["n"],
           "wall_s": round(time.time() - t_start, 1), "phase": list(map(float, phases)),
           "obs": list(map(float, b.get_value("fluxes", dataset="lc01", context="dataset"))),
           "model": list(map(float, b.get_value("fluxes", dataset="lc01", model="qprof")))}
    os.makedirs(a.outdir, exist_ok=True)
    tag = f"{a.star.replace(' ', '_')}_contact_q{a.q:.2f}_f{a.f:.2f}_max{a.f_max:.2f}"
    out["f_start"], out["f_max"] = a.f, a.f_max
    with open(os.path.join(a.outdir, tag + ".json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle)
    print(f"{tag}: chi2 {c_final:.1f} (red {out['chi2_red']:.1f}) T2 {t2:.0f} ({t2/t1:.3f} T1) "
          f"i {inc:.2f} f {f:.3f} r1 {out['r1_a']:.3f} r2 {out['r2_a']:.3f} "
          f"({evals['n']} models, {out['wall_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
