"""
================================================================================
AUDIT — ERROR BUDGET OF EACH FINISHED STAR (statistical and systematic terms)
================================================================================
Summary:
    The results files quote MCMC intervals only. This audit lists, per finished
    star (a results file whose input_bundle_sha is the current bundle's), every
    term the thesis tables need (Pipeline_Log Entries 110, 112, 115-118, 124, 126):

      stat      the MCMC 1-sigma (results file), and beta, a correlated-residual
                factor from the final residuals (time-averaging method: beta =
                measured scatter of N-point binned residuals / (sigma_1 / sqrt(N)
                * sqrt(M / (M - 1))), mean over N = 2..15; Winn et al. 2008, read
                through Cubillos et al. 2017, who show it can underestimate
                correlated noise: a lower bound);
      q         contact stars: incl/fill-out change over the q range
                (Audits/q_systematics.csv); semi-detached: q is fitted;
      T1        sigma_T1 (Audits/t1_anchor.csv) x dT2/dT1 at a fixed TESS-band
                surface-brightness ratio (blackbodies, 600-1000 nm, as Entry 110):
                T2_sys_T1, blackbody only, an upper bound for ramp stars.
                T2_sys_T1_lo/_hi (Entry 135) add the gravity-darkening change: for a
                non-contact star, moving T1 by -/+sigma also moves the primary along
                the 6600-8000 K ramp (gravdark.py), by S x (change in ramp fraction);
      gd        (Entry 135) the gravity-darkening choice for non-contact stars: the
                T2 shift from the ramp to the two end members (step8000, step6600)
                at the adopted T1; zero outside 6600-8000 K, where all three agree.
                Contact stars use step8000, chosen by chi-squared (Entry 93): blank.
                S = dT2 per unit ramp fraction of the primary (beta 0.32 -> 1.0,
                albedo 0.6 -> 1.0), from converged Nelder-Mead fits in
                Audits/model_tests.csv that differ only in the primary's beta
                (slope of T2 against beta1, median over matching groups). A
                stalled fit counts only if named in ACCEPT_STALLED (RZ Dra G1,
                Entry 139); gd_S_src then says so. Stars
                without such fits borrow the measured range: the smaller |S| for
                the T1 term, the larger for the gd term (both conservative). The
                secondary's own ramp position is not varied (secondaries of the
                finished stars are below 6600 K);
      spot      a star's own shifts where it has converged spotted fits
                (Audits/spot_terms.csv); otherwise, for stars with |O'Connell|
                >= 1 % (Audits/oconnell.csv), the sample-wide term from the spot
                tests on six stars (i -0.9..+1.0 deg, q -4..+3 %, f -0.001..+0.008,
                T2 <= 81 K, detached <= 120 K): interim values under the spot-term
                rule of 11 Nov 2026 (see SPOT below);
      LD        sample-wide i +0.8..+2.1 deg (linear 0.6 against table LD);
      l3        fitted, fixed or none (from the flags). Fitted and fixed values are
                converted to the share of the mean model flux (Entries 136-137):
                in PHOEBE's dataset-scaled mode l3_frac puts k times more third
                light into the model, l3_eff/(1-l3_eff) = k l3_frac/(1-l3_frac),
                k from Audits/l3_scaling.csv (k_final, measured on the final
                bundle; used only if its l3_frac_med equals the results file's).
                Columns l3_frac (raw), l3_k, l3_pct, l3_pct_lo/_hi (16/84 %);
      floor     share of bins where the 0.003 error floor dominates.

    Semi-detached stars: r2 (constrained to the lobe) gets an error from the q
    posterior: for 300 posterior draws, q and requiv@primary are set in the final
    bundle, the constraints run, and r2 = requiv@secondary / sma is read; the same
    pass gives the primary's fraction of its lobe (requiv / requiv_max) and the
    share of draws above 0.97.

    Read-only: loads the final bundles, computes no light curve.
Usage (WSL, from Code/; about a minute per star):
    python3 Audits/audit_error_budget.py [Targets/phaseF_queue.csv]
Output:
    Audits/error_budget.csv; a table on stdout.
================================================================================
"""
import csv
import hashlib
import logging
import math
import os
import sys

import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
OUT = os.path.join(CODE, "Audits", "error_budget.csv")
BURN = 350
N_DRAWS = 300
# Spot term (Entries 115-118, widened in 146-147 by TZ Lyr's continued spot fits:
# i +0.23/+0.45 deg, q -1.0/-3.7 %, T2 +14/+33 K). q is two-sided. A minimum,
# not a per-star error. Since Entry 163 i and f are two-sided as well: V1128 Tau's
# spotted fit moved i by -0.85 deg and f by -0.001 (the first negative shifts), so
# the ranges now span the spot tests on six stars (V842 Her, V358 Cep, PX Car,
# TZ Lyr, OT Cep, V1128 Tau). T2 is kept at <=35 K (OT Cep +41, V1128 Tau +81 K
# exceed it; both carry their own term).
SPOT = {"i_deg": "-0.9..+1.0", "q_pct": "-4..+3", "f": "-0.001..+0.008", "T2_K": "<=81"}
# T2: <= 81 K is the largest shift among the contact and semi-detached spot tests
# (V1128 Tau +81, OT Cep +41, the others <= 35 K); detached stars keep PX Car's
# +120 K, quoted separately since Entry 117 (interim values, user decision 11 Nov).
SPOT_T2_DETACHED = "<=120"
#
# SPOT-TERM RULE, fixed 11 Nov 2026 before any further spot fits (Entry 164).
# Fixed in advance so that the term cannot drift towards whatever looks best.
#  1. A spot shift = a CONVERGED spotted fit minus a no-spot reference fit of the
#     same batch and settings (the better-chi2 converged spot fit if both stars were
#     tried). Fits stopped at the model cap are continued to convergence or not used.
#  2. Residual wave A_res of a finished star: the rms of a 4th-order Fourier series
#     fitted by least squares to the 151 binned residuals (data - model) of its
#     production fit, in % of the normalised flux (the constant term is fitted but
#     not counted: an offset is not a wave). The Fourier series keeps the
#     coherent wave and drops bin-to-bin noise. It measures any coherent misfit;
#     the spot fit tests whether a spot explains it. (Reason: V1128 Tau's maxima
#     differ by only 0.47 %, yet its residuals carry a +-1.5 % wave and it gave the
#     largest T2 shift: the O'Connell size alone misses spots.)
#  3. Trigger: a star carries a spot term if |O'Connell| >= 1 % or A_res >= 0.3 %
#     (the systematic error floor of 04, SYS_ERROR_FLOOR). Each such finished star
#     is proposed for spot fits (spot on each star, start longitudes 90 and 270, and
#     the reference); the user approves each batch.
#  4. A star with its own converged spotted fits uses its own shifts (spot_terms.csv).
#  5. Any other triggered star: per quantity (i in deg, q in %, f, T2 in % of T2),
#     the term is +-k * A_res, k = the 90th percentile of |shift| / A_res over the
#     converged fits of its group (contact and semi-detached together; detached
#     apart), PROVIDED the group has >= 8 converged fits and a one-sided Spearman
#     test of |shift| against A_res gives p < 0.05 for that quantity. Otherwise the
#     flat term: the largest |shift| (two-sided) while the group has fewer than 10
#     converged fits, the 90th percentile of |shift| from 10 on.
#  6. The term is recomputed ONCE, in the January audit pass after the production
#     freeze, and reported with N, the statistic used and the test result; it is
#     not retuned star by star.
#  7. Until then (interim): the flat values above (SPOT, SPOT_T2_DETACHED), which
#     include capped fits, and the trigger |O'Connell| >= 1 % (A_res is not computed
#     yet).
# Per-star spot terms (user decision 6 Nov, option 2; first used Entry 162):
# Audits/spot_terms.csv (star, i_deg, q_pct, f, T2_K, source) replaces the
# sample-wide term for a star whose own spotted fits exist, whatever its O'Connell.
SPOT_TERMS_CSV = os.path.join(CODE, "Audits", "spot_terms.csv")
# Blend terms (Entry 166): for a star whose TESS aperture holds a variable neighbour that
# was NOT removed (Targets/variable_neighbours.csv lists the removed ones), the shift that
# removing it produced in test fits (Audits/blend_terms.csv: star, i_deg, f, T2_K,
# source). Optional; a missing file means no blend terms.
BLEND_TERMS_CSV = os.path.join(CODE, "Audits", "blend_terms.csv")
LD_I = "+0.8..+2.1"
GD_KEY = ("star", "label", "geometry", "ld", "irrad", "l3", "spot", "free_q", "T1")
# Non-converged fits accepted in the slope, each named here, keyed by
# (star, label, beta1, chi2) as written in model_tests.csv (Entry 139, user decision
# 4 Nov). RZ Dra G1 (label GD, beta1 0.674, T1 8015): it stopped at the 500-model cap
# against the r2 = lobe boundary; a further 500 models moved chi2 only 89.996 -> 89.992
# and left T2 unchanged. Only that lower-chi2 row is used.
ACCEPT_STALLED = {("RZ Dra", "GD", "0.67409", "89.99152"): "RZ Dra G1"}


def ramp_frac(t):
    """The primary's position on the gravity-darkening ramp (0 convective, 1 radiative)."""
    from gravdark import BETA_CONVECTIVE, BETA_RADIATIVE, envelope_coeffs
    return (envelope_coeffs(t, "ramp")[0] - BETA_CONVECTIVE) / (BETA_RADIATIVE - BETA_CONVECTIVE)


def gd_sensitivity(path):
    """{star: (S, n_groups, n_groups_with_a_stalled_fit)}: dT2 per unit ramp fraction
    of the primary, from fits that differ only in the primary's beta (converged, or
    named in ACCEPT_STALLED; same star/label/geometry/LD/irradiation/l3/spot/
    q-freedom/T1)."""
    from gravdark import BETA_CONVECTIVE, BETA_RADIATIVE
    groups = {}
    try:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
    except OSError:
        return {}
    for r in rows:
        stalled = (r.get("star"), r.get("label"), r.get("beta1"), r.get("chi2")) in ACCEPT_STALLED
        if (r.get("converged") != "True" and not stalled) or r.get("geometry") == "contact":
            continue                      # contact stars do not use the ramp
        try:
            x = (float(r["beta1"]) - BETA_CONVECTIVE) / (BETA_RADIATIVE - BETA_CONVECTIVE)
            t2 = float(r["T2"])
        except (KeyError, ValueError):
            continue
        groups.setdefault(tuple(r.get(k, "") for k in GD_KEY), []).append((x, t2, stalled))
    per_star = {}
    for key, pts in groups.items():
        xs = np.array([p[0] for p in pts])
        if xs.max() - xs.min() < 0.2:
            continue
        slope = float(np.polyfit(xs, [p[1] for p in pts], 1)[0])
        per_star.setdefault(key[0], []).append((slope, any(p[2] for p in pts)))
    return {s: (float(np.median([v[0] for v in vs])), len(vs), sum(v[1] for v in vs))
            for s, vs in per_star.items()}


def file_sha(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()[:12]


def results(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8", errors="replace")):
        def num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return math.nan
        out[r["Symbol"]] = (r["Value"], num(r["Value"]), num(r["Sigma_minus"]), num(r["Sigma_plus"]))
    return out


def band_ln_slope(T):
    """d ln S / dT for a blackbody integrated over 600-1000 nm (energy units)."""
    lam = np.linspace(600e-9, 1000e-9, 400)
    c2 = 1.4387769e-2

    def S(t):
        trap = getattr(np, "trapezoid", None) or np.trapz
        return trap(lam ** -5 / np.expm1(c2 / (lam * t)), lam)
    dT = 1.0
    return (np.log(S(T + dT)) - np.log(S(T - dT))) / (2 * dT)


def dT2_dT1(t1, t2):
    """Change of T2 per kelvin of T1 at a fixed band surface-brightness ratio."""
    return band_ln_slope(t1) / band_ln_slope(t2)


def beta_factor(res):
    res = np.asarray(res, float)
    s1 = np.std(res, ddof=1)
    out = []
    for n in range(2, 16):
        m = len(res) // n
        if m < 5:
            break
        binned = res[: m * n].reshape(m, n).mean(axis=1)
        expect = s1 / math.sqrt(n) * math.sqrt(m / (m - 1))
        out.append(np.std(binned, ddof=1) / expect)
    return float(np.mean(out)) if out else math.nan


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def l3_share(l3_frac, k):
    """Share of the mean model flux that l3_frac puts into the model in PHOEBE's
    dataset-scaled mode: l3_eff/(1-l3_eff) = k l3_frac/(1-l3_frac) (Entries 136-137)."""
    x = k * l3_frac / (1.0 - l3_frac)
    return x / (1.0 + x)


def table(path, key="star"):
    try:
        return {r[key]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    except OSError:
        return {}


def main() -> int:
    logging.disable(logging.WARNING)
    import phoebe
    cat = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CODE, "Targets", "phaseF_queue.csv")
    stars = [r["name"] for r in csv.DictReader(open(cat, encoding="utf-8"))]
    t1a = table(os.path.join(CODE, "Audits", "t1_anchor.csv"))
    qsys = table(os.path.join(CODE, "Audits", "q_systematics.csv"))
    spot_terms = table(SPOT_TERMS_CSV)
    blend_terms = table(BLEND_TERMS_CSV)
    oc = table(os.path.join(CODE, "Audits", "oconnell.csv"))
    floor = table(os.path.join(CODE, "Audits", "error_floor.csv"))
    l3k = table(os.path.join(CODE, "Audits", "l3_scaling.csv"))
    gds = gd_sensitivity(os.path.join(CODE, "Audits", "model_tests.csv"))
    s_all = [abs(v[0]) for v in gds.values()]
    s_small, s_large = (-min(s_all), -max(s_all)) if s_all else (math.nan, math.nan)
    print("gravity-darkening sensitivity S (K per unit ramp fraction), measured: "
          + ", ".join(f"{k} {v[0]:.0f} ({v[1]} group{'s' if v[1] > 1 else ''}"
                      f"{f', {v[2]} with a stalled fit' if v[2] else ''})" for k, v in sorted(gds.items())))
    rows = []
    for star in stars:
        safe = star.replace(" ", "_")
        rp = os.path.join(CODE, "data", "results", f"results_{safe}.csv")
        if not os.path.exists(rp):
            continue
        r = results(rp)
        if r.get("input_bundle_sha", ("",))[0] != file_sha(os.path.join(CODE, "data", "bundles", f"bundle_{safe}.phoebe")):
            continue                                   # not a result of the current bundle
        morph = r.get("morphology", ("",))[0]
        g = lambda k, i=1: r.get(k, ("", math.nan, math.nan, math.nan))[i]
        row = {"star": star, "morphology": morph, "chi2_red": g("chi2_red"),
               "T1_K": g("T1_K"), "T2_K": g("T2_K"), "T2_sig": 0.5 * (g("T2_K", 2) + g("T2_K", 3)),
               "T2_over_T1": g("T2_over_T1"), "T2T1_sig": 0.5 * (g("T2_over_T1", 2) + g("T2_over_T1", 3)),
               "incl": g("incl_deg"), "incl_sig": 0.5 * (g("incl_deg", 2) + g("incl_deg", 3)),
               "q": g("q"), "q_sig": 0.5 * (g("q", 2) + g("q", 3)),
               "fillout": g("fillout_factor"), "fillout_sig": 0.5 * (g("fillout_factor", 2) + g("fillout_factor", 3)),
               "flags": r.get("flags", ("",))[0]}
        fb = os.path.join(CODE, "data", "bundles", f"bundle_{safe}_final.phoebe")
        b = phoebe.load(fb)
        obs = np.asarray(b.get_value("fluxes", dataset="lc01", context="dataset"), float)
        mod = np.asarray(b.get_value("fluxes", dataset="lc01", model="solution_model"), float)
        times = np.asarray(b.get_value("times", dataset="lc01", context="dataset"), float)
        order = np.argsort(times)
        resid = (obs - mod)[order]
        row["resid_rms"] = float(np.std(resid, ddof=1))
        row["beta"] = beta_factor(resid)
        # T1 term
        a = t1a.get(star, {})
        try:
            s_t1 = float(a.get("sigma_T1", "nan"))
        except ValueError:
            s_t1 = math.nan
        row["sigma_T1"] = s_t1
        row["dT2_dT1_bb"] = dT2_dT1(row["T1_K"], row["T2_K"])
        row["T2_sys_T1"] = row["dT2_dT1_bb"] * s_t1
        # gravity darkening (Entry 135)
        t1v = row["T1_K"]
        row.update(gd_x1="", gd_S_K="", gd_S_src="", T2_sys_gd_lo="", T2_sys_gd_hi="")
        row["T2_sys_T1_lo"] = -row["T2_sys_T1"]
        row["T2_sys_T1_hi"] = row["T2_sys_T1"]
        if morph != "contact" and math.isfinite(t1v):
            x0 = ramp_frac(t1v)
            meas = gds.get(star)
            s_t1term, s_gd = (meas[0], meas[0]) if meas else (s_small, s_large)
            row["gd_x1"] = x0
            row["gd_S_K"] = meas[0] if meas else f"{s_small:.0f}..{s_large:.0f}"
            row["gd_S_src"] = (f"measured ({meas[1]}" + (f"; {meas[2]} with a stalled fit" if meas[2] else "")
                               + ")") if meas else "borrowed"
            if math.isfinite(s_t1):
                lo = -row["dT2_dT1_bb"] * s_t1 + s_t1term * (ramp_frac(t1v - s_t1) - x0)
                hi = row["dT2_dT1_bb"] * s_t1 + s_t1term * (ramp_frac(t1v + s_t1) - x0)
                row["T2_sys_T1_lo"], row["T2_sys_T1_hi"] = min(lo, hi), max(lo, hi)
            x_s8000 = 1.0 if t1v >= 8000.0 else 0.0
            x_s6600 = 1.0 if t1v > 6600.0 else 0.0
            ends = [s_gd * (x_s8000 - x0), s_gd * (x_s6600 - x0)]
            row["T2_sys_gd_lo"], row["T2_sys_gd_hi"] = min(ends + [0.0]) + 0.0, max(ends + [0.0]) + 0.0   # no -0
        # q term (contact)
        qs = qsys.get(star, {})
        row["incl_sys_q"] = qs.get("incl_sys_q", "") if morph == "contact" else ""
        row["fillout_sys_q"] = qs.get("fillout_sys_q", "") if morph == "contact" else ""
        # spot term
        try:
            ocp = float(oc.get(star, {}).get("oconnell_pct", "nan"))
        except ValueError:
            ocp = math.nan
        row["oconnell_pct"] = ocp
        spot = math.isfinite(ocp) and abs(ocp) >= 1.0
        own = spot_terms.get(star)
        for k, v in SPOT.items():
            if k == "T2_K" and morph == "detached":
                v = SPOT_T2_DETACHED
            row[f"spot_{k}"] = own.get(k, "") if own else (v if spot else "")
        row["spot_src"] = (own.get("source", "own fits") if own else ("sample-wide" if spot else ""))
        bl = blend_terms.get(star, {})
        for k in ("i_deg", "f", "T2_K"):
            row[f"blend_{k}"] = bl.get(k, "")
        row["blend_src"] = bl.get("source", "")
        row["ld_incl"] = LD_I
        fl = row["flags"]
        row.update(l3="none", l3_frac=math.nan, l3_k=math.nan, l3_pct=math.nan,
                   l3_pct_lo=math.nan, l3_pct_hi=math.nan)
        ks = l3k.get(star, {})
        k_fin, k_in = _num(ks.get("k_final")), _num(ks.get("k_input"))
        if "third_light_fitted" in fl:
            med = g("l3_frac")
            lo, hi = med - g("l3_frac", 2), med + g("l3_frac", 3)
            row["l3_frac"] = med
            if math.isfinite(k_fin) and abs(_num(ks.get("l3_frac_med")) - med) < 1e-6:
                row.update(l3_k=k_fin, l3_pct=100 * l3_share(med, k_fin),
                           l3_pct_lo=100 * l3_share(lo, k_fin), l3_pct_hi=100 * l3_share(hi, k_fin))
                row["l3"] = f"fitted {row['l3_pct']:.1f} % [{row['l3_pct_lo']:.1f}, {row['l3_pct_hi']:.1f}]"
            else:
                row["l3"] = f"fitted l3_frac {med:.4f} (no current k: run Audits/audit_l3_scaling.py)"
        elif "third_light_fixed" in fl:
            try:
                v = float(b.get_value("l3_frac", dataset="lc01", context="dataset"))
            except Exception:
                v = math.nan
            k = k_fin if math.isfinite(k_fin) else k_in
            row["l3_frac"] = v
            if math.isfinite(v) and math.isfinite(k):
                row.update(l3_k=k, l3_pct=100 * l3_share(v, k))
                row["l3"] = f"fixed {row['l3_pct']:.1f} %"
            else:
                row["l3"] = f"fixed l3_frac {v:.5f} (no k: run Audits/audit_l3_scaling.py)"
        row["floor_share"] = floor.get(star, {}).get("frac_floor_dominated", "")
        # derived geometry for semi-detached stars
        row.update(r2=math.nan, r2_lo=math.nan, r2_hi=math.nan, r1_lobe_med=math.nan,
                   r1_lobe_p97=math.nan)
        if morph == "semidetached":
            ch = np.asarray(b.get_value("samples", solution="best_solution"), float)
            tw = [str(t) for t in b.get_value("fitted_twigs", solution="best_solution")]
            x = ch[BURN:].reshape(-1, ch.shape[-1])
            iq = next(i for i, t in enumerate(tw) if t.startswith("q@"))
            ir = next(i for i, t in enumerate(tw) if t.startswith("requiv@primary"))
            rng = np.random.default_rng(1)
            idx = rng.choice(len(x), size=min(N_DRAWS, len(x)), replace=False)
            sma = float(b.get_value("sma@binary@component"))
            r2s, fr = [], []
            for k in idx:
                b.set_value("q@binary@component", float(x[k, iq]))
                b.set_value("requiv@primary@component", float(x[k, ir]))
                b.run_delayed_constraints()
                r2s.append(float(b.get_value("requiv@secondary@component")) / sma)
                fr.append(float(x[k, ir]) / float(b.get_value("requiv_max@primary@component")))
            p = np.percentile(r2s, [16, 50, 84])
            row.update(r2=p[1], r2_lo=p[1] - p[0], r2_hi=p[2] - p[1],
                       r1_lobe_med=float(np.median(fr)), r1_lobe_p97=float(np.mean(np.asarray(fr) > 0.97)),
                       r2_results=g("r2_frac"))
        rows.append(row)
        print(f"{star:9s} {morph[:4]} chi2r {row['chi2_red']:.2f} beta {row['beta']:.2f} "
              f"T2 {row['T2_K']:.0f} +-{row['T2_sig']:.0f} (stat) +-{row['T2_sys_T1']:.0f} (T1: "
              f"{s_t1:.0f} K x {row['dT2_dT1_bb']:.2f}; with gd {row['T2_sys_T1_lo']:+.0f}/{row['T2_sys_T1_hi']:+.0f})"
              + (f"  gd-choice {row['T2_sys_gd_lo']:+.0f}/{row['T2_sys_gd_hi']:+.0f} (x1 {row['gd_x1']:.2f}, S {row['gd_S_K'] if isinstance(row['gd_S_K'], str) else format(row['gd_S_K'], '.0f')}"
                 f"{' ' + row['gd_S_src'] if row['gd_S_src'] else ''})" if row["gd_x1"] != "" else "  gd-choice - (contact)")
              + f"  i {row['incl']:.2f} +-{row['incl_sig']:.2f} "
              f"q-sys {row['incl_sys_q'] or '-'}  spot {'own' if own else ('yes' if spot else 'no')} ({ocp:+.2f} %)  l3 {row['l3']}"
              + (f"  r2 {row['r2']:.4f} +{row['r2_hi']:.4f}/-{row['r2_lo']:.4f} (results {row['r2_results']:.4f})"
                 f"  r1/lobe {row['r1_lobe_med']:.3f} P(>0.97) {row['r1_lobe_p97']:.2f}"
                 if morph == "semidetached" else ""))
    cols = ["star", "morphology", "chi2_red", "beta", "resid_rms", "floor_share",
            "T1_K", "sigma_T1", "T2_K", "T2_sig", "dT2_dT1_bb", "T2_sys_T1",
            "T2_sys_T1_lo", "T2_sys_T1_hi", "gd_x1", "gd_S_K", "gd_S_src",
            "T2_sys_gd_lo", "T2_sys_gd_hi",
            "T2_over_T1", "T2T1_sig", "incl", "incl_sig", "incl_sys_q", "q", "q_sig",
            "fillout", "fillout_sig", "fillout_sys_q", "oconnell_pct", "spot_i_deg",
            "spot_q_pct", "spot_f", "spot_T2_K", "ld_incl", "l3", "l3_frac", "l3_k", "l3_pct",
            "l3_pct_lo", "l3_pct_hi", "r2", "r2_lo", "r2_hi",
            "r1_lobe_med", "r1_lobe_p97", "flags", "spot_src", "blend_i_deg", "blend_f",
            "blend_T2_K", "blend_src"]
    with open(OUT, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) and math.isfinite(v) else
                            ("" if isinstance(v, float) else v)) for k, v in row.items()})
    print("table:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
