"""
================================================================================
AUDIT — RESULTS LEDGER: WHAT PRODUCED EACH results_<star>.csv, AND IS IT VALID?
================================================================================
Summary:
    One row per published result (data/results/results_<star>.csv), joining
    everything known about how it was produced and what is now known to be
    wrong with its inputs (Pipeline_Log Entry 76):

      * CODE ERA, from the results file date against the dates of the fixes in
        Pipeline_Log (σ floor, convergence gate, fill-out / inclination floors,
        5D-b seeding fix). Pre-gate eras ran a fixed 2850 steps.
      * STOP, from the surviving Script 05 logs (batches 3-6 + one-offs):
        "converged at N" or "ceiling N"; older runs have no log.
      * RAILING: a posterior median within one RAW σ (CSV σ / err_scale) of a
        prior bound of its era; INTERVALS that cross a bound once scaled.
      * INPUT DEFECTS from the other audits: fold drift at P_new
        (period_fold_audit.csv), crowding / variable contaminant / PDCSAP
        distortion (crowding_pdc_audit.csv), ToM O−C scatter
        (tom_estimator_audit.csv); morphology routing from the GCVS type;
        the irradiation mismatch (contact stars: sampled with Horvat, reported
        with Wilson — 05Optimization.py:1306).
      * VERDICT: the Tier-1 reasons that invalidate the result, if any.

    Read-only: parses CSVs and logs; loads no bundle and runs no model.

Inputs:
    data/results/results_*.csv, Targets/neglected_targets*.csv, Runs/*05*.log,
    Audits/{period_fold,crowding_pdc,tom_estimator}_audit.csv.

Outputs:
    Audits/results_ledger.csv, and a table on stdout.

Usage (from Code/; light, seconds):
    python Audits/audit_results_ledger.py

Dependencies:
    numpy
================================================================================
"""

import csv
import datetime as dt
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths                                              # noqa: E402

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
LEDGER_CSV = os.path.join(AUDIT_DIR, "results_ledger.csv")

# Fix dates (Pipeline_Log entry headers). A result written before a date predates the fix.
ERA_BOUNDS = [
    (dt.datetime(2026, 8, 14, 23, 59), "E1 batch-1 retrofit (fixed 2850, i≥50, no f floor, σ floor by rescale)"),
    (dt.datetime(2026, 8, 21, 23, 59), "E2 pre-gate (fixed 2850, i≥50, no f floor)"),
    (dt.datetime(2026, 9, 3, 12, 0), "E3 gate v1, old floors (i≥50, no f floor)"),
    (dt.datetime(2026, 9, 18, 12, 0), "E4 floors, pre-5D-b seeding fix"),
    (dt.datetime(2100, 1, 1), "E5 current code (post-5D-b)"),
]
DRIFT_LIMIT_CYC = 0.02          # ~3 phase bins of 0.0067
PDC_RATIO_LIMIT = 3.0
CONTAMINANT_PTP_LIMIT = 0.05
CONTAMINANT_CROWDSAP_LIMIT = 0.9   # a neighbour must contribute > 10 % of the aperture
RAIL_SIGMAS = 2.0


def era_of(mtime: dt.datetime) -> str:
    for bound, label in ERA_BOUNDS:
        if mtime <= bound:
            return label
    return ERA_BOUNDS[-1][1]


def read_results(path: str) -> dict:
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            def num(x):
                try:
                    v = float(x)
                    return v if math.isfinite(v) else None
                except (TypeError, ValueError):
                    return None
            out[row["Symbol"]] = (num(row["Value"]), num(row["Sigma_minus"]), num(row["Sigma_plus"]))
    return out


def read_csv_by_star(name: str, key: str = "star", filt=None) -> dict:
    p = os.path.join(AUDIT_DIR, name)
    if not os.path.exists(p):
        return {}
    with open(p, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if (filt is None or filt(r))]
    return {r[key]: r for r in rows}


def catalogue_types() -> dict:
    out = {}
    for p in glob.glob(os.path.join(paths.TARGETS_DIR, "neglected_targets*.csv")):
        with open(p, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row.get("name"):
                    out[row["name"].strip()] = row.get("type", "").strip()
    return out


def stop_info() -> dict:
    """star → last 'MCMC sampling complete' message in the surviving 05 logs."""
    out = {}
    for log in sorted(glob.glob(os.path.join("Runs", "*05*.log"))):
        star = None
        with open(log, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = re.search(r"OPTIMISATION:\s+(.+?)\s+\(", line)
                if m:
                    star = m.group(1).strip()
                m = re.search(r"MCMC sampling complete \((.+?)\)", line)
                if m and star:
                    msg = m.group(1)
                    msg = re.sub(r"reached ceiling (\d+).*", r"ceiling \1 (gate NOT met)", msg)
                    out[star] = f"{msg} [{os.path.basename(log)}]"
    return out


def routing_verdict(gcvs: str, flag: str) -> str:
    g = gcvs.upper().replace(":", "")
    base, _, cls = g.partition("/")
    built_contact = "CONTACT" in (flag or "").upper() or (flag == "" and base == "EW")
    if cls in ("KE", "K"):
        return "should be CONTACT (GCVS physical class)" if not built_contact else "ok"
    if cls == "SD":
        return "should be SEMI-DETACHED (no path yet)"
    if cls == "DW":
        return "should be DETACHED (EW/DW = not in contact)" if built_contact else "ok"
    if base == "EB" and not built_contact:
        return "EB (β Lyr): detached model; check lobe railing"
    return "ok"


def eggleton(q: float) -> float:
    q13, q23 = q ** (1 / 3), q ** (2 / 3)
    return 0.49 * q23 / (0.6 * q23 + math.log(1 + q13))


def railing(res: dict, era: str, contact: bool) -> tuple[list, list]:
    """(railed params, params whose scaled interval crosses a bound)."""
    es = (res.get("err_scale") or (None,))[0] or 1.0
    t1 = (res.get("T1_K") or (None,))[0]
    old_floors = era.startswith(("E1", "E2", "E3"))
    bounds = {}
    if contact:
        bounds["incl_deg"] = (50.0 if old_floors else 40.0, 90.0)
        # The 0.70 -> 0.50 ceiling predates every result (Pipeline_Log ~line 787);
        # the 0.03 floor arrived with Entry 35.
        bounds["fillout_factor"] = (0.0, 0.50) if old_floors else (0.03, 0.50)
        bounds["T2_over_T1"] = (0.80, 1.25)
    else:
        bounds["incl_deg"] = (60.0, 90.0)
        bounds["T2_over_T1"] = (0.70, 1.10)
        bounds["q"] = (0.1, 1.0)
        bounds["r2_frac"] = (0.05, 1.0 * eggleton(0.8))     # cap at the seed q = 0.8
        bounds["r1_frac"] = (0.05, 1.0 * eggleton(1 / 0.8))
    if t1:                                                   # 8000 K regime clamp on T2
        lo, hi = bounds["T2_over_T1"]
        if t1 * lo < 8000 < t1 * hi:
            t2 = (res.get("T2_K") or (None,))[0] or t1
            bounds["T2_over_T1"] = (8000 / t1, hi) if t2 >= 8000 else (lo, 8000 / t1)
    railed, crossing = [], []
    for sym, (lo, hi) in bounds.items():
        v, sm, sp = res.get(sym, (None, None, None))
        if v is None:
            continue
        raw_m = (sm or 0) / es
        raw_p = (sp or 0) / es
        # Railed: within RAIL_SIGMAS raw σ of a bound, or within 0.1 % of the range.
        tol_lo = max(RAIL_SIGMAS * raw_m, 1e-3 * (hi - lo))
        tol_hi = max(RAIL_SIGMAS * raw_p, 1e-3 * (hi - lo))
        if v - lo <= tol_lo or hi - v <= tol_hi:
            side = "low" if v - lo <= hi - v else "high"
            railed.append(f"{sym}@{side}({v:.4g})")
        if (sm is not None and v - sm < lo) or (sp is not None and v + sp > hi):
            crossing.append(sym)
    return railed, crossing


COLUMNS = ["star", "results_date", "era", "gcvs_type", "morphology", "routing", "stop",
           "chi2_red", "err_scale", "railed", "interval_crosses_bound", "irrad_mismatch",
           "drift_cyc", "amp_loss_pct", "crowdsap", "variable_contaminant", "pdc_over_sap",
           "tom_oc_rms_s", "tier1_reasons", "priority", "decision"]
CAVEATS_CSV = os.path.join(AUDIT_DIR, "kept_results_caveats.csv")
CAVEAT_COLUMNS = ["star", "decision", "amp_loss_pct", "drift_cyc", "era", "stop", "chi2_red",
                  "err_scale", "railed", "interval_crosses_bound", "irrad_mismatch",
                  "crowdsap", "tier1_reasons"]


def keep_list_from_05() -> set:
    """KEEP_PRE_AUDIT_RESULTS as defined in 05Optimization.py (single source of truth)."""
    import ast
    src = open(os.path.join(os.path.dirname(AUDIT_DIR), "05Optimization.py"), encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", "") == "KEEP_PRE_AUDIT_RESULTS" for t in targets):
                call = node.value
                return set(ast.literal_eval(call.args[0] if isinstance(call, ast.Call) else call))
    return set()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console is cp1252
    import runstats
    _run_tracker = runstats.RunTracker("audit_results_ledger", per_star=False)

    KEEP = keep_list_from_05()
    fold = read_csv_by_star("period_fold_audit.csv")
    crowd = read_csv_by_star("crowding_pdc_audit.csv", filt=lambda r: r.get("selected") == "True")
    tom = read_csv_by_star("tom_estimator_audit.csv")
    types = catalogue_types()
    stops = stop_info()

    rows = []
    for path in sorted(glob.glob(os.path.join(paths.RESULTS_DIR, "results_*.csv"))):
        safe = os.path.basename(path)[len("results_"):-len(".csv")]
        if safe.endswith(("_bug", "_walker8bug")) or "_superseded" in safe:
            continue                                     # archived, not published
        star = safe.replace("_", " ")
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(path))
        era = era_of(mtime)
        res = read_results(path)
        flag = ""
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row["Symbol"] == "morphology_flag":
                    flag = row["Value"]
        contact = ("fillout_factor" in res) or "CONTACT" in flag.upper()
        railed, crossing = railing(res, era, contact)
        f = fold.get(star, {})
        c = crowd.get(star, {})
        drift = float(f["drift_new_cyc"]) if f.get("drift_new_cyc") else float("nan")
        try:
            contam = (float(c["crowdsap"]) < CONTAMINANT_CROWDSAP_LIMIT
                      and float(c["resid_peak_ptp"]) > CONTAMINANT_PTP_LIMIT
                      and float(c["resid_peak_period_d"]) < 2.0
                      and c.get("resid_peak_harmonic") != "True")
        except (KeyError, ValueError, TypeError):
            contam = None
        try:
            pdc = float(c["pdc_over_sap"])
        except (KeyError, ValueError, TypeError):
            pdc = float("nan")
        gcvs = types.get(star, "")
        routing = routing_verdict(gcvs, "CONTACT" if contact else "DETACHED")
        if era.startswith(("E1", "E2")):
            stop = "fixed 2850 (no gate)"
        else:
            stop = stops.get(star, "no surviving log")

        reasons = []
        if math.isfinite(drift) and drift > DRIFT_LIMIT_CYC:
            reasons.append(f"period drift {drift:.3f}")
        if contact:
            reasons.append("irrad mismatch")
        if not routing.startswith("ok") and not routing.startswith("EB (β"):
            reasons.append("routing")
        if contam:
            reasons.append("variable contaminant")
        if math.isfinite(pdc) and pdc > PDC_RATIO_LIMIT:
            reasons.append(f"PDCSAP distortion ×{pdc:.0f}")
        if era.startswith(("E1", "E2")):
            reasons.append("pre-gate fixed length")
        if railed:
            reasons.append("railed")
        priority = ("BLOCKED (data)" if contam else
                    "must re-run" if reasons else "keep")
        decision = ("KEPT, documented (Entry 78)" if star in KEEP
                    else "RE-RUN (Entry 78)")
        try:
            amp_loss = float(f["amp_loss_pct"])
        except (KeyError, ValueError, TypeError):
            amp_loss = float("nan")

        rows.append({
            "star": star, "results_date": mtime.strftime("%Y-%m-%d"), "era": era,
            "gcvs_type": gcvs, "morphology": "contact" if contact else "detached",
            "routing": routing, "stop": stop,
            "chi2_red": (res.get("chi2_red") or (None,))[0],
            "err_scale": (res.get("err_scale") or (None,))[0],
            "railed": "; ".join(railed), "interval_crosses_bound": "; ".join(crossing),
            "irrad_mismatch": "yes" if contact else "no",
            "drift_cyc": drift, "crowdsap": c.get("crowdsap", ""),
            "variable_contaminant": {True: "yes", False: "no", None: "n/a"}[contam],
            "pdc_over_sap": pdc,
            "tom_oc_rms_s": tom.get(star, {}).get("fixed_oc_rms_s", ""),
            "tier1_reasons": "; ".join(reasons), "priority": priority,
            "decision": decision, "amp_loss_pct": amp_loss,
        })

    rows.sort(key=lambda r: (-(r["drift_cyc"] if math.isfinite(r["drift_cyc"]) else 0)))
    with open(LEDGER_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([f"{r[k]:.4g}" if isinstance(r[k], float) else r[k] for k in COLUMNS])

    with open(CAVEATS_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CAVEAT_COLUMNS)
        for r in sorted(rows, key=lambda r: r["star"]):
            if r["decision"].startswith("KEPT"):
                w.writerow([f"{r[k]:.4g}" if isinstance(r[k], float) else r[k] for k in CAVEAT_COLUMNS])

    for r in rows:
        print(f"{r['decision'][:6]:<6} amp-loss {r['amp_loss_pct']:5.1f}% | "
              f"{r['star']:<10} {r['era'][:2]} {r['morphology'][:4]} chi2={r['chi2_red'] or 0:7.2f} "
              f"drift={r['drift_cyc']:.3f} | {r['priority']:<14} | {r['tier1_reasons']}"
              + (f" | railed: {r['railed']}" if r["railed"] else "")
              + (f" | crosses: {r['interval_crosses_bound']}" if r["interval_crosses_bound"] else "")
              + (f" | {r['routing']}" if not r["routing"].startswith("ok") else ""))
    from collections import Counter
    print("\n" + ", ".join(f"{k}: {v}" for k, v in Counter(r["priority"] for r in rows).items()))
    print(", ".join(f"{k[:2]}: {v}" for k, v in sorted(Counter(r["era"] for r in rows).items())))
    print(", ".join(f"{k}: {v}" for k, v in Counter(r["decision"] for r in rows).items()))
    print(f"CSV: {LEDGER_CSV}\nKept-results caveats: {CAVEATS_CSV}")
