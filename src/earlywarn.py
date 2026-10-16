"""
earlywarn.py — catch a wrongly set-up star before MCMC hours are spent on it.

Script 05 calls three checkpoints (Pipeline_Log Entry 85):

  1. preflight   (seconds, before Nelder-Mead; file checks also on resume)
     bundle geometry matches the morphology routing, upstream products are not
     newer than the bundle, the bundle period is the Script-03 period, the data
     are sorted/finite, exposure integration and irradiation are as intended.
  2. post-NM     (no extra cost; between the pre-wash and the emcee ball)
     a DETACHED star whose Nelder-Mead solution fills a Roche lobe is parked —
     the V358 Cep / OT Cep signature (Entries 75, 84).
  3. per chunk   (after every checkpoint)
     live walkers, Roche-overflow share of the chunk's evaluations, walkers
     pinned at the lobe, frozen walkers, acceptance, railing.

A PARK finding stops that star only: 05 writes data/results/parked_<star>.txt
(reason, values, suggested fix), keeps any checkpoint, and moves on to the next
star. The park is keyed to the input bundle's sha, so rebuilding the bundle
(e.g. after adding a MORPHOLOGY_OVERRIDE and re-running 04) clears it
automatically. IGNORE_PARK="Star A;Star B" runs named stars anyway (a false
positive), downgrading their PARK findings to warnings.

WARN findings never stop anything; they are printed and logged to
Runs/run_stats/05Optimization_health.csv, which Audits/run_health.py reads.

The module is pure Python/numpy apart from duck-typed PHOEBE bundle calls, so
its decision functions can be tested on saved chains without PHOEBE or MPI.
"""
from __future__ import annotations

import csv
import math
import os
import time
from typing import Optional

import numpy as np

CONTACT, DETACHED, SEMIDETACHED = "contact", "detached", "semidetached"

# ── Thresholds (calibration in Pipeline_Log Entry 85) ─────────────────────────
# Post-NM: r / requiv_max (PHOEBE's exact critical radius at the NM q). Measured
# NM end states: lobe-filling stars 0.983 (V358 Cep, batch 6), 0.984 (OT Cep,
# batch 6), 1.002 (OT Cep, Phase E; Eggleton-based); healthy detached r1 0.32-0.46.
# No known-good detached star sits near this value, so false positives are untested.
LOBE_FILL_PARK: float = 0.97
# Per chunk: median live-walker r / Eggleton lobe. Eggleton (1983) is good to ~1 %;
# OT Cep's live walkers sat at 1.002 (accepted by PHOEBE's exact check, i.e.
# Eggleton overestimates there), so 0.985 means "pressed against the lobe".
LOBE_PINNED_PARK: float = 0.985
# Per chunk: Roche-overflow failures as a fraction of the chunk's evaluations
# (steps x walkers). OT Cep averaged 0.37 over its first 400 steps (dead-walker
# proposals included); V358 Cep (semi-detached) and the trial stars had none.
OVERFLOW_PARK_FRAC: float = 0.25
# Acceptance collapse is a known, accepted condition (Entries 48, 56): warn only.
ACCEPT_WARN: float = 0.05
# A value is "railed" within this fraction of its prior range from a bound — the
# same 2 % as the emcee initialisation ball's CLIPPED test in Script 05 (5D-b).
RAIL_FRAC: float = 0.02
# Exposure integration is expected at and above this cadence (Script 04).
FTI_MIN_CADENCE_S: float = 300.0

HEALTH_COLUMNS = ["ts", "run_id", "star", "stage", "level", "code", "detail"]

# What to do when a DETACHED fit puts a star on its Roche lobe. The semi-detached
# path (Script 04) models a lobe-filling SECONDARY only.
_LOBE_FIX = {
    "secondary": ("likely semi-detached: add the star to MORPHOLOGY_OVERRIDE in morphology.py "
                  "as SEMIDETACHED, re-run Script 04 for it, relaunch"),
    "primary": ("the PRIMARY (hotter star) fills its lobe: the semi-detached path only models a "
                "lobe-filling secondary, so this needs a decision (near-contact / contact "
                "routing, or accept the detached fit with IGNORE_PARK)"),
}


class StarParked(Exception):
    """Raised inside Script 05's setup when a PARK finding is not overridden."""

    def __init__(self, stage: str, findings: list, values: Optional[dict] = None):
        self.stage    = stage
        self.findings = findings            # [(code, detail, fix), ...]
        self.values   = values or {}
        super().__init__("; ".join(f"{c}: {d}" for c, d, _ in findings))


# ── small helpers ─────────────────────────────────────────────────────────────
def eggleton(q: float) -> float:
    """Eggleton (1983) volume-equivalent Roche-lobe radius r_L/a of the star whose
    mass ratio to its companion is q (same formula as Script 05's)."""
    q = float(q)
    a = q ** (2.0 / 3.0)
    return 0.49 * a / (0.6 * a + math.log(1.0 + q ** (1.0 / 3.0)))


def ignored_stars() -> set:
    """Stars named in IGNORE_PARK (";"-separated): PARK findings become warnings."""
    raw = os.environ.get("IGNORE_PARK", "")
    return {s.strip() for s in raw.split(";") if s.strip()}


def bundle_kind(bundle) -> str:
    """Geometry the bundle was BUILT with: contact envelope → contact; secondary
    radius held by PHOEBE's semidetached constraint → semidetached; else detached."""
    try:
        if "contact_envelope" in list(bundle.components):
            return CONTACT
    except Exception:
        pass
    try:
        ps = bundle.filter(context="constraint", qualifier="requiv",
                           component="secondary", check_visible=False)
        for param in ps.to_list():
            if getattr(param, "constraint_func", "") == "semidetached":
                return SEMIDETACHED
    except Exception:
        pass
    return DETACHED


def read_ephem_period(timing_dir: str, safe: str) -> Optional[float]:
    """P_new from Script 03's ephem_<star>.txt (same format Script 04 parses)."""
    path = os.path.join(timing_dir, f"ephem_{safe}.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("P_new") and "=" in line:
                try:
                    return float(line.split("=", 1)[1].split("#")[0].strip())
                except ValueError:
                    return None
    return None


# ── 1. preflight ──────────────────────────────────────────────────────────────
def preflight_file_checks(safe: str, bundle_path: str, lightcurve_dir: str,
                          timing_dir: str) -> list:
    """Upstream products must not be newer than the bundle built from them."""
    out = []
    if not os.path.exists(bundle_path):
        return out
    b_mtime = os.path.getmtime(bundle_path)
    for label, path in (("clean light curve", os.path.join(lightcurve_dir, f"lc_{safe}_clean.fits")),
                        ("ephemeris", os.path.join(timing_dir, f"ephem_{safe}.txt"))):
        if os.path.exists(path) and os.path.getmtime(path) > b_mtime + 1.0:
            out.append(("stale_bundle",
                        f"{label} {os.path.basename(path)} is newer than the bundle "
                        f"({time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(path)))} "
                        f"> {time.strftime('%Y-%m-%d %H:%M', time.localtime(b_mtime))})",
                        "re-run Script 04 for this star (and 03 first if the ephemeris is stale)"))
    return out


def preflight_bundle_checks(bundle, kind: str, ephem_period: Optional[float],
                            check_data: bool = True, q_adopted: Optional[dict] = None,
                            q_required: bool = False) -> list:
    """Bundle geometry, period, data and compute options as Script 05 expects.

    For a contact star, `q_adopted` is morphology.read_q_adopted()'s dict: the
    bundle must carry that mass ratio (Entry 90: q is fixed in the fit, and 0.5
    for every star biased i by 5-16 degrees). With q_required, a contact star
    without an adopted q is a finding too."""
    out = []
    if kind == CONTACT and (q_adopted or q_required):
        if not q_adopted:
            out.append(("q_not_adopted", "no data/qsearch/q_adopted_<star>.txt for this contact star",
                        "run ContactQSearch.py for it, then re-run Script 04"))
        else:
            try:
                q_bundle = float(bundle.get_value("q", component="binary", context="component"))
                if abs(q_bundle - q_adopted["q_adopted"]) > 1e-6 * max(1.0, q_adopted["q_adopted"]):
                    out.append(("q_mismatch", f"bundle q = {q_bundle:.4f}, adopted q = "
                                f"{q_adopted['q_adopted']:.4f} ({q_adopted.get('q_source', '?')})",
                                "re-run Script 04 for this star (it reads the adopted q)"))
            except Exception as error:
                out.append(("q_unreadable", str(error), "inspect the bundle"))
    built = bundle_kind(bundle)
    if built != kind:
        out.append(("morphology_mismatch",
                    f"routing says {kind} but the bundle was built {built}",
                    "re-run Script 04 for this star (morphology.py changed after the build)"))
    if ephem_period is not None:
        try:
            p_bundle = float(bundle.get_value("period", component="binary", context="component"))
            if abs(p_bundle - ephem_period) > 1e-9 * ephem_period:
                out.append(("period_mismatch",
                            f"bundle P = {p_bundle:.10f} d, Script 03 P_new = {ephem_period:.10f} d",
                            "re-run Script 04 for this star (03 was re-run after the build)"))
        except Exception as error:
            out.append(("period_unreadable", str(error), "inspect the bundle"))
    if check_data:
        try:
            t = np.asarray(bundle.get_value("times", dataset="lc01", context="dataset"), float)
            f = np.asarray(bundle.get_value("fluxes", dataset="lc01", context="dataset"), float)
            s = np.asarray(bundle.get_value("sigmas", dataset="lc01", context="dataset"), float)
            if t.size < 2 or np.any(np.diff(t) <= 0):
                out.append(("times_not_sorted",
                            f"{int(np.sum(np.diff(t) <= 0))} non-increasing time steps",
                            "re-run 02 and 04 (light-curve stitching/sorting)"))
            if not (np.all(np.isfinite(f)) and np.all(np.isfinite(s))):
                out.append(("nonfinite_data", "NaN/inf in fluxes or sigmas", "re-run 02 and 04"))
            if np.any(s <= 0):
                out.append(("bad_sigmas", f"{int(np.sum(s <= 0))} sigmas <= 0", "re-run 02 and 04"))
        except Exception as error:
            out.append(("data_unreadable", str(error), "inspect the bundle"))
    try:
        exptime = float(bundle.get_value("exptime", dataset="lc01", context="dataset",
                                         check_visible=False))
        fti = bundle.get_value("fti_method", compute="phoebe01", dataset="lc01",
                               check_visible=False)
        if exptime >= FTI_MIN_CADENCE_S and fti != "oversample":
            out.append(("fti_off", f"exptime {exptime:.0f} s but fti_method = {fti}",
                        "bug in 04/05 dataset handling; do not fit until fixed"))
    except Exception:
        pass
    try:
        irrad = bundle.get_value("irrad_method", compute="phoebe01", check_visible=False)
        want = "wilson" if kind == CONTACT else "horvat"
        if irrad != want:
            out.append(("irrad_mismatch", f"irrad_method = {irrad}, expected {want}",
                        "bug in 05 setup (5A); do not fit until fixed"))
    except Exception:
        pass
    return out


# ── 2. post-NM ────────────────────────────────────────────────────────────────
def post_nm_assess(bundle, bounds: dict, kind: str) -> tuple:
    """(parks, warns, values) for the adopted Nelder-Mead solution.
    `bounds` maps each fitted twig to its (lo, hi) prior."""
    parks, warns, values = [], [], {}
    for twig, (lo, hi) in bounds.items():
        try:
            v = float(bundle.get_value(twig, context=["component", "dataset"]))
        except Exception:
            continue
        values[twig] = v
        span = hi - lo
        if span > 0 and (v - lo < RAIL_FRAC * span or hi - v < RAIL_FRAC * span):
            side = "lower" if v - lo < RAIL_FRAC * span else "upper"
            warns.append(("nm_railed", f"{twig} = {v:.5g} at its {side} prior bound [{lo:.5g}, {hi:.5g}]"))
    if kind == DETACHED:
        for comp in ("primary", "secondary"):
            try:
                r     = float(bundle.get_value("requiv", component=comp, context="component"))
                r_max = float(bundle.get_value("requiv_max", component=comp, context="component"))
            except Exception:
                continue
            ratio = r / r_max if r_max > 0 else float("nan")
            values[f"requiv/requiv_max@{comp}"] = ratio
            if ratio >= LOBE_FILL_PARK:
                parks.append(("nm_lobe_filling",
                              f"{comp} fills {ratio:.3f} of its Roche lobe after Nelder-Mead "
                              f"(park at >= {LOBE_FILL_PARK})", _LOBE_FIX[comp]))
    elif kind == SEMIDETACHED:
        # The secondary fills its lobe by construction. A primary near its own lobe
        # is a legitimate near-contact solution — RZ Dra's literature solution has
        # 0.98 (Entry 87) — so it warns and never parks.
        try:
            r     = float(bundle.get_value("requiv", component="primary", context="component"))
            r_max = float(bundle.get_value("requiv_max", component="primary", context="component"))
            ratio = r / r_max if r_max > 0 else float("nan")
            values["requiv/requiv_max@primary"] = ratio
            if ratio >= LOBE_FILL_PARK:
                warns.append(("primary_near_lobe",
                              f"primary at {ratio:.3f} of its Roche lobe (near-contact; both "
                              "stars at their lobes if ~1.0)"))
        except Exception:
            pass
    return parks, warns, values


# ── 3. per chunk ──────────────────────────────────────────────────────────────
def failed_by_reason(failed_samples) -> dict:
    """{message: count} from PHOEBE's cumulative failed_samples dict."""
    return {str(msg): len(lst) for msg, lst in (failed_samples or {}).items()}


def chunk_assess(samples_sf: np.ndarray, lnp_sf: np.ndarray, twigs: list,
                 bounds: dict, kind: str, sma: float, prev_done: int, done: int,
                 failed_now: dict, failed_before: dict, min_live: int) -> tuple:
    """(parks, warns, summary) for the chunk [prev_done, done).
    samples_sf: (n_steps, n_walkers, n_params); lnp_sf: (n_steps, n_walkers)."""
    parks, warns = [], []
    n_steps, n_walk = samples_sf.shape[0], samples_sf.shape[1]
    lo_i  = max(0, min(prev_done, n_steps - 1))
    live  = [w for w in range(n_walk) if np.isfinite(lnp_sf[-1, w])]
    seg   = samples_sf[lo_i:done]
    moved = np.any(np.any(np.diff(seg, axis=0) != 0, axis=2), axis=0) if seg.shape[0] > 1 \
        else np.ones(n_walk, bool)
    frozen = [w for w in live if not moved[w]]
    acc = float(np.mean(np.any(np.diff(seg, axis=0) != 0, axis=2))) if seg.shape[0] > 1 else float("nan")
    evaluations = max(1, (done - prev_done) * n_walk)
    overflow = sum(c - failed_before.get(m, 0) for m, c in failed_now.items() if "overflow" in m.lower())
    failed   = sum(c - failed_before.get(m, 0) for m, c in failed_now.items())
    of_frac, fail_frac = overflow / evaluations, failed / evaluations
    summary = {"live": len(live), "walkers": n_walk, "frozen": len(frozen),
               "accept": acc, "overflow_frac": of_frac, "failed_frac": fail_frac}

    if len(live) < min_live:
        parks.append(("dead_walkers", f"only {len(live)}/{n_walk} walkers at a valid model "
                      f"(need >= {min_live})",
                      "inspect the init ball / NM solution; the chain cannot converge"))
    elif len(live) < n_walk:
        warns.append(("dead_walkers", f"{n_walk - len(live)} walker(s) at lnprob = -inf"))
    if of_frac >= OVERFLOW_PARK_FRAC:
        if kind == DETACHED:
            parks.append(("roche_overflow", f"{of_frac:.2f} of this chunk's evaluations failed "
                          f"with Roche overflow (park at >= {OVERFLOW_PARK_FRAC})",
                          "the fit wants a star larger than its lobe: likely semi-detached — add "
                          "to MORPHOLOGY_OVERRIDE, re-run 04"))
        else:
            # Semi-detached: a primary sitting near its lobe (RZ Dra, V358 Cep) rejects
            # the proposals beyond it — a hard physical wall, not a wrong model.
            warns.append(("roche_overflow", f"{of_frac:.2f} of this chunk's evaluations failed "
                          "with Roche overflow (a star is sampling against its lobe)"))
    if frozen:
        warns.append(("frozen_walkers", f"walkers {frozen} made no move in this chunk"))
    if acc == acc and acc < ACCEPT_WARN:
        warns.append(("low_acceptance", f"acceptance {acc:.3f} < {ACCEPT_WARN}"))

    if live:
        last = samples_sf[-1, live, :]
        med  = {t: float(np.median(last[:, i])) for i, t in enumerate(twigs)}
        for t, (lo, hi) in bounds.items():
            # bounds use 05's short twigs ("q@binary"), fitted twigs are full
            # ("q@binary@orbit@component").
            key = next((tw for tw in twigs if tw == t or tw.startswith(t + "@")), None)
            if key is None or not (hi > lo):
                continue
            v = med[key]
            if v - lo < RAIL_FRAC * (hi - lo) or hi - v < RAIL_FRAC * (hi - lo):
                warns.append(("railed", f"{t} median {v:.5g} at a prior bound [{lo:.5g}, {hi:.5g}]"))
        if kind == DETACHED and sma > 0:
            qkey = next((tw for tw in twigs if tw.startswith("q@")), None)
            for comp in ("primary", "secondary"):
                rkey = next((tw for tw in twigs if tw.startswith("requiv@" + comp)), None)
                if rkey is None or qkey is None:
                    continue
                qi, ri = twigs.index(qkey), twigs.index(rkey)
                ratios = [last[k, ri] / sma / eggleton(last[k, qi] if comp == "secondary"
                                                       else 1.0 / last[k, qi])
                          for k in range(last.shape[0])]
                med_ratio = float(np.median(ratios))
                summary[f"lobe_{comp}"] = med_ratio
                if med_ratio >= LOBE_PINNED_PARK:
                    parks.append(("lobe_pinned", f"{comp}: median live walker at {med_ratio:.3f} of "
                                  f"its Eggleton lobe (park at >= {LOBE_PINNED_PARK})",
                                  _LOBE_FIX[comp]))
    return parks, warns, summary


def summary_line(summary: dict, parks: list, warns: list) -> str:
    """One-line [health] report for the run log."""
    parts = [f"live {summary['live']}/{summary['walkers']}",
             f"overflow {summary['overflow_frac']:.2f}",
             f"failed {summary['failed_frac']:.2f}",
             f"frozen {summary['frozen']}"]
    if summary.get("accept") == summary.get("accept"):
        parts.append(f"accept {summary['accept']:.2f}")
    for comp in ("primary", "secondary"):
        if f"lobe_{comp}" in summary:
            parts.append(f"r/lobe {comp[0]} {summary[f'lobe_{comp}']:.3f}")
    state = "PARK" if parks else ("WARN" if warns else "OK")
    return "  [health] " + " · ".join(parts) + f" · {state}"


# ── parking ───────────────────────────────────────────────────────────────────
def parked_path(results_dir: str, safe: str) -> str:
    return os.path.join(results_dir, f"parked_{safe}.txt")


def read_parked(path: str) -> dict:
    """{'input_bundle_sha': ..., 'stage': ..., 'reason': ...} from a parked file."""
    meta = {}
    if not os.path.exists(path):
        return meta
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if ":" in line and not line.startswith(" "):
                key, value = line.split(":", 1)
                meta.setdefault(key.strip(), value.strip())
    return meta


def write_parked(path: str, star: str, run_id: str, input_sha: str, stage: str,
                 findings: list, values: Optional[dict] = None,
                 checkpoint: Optional[str] = None) -> None:
    lines = [f"star: {star}",
             f"parked: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"run_id: {run_id}",
             f"input_bundle_sha: {input_sha}",
             f"stage: {stage}",
             f"reason: {'; '.join(code for code, _, _ in findings)}",
             "", "findings:"]
    for code, detail, fix in findings:
        lines += [f"  - {code}: {detail}", f"    fix: {fix}"]
    if values:
        lines += ["", "values at park time:"]
        lines += [f"  {k} = {v:.6g}" if isinstance(v, float) else f"  {k} = {v}"
                  for k, v in values.items()]
    if checkpoint:
        lines += ["", f"checkpoint kept: {checkpoint} (resumable if this is a false positive)"]
    lines += ["", "to un-park:",
              "  * fix the cause and rebuild the bundle (Script 04): the new bundle sha clears",
              "    this park automatically on the next 05 run; or",
              f"  * if this is a false positive: IGNORE_PARK=\"{star}\" (forward with mpirun -x),",
              "    or delete this file."]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


def resolve_parked(path: str) -> str:
    """Rename a park whose bundle has since been rebuilt; returns the new path."""
    new = path.replace(".txt", f"_resolved_{time.strftime('%Y%m%d')}.txt")
    os.replace(path, new)
    return new


def health_event(csv_path: str, run_id: str, star: str, stage: str, level: str,
                 code: str, detail: str) -> None:
    """Append one row to the health-event CSV. Never raises."""
    try:
        new = not os.path.exists(csv_path)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEALTH_COLUMNS)
            if new:
                writer.writeheader()
            writer.writerow({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "run_id": run_id,
                             "star": star, "stage": stage, "level": level, "code": code,
                             "detail": detail})
    except Exception as error:
        print(f"  [health] event not written: {error}", flush=True)
