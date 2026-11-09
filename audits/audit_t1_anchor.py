"""
================================================================================
AUDIT — HOW WELL IS EACH STAR'S FIXED PRIMARY TEMPERATURE T1 ANCHORED?
================================================================================
Summary:
    Script 05 holds T1 fixed at the value Script 04 put in the bundle, mostly Gaia
    DR3 teff_msc1 (04Modeling.py GAIA_TEFF_OVERRIDE_K). The light curve measures
    T2/T1, so every T2 inherits T1's error (Pipeline_Log Entry 110). This audit
    puts the independent estimates beside the pipeline value, one row per star:

      * pipeline T1 and its source (parsed from the GAIA_TEFF_OVERRIDE_K lines);
      * Gaia DR3 astrophysical parameters (VizieR I/355/paramp, by the source_id
        of the brightest Gaia DR3 source within 5" of the TIC position, the route
        04 documents): MSC Teff1/Teff2 with their lower/upper bounds, GSP-Phot,
        GSP-Spec (RVS spectra), ESP-HS (hot stars);
      * SIMBAD spectral type and its bibcode (TAP), converted to a dwarf Teff
        with Pecaut & Mamajek (2013) as tabulated by E. Mamajek (read online at
        run time, not stored). Am stars use the hydrogen-line type; giants and
        unreferenced types are flagged;
      * LAMOST DR11 LRS/MRS Teff where the star is covered (VizieR V/162);
      * literature T1 values already read for this project (LITERATURE_T1).

    Per star it reports sigma_T1 = max(250 K, half the MSC interval, half the
    spread of the credible estimates) and whether the change rule fires:
    at least two estimates, one of them not from Gaia, agree within 300 K and
    differ from the pipeline T1 by more than 500 K. GSP-Phot and MSC come from the
    same BP/RP spectra and count as one Gaia estimate.

    Read-only on the pipeline; network reads only (VizieR, SIMBAD, the Mamajek
    table); nothing is downloaded to disk.

Usage (WSL or Windows python with astroquery; from Code/; a few minutes):
    python3 Audits/audit_t1_anchor.py [Targets/phaseF_queue.csv]
Output:
    Audits/t1_anchor.csv; a table on stdout.
================================================================================
"""
import csv
import json
import math
import os
import re
import sys
import urllib.parse
import urllib.request
import warnings

warnings.filterwarnings("ignore")
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(CODE, "Audits", "t1_anchor.csv")
POS_JSON = os.path.join(CODE, "Runs", "modeltype", "catalog_lookup_60.json")
MAMAJEK_URL = "https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt"
SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"

# Literature primary temperatures read for this project:
# (T1, note with check level, bibcode of the source, depends on Gaia/LAMOST?).
# A source that is also SIMBAD's spectral-type reference is not counted twice.
LITERATURE_T1 = {
    "GW Gem":   (7700, "Lee et al. 2009 (PASP 121, 104): fixed from B-V 0.24 [R full text]",
                 "2009PASP..121..104L", False),
    "TY UMa":   (6250, "Li et al. 2015 (AJ 149, 120): F8 V assumed [R full text]",
                 "2015AJ....149..120L", False),
    "V576 Lyr": (7514, "Guo et al. 2025 (AJ 170, 115): mean of Gaia/TESS/LAMOST, corrected [R MRT]",
                 "", True),
    "EG Cep":   (7760, "Rucinski et al. 2008 (DDO XIII): A7V spectral type [R full text]",
                 "2008AJ....136..586R", False),
    "RZ Dra":   (7910, "Erdem et al. 2011 (NewA 16, 6): A6 primary -> A6 V [R abstract]",
                 "2011NewA...16....6E", False),
    "V404 Lyr": (7330, "Lee et al. 2019: Teff,1 7330 +- 150 K from the disentangled primary spectrum [R full text]",
                 "", False),
}
CHANGE_AGREE_K = 300.0
CHANGE_DIFF_K = 500.0
SIGMA_FLOOR_K = 250.0


sys.path.insert(0, CODE)
from morphology import resolve_morphology_kind        # noqa: E402


def file_sha(path: str) -> str:
    """sha256 of a file, first 12 hex digits (as 05Optimization.file_sha)."""
    import hashlib
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()[:12]


def pipeline_t1() -> dict:
    """{star: (T1, comment)} parsed from 04Modeling.py's GAIA_TEFF_OVERRIDE_K."""
    out, inside = {}, False
    for line in open(os.path.join(CODE, "04Modeling.py"), encoding="utf-8"):
        if line.startswith("GAIA_TEFF_OVERRIDE_K"):
            inside = True
            continue
        if inside and line.startswith("}"):
            break
        m = re.match(r'\s*"([^"]+)":\s*([0-9.]+),\s*#?\s*(.*)', line) if inside else None
        if m:
            out[m.group(1)] = (float(m.group(2)), m.group(3).strip())
    return out


def mamajek_table() -> dict:
    """{'A6V': 7910, ...} from the online Mamajek dwarf table (SpT and Teff columns)."""
    text = urllib.request.urlopen(MAMAJEK_URL, timeout=60).read().decode("latin-1")
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) > 2 and re.fullmatch(r"[OBAFGKMLTY][0-9.]+V", parts[0]):
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
        if out and line.startswith("#SpT") and len(out) > 50:
            break
    return out


def sptype_teff(sp: str, table: dict) -> tuple:
    """(Teff or nan, parsed type, note). Am stars use the hydrogen type; 'A3/4' -> mean."""
    if not sp:
        return math.nan, "", "no type"
    note = []
    s = sp.strip()
    m = re.search(r"h([OBAFGKM][0-9.]+)", s)            # Am notation kA4hF0mF0
    if m and s.startswith("k"):
        core, note = m.group(1), ["Am: hydrogen type used"]
    else:
        m = re.match(r"([OBAFGKM])([0-9.]*)(?:/([OBAFGKM]?)([0-9.]+))?\s*([IV]*)", s)
        if not m or not m.group(2):
            return math.nan, s, "no subtype"
        lum = m.group(5)
        if lum and lum not in ("V", "IV-V", "IV"):
            note.append(f"luminosity class {lum}: dwarf Teff used")
        elif lum == "IV":
            note.append("subgiant: dwarf Teff used")
        if m.group(4):                                     # A3/4 or A9/F0
            l2 = m.group(3) or m.group(1)
            t1 = table.get(f"{m.group(1)}{m.group(2)}V")
            t2 = table.get(f"{l2}{m.group(4)}V")
            if t1 and t2:
                return 0.5 * (t1 + t2), s, "; ".join(note + ["mean of the two types"])
        core = f"{m.group(1)}{m.group(2)}"
    if ":" in s:
        note.append("uncertain (:)")
    t = table.get(core + "V")
    if t is None and "." not in core:
        t = table.get(core + ".0V")
    return (t if t else math.nan), s, "; ".join(note) or ""


def simbad_types(names: list) -> dict:
    ids = ",".join("'V* %s'" % n for n in names)
    q = ("SELECT i.id, b.sp_type, b.sp_qual, b.sp_bibcode FROM ident AS i "
         f"JOIN basic AS b ON b.oid = i.oidref WHERE i.id IN ({ids})")
    url = SIMBAD_TAP + "?" + urllib.parse.urlencode(
        {"request": "doQuery", "lang": "adql", "format": "csv", "query": q})
    rows = list(csv.DictReader(urllib.request.urlopen(url, timeout=90).read().decode().splitlines()))
    return {r["id"].replace("V* ", "").strip(): r for r in rows}


def val(t, col):
    try:
        v = float(t[col][0])
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def main() -> int:
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    cat = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CODE, "Targets", "phaseF_queue.csv")
    cat_rows = list(csv.DictReader(open(cat, encoding="utf-8")))
    stars = [r["name"] for r in cat_rows]
    types = {r["name"]: r.get("type", "") for r in cat_rows}
    pos = json.load(open(POS_JSON, encoding="utf-8"))
    pipe = pipeline_t1()
    pm13 = mamajek_table()
    print(f"Mamajek table: {len(pm13)} dwarf types read")
    simb = simbad_types(stars)
    v = Vizier(columns=["**", "+_r"], row_limit=50)
    rows = []
    for k, star in enumerate(stars, 1):
        row = {"queue_pos": k, "star": star}
        t1, src = pipe.get(star, (math.nan, "not in GAIA_TEFF_OVERRIDE_K"))
        row.update(T1_pipeline=t1, T1_source=src)
        safe = star.replace(" ", "_")
        res = os.path.join(CODE, "data", "results", f"results_{safe}.csv")
        # Done in this campaign = a results file whose input_bundle_sha is the sha of
        # the current bundle (05's own staleness test); an older result is "old".
        row["status"], row["T2_over_T1_fit"], res_sha = "pending", math.nan, ""
        if os.path.exists(res):
            for r in csv.DictReader(open(res, encoding="utf-8", errors="replace")):
                if r["Symbol"] == "T2_over_T1":
                    row["T2_over_T1_fit"] = float(r["Value"])
                if r["Symbol"] == "input_bundle_sha":
                    res_sha = r["Value"]
            cur = file_sha(os.path.join(CODE, "data", "bundles", f"bundle_{safe}.phoebe"))
            row["status"] = "done" if res_sha and res_sha == cur else "pending (old result)"
        row["routing"] = resolve_morphology_kind(star, types.get(star, ""))
        # Gaia DR3 source: the TIC's own Gaia id (TIC 8.2 lists the DR2 id, which is
        # the DR3 id for almost every star), so proper motion cannot pick a neighbour
        # (VW Cep). Fallback: the brightest source within 5" of the TIC position.
        p = pos.get(star, {})
        c = SkyCoord(p["ra"], p["dec"], unit="deg") if p else None
        ap = None
        try:
            tic = v.query_constraints(catalog="IV/39/tic82", TIC=str(p.get("tic", "")))[0]
            gid = str(tic["GAIA"][0])
            g = v.query_constraints(catalog="I/355/gaiadr3", Source=gid)
            if len(g) and len(g[0]):
                row.update(gaia_source=gid, Gmag=float(g[0]["Gmag"][0]), gaia_match="TIC Gaia id")
            else:
                raise LookupError("TIC Gaia id not in DR3")
        except Exception as e:
            try:
                g = v.query_region(c, radius=5 * u.arcsec, catalog="I/355/gaiadr3")[0]
                j = int(min(range(len(g)), key=lambda i: float(g["Gmag"][i])))
                row.update(gaia_source=str(g["Source"][j]), Gmag=float(g["Gmag"][j]),
                           gaia_match=f"cone 5\" (brightest of {len(g)}; {str(e)[:30]})")
            except Exception as e2:
                row["gaia_match"] = "none: " + str(e2)[:60]
        if row.get("gaia_source"):
            try:
                ap = v.query_constraints(catalog="I/355/paramp", Source=row["gaia_source"])
                ap = ap[0] if len(ap) else None
            except Exception:
                ap = None
        for out_col, col in (("msc_T1", "Teff1-MSC"), ("msc_T1_lo", "b_Teff1-MSC"),
                             ("msc_T1_hi", "B_Teff1-MSC"), ("msc_T2", "Teff2-MSC"),
                             ("msc_T2_lo", "b_Teff2-MSC"), ("msc_T2_hi", "B_Teff2-MSC"),
                             ("msc_logg1", "logg1-MSC"), ("msc_flag", "Flag-MSC"),
                             ("gspphot", "Teff"), ("gspphot_lo", "b_Teff"), ("gspphot_hi", "B_Teff"),
                             ("gspspec", "Teff-S"), ("esphs", "Teff-HS")):
            row[out_col] = val(ap, col) if ap is not None else math.nan
        # SIMBAD spectral type
        s = simb.get(star, {})
        row["sp_type"], row["sp_bibcode"] = s.get("sp_type", ""), s.get("sp_bibcode", "")
        row["sp_qual"] = s.get("sp_qual", "")
        tsp, parsed, note = sptype_teff(row["sp_type"], pm13)
        row["sp_teff"], row["sp_note"] = tsp, note
        # LAMOST DR11 stellar parameters (VizieR V/162): LRS (dr11sl, Teff) and MRS
        # (dr11sm, Tefflasp), every epoch within 3". A spectrum at a random phase
        # is the combined light, so the median over epochs is used and the spread kept.
        row["lamost_teff"], row["lamost_n"], row["lamost_range"] = math.nan, 0, ""
        try:
            vals = []
            lt = v.query_region(c, radius=3 * u.arcsec, catalog=["V/162/dr11sl", "V/162/dr11sm"])
            for key in lt.keys():
                tab = lt[key]
                col = "Teff" if "Teff" in tab.colnames else ("Tefflasp" if "Tefflasp" in tab.colnames else None)
                if col:
                    vals += [float(x) for x in tab[col] if x is not None and str(x) not in ("--", "")
                             and math.isfinite(float(x))]
            if vals:
                vals.sort()
                row["lamost_teff"] = vals[len(vals) // 2] if len(vals) % 2 else 0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2])
                row["lamost_n"], row["lamost_range"] = len(vals), f"{vals[0]:.0f}-{vals[-1]:.0f}"
        except Exception as e:
            row["lamost_range"] = "query error: " + str(e)[:40]
        lit = LITERATURE_T1.get(star)
        row["lit_T1"], row["lit_note"] = (lit[0], lit[1]) if lit else (math.nan, "")
        # estimates: (label, value, is_gaia). MSC and GSP-Phot both come from the
        # BP/RP spectra: a pair of them never counts as independent agreement.
        est = []
        for lab in ("msc_T1", "gspphot", "gspspec", "esphs"):
            if math.isfinite(row[lab]) and not (lab == "msc_T1" and "ceiling" in row["T1_source"]):
                est.append((lab.replace("_T1", ""), row[lab], True))
        if math.isfinite(tsp) and row["sp_bibcode"]:
            est.append(("sptype", tsp, False))
        if math.isfinite(row["lamost_teff"]):
            est.append(("lamost", row["lamost_teff"], False))
        if lit and not (lit[2] and lit[2] == row["sp_bibcode"]):
            est.append(("lit", lit[0], lit[3]))           # lit[3]: depends on Gaia
        flags = []
        if math.isfinite(row["msc_T1_hi"]) and row["msc_T1_hi"] >= 7999:
            flags.append("msc_upper_bound_8000")
        if math.isfinite(row["msc_T1"]) and 7190 <= row["msc_T1"] <= 7420:
            flags.append("msc_cluster_7200_7420")
        if row["sp_type"] and not row["sp_bibcode"]:
            flags.append("sptype_unreferenced")
        if math.isfinite(tsp) and math.isfinite(t1) and abs(tsp - t1) > CHANGE_DIFF_K:
            flags.append(f"sptype_differs_{tsp - t1:+.0f}K")
        if (row["routing"] != "contact" and math.isfinite(row["T2_over_T1_fit"])
                and math.isfinite(row["msc_T2"]) and math.isfinite(row["msc_T1"])):
            r_msc = row["msc_T2"] / row["msc_T1"]
            if abs(r_msc - row["T2_over_T1_fit"]) > 0.15:
                flags.append(f"msc_split_vs_lightcurve_{r_msc:.2f}_vs_{row['T2_over_T1_fit']:.2f}")
        vals = [e[1] for e in est] + ([t1] if math.isfinite(t1) else [])
        spread = 0.5 * (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        half_msc = (0.5 * (row["msc_T1_hi"] - row["msc_T1_lo"])
                    if math.isfinite(row["msc_T1_hi"]) and math.isfinite(row["msc_T1_lo"]) else 0.0)
        row["sigma_T1"] = round(max(SIGMA_FLOOR_K, spread, half_msc))
        # change rule
        fire = ""
        for a in range(len(est)):
            for b in range(a + 1, len(est)):
                ea, eb = est[a], est[b]
                if ea[2] and eb[2]:
                    continue                                   # two Gaia values: one estimate
                if abs(ea[1] - eb[1]) <= CHANGE_AGREE_K and math.isfinite(t1):
                    mean = 0.5 * (ea[1] + eb[1])
                    if abs(mean - t1) > CHANGE_DIFF_K:
                        fire = f"{ea[0]}+{eb[0]} agree at {mean:.0f} K ({mean - t1:+.0f} K)"
        row["change_rule"] = fire
        row["estimates"] = "; ".join(f"{e[0]} {e[1]:.0f}" for e in est)
        row["flags"] = "; ".join(flags)
        rows.append(row)
        print(f"{k:2d} {star:10s} {row['status']:8s} T1 {t1:6.0f}  MSC {row['msc_T1']:6.0f} "
              f"[{row['msc_T1_lo']:5.0f},{row['msc_T1_hi']:5.0f}]  T2msc {row['msc_T2']:5.0f}  "
              f"GSP {row['gspphot']:5.0f}  SpT {row['sp_type'][:12]:12s} {tsp:5.0f}  "
              f"LAM {row['lamost_teff']:5.0f}  sig {row['sigma_T1']:4d}  {fire}  {row['flags']}")
    cols = ["queue_pos", "star", "status", "routing", "T1_pipeline", "T1_source", "msc_T1", "msc_T1_lo",
            "msc_T1_hi", "msc_T2", "msc_T2_lo", "msc_T2_hi", "msc_logg1", "msc_flag", "gspphot",
            "gspphot_lo", "gspphot_hi", "gspspec", "esphs", "sp_type", "sp_qual", "sp_bibcode",
            "sp_teff", "sp_note", "lamost_teff", "lamost_n", "lamost_range", "lit_T1", "lit_note", "T2_over_T1_fit",
            "gaia_source", "gaia_match", "Gmag", "estimates", "sigma_T1",
            "change_rule", "flags"]
    with open(OUT, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(x, 1) if isinstance(x, float) and math.isfinite(x) else
                            ("" if isinstance(x, float) else x)) for k, x in r.items()})
    print("table:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
