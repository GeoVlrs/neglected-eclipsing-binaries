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
      * literature T1 values already read for this project (LITERATURE_T1);
      * colour temperatures from Audits/colour_teff.csv (audit_colour_teff.py), when
        that table exists: 2MASS J-Ks (non-Gaia) and Gaia BP-RP dereddened with the
        Bayestar19 map (Gaia; only where the map is reliable at the star's distance).

    Per star it reports sigma_T1 = max(250 K, half the MSC interval, half the
    spread of the credible estimates) and whether the change rule fires:
    at least two estimates, one of them not from Gaia, agree within 300 K and
    differ from the pipeline T1 by more than 500 K. GSP-Phot, MSC and the BP-RP colour
    come from the same BP/RP photometry and count as one Gaia estimate.

    Read-only on the pipeline; network reads only (VizieR, SIMBAD, the Mamajek
    table); nothing is downloaded to disk.

Usage (WSL or Windows python with astroquery; from Code/; a few minutes):
    python3 Audits/audit_t1_anchor.py [Targets/phaseF_queue.csv]
    python3 Audits/audit_t1_anchor.py --offline     # no network: rebuild from the existing
        table (catalogue values kept; T1, status, literature and colours refreshed)
    --no-colour ignores Audits/colour_teff.csv
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
COLOUR_CSV = os.path.join(CODE, "Audits", "colour_teff.csv")    # audit_colour_teff.py (optional)
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
from morphology import resolve_morphology_kind, read_third_light        # noqa: E402

# Stars fitted with third light in production (mode free or fixed), as 05 reads them.
THIRD_LIGHT = read_third_light(os.path.join(CODE, "Targets"))
ESPHS_MIN_K = 7500.0      # ESP-HS's domain: stars hotter than 7500 K (Creevey et al. 2023)


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


def colour_table() -> dict:
    """{star: row} of Audits/colour_teff.csv (audit_colour_teff.py), or {} if absent."""
    try:
        return {r["star"]: r for r in csv.DictReader(open(COLOUR_CSV, encoding="utf-8"))}
    except OSError:
        return {}


def fnum(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def result_status(star: str) -> tuple:
    """(status, fitted T2/T1). Done in this campaign = a results file whose
    input_bundle_sha is the sha of the current bundle (05's own staleness test);
    an older result is "pending (old result)"."""
    safe = star.replace(" ", "_")
    res = os.path.join(CODE, "data", "results", f"results_{safe}.csv")
    status, ratio, res_sha = "pending", math.nan, ""
    if os.path.exists(res):
        for r in csv.DictReader(open(res, encoding="utf-8", errors="replace")):
            if r["Symbol"] == "T2_over_T1":
                ratio = float(r["Value"])
            if r["Symbol"] == "input_bundle_sha":
                res_sha = r["Value"]
        cur = file_sha(os.path.join(CODE, "data", "bundles", f"bundle_{safe}.phoebe"))
        status = "done" if res_sha and res_sha == cur else "pending (old result)"
    return status, ratio


def assess(row: dict, colour: dict) -> None:
    """Fill estimates, flags, sigma_T1 and change_rule of one star's row (in place).

    Estimates are (label, Teff, is_gaia). MSC, GSP-Phot and the Gaia BP-RP colour all
    come from Gaia's BP/RP photometry: two of them never count as independent agreement.
    The 2MASS J-Ks colour temperature (audit_colour_teff.py) counts as non-Gaia when that
    audit marks it usable (good photometry; dust and colour error each <= 250 K in T);
    the BP-RP one where the dust map is reliable at the star's distance, or the star is
    so near that the reddening bracket moves T by <= 250 K.
    Colours count for contact stars without third light in production only, and the
    ESP-HS value only for a star that another estimate also puts at >= 7500 K (see the
    comments below).
    """
    t1, tsp = row["T1_pipeline"], row["sp_teff"]
    lit = LITERATURE_T1.get(row["star"])
    est = []
    for lab in ("msc_T1", "gspphot", "gspspec"):
        if math.isfinite(row[lab]) and not (lab == "msc_T1" and "ceiling" in row["T1_source"]):
            est.append((lab.replace("_T1", ""), row[lab], True))
    if math.isfinite(tsp) and row["sp_bibcode"]:
        est.append(("sptype", tsp, False))
    if math.isfinite(row["lamost_teff"]):
        est.append(("lamost", row["lamost_teff"], False))
    if lit and not (lit[2] and lit[2] == row["sp_bibcode"]):
        est.append(("lit", lit[0], lit[3]))               # lit[3]: depends on Gaia
    flags = []
    # ESP-HS analyses only stars it classifies as hotter than 7500 K (O, B, A), and some
    # cooler stars were misclassified and processed anyway (Creevey et al. 2023, Sect. 3).
    # Its value counts only where the star is hot by other evidence: another estimate at
    # >= 7500 K, or a colour temperature with NO dust correction at >= 7500 K (reddening
    # only cools a colour, so that is a lower limit). II UMa: ESP-HS 14310 K against MSC
    # 6245 K and uncorrected colours of ~6600 K: not counted. SU Cep: uncorrected J-Ks
    # gives an A star: counted.
    if math.isfinite(row["esphs"]):
        c0 = colour.get(row["star"], {})
        floor = max([fnum(c0.get("T_bprp_unred")), fnum(c0.get("T_jk_unred"))],
                    key=lambda v: v if math.isfinite(v) else -1.0)
        if any(e[1] >= ESPHS_MIN_K for e in est) or (math.isfinite(floor) and floor >= ESPHS_MIN_K):
            est.insert(min(len(est), 3), ("esphs", row["esphs"], True))
        else:
            flags.append(f"esphs_not_counted_{row['esphs']:.0f}K_no_other_hot_evidence")
    c = colour.get(row["star"])
    row["colour_jk_T"] = row["colour_bprp_T"] = math.nan
    row["colour_bprp_range"] = row["map_status"] = ""
    if c:
        row["map_status"] = c.get("map_status", "")
        tjk, tbp = fnum(c.get("T_jk")), fnum(c.get("T_bprp_est"))
        lo, hi = fnum(c.get("T_bprp_16")), fnum(c.get("T_bprp_84"))
        # A colour temperature is that of the combined light. In a contact binary the
        # two stars have similar temperatures, so it estimates T1; with a much cooler
        # secondary (semi-detached, detached) it is pulled towards T2, the infrared
        # (J-Ks) most (GW Gem: J-Ks 6425 K against T1 7204 K at T2/T1 0.67; V404 Lyr,
        # Entry 131). Those stars' colours are shown but not counted. The same holds for a
        # contact star with third light in production (Targets/third_light.csv, mode free
        # or fixed): its colours include the third star's light (TZ Boo: a second binary
        # gives up to 28 % of the light, Pribulla et al. 2009).
        third = row["star"] in THIRD_LIGHT
        counts = row["routing"] == "contact" and not third
        if c.get("jk_counts") == "True" and math.isfinite(tjk):
            row["colour_jk_T"] = tjk
            if counts:
                est.append(("colour_jk", tjk, False))
        if c.get("bprp_counts") == "True" and math.isfinite(tbp):
            row["colour_bprp_T"] = tbp
            if counts:
                est.append(("colour_bprp", tbp, True))
        if not counts and (math.isfinite(row["colour_jk_T"]) or math.isfinite(row["colour_bprp_T"])):
            flags.append("colours_not_counted_third_light" if row["routing"] == "contact" and third
                         else "colours_not_counted_unequal_pair")
        if math.isfinite(lo):
            row["colour_bprp_range"] = f"{lo:.0f}-{hi:.0f}" if math.isfinite(hi) else f">={lo:.0f}"
        if c.get("bprp_counts") != "True" and math.isfinite(lo) and math.isfinite(t1):
            # not an estimate, but a bound: BP-RP with no reddening is a lower limit
            if lo - t1 > CHANGE_DIFF_K:
                flags.append(f"bprp_lower_bound_{lo:.0f}K_above_T1")
            if math.isfinite(hi) and t1 - hi > CHANGE_DIFF_K:
                flags.append(f"bprp_upper_bound_{hi:.0f}K_below_T1")
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


NUMERIC = ("T1_pipeline", "msc_T1", "msc_T1_lo", "msc_T1_hi", "msc_T2", "msc_T2_lo", "msc_T2_hi",
           "msc_logg1", "msc_flag", "gspphot", "gspphot_lo", "gspphot_hi", "gspspec", "esphs",
           "sp_teff", "lamost_teff", "lit_T1", "T2_over_T1_fit", "Gmag")


def offline(colour: dict) -> list:
    """Rows rebuilt from the existing t1_anchor.csv (no VizieR or SIMBAD): the
    catalogue values are kept; T1 and its source (04's table), the result status and
    T2/T1, the literature T1 and the colour estimates are refreshed; estimates,
    sigma_T1 and the change rule are recomputed."""
    pipe = pipeline_t1()
    rows = []
    for r in csv.DictReader(open(OUT, encoding="utf-8")):
        row = dict(r)
        for k in NUMERIC:
            row[k] = fnum(row.get(k))
        star = row["star"]
        t1, src = pipe.get(star, (math.nan, "not in GAIA_TEFF_OVERRIDE_K"))
        row["T1_pipeline"], row["T1_source"] = t1, src
        row["status"], row["T2_over_T1_fit"] = result_status(star)
        lit = LITERATURE_T1.get(star)
        row["lit_T1"], row["lit_note"] = (lit[0], lit[1]) if lit else (math.nan, "")
        assess(row, colour)
        rows.append(row)
    return rows


def write_table(rows: list, cols: list) -> None:
    with open(OUT, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(x, 1) if isinstance(x, float) and math.isfinite(x) else
                            ("" if isinstance(x, float) else x)) for k, x in r.items()})


def main() -> int:
    flags_cli = [a for a in sys.argv[1:] if a.startswith("--")]
    pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    colour = {} if "--no-colour" in flags_cli else colour_table()
    print(f"colour temperatures: {len(colour)} stars from {COLOUR_CSV}" if colour
          else "colour temperatures: none (no table, or --no-colour)")
    if "--offline" in flags_cli:
        rows = offline(colour)
        return finish(rows)
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    cat = pos_args[0] if pos_args else os.path.join(CODE, "Targets", "phaseF_queue.csv")
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
        row["status"], row["T2_over_T1_fit"] = result_status(star)
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
        assess(row, colour)
        rows.append(row)
    return finish(rows)


# New columns go at the end so that readers keyed on the old order are unaffected.
COLS = ["queue_pos", "star", "status", "routing", "T1_pipeline", "T1_source", "msc_T1", "msc_T1_lo",
        "msc_T1_hi", "msc_T2", "msc_T2_lo", "msc_T2_hi", "msc_logg1", "msc_flag", "gspphot",
        "gspphot_lo", "gspphot_hi", "gspspec", "esphs", "sp_type", "sp_qual", "sp_bibcode",
        "sp_teff", "sp_note", "lamost_teff", "lamost_n", "lamost_range", "lit_T1", "lit_note", "T2_over_T1_fit",
        "gaia_source", "gaia_match", "Gmag", "estimates", "sigma_T1",
        "change_rule", "flags", "colour_jk_T", "colour_bprp_T", "colour_bprp_range", "map_status"]


def finish(rows: list) -> int:
    for row in rows:
        print(f"{row['queue_pos']:>2} {row['star']:10s} {row['status'][:8]:8s} T1 {row['T1_pipeline']:6.0f}  "
              f"MSC {row['msc_T1']:6.0f} [{row['msc_T1_lo']:5.0f},{row['msc_T1_hi']:5.0f}]  "
              f"GSP {row['gspphot']:5.0f}  SpT {row['sp_type'][:12]:12s} {row['sp_teff']:5.0f}  "
              f"LAM {row['lamost_teff']:5.0f}  JK {row['colour_jk_T']:5.0f}  BPRP {row['colour_bprp_T']:5.0f} "
              f"({row['colour_bprp_range']})  sig {row['sigma_T1']:4d}  {row['change_rule']}  {row['flags']}")
    write_table(rows, COLS)
    print("table:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
