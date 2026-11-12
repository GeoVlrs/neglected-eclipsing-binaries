"""
================================================================================
AUDIT — DUST-CORRECTED COLOUR TEMPERATURES (an independent check of the fixed T1)
================================================================================
Summary:
    Script 05 holds T1 fixed (04Modeling.py GAIA_TEFF_OVERRIDE_K); audit_t1_anchor.py
    sets beside it every independent estimate and applies the change rule. This audit
    adds two colour temperatures per star, which audit_t1_anchor.py reads from
    Audits/colour_teff.csv:

      * T_bprp: Gaia DR3 BP-RP, dereddened with the Bayestar19 3D dust map (Green et al.
        2019) at the star's parallax distance and the Gaia extinction coefficients
        k_X(BP-RP, A0) of Babusiaux et al. (2018, Eq. 1, Table 1);
      * T_jk: 2MASS J-Ks (Skrutskie et al. 2006), dereddened with the same map value
        and the map's own 2MASS coefficients (Green et al. 2019, Table 1).

    Colours are converted to Teff with the Pecaut & Mamajek (2013) dwarf table as
    tabulated by E. Mamajek (Audits/EEM_dwarf_table.txt, version 2022.04.16).

    The map (Harvard Dataverse doi:10.7910/DVN/2EJ9TX, bayestar2019.h5, 728 MB) is read
    ONE PIXEL per star with h5py; the dustmaps package would load the whole map
    (~12 GB of RAM). The lookup is the one of dustmaps 1.0.14 (bayestar.py): nested
    HEALPix at each nside level, linear interpolation in distance modulus between the
    120 bins, linear scaling below the first bin, percentiles over the 5 samples.

    Map status per star:
      reliable              distance inside the pixel's DM_reliable_min..max
      nearer_than_reliable  closer than DM_reliable_min: the map is not constrained
                            there; E is bracketed between 0 and the map median at
                            DM_reliable_min (reddening grows with distance)
      beyond_reliable       farther than DM_reliable_max: the map value is used, flagged
      not_converged         the pixel's fit did not converge
      no_coverage           south of dec -30 deg (outside the map)

Limitations:
    A colour temperature is the combined light of an unresolved pair matched to a single
    dwarf; it is evidence for T1, not a replacement. The Babusiaux et al. coefficients
    were fitted on the DR2 passbands (applied here to DR3 colours). A0 per map unit is
    interpolated between the PS1 g and r coefficients at 541.4 nm (as in the thesis's
    T1 section). 2MASS is a single epoch at an unknown orbital phase; J, H and Ks were
    taken simultaneously, so the colour is less affected than either magnitude.

Inputs:
    catalogue CSV (column "name"; default Targets/phaseF_queue.csv); Audits/t1_anchor.csv
    (gaia_source, resolved through the TIC); Runs/modeltype/catalog_lookup_60.json
    (ra, dec); network: ESA Gaia TAP and IRSA TAP (not CDS). The answers are cached in
    Audits/colour_inputs.csv; --cached reruns from the cache without the network.
    Environment: BAYESTAR_MAP (default ~/dustmaps_data/bayestar/bayestar2019.h5).

Usage (WSL, from Code/; needs numpy, h5py, healpy; seconds per star):
    PYTHONPATH=~/dustmaps_pkgs python3 Audits/audit_colour_teff.py [catalogue] [--cached]
Output:
    Audits/colour_teff.csv (one row per star); Audits/colour_inputs.csv; table on stdout.
================================================================================
"""
import csv
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(CODE, "Audits", "colour_teff.csv")
CACHE = os.path.join(CODE, "Audits", "colour_inputs.csv")
T1A = os.path.join(CODE, "Audits", "t1_anchor.csv")
POS_JSON = os.path.join(CODE, "Runs", "modeltype", "catalog_lookup_60.json")
EEM = os.path.join(CODE, "Audits", "EEM_dwarf_table.txt")
MAP = os.environ.get("BAYESTAR_MAP", os.path.expanduser("~/dustmaps_data/bayestar/bayestar2019.h5"))
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP/sync"

# Green et al. (2019), Table 1: extinction vector A_b = R_b E (E = one map unit).
# Checked against Sec II.1: R_g - R_r = 0.901 (E(g-r) per unit E) and R_H/R_Ks = 1.55.
R_G, R_R, R_J, R_H, R_KS = 3.518, 2.617, 0.7927, 0.4690, 0.3026
L_G, L_R, L_0 = 481.0, 617.0, 541.4          # nm: PS1 g, r effective (Tonry et al. 2012); Gaia A0
A0_PER_E = R_R + (1 / L_0 - 1 / L_R) / (1 / L_G - 1 / L_R) * (R_G - R_R)   # 3.062, interp. in 1/lambda
# Babusiaux et al. (2018), Eq. 1 and Table 1: k = A_X/A0 = c1 + c2 x + c3 x^2 + c4 x^3
# + c5 A0 + c6 A0^2 + c7 x A0, with x = (BP-RP)0. Fit range 3500-10000 K, A0 0.01-5.
K_BP = (1.1517, -0.0871, -0.0333, 0.0173, -0.0230, 0.0006, 0.0043)
K_RP = (0.6104, -0.0170, -0.0026, -0.0017, -0.0078, 0.00005, 0.0006)
TWOMASS_RADIUS_AS = 5.0      # TUNABLE: 2MASS match radius; raising it risks a neighbour,
                             # lowering it loses high-proper-motion stars (2MASS epoch ~2000)
JK_BRACKET_MAX_K = 250.0     # TUNABLE: a J-Ks estimate whose reddening bracket, or whose
                             # photometric error (half range), moves it by more than this (the
                             # T1 sigma floor of audit_t1_anchor) is not counted (flag only).
                             # J-Ks flattens for A stars, so hot stars drop out here.
TWOMASS_EPOCH = 2000.0       # 2MASS observed 1997-2001 (Skrutskie et al. 2006)
DEC_LIMIT = -30.0            # Bayestar19 covers the sky north of dec -30 deg (Green et al. 2019)


def k_coef(c, x, a0):
    return c[0] + c[1] * x + c[2] * x * x + c[3] * x ** 3 + c[4] * a0 + c[5] * a0 * a0 + c[6] * x * a0


def deredden_bprp(bprp, a0):
    """(BP-RP)0 for extinction a0 (mag at 541.4 nm), iterating on the colour-dependent k."""
    x = bprp
    for _ in range(6):
        x = bprp - (k_coef(K_BP, x, a0) - k_coef(K_RP, x, a0)) * a0
    return x


def dwarf_table():
    """Arrays (Teff, BP-RP, J-Ks) of the Mamajek dwarf table, Teff decreasing."""
    rows = []
    for line in open(EEM, encoding="latin-1"):
        p = line.split()
        if not p or p[0].startswith("#") or len(p) < 21 or not p[0].endswith("V") or p[0][0] not in "OBAFGKM":
            continue
        try:
            t = float(p[1])
        except ValueError:
            continue
        def f(i):
            try:
                return float(p[i])
            except ValueError:
                return math.nan
        rows.append((t, f(11), f(19) + f(20)))       # Teff, Bp-Rp, J-H + H-Ks
    rows.sort(reverse=True)
    return np.array(rows).T


TT, BPRP_T, JK_T = dwarf_table()


TABLE_TMIN, TABLE_TMAX = 3000.0, 15000.0   # conversion range; results outside the
FIT_TMIN, FIT_TMAX = 3500.0, 10000.0       # Babusiaux et al. fit range are flagged


def _monotonic(col):
    """(colour, Teff) for TABLE_TMIN-TABLE_TMAX, ordered by Teff, keeping only rows whose
    colour rises as Teff falls (drops the few table rows where a colour turns over)."""
    ok = np.isfinite(col) & (TT >= TABLE_TMIN) & (TT <= TABLE_TMAX)
    t, c = TT[ok][::-1], col[ok][::-1]               # Teff increasing
    keep, last = [], math.inf
    for i in range(len(t) - 1, -1, -1):              # from hot to cool: colour must increase
        if not keep or c[i] > last:
            keep.append(i)
            last = c[i]
    keep = keep[::-1]
    return c[keep][::-1], t[keep][::-1]               # colour increasing, Teff decreasing


_CURVES = {}


def colour_to_teff(colour, col):
    """Teff for a colour (linear interpolation in the dwarf table); nan outside its range."""
    key = id(col)
    if key not in _CURVES:
        _CURVES[key] = _monotonic(col)
    c, t = _CURVES[key]
    if not math.isfinite(colour) or colour < c[0] or colour > c[-1]:
        return math.nan
    return float(np.interp(colour, c, t))


# ─────────────────────────── network inputs (cached) ───────────────────────────
def tap(url, query, tries=4):
    data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
                                   "QUERY": query}).encode()
    for k in range(tries):
        try:
            txt = urllib.request.urlopen(url, data=data, timeout=120).read().decode("utf-8")
            if txt.lstrip().startswith("<"):
                raise IOError(txt[:200])
            return list(csv.DictReader(txt.splitlines()))
        except Exception as e:                      # overloaded service: wait and retry
            if k == tries - 1:
                raise
            print(f"  TAP retry {k + 1} ({str(e)[:60]})")
            time.sleep(30 * (k + 1))
    return []


def fetch_inputs(stars, gaia_ids, pos):
    ids = [gaia_ids[s] for s in stars if gaia_ids.get(s)]
    q = ("SELECT source_id, ra, dec, pmra, pmdec, l, b, parallax, parallax_error, phot_g_mean_mag, bp_rp, "
         "phot_bp_rp_excess_factor, ruwe FROM gaiadr3.gaia_source WHERE source_id IN (%s)" % ",".join(ids))
    gaia = {r["source_id"]: r for r in tap(GAIA_TAP, q)}
    out = []
    for s in stars:
        row = {"star": s, "gaia_source": gaia_ids.get(s, "")}
        g = gaia.get(row["gaia_source"], {})
        for k in ("ra", "dec", "pmra", "pmdec", "l", "b", "parallax", "parallax_error", "phot_g_mean_mag",
                  "bp_rp", "phot_bp_rp_excess_factor", "ruwe"):
            row[k] = g.get(k, "")
        ra = float(g["ra"]) if g.get("ra") else pos.get(s, {}).get("ra")
        de = float(g["dec"]) if g.get("dec") else pos.get(s, {}).get("dec")
        if g.get("pmra") and g.get("pmdec"):
            # Gaia DR3 positions are at epoch 2016.0; move them back to the 2MASS epoch
            # (~2000), so that a fast-moving star (VW Cep, ~1.3"/yr) is still matched.
            dt = TWOMASS_EPOCH - 2016.0
            de = de + float(g["pmdec"]) * dt / 3.6e6
            ra = ra + float(g["pmra"]) * dt / 3.6e6 / math.cos(math.radians(de))
        row["tmass_n"] = 0
        if ra is not None:
            r = TWOMASS_RADIUS_AS / 3600.0
            q2 = ("SELECT designation, ra, dec, j_m, j_msigcom, h_m, h_msigcom, k_m, k_msigcom, ph_qual, cc_flg "
                  f"FROM fp_psc WHERE CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{de},{r}))=1")
            try:
                tm = tap(IRSA_TAP, q2)
            except Exception as e:
                tm, row["tmass_error"] = [], str(e)[:80]
            def sep(t):
                dra = (float(t["ra"]) - ra) * math.cos(math.radians(de))
                return 3600.0 * math.hypot(dra, float(t["dec"]) - de)
            tm.sort(key=sep)
            row["tmass_n"] = len(tm)
            if tm:
                t = tm[0]
                row.update(tmass_id=t["designation"], tmass_sep=f"{sep(t):.2f}", j_m=t["j_m"],
                           j_e=t["j_msigcom"], h_m=t["h_m"], h_e=t["h_msigcom"], k_m=t["k_m"],
                           k_e=t["k_msigcom"], ph_qual=t["ph_qual"], cc_flg=t["cc_flg"])
        out.append(row)
        print(f"  inputs {s:10s} Gaia {'ok' if g else 'MISSING'}  2MASS {row['tmass_n']} within "
              f"{TWOMASS_RADIUS_AS:.0f}\"")
    return out


# ─────────────────────────── Bayestar19, one pixel ───────────────────────────
class PixelMap:
    def __init__(self, path):
        import h5py
        self.f = h5py.File(path, "r")
        info = self.f["pixel_info"]
        self.nside = info.fields("nside")[:]
        self.hpix = info.fields("healpix_index")[:]
        self.info = info
        self.bins = info.attrs["DM_bin_edges"]

    def row(self, l, b):
        import healpy as hp
        theta, phi = np.radians(90.0 - b), np.radians(l)
        for n in np.unique(self.nside):
            ip = hp.ang2pix(int(n), theta, phi, nest=True)
            hit = np.nonzero((self.nside == n) & (self.hpix == ip))[0]
            if hit.size:
                return int(hit[0])
        return None

    def e_at(self, samp, dm):
        bins = self.bins
        k = int(np.searchsorted(bins, dm))
        if k == 0:
            return 10.0 ** (0.2 * (dm - bins[0])) * samp[:, 0]
        if k == len(bins):
            return samp[:, -1]
        a = (bins[k] - dm) / (bins[k] - bins[k - 1])
        return (1 - a) * samp[:, k] + a * samp[:, k - 1]

    def query(self, l, b, d_pc):
        r = self.row(l, b)
        if r is None:
            return None
        samp = self.f["samples"][r]                 # (5 samples, 120 distance bins)
        pinf = self.info[r]
        dm = 5.0 * math.log10(d_pc) - 5.0
        lo, hi = float(pinf["DM_reliable_min"]), float(pinf["DM_reliable_max"])
        e = np.percentile(self.e_at(samp, dm), [16, 50, 84])
        if not bool(pinf["converged"]):
            status = "not_converged"
        elif not (math.isfinite(lo) and math.isfinite(hi)):
            status = "not_converged"
        elif dm < lo:
            status = "nearer_than_reliable"
        elif dm > hi:
            status = "beyond_reliable"
        else:
            status = "reliable"
        e_lo_edge = float(np.median(self.e_at(samp, lo))) if math.isfinite(lo) else math.nan
        return dict(row=r, dm=dm, e=e, status=status, dm_min=lo, dm_max=hi, e_at_dm_min=e_lo_edge)


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cached = "--cached" in sys.argv
    cat = args[0] if args else os.path.join(CODE, "Targets", "phaseF_queue.csv")
    stars = [r["name"] for r in csv.DictReader(open(cat, encoding="utf-8"))]
    if cached:
        inp = {r["star"]: r for r in csv.DictReader(open(CACHE, encoding="utf-8"))}
        rows_in = [inp[s] for s in stars if s in inp]
    else:
        gaia_ids = {r["star"]: r["gaia_source"] for r in csv.DictReader(open(T1A, encoding="utf-8"))}
        pos = json.load(open(POS_JSON, encoding="utf-8"))
        rows_in = fetch_inputs(stars, gaia_ids, pos)
        cols_in = ["star", "gaia_source", "ra", "dec", "pmra", "pmdec", "l", "b", "parallax", "parallax_error",
                   "phot_g_mean_mag", "bp_rp", "phot_bp_rp_excess_factor", "ruwe", "tmass_n", "tmass_id",
                   "tmass_sep", "j_m", "j_e", "h_m", "h_e", "k_m", "k_e", "ph_qual", "cc_flg", "tmass_error"]
        old = {}
        if os.path.exists(CACHE):                   # keep stars not in this catalogue
            old = {r["star"]: r for r in csv.DictReader(open(CACHE, encoding="utf-8"))}
        for r in rows_in:
            old[r["star"]] = r
        with open(CACHE, "w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=cols_in, extrasaction="ignore")
            w.writeheader()
            for r in old.values():
                w.writerow(r)
    pm = PixelMap(MAP)
    out = []
    print(f"A0 per map unit {A0_PER_E:.3f}; E(J-Ks) per map unit {R_J - R_KS:.4f}")
    for r in rows_in:
        s = r["star"]
        o = {"star": s, "gaia_source": r.get("gaia_source", "")}
        plx, eplx, bprp = fnum(r.get("parallax")), fnum(r.get("parallax_error")), fnum(r.get("bp_rp"))
        l, b, dec = fnum(r.get("l")), fnum(r.get("b")), fnum(r.get("dec"))
        flags = ["combined light of an unresolved pair"]
        d = 1000.0 / plx if plx > 0 else math.nan
        o.update(d_pc=d, plx_rel_err=eplx / plx if plx > 0 else math.nan, bp_rp=bprp,
                 bp_rp_excess=fnum(r.get("phot_bp_rp_excess_factor")), ruwe=fnum(r.get("ruwe")))
        # ---- map ----
        m = None
        if not math.isfinite(d) or not math.isfinite(l):
            o["map_status"] = "no_gaia_distance"
        elif dec < DEC_LIMIT:
            o["map_status"] = "no_coverage"
        else:
            m = pm.query(l, b, d)
            o["map_status"] = m["status"] if m else "no_coverage"
        if m:
            o.update(map_row=m["row"], dm=round(m["dm"], 3), E16=m["e"][0], E50=m["e"][1], E84=m["e"][2],
                     dm_reliable_min=m["dm_min"], dm_reliable_max=m["dm_max"], E_at_dm_min=m["e_at_dm_min"])
            if plx > 0 and eplx > 0:
                o["E50_plx_minus"] = float(np.median(pm.e_at(pm.f["samples"][m["row"]],
                                                              5 * math.log10(1000 / (plx - eplx)) - 5)))
                o["E50_plx_plus"] = float(np.median(pm.e_at(pm.f["samples"][m["row"]],
                                                             5 * math.log10(1000 / (plx + eplx)) - 5)))
        # E bracket used for the colours: (lo, mid, hi) in map units
        st = o["map_status"]
        if st in ("reliable", "beyond_reliable"):
            eb = (o["E16"], o["E50"], o["E84"])
        elif st == "nearer_than_reliable":
            # Upper bound: the map median at DM_reliable_min scaled linearly with distance.
            # ASSUMPTION: the dust density along the sight line does not fall towards the
            # Sun inside DM_reliable_min (true for a line leaving the low-density Local
            # Bubble), so E grows at least linearly out to that distance. The bound from
            # monotonic reddening alone, E_at_dm_min, is kept in the table as well.
            d_min = 10.0 ** (0.2 * o["dm_reliable_min"] + 1.0)
            o["E_upper"] = o["E_at_dm_min"] * d / d_min
            eb = (0.0, math.nan, o["E_upper"])
        else:
            eb = (0.0, math.nan, math.nan)
        if st == "beyond_reliable":
            flags.append("map beyond its reliable range")
        # ---- BP-RP ----
        def t_bprp(e):
            if not (math.isfinite(e) and math.isfinite(bprp)):
                return math.nan
            return colour_to_teff(deredden_bprp(bprp, A0_PER_E * e), BPRP_T)
        o["T_bprp_unred"] = t_bprp(0.0)
        o["T_bprp_16"], o["T_bprp_50"], o["T_bprp_84"] = (t_bprp(eb[0]), t_bprp(eb[1]), t_bprp(eb[2]))
        # ---- J-Ks ----
        jk = fnum(r.get("j_m")) - fnum(r.get("k_m"))
        jk_e = math.hypot(fnum(r.get("j_e")), fnum(r.get("k_e")))
        o.update(tmass_id=r.get("tmass_id", ""), tmass_sep=fnum(r.get("tmass_sep")),
                 tmass_n=r.get("tmass_n", ""), ph_qual=r.get("ph_qual", ""), cc_flg=r.get("cc_flg", ""),
                 J_Ks=jk, J_Ks_err=jk_e)
        def t_jk(e, dc=0.0):
            if not (math.isfinite(e) and math.isfinite(jk)):
                return math.nan
            return colour_to_teff(jk - (R_J - R_KS) * e + dc, JK_T)
        e_mid = eb[1] if math.isfinite(eb[1]) else 0.0
        o["T_jk"] = t_jk(e_mid)
        o["T_jk_unred"] = t_jk(0.0)        # no dust correction: a lower limit on T
        o["T_jk_colour_lo"], o["T_jk_colour_hi"] = t_jk(e_mid, +jk_e), t_jk(e_mid, -jk_e)
        o["T_jk_red_lo"], o["T_jk_red_hi"] = t_jk(eb[0]), t_jk(eb[2])
        good2mass = (o["ph_qual"] == "AAA" and str(o["cc_flg"]) in ("000", "0", "0.0")
                     and str(o["tmass_n"]) == "1" and math.isfinite(jk))
        if not good2mass and math.isfinite(jk):
            flags.append(f"2MASS quality {o['ph_qual']} cc {o['cc_flg']} n{o['tmass_n']} within "
                         f"{TWOMASS_RADIUS_AS:.0f}\"")
        mapped = st in ("reliable", "beyond_reliable", "nearer_than_reliable")
        bracket = (abs(o["T_jk_red_hi"] - o["T_jk_red_lo"]) if mapped
                   and math.isfinite(o["T_jk_red_hi"]) and math.isfinite(o["T_jk_red_lo"]) else math.nan)
        cerr = (0.5 * abs(o["T_jk_colour_hi"] - o["T_jk_colour_lo"])
                if math.isfinite(o["T_jk_colour_hi"]) and math.isfinite(o["T_jk_colour_lo"]) else math.nan)
        o["T_jk_bracket_K"], o["T_jk_colour_err_K"] = bracket, cerr
        o["jk_counts"] = bool(good2mass and math.isfinite(o["T_jk"]) and bracket <= JK_BRACKET_MAX_K
                              and cerr <= JK_BRACKET_MAX_K)
        # BP-RP counts where the map is reliable (median, 16-84 % range), or where the star
        # is nearer than that range but the reddening bracket moves T by <= JK_BRACKET_MAX_K
        # (very nearby stars, where even the upper bound on E is small); then the middle
        # of the bracket is used. Same test as for J-Ks.
        if st == "reliable":
            o["T_bprp_est"] = o["T_bprp_50"]
        elif st == "nearer_than_reliable" and math.isfinite(o["T_bprp_16"]) and math.isfinite(o["T_bprp_84"])                 and abs(o["T_bprp_84"] - o["T_bprp_16"]) <= JK_BRACKET_MAX_K:
            o["T_bprp_est"] = 0.5 * (o["T_bprp_16"] + o["T_bprp_84"])
        else:
            o["T_bprp_est"] = math.nan
        o["bprp_counts"] = bool(math.isfinite(o["T_bprp_est"]) and FIT_TMIN <= o["T_bprp_est"] <= FIT_TMAX)
        if good2mass and not o["jk_counts"]:
            if not mapped:
                flags.append("no map: J-Ks not dereddened, not counted")
            elif not math.isfinite(o["T_jk"]) or not math.isfinite(bracket) or not math.isfinite(cerr):
                flags.append("J-Ks outside the dwarf table after dereddening")
            else:
                flags.append(f"J-Ks not counted: reddening bracket {bracket:.0f} K, colour error "
                             f"+-{cerr:.0f} K (limit {JK_BRACKET_MAX_K:.0f} K)")
        for lab in ("T_bprp_unred", "T_bprp_50"):
            if math.isfinite(o[lab]) and not FIT_TMIN <= o[lab] <= FIT_TMAX:
                flags.append(f"{lab} {o[lab]:.0f} K outside the extinction-coefficient fit range "
                             f"({FIT_TMIN:.0f}-{FIT_TMAX:.0f} K)")
                break
        if not math.isfinite(o["T_bprp_unred"]) and math.isfinite(bprp):
            flags.append("BP-RP outside the dwarf table")
        o["flags"] = "; ".join(flags)
        out.append(o)
        print(f"{s:10s} d {d:6.1f} pc  map {st:20s} E {o.get('E50', math.nan):5.3f} "
              f"[{eb[0]:.3f},{eb[2]:.3f}]  BP-RP {bprp:5.3f} T {o['T_bprp_unred']:5.0f}/{o['T_bprp_50']:5.0f}"
              f" [{o['T_bprp_16']:5.0f},{o['T_bprp_84']:5.0f}]  J-Ks {jk:5.3f}+-{jk_e:.3f} T {o['T_jk']:5.0f}"
              f" (col +-{cerr:4.0f}; red {bracket:4.0f})"
              f"  counts jk {o['jk_counts']:d} bprp {o['bprp_counts']:d}")
    cols = ["star", "gaia_source", "d_pc", "plx_rel_err", "map_status", "map_row", "dm", "E16", "E50", "E84",
            "E50_plx_minus", "E50_plx_plus", "dm_reliable_min", "dm_reliable_max", "E_at_dm_min", "E_upper",
            "bp_rp", "bp_rp_excess", "ruwe", "T_bprp_unred", "T_bprp_16", "T_bprp_50", "T_bprp_84",
            "tmass_id", "tmass_sep", "tmass_n", "ph_qual", "cc_flg", "J_Ks", "J_Ks_err", "T_jk", "T_jk_unred",
            "T_jk_colour_lo", "T_jk_colour_hi", "T_jk_colour_err_K", "T_jk_red_lo", "T_jk_red_hi", "T_jk_bracket_K",
            "T_bprp_est", "jk_counts", "bprp_counts", "flags"]
    with open(OUT, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for o in out:
            w.writerow({k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else
                            ("" if isinstance(v, float) else v)) for k, v in o.items()})
    print("table:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
