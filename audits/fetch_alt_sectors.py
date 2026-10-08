"""
================================================================================
FETCH — OTHER TESS SECTORS FOR STARS FLAGGED BY audit_crowding_pdc.py
================================================================================
Summary:
    The PDCSAP-distortion decision (Pipeline_Log Entry 76: "hybrid — best clean
    sector, else SAP") needs every available sector of the flagged stars scored
    with the same PDCSAP-vs-SAP test. Only the selected sector is usually in the
    local lightkurve cache, so this helper lists (default) or downloads
    (--download) the star's other light-curve products from MAST into that same
    cache, with Script 02's author order: SPOC 2-min, else TESS-SPOC FFI. QLP is
    skipped (it has no PDCSAP to compare).

    Downloads go to ~/.lightkurve/cache — exactly where Script 02 would put them —
    so nothing in Code/ changes. Re-run audit_crowding_pdc.py afterwards to score
    the new sectors.

Usage (from Code/; network, light CPU):
    python Audits/fetch_alt_sectors.py "GK Cep" "FN Cam" "EF Boo" "V899 Her"
    python Audits/fetch_alt_sectors.py --download "GK Cep" "FN Cam" "EF Boo" "V899 Her"

Dependencies:
    pip install lightkurve
================================================================================
"""

import csv
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import lightkurve as lk                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths                                              # noqa: E402

MAX_VALID_SECTOR: int = 200           # 02 rejects bogus sector IDs (e.g. the "1751" stub)


def tic_of(star: str) -> int:
    for p in glob.glob(os.path.join(paths.TARGETS_DIR, "neglected_targets*.csv")):
        with open(p, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row.get("name", "").strip() == star and row.get("tic"):
                    return int(float(row["tic"]))
    raise KeyError(f"{star}: no TIC in the catalogue CSVs")


def products(star: str):
    """One search row per sector: SPOC if available, else TESS-SPOC."""
    tic = tic_of(star)
    chosen = {}
    for author in ("SPOC", "TESS-SPOC"):
        res = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author=author)
        for i in range(len(res)):
            row = res[i]
            sector = int(str(row.mission[0]).split()[-1])
            if sector > MAX_VALID_SECTOR or sector in chosen:
                continue
            if author == "SPOC" and float(row.exptime[0].value) > 200:
                continue                              # SPOC 20 s / 2 min only; FFI comes from TESS-SPOC
            chosen[sector] = (author, row)
    return tic, dict(sorted(chosen.items()))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--download"]
    download = "--download" in sys.argv
    for star in args:
        tic, rows = products(star)
        print(f"{star} (TIC {tic}): {len(rows)} sector(s) with SPOC/TESS-SPOC light curves")
        for sector, (author, row) in rows.items():
            exp = float(row.exptime[0].value)
            line = f"   S{sector:<3} {author:<9} {exp:>5.0f} s"
            if download:
                try:
                    lc = row.download(quality_bitmask="none")
                    line += f"  downloaded ({len(lc)} cadences)"
                except Exception as err:
                    line += f"  FAILED ({err})"
            print(line)
