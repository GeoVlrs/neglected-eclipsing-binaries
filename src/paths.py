"""
Single source of truth for the pipeline's on-disk layout.

Every numbered stage (02-06) and GenerateThesisTables imports this module and routes
each product read/write through the constants below, so the folder structure is defined
in exactly one place and the stages cannot drift apart (Stage 03 must read exactly where
Stage 02 wrote). Scripts are run FROM the Code/ directory, so every path here is relative
to Code/.

    Code/
      <scripts>.py, paths.py
      Targets/                      inputs (neglected_targets_*.csv)
      data/                         all products, typed
        lightcurves/  lc_*_clean.fits, lc_*_folded.fits, cadence_*.txt      (02)
        timing/       tominima_*.csv, ephem_*.txt                            (03)
        bundles/      bundle_*.phoebe (+ _mcmc transient, _final)            (04/05)
        results/      results_*.csv                                          (05)
        plots/  lightcurves/ oc/ models_init/ models_final/ mesh/ thesis/ mcmc_diag/
      Runs/  Audits/  Notes/
"""
import os

# --- Inputs -----------------------------------------------------------------
TARGETS_DIR = "Targets"                                    # neglected_targets_*.csv

# --- Product root (renamed from the old "02,03,04 Data") --------------------
DATA_DIR = "data"

# --- Typed product sub-folders ----------------------------------------------
LIGHTCURVE_DIR = os.path.join(DATA_DIR, "lightcurves")     # lc_*_clean/folded.fits, cadence_*.txt (02)
TIMING_DIR     = os.path.join(DATA_DIR, "timing")          # tominima_*.csv, ephem_*.txt            (03)
BUNDLE_DIR     = os.path.join(DATA_DIR, "bundles")         # bundle_*.phoebe (+ _mcmc, _final)      (04/05)
RESULTS_DIR    = os.path.join(DATA_DIR, "results")         # results_*.csv                          (05)

# --- Plots (typed) ----------------------------------------------------------
PLOT_DIR        = os.path.join(DATA_DIR, "plots")
PLOT_LC_DIR     = os.path.join(PLOT_DIR, "lightcurves")    # lc_*_raw_vs_clean.png       (02)
PLOT_OC_DIR     = os.path.join(PLOT_DIR, "oc")             # oc_*.png                    (03)
PLOT_INIT_DIR   = os.path.join(PLOT_DIR, "models_init")    # phoebe_init_*.png           (04)
PLOT_FINAL_DIR  = os.path.join(PLOT_DIR, "models_final")   # phoebe_final_*.png          (05)
PLOT_MESH_DIR   = os.path.join(PLOT_DIR, "mesh")           # mesh_*_*.png + .gif         (06)
PLOT_THESIS_DIR = os.path.join(PLOT_DIR, "thesis")         # gallery + roche schematics  (06)
PLOT_MCMC_DIR   = os.path.join(PLOT_DIR, "mcmc_diag")      # mcmc_chains_*.png           (audit)

# --- Hand-curated thesis figure drop (lives in the thesis text tree) --------
THESIS_FIGURES_DIR = os.path.join("..", "Text", "Figures")  # from Code/ (was ../../ from the old data folder)

_ALL_DIRS = (
    LIGHTCURVE_DIR, TIMING_DIR, BUNDLE_DIR, RESULTS_DIR,
    PLOT_LC_DIR, PLOT_OC_DIR, PLOT_INIT_DIR, PLOT_FINAL_DIR, PLOT_MESH_DIR, PLOT_THESIS_DIR,
    PLOT_MCMC_DIR,
)


def ensure_dirs() -> None:
    """Create every product/plot sub-folder (idempotent). Call once at script startup."""
    for d in _ALL_DIRS:
        os.makedirs(d, exist_ok=True)
