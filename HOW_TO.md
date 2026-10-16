# HOW TO run the pipeline

An operating manual for every script in this project: what each one does, how to
run it, what it reads and writes, how long it takes, and what can go wrong.

The pipeline models short-period eclipsing binaries from TESS photometry. Six stages
take a target list to fitted physical parameters and rendered geometry. The fit
itself is a PHOEBE 2 forward model sampled with the emcee MCMC sampler.

> **Names.** This manual uses the names of the working tree (`01TargetSelection.py`
> … `06Visualization.py`, `Targets/`, `Audits/`). The published repository renames
> some of them; see [§9 Published layout](#9-publishing-and-the-published-layout).

---

## Contents

1. [Overview](#1-overview)
2. [Environments and installation](#2-environments-and-installation)
3. [Layout and conventions](#3-layout-and-conventions)
4. [Quick start: one batch, end to end](#4-quick-start-one-batch-end-to-end)
5. [Script reference](#5-script-reference)
6. [Running Script 05 (the MCMC stage)](#6-running-script-05-the-mcmc-stage)
7. [Audits](#7-audits)
8. [Run statistics and provenance](#8-run-statistics-and-provenance)
9. [Publishing and the published layout](#9-publishing-and-the-published-layout)
10. [Safety rules (checklist)](#10-safety-rules-checklist)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Overview

```
 01 TargetSelection ──► Targets/neglected_targets*.csv
          │
 02 DataAcquisition ──► data/lightcurves/lc_<star>_clean.fits   (TESS light curve)
          │
 03 OCAnalysis      ──► data/timing/ephem_<star>.txt           (period, epoch)
          │
 04 Modeling        ──► data/bundles/bundle_<star>.phoebe      (PHOEBE model)
          │
 05 Optimization    ──► data/results/results_<star>.csv        (posterior)
          │             data/bundles/bundle_<star>_final.phoebe
 06 Visualization   ──► data/plots/mesh/…                      (3-D renders)
```

| Stage | Does | Cost per star | Runs on |
|---|---|---|---|
| 01 | Selects neglected binaries from the Kreiner (2004) catalogue, GCVS, VSX and MAST | network-bound, ~1 h for the full catalogue | Windows/Linux |
| 02 | Downloads TESS light curves and picks a clean sector | seconds (downloads cached) | Windows/Linux |
| 03 | Times every eclipse minimum and refines the period | ~1 s | Windows/Linux |
| 04 | Builds the PHOEBE bundle (geometry, physics, binned data) | ~5–15 s | Linux / WSL |
| 05 | Nelder–Mead pre-fit, then MCMC with a convergence gate | **hours to ~10 days** | Linux / WSL, under `mpirun` |
| 06 | Renders surface meshes and orbit GIFs (Mayavi) | seconds | Windows (Mayavi) |

The utilities are `GenerateThesisTables.py` (LaTeX results table) and two
illustration scripts, `SchematicRocheGeometry.py` and
`SchematicRocheEquipotentials.py`. The audits are described in §7.

Each stage reads the previous stage's files from `data/`. Re-running a stage
**overwrites** its outputs for the stars in the catalogue you give it. Stage 05 is
the exception: it skips stars whose results are already current (§6.6).

---

## 2. Environments and installation

### 2.1 Why two environments

- **Stage 05 needs MPI** (`mpi4py` + Open MPI) and runs under `mpirun`. On Windows
  that is impractical, so it runs in **Linux / WSL 2**. Stage 04 also runs there,
  so bundles are built by the same PHOEBE installation that fits them.
- **Stage 06 and `SchematicRocheGeometry.py` need Mayavi/VTK/PyQt5**, which install
  most easily on **Windows**.
- 01, 02, 03, `SchematicRocheEquipotentials.py`, GenerateThesisTables and the light
  audits run anywhere.

Tested versions (both environments unless noted):

| Package | Version |
|---|---|
| phoebe | 2.4.22 |
| emcee | 3.1.6 |
| lightkurve | 2.6.0 |
| astropy / astroquery | 7.2.0 / 0.4.11 |
| numpy / scipy / matplotlib / pandas | 2.4.4 / 1.17.1 / 3.10 / 2.3.3 |
| mpi4py / Open MPI (Linux only) | 3.1.5 / 4.1.6 |
| mayavi / vtk / PyQt5 (Windows only) | 4.9.0 / 9.7.0 / 5.15.11 |
| Python | 3.12 (Linux), 3.14 (Windows) |

### 2.2 Install

Linux / WSL (stages 04–05):

```bash
sudo apt install openmpi-bin libopenmpi-dev python3-venv python3-dev build-essential
pip install phoebe==2.4.22 emcee==3.1.6 mpi4py numpy scipy matplotlib astropy astroquery lightkurve pandas
```

Windows (stages 01–03, 06, schematics):

```bash
pip install phoebe==2.4.22 emcee==3.1.6 lightkurve numpy scipy matplotlib astropy astroquery pandas requests
pip install mayavi vtk pyqt5 pillow
```

Pin phoebe and emcee. A different version changes the numbers, and the run
statistics record the versions used (§8).

### 2.3 PHOEBE passbands offline

Scripts 05, 06 and `SchematicRocheGeometry.py` set `PHOEBE_ENABLE_ONLINE_PASSBANDS=FALSE` before
importing PHOEBE. Otherwise, on a host without internet access, PHOEBE tries to reach
its passband server on almost every model and each attempt blocks for tens of
seconds. The TESS passband must therefore be installed locally once. Set the
variable to `TRUE` only to download a new passband.

### 2.4 Things that stop a long run (WSL)

- **Closing the terminal kills the run.** `mpirun` aborts on hang-up. Keep the
  terminal open, or use `tmux` (§6.4).
- **Windows sleep or hibernation freezes WSL**, so the run pauses. Set sleep to
  *Never* (on mains power) for the duration of a run.
- **Windows Update restarts** end the run. Set active hours or pause updates. The
  run resumes afterwards (§6.3), losing at most one chunk.
- **WSL updates itself.** On 13 Oct 2026 Windows upgraded WSL in the middle of a
  run and killed every process in it, without a reboot. The wrapper cannot
  survive that; the queue stays down until it is relaunched by hand. Pause
  Windows and Microsoft Store updates for the length of a campaign.

---

## 3. Layout and conventions

### 3.1 Run everything from `Code/`

All paths come from `paths.py` and are **relative to the current directory**.
Always `cd` into `Code/` first. The only exceptions are `runstats.py` (anchored to
its own folder) and `run_05_autorestart.sh` (it `cd`s to its own folder itself).

```
Code/
├── 01TargetSelection.py … 06Visualization.py
├── GenerateThesisTables.py, SchematicRocheGeometry.py, SchematicRocheEquipotentials.py
├── run_05_autorestart.sh            # supervised launcher for stage 05
├── paths.py, runstats.py, morphology.py   # shared modules (imported, not run)
├── Targets/                         # target catalogues (CSV)
├── Audits/                          # diagnostic scripts and their CSV outputs
├── Runs/                            # logs, run statistics, backups
│   └── run_stats/                   # automatic per-run / per-star statistics
└── data/
    ├── lightcurves/   lc_<star>_clean.fits, lc_<star>_folded.fits, cadence_<star>.txt   (02)
    ├── timing/        ephem_<star>.txt, tominima_<star>.csv                            (03)
    ├── bundles/       bundle_<star>.phoebe (04), _mcmc / _final (05), *.lock (05)
    ├── results/       results_<star>.csv                                               (05)
    └── plots/         lightcurves/ oc/ models_init/ models_final/ mcmc_diag/ mesh/ thesis/
```

`<star>` is the star name with spaces replaced by underscores (`V899 Her` →
`V899_Her`).

### 3.2 Catalogue format

```
name,type,period,sectors,tic
V899 Her,EW,0.4211676,78;25;51;24;79;52,8673277
```

| Column | Used by | Meaning |
|---|---|---|
| `name` | all | Star name (GCVS form). |
| `type` | 04, 05, 06 | GCVS type string, e.g. `EW/KW`, `EB/SD:`. Decides the morphology (§5.10). |
| `period` | 02, 03 | Catalogue period in days (starting value only). |
| `sectors` | audits | TESS sectors, `;`-separated. |
| `tic` | 02, audits | TESS Input Catalog ID. **Search MAST by TIC**: names can collide (e.g. "NU Cyg" vs ν Cyg). |
| `skip_sectors` | 02 (optional) | `;`-separated sectors never to use. |

**Always pass a catalogue.** Every stage defaults to `neglected_targets.csv` in the
current directory, which only exists right after re-running stage 01.

### 3.3 The catalogues in `Targets/`

| File | Contents |
|---|---|
| `neglected_targets.csv` | Full selection (393 stars) from stage 01, with TIC added. |
| `neglected_targets_batch1.csv` … `batch8.csv` | The working sample (60 stars). Batches 1–5: contact (EW). 6–7: EB. 8: semi-detached and other classes. |
| `neglected_targets_batch9_held.csv` | 5 stars held back. |
| `phaseE_queue.csv` | The current re-run queue for stage 05 (re-runs first, then never-run stars). |
| `remediation_*.csv` | Small lists used for targeted re-processing in the past (`remediation_resector.csv` carries `skip_sectors`). |
| `triage_neglected_targets.csv` | Prioritisation of the full selection. |
| `rejected_targets.csv` | Stage 01's rejection log. |
| `trial_*.csv` | **Do not use.** They contain real stars; a trial against them would overwrite real results. Use a scratch star (§6.8). |

---

## 4. Quick start: one batch, end to end

```bash
cd <path-to>/Code

# Windows (or Linux) — data and timing
python 02DataAcquisition.py Targets/neglected_targets_batch1.csv
python 03OCAnalysis.py      Targets/neglected_targets_batch1.csv
python Audits/audit_period_fold.py          # check: drift < 0.01 cycle for every star

# Linux / WSL — model and fit
python3 04Modeling.py Targets/neglected_targets_batch1.csv
# contact stars only: fix the mass ratio (§5.4b, heavy), then rebuild their bundles
bash Runs/qsearch/run_qsearch.sh Targets/neglected_targets_batch1.csv "coarse refine literature" batch1
python3 ContactQSearch.py adopt Targets/neglected_targets_batch1.csv
python3 04Modeling.py Targets/neglected_targets_batch1.csv
bash run_05_autorestart.sh Targets/neglected_targets_batch1.csv Runs/batch1_05.log "BATCH 1"

# Windows — renders and the table
python 06Visualization.py Targets/neglected_targets_batch1.csv
python GenerateThesisTables.py Targets/neglected_targets_batch1.csv > results_table.tex
```

Stage 01 is only needed to rebuild the target selection from scratch (§5.1).

---

## 5. Script reference

### 5.1 `01TargetSelection.py`: target selection

- **Purpose.** Finds "neglected" eclipsing binaries:
  - Kreiner (2004) catalogue, period < 1 d;
  - GCVS type containing EW or EB;
  - at least one TESS target-pixel file on MAST;
  - last VSX timing epoch ≥ 10 years old.
- **Run.** `python 01TargetSelection.py`. No arguments.
- **Writes**, into the current directory (not `Targets/`):
  - `neglected_targets.csv` (`name,type,period,sectors`, **without** `tic`);
  - `rejected_targets.csv`.
- **Network:** VizieR (Kreiner, GCVS), the AAVSO VSX API and MAST, with a short
  pause per star. Expect about an hour for the full catalogue. If VizieR is
  unreachable it falls back to a small demo list.
- **Traps:**
  - The output has no TIC column. The curated catalogues in `Targets/` have one;
    add TICs before using a fresh output.
  - The script runs on import (no `__main__` guard).
  - On Windows, set `PYTHONIOENCODING=utf-8` if you redirect its output to a file.

### 5.2 `02DataAcquisition.py`: TESS light curves

- **Purpose.** Downloads each star's light curve, trying in order: SPOC 2-min, then
  TESS-SPOC FFI (200/600/1800 s), then QLP. It searches by TIC when the catalogue
  has one.
  - **Every candidate sector is scored PDCSAP vs SAP.** PDC's systematics
    correction can absorb part of a large-amplitude star's own signal; then the
    eclipse depth changes from cycle to cycle in PDCSAP but not in SAP. A sector is
    *clean* when its PDCSAP cycle-amplitude scatter is ≤ 1.5× SAP's, or below
    0.5 %.
  - **Selection:**
    1. the newest clean sector at ≤ 200 s cadence;
    2. else the newest clean sector at ≤ 600 s;
    3. else SAP flux, crowding-corrected exactly as SPOC does, with slow trends
       removed by orbit-averaged means.
  - It rejects bogus sector IDs, baselines < 10 d and fewer than 500 points, and
    honours `skip_sectors`.
  - If the chosen sector has fewer than 5000 points, further clean sectors are
    stitched (sorted in time).
  - Finally a 4σ clip and normalisation.
- **Run.** `python 02DataAcquisition.py Targets/<catalogue>.csv`
- **Writes:**
  - `data/lightcurves/lc_<star>_clean.fits`, whose header records `FLUXSRC`
    (PDCSAP / SAP_CROWDCORR / QLP_SAP), `CROWDSAP`, `FLFRCSAP`, `PDCRATIO`,
    `SECTSEL` (the reason for the choice), `SECTOR` and `STACKED`;
  - `lc_<star>_folded.fits` and `cadence_<star>.txt` (diagnostic);
  - `data/plots/lightcurves/lc_<star>_raw_vs_clean.png`.
- **Network:** MAST. Files are cached in `~/.lightkurve/cache`, so a re-run is fast.
- **Traps:** It always overwrites. A new light curve means re-running 03 and 04 for
  that star.

### 5.3 `03OCAnalysis.py`: times of minimum and the period

- **Purpose.** Times every eclipse minimum with the Kwee & van Woerden (1956) method:
  - a fixed symmetric reflection range of ±0.06 P;
  - a parabola on centred times, its vertex taken as the minimum;
  - a parabola fallback when a window has a gap.
  Then it fits the linear ephemeris by weighted least squares:
  - epochs counted from the Kreiner (2004) baseline at **half-cycle** resolution;
  - primaries and secondaries both used, with a separate zero-point for
    secondaries;
  - errors scaled to the observed scatter.
  **Self-check:** if the refined period folds the data more than 1 % worse (by
  phase dispersion) than the Kreiner period, it reverts to Kreiner. A
  phase-dispersion period is computed as an independent cross-check.
- **Run.** `python 03OCAnalysis.py Targets/<catalogue>.csv`
- **Writes:**
  - `data/timing/ephem_<star>.txt` with `P_new`, `T0_new`, their errors,
    `PERIOD_REVERTED`, `SEC_OFFSET_days`, counts, `DISP_NEW`/`DISP_KREINER`,
    `P_best_pdm` and **`DRIFT_VS_PDM_cycles`**;
  - `tominima_<star>.csv` (with a P/S column);
  - `data/plots/oc/oc_<star>.png`.
- **Network:** one VizieR fetch at the start.
- **Check after every run:** `DRIFT_VS_PDM_cycles` (and `Audits/audit_period_fold.py`)
  must be **< 0.01 cycle**. One 151-bin phase bin is 0.0066 cycle. A larger drift
  smears the folded light curve and biases every fitted parameter. 03 prints a
  warning when it happens.

### 5.4 `04Modeling.py`: PHOEBE bundles

- **Purpose.** Builds one PHOEBE 2 bundle per star:
  - **Morphology** from `morphology.py` (§5.10). Contact uses the "double flip", so
    fill-out drives the Roche potential. Semi-detached uses PHOEBE's
    `semidetached` constraint: the secondary's radius equals its Roche lobe.
  - **Atmospheres and physics:** blackbody atmospheres; gravity darkening and
    reflection by model type (`gravdark.py`, §5.10): contact stars switch at
    8000 K, other stars follow a ramp over 6600–8000 K.
  - **Contact stars: the adopted mass ratio.** q, and the starting inclination
    and fill-out, come from `data/qsearch/q_adopted_<star>.txt` (§5.4b). Without
    that file the bundle gets the placeholder q = 0.5 and stage 05 parks the
    star.
  - **Data:** the light curve folded with stage 03's ephemeris and binned to **151**
    phase bins (odd, so one bin is centred on phase 0), with a 0.003 systematic
    error floor.
  - **Phase 0** = the deeper minimum, measured on the unbinned fold.
  - **Compute options:** irradiation stored in the compute options (contact Wilson,
    others Horvat), so the MCMC samples the same model it reports. For cadence
    ≥ 300 s, exposure-time integration (`exptime` + `fti_method="oversample"`,
    ×5).
  - **Fixed primary temperature:** Gaia DR3 where pre-resolved, else a live Gaia
    query, else a period–spectral-type estimate.
  - An initial forward model and plot as a sanity check.
- **Run (Linux/WSL).** `python3 04Modeling.py Targets/<catalogue>.csv`
- **Writes:** `data/bundles/bundle_<star>.phoebe` (overwrites) and
  `data/plots/models_init/phoebe_init_<star>.png`.
- **Traps:** Rebuilding a bundle changes its checksum, so stage 05 will supersede
  and refit that star (§6.6). Only rebuild the stars you mean to refit. Messages
  `Constraint 'requiv@…' raised` during the double flip are a known, harmless
  transient.

### 5.4b `ContactQSearch.py`: the mass ratio of contact stars

Single-band photometry of a partially eclipsing contact binary constrains q
poorly, and q fixed at 0.5 for every star biased the inclination by 5–16°. This
stage fixes q per star.

- **What it does.** For each contact star, Nelder–Mead fits of (T₂, i, fill-out)
  at fixed q on a grid (q = 0.08 … 12; q > 1 means W-type: star 1 is the hotter
  star), from two fill-out starts, on the star's own stage-04 bundle.
- **Adoption** (`adopt`): the published **spectroscopic** q from
  `Targets/contact_q_literature.csv` where there is one, on the better of the two
  branches q and 1/q; otherwise the q-search minimum.
- **How far to trust it.** Against 22 spectroscopic values the q-search minimum
  was within about 15 % for 13 stars and well off for 9. A finer mesh does not
  change that (Entry 94). Every q taken from the q-search is therefore marked
  weak (`q_weak_photometric`).
- **Run (Linux/WSL, heavy: about 36 fits per star, 15–30 min each).**
  ```bash
  bash Runs/qsearch/run_qsearch.sh Targets/<cat>.csv "coarse refine literature" <tag>
  python3 ContactQSearch.py adopt Targets/<cat>.csv
  ```
  Finished fits are skipped on a relaunch. Then re-run stage 04 for those stars.
- **Writes:** `data/qsearch/jobs/<star>/*.json`, `q_adopted_<star>.txt`,
  `qsearch_<star>.csv|png`, `Audits/contact_q_validation.csv`.
- **Traps.** PHOEBE cannot build the contact model near q = 10.0–10.35; fits that
  never left their start are dropped at adoption. `QSEARCH_NTRIANGLES=4000`
  repeats fits on a finer mesh into `data/qsearch/jobs_n4000/`.

### 5.5 `05Optimization.py`: posterior sampling

The expensive stage; see **§6** for operations. In short:

- **Nelder–Mead** pre-wash, then **emcee** with `MCMC_NWALKERS` walkers, in
  checkpointed chunks of `MCMC_CHECKPOINT` steps.
  - **Every star: multi-start.** Several starting points each get a short scout
    run (`NM_SCOUT_MAXITER` iterations); the best is polished with the full
    `NM_MAXITER`. The log shows one `[multi-start]` line per start.
  - **Contact starts:** `seed` (the q-search solution), `shallow` (fill-out 0.15,
    i 82°), `deep` (fill-out 0.70, i 70°).
  - **Mesh gate.** PHOEBE's surface mesh changes in steps when a parameter moves
    by a hair, and χ² jumps with it. After the scouts, the scatter of χ² about a
    smooth curve is measured at the best start (23 models). If it exceeds
    `MESH_NOISE_THRESHOLD` (2 = twice a 1σ step, after pre-scaling), the mesh is
    raised from 2000 to 3000 and then 4000 triangles; if none passes, the
    quietest is used and the star is flagged. A finer mesh costs 1.7–2.1× (3000)
    or 2.6–3.7× (4000) per model. Log lines start with `[mesh]`.
  - **Why several starts.** A single start (small hot star) trapped semi-detached
    fits in a wrong basin where q runs to its limit: RZ Dra, whose spectroscopic
    q is 0.40, railed at q = 1 from that start and lands on q ≈ 0.40–0.42 with
    the multi-start.
    - Semi-detached starts: `prod` (r1 = 0.17 a), `bigP` (r1 = 0.9 × its Roche
      lobe), `bigP_q0.4`, `bigP_q1.5`.
    - Detached starts: `prod` (r1 0.17 a, r2 0.24 a), `swap`, `bigP`, `bigS`.
    - Cost: about 1.6 extra pre-washes (tens of minutes; ~2–3 h for a 600 s star).
- **Convergence gate.** Stops early only when every parameter has chain length
  > 50 τ **and** post-burn-in R̂ < 1.1; otherwise it runs to `MCMC_MAX_NITERS`.
- **Burn-in** = max(`MCMC_BURNIN`, 2 τ_max), capped at half the chain.
- **Free parameters:**

  | Morphology | Free | Fixed |
  |---|---|---|
  | contact | T₂, i, fill-out (0.03–0.95) | q = the adopted value (§5.4b) |
  | detached | T₂, i, q (0.1–1), r₁, r₂ | — |
  | semi-detached | T₂, i, q (0.1–3), r₁ | r₂ = Roche lobe |

  For non-contact stars T₂ may range over 0.50–1.10 T₁ (contact: 0.80–1.25 T₁).
  A semi-detached q above 1 means the lobe-filling star is the more massive one;
  the result is flagged.

  T₁ is always fixed. Flux is scaled to the data at every step
  (`pblum_mode="dataset-scaled"`).
- **Surface physics.** Limb darkening: linear, 0.6 for every star. This is an
  effective value: it fits better than the Claret (2017) table values, linear or
  two-coefficient, on every star tested, and a freed coefficient lands at
  0.59–0.60 (Entry 93). Gravity darkening and albedo: contact stars 0.32 / 0.6
  below 8000 K; other stars a ramp over 6600–8000 K at each star's own
  temperature, with the secondary's values set from its fitted temperature before
  sampling. For non-contact stars the T₂ prior is no longer locked to one side of
  8000 K.
- **Third light.** Stars listed in `Targets/third_light.csv` get a third-light
  fraction, fitted (`free`) or held (`fixed`).
- **Uncertainties.** After the polish, one model gives χ²ν; the data
  uncertainties are multiplied by max(1, √χ²ν) **before** sampling
  (`SIGMA_PRESCALE`; tested: same medians, 2.5× the acceptance). The quoted
  errors are the 16/50/84 % posterior percentiles, times a residual
  `err_scale` (≈ 1), clipped at the prior bounds. χ²ν in the results is always
  against the original uncertainties.
- **Run (Linux/WSL, always under MPI).** Use the wrapper:
  `bash run_05_autorestart.sh Targets/<cat>.csv Runs/<log>.log "<label>"`.
  Directly: `mpirun --use-hwthread-cpus -n 8 python3 -u 05Optimization.py Targets/<cat>.csv`.

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `MCMC_NWALKERS` | 14 | Walkers. Useful MPI processes = walkers/2 + 1. |
| `MCMC_NITERS` | (unset) | If set: fixed-length run of this many steps; the gate is off. |
| `MCMC_MIN_NITERS` / `MCMC_MAX_NITERS` | 1500 / 6000 | Gate floor / ceiling. |
| `MCMC_CHECKPOINT` | 100 | Steps per chunk (= checkpoint interval). |
| `MCMC_BURNIN` | 350 | Burn-in floor. |
| `NM_MAXITER` | 200 | Nelder–Mead iteration cap of the polish run. |
| `NM_SCOUT_MAXITER` | 80 | Iterations of each multi-start scout run. |
| `SIGMA_PRESCALE` | 1 | 0 samples against the raw uncertainties (comparison only). |
| `MESH_NOISE_THRESHOLD` | 2 | Mesh gate limit; 0 turns the gate off (mesh stays at 2000). |
| `FILLOUT_FLOOR` | 0.03 | Lower prior bound on contact fill-out. |
| `ROCHE_LOBE_CAP` | 1.0 | Detached radius cap, as a fraction of the largest Roche lobe the q prior allows. |
| `PHOEBE_ENABLE_ONLINE_PASSBANDS` | FALSE | §2.3. |

Under `mpirun`, environment variables must be forwarded: `mpirun -x MCMC_NITERS …`.

### 5.6 `06Visualization.py`: mesh renders

- **Purpose.** Off-screen Mayavi renders coloured by effective temperature.
  - An *init* tier from `bundle_<star>.phoebe` (quick, no irradiation).
  - A *final* tier from `bundle_<star>_final.phoebe`: a still at quadrature plus a
    36-frame orbit GIF starting at the primary eclipse, using the same irradiation
    scheme as the fit.
  - A gallery once ≥ 10 final bundles exist.
- **Run (Windows).** `python 06Visualization.py Targets/<catalogue>.csv`
- **Writes:** `data/plots/mesh/mesh_init_<star>.png`, `mesh_final_<star>.png` / `.gif`,
  `data/plots/thesis/appendix_mesh_gallery.png` (also copied to `../Text/Figures/`).
- **Re-run behaviour:** it renders only images missing or older than their bundle.
  `FORCE_RERENDER=1` renders everything.

### 5.7 `GenerateThesisTables.py`: LaTeX results table

- **Run.** `python GenerateThesisTables.py [Targets/<catalogue>.csv] > table.tex`.
  With no argument it lists every `neglected_targets_batch<N>.csv` star (not the
  held batch).
- **Columns:** model (C contact / D detached / SD semi-detached), T₁ (fixed), T₂, i,
  q († = fixed), fill-out or r₁, r₂ (‡ = set by the Roche-lobe constraint), χ²ν,
  and **Notes**:
  - **a**: pre-audit result, kept;
  - **b**: stopped at the step ceiling;
  - **c**: railed at a prior bound;
  - **d**: blended with a variable neighbour;
  - **e**: SAP flux;
  - **f**: interval clipped at a bound.

  Missing results print *[Pending Optimization]*.
- **Trap:** the run-statistics summary block is printed to stdout **after**
  `\end{table}`. Delete it from the `.tex` file, or cut the output at
  `\end{table}`.

### 5.8 `SchematicRocheGeometry.py`

Illustrative (not fitted) Mayavi meshes of a generic detached and a generic contact
binary.
- **Run:** `python SchematicRocheGeometry.py` (Windows).
- **Writes:** `data/plots/thesis/roche_schematic_detached.png` and `_contact.png`.

### 5.9 `SchematicRocheEquipotentials.py`

Analytic Roche equipotentials, Lagrange points and critical surfaces for q = 0.5
(matplotlib only).
- **Run:** `python SchematicRocheEquipotentials.py`.
- **Writes:** `data/plots/thesis/roche_schematic_equipotentials.png`.

### 5.10 Shared modules (imported, never run)

- **`paths.py`**: every folder constant (§3.1) and `ensure_dirs()`.
- **`runstats.py`**: run timing, provenance and statistics (§8). Also provides
  `chain_acceptance()`. **Never** use PHOEBE's `acceptance_fractions`: it is wrong
  after a resumed chain.
- **`morphology.py`**: the **only** place morphology is decided. 04, 05 and 06 import
  it, so a star is built, fitted and rendered with the same geometry.

  | GCVS physical class | Model |
  |---|---|
  | `KE`, `KW`, `K` | contact |
  | `SD` | semi-detached |
  | `DW`, `DM`, `DS`, `D` | detached |
  | none: `EW` | contact |
  | none: `EA`, `EB` | detached |

  Plain `EB` (β Lyrae type, no physical class) is often really semi-detached:
  V358 Cep and OT Cep both turned out lobe-filling. Stage 05's post-NM check parks
  such a star before any MCMC (§6.11).

  `MORPHOLOGY_OVERRIDE` sets a star explicitly. **Workflow to change a star's
  model:**
  1. edit the override;
  2. re-run 04 for that star;
  3. re-run 05; it notices the new bundle and refits (§6.6).
- **`earlywarn.py`**: Stage 05's early-warning checks (preflight, post-NM, per
  chunk) and the parking files (§6.11). Pure Python/numpy, so its decisions can be
  tested on saved chains without PHOEBE or MPI.
- **`limbdark.py`**: the Claret (2017) TESS limb-darkening tables, linear and
  two-coefficient, by temperature. Used by the test tools only: the pipeline keeps
  the single linear coefficient 0.6 (§5.5).
- **`gravdark.py`**: gravity-darkening exponent and albedo by temperature and
  model type. `rule_for(kind)` is the pipeline's rule (contact: 8000 K step;
  others: ramp over 6600–8000 K); 04 and 05 both call `coeffs_for`.

---

## 6. Running Script 05 (the MCMC stage)

### 6.1 Always use the wrapper

`run_05_autorestart.sh` runs 05 under `mpirun` and:
- appends to the log after a marker line
  (`########## <label> — <date> ##########`);
- **restarts automatically** if 05 dies (up to `MAX_RESTARTS`, default 3), e.g.
  PHOEBE's occasional segfault (§11);
- refuses to start a second copy by accident.

```bash
cd <path-to>/Code
bash run_05_autorestart.sh Targets/<catalogue>.csv Runs/<name>.log "<label>"
```

| Variable | Default | Meaning |
|---|---|---|
| `NPROCS` | 8 | MPI processes per queue (§6.9). |
| `MAX_RESTARTS` | 3 | Automatic restarts before giving up. |
| `PAUSE_S` | 60 | Wait before a restart (Ctrl-C cancels). |
| `ALLOW_CONCURRENT` | 0 | Set to 1 to start a **second** queue while another runs (§6.5). |
| `RUN_CMD` | — | Replaces the mpirun command; for tests only (§6.8). |

The wrapper does **not** forward `MCMC_*` variables to `mpirun`. For non-default
settings, put the full `mpirun -x …` command in `RUN_CMD` (§6.8).

### 6.2 Start, pause, continue

| Action | Do this |
|---|---|
| **Start** | the wrapper command above, in a terminal you keep open |
| **Pause** | **Ctrl-C** in that terminal. The wrapper stops and does not restart. |
| **Continue** | run **the same command** again |

- **What a pause costs:** only the chunk in progress. Checkpoints are written every
  `MCMC_CHECKPOINT` steps (100). A chunk takes ~20 min for a fast contact star and
  up to ~4 h for a 600 s-cadence star with exposure integration.
- **When to pause:** just after a `[checkpoint] … saved` line.
  ```bash
  grep "\[checkpoint\]" Runs/<name>.log | tail -1
  ```
- **What continuing does:** finished stars are skipped, an interrupted star resumes
  from its last checkpoint, and locks left by the stopped run are taken over
  automatically.

### 6.3 Power cut, closed terminal, crash, reboot

Nothing is lost beyond the chunk in progress:
- checkpoints are saved atomically (a temporary file renamed into place), so a
  power cut cannot corrupt one;
- finished results are kept;
- dead processes' locks are taken over.

To recover:

```bash
cd <path-to>/Code
pgrep -fa "python3 -u 05Optimization.py" | wc -l     # must print 0
# if not 0:  pkill -f 05Optimization.py              (stops ALL queues)
bash run_05_autorestart.sh Targets/<catalogue>.csv Runs/<name>.log "<label>"
```

If the wrapper itself gave up after `MAX_RESTARTS` crashes, its log ends with
`restart limit (3) reached`. Run the same command again.

### 6.4 Surviving a closed window: `tmux`

```bash
tmux new -s A        # then run the wrapper inside it
# detach: Ctrl-b then d      re-attach: tmux attach -t A
```

`tmux` survives closing the window, not a reboot or power cut (for those, §6.3).

### 6.5 Two queues at once

One star can use at most 8 processes (§6.9). On a 16-thread machine a **second
queue** fits beside the first. Both may read the **same catalogue**: each star is
claimed by a lock, so the queues never fit the same star.

```bash
# terminal 1
bash run_05_autorestart.sh Targets/<catalogue>.csv Runs/<name>_A.log "<label> A"
# terminal 2 — whichever queue starts second needs ALLOW_CONCURRENT=1
ALLOW_CONCURRENT=1 bash run_05_autorestart.sh Targets/<catalogue>.csv Runs/<name>_B.log "<label> B"
```

Rules:
- one log per queue (enforced);
- stop the second queue with Ctrl-C; its star resumes later from its checkpoint;
- the gain is ~1.1–1.4× (§6.9), and the machine is fully loaded. Run a second queue
  only when the computer is not otherwise in use.

### 6.6 Which stars are skipped, refitted or resumed

For each star in the catalogue, in order:

1. **Kept list** (`KEEP_PRE_AUDIT_RESULTS` in 05): if the star is listed and has
   results, it is skipped. These results are deliberately retained and documented;
   their caveats are in `Audits/kept_results_caveats.csv`.
2. **Lock:** if another live queue holds `bundle_<star>.lock`, it is skipped. A lock
   whose process is dead is taken over.
3. **Parked:** if `data/results/parked_<star>.txt` exists for the **current**
   bundle, it is skipped (§6.11). If the bundle was rebuilt since, the park is
   renamed `…_resolved_<date>.txt` and the star is fitted. `IGNORE_PARK` overrides.
4. **Current results:** if `results_<star>.csv` exists **and** its
   `input_bundle_sha` equals the checksum of the current `bundle_<star>.phoebe`, it
   is skipped (done).
5. **Outdated results** (older bundle, or no checksum row): the old results, final
   bundle and final plot are renamed `…_superseded_<date>…` (**never deleted**) and
   the star is refitted.
6. **Checkpoint:** it is resumed only if its sidecar
   `bundle_<star>_mcmc.phoebe.input_sha` matches the current bundle. Otherwise it is
   set aside as `…_mcmc_stale_<date>.phoebe` and the star starts fresh.

So **rebuilding a bundle with 04 is enough to make the next 05 run refit that
star**, and to clear its park. Do not rebuild bundles you do not want refitted.

### 6.7 Checking progress

The quickest overview is the **health board** (read-only, seconds, safe beside a
live run): one row per catalogue star with its state (done / kept /
running(A|B) / paused / PARKED / pending), steps, s/step, acceptance, failure rate
and any early-warning codes. It exits with 1 if a star is parked.

```bash
python Audits/run_health.py Targets/<catalogue>.csv
```

Details from the logs:

```bash
grep "\[health\]" Runs/<name>.log | tail -1       # live walkers, overflow, lobe, OK/WARN/PARK
grep "\[eta\]" Runs/<name>.log | tail -1          # steps done, s/step, ETA
grep "\[checkpoint\]" Runs/<name>.log | tail -1   # last save
pgrep -fa "python3 -u 05Optimization.py" | wc -l  # 8 per running queue
tail -3 Runs/run_stats/05Optimization_chunks.csv  # one row per chunk
```

Read only the current run's part of a log: `awk '/########## <label>/{f=1} f' Runs/<name>.log`.

Two progress bars appear: **Nelder–Mead** (`/200`) and then **MCMC** (`/101` per
chunk). Do not estimate the MCMC time from the Nelder–Mead bar.

### 6.8 Trials: always on a scratch star

**Never run 05 against a real star just to test something.** It would supersede
that star's results. Copy the bundle under a throwaway name instead:

```bash
cp data/bundles/bundle_V899_Her.phoebe          data/bundles/bundle_ZZ_Trial.phoebe
cp data/lightcurves/lc_V899_Her_clean.fits      data/lightcurves/lc_ZZ_Trial_clean.fits
printf 'name,type,period,sectors,tic\nZZ Trial,EW,0.4211676,52,8673277\n' > Runs/trial_scratch.csv

export MCMC_NITERS=50 MCMC_CHECKPOINT=25 MCMC_BURNIN=10 NM_MAXITER=20
mpirun --use-hwthread-cpus -n 8 -x MCMC_NITERS -x MCMC_CHECKPOINT -x MCMC_BURNIN -x NM_MAXITER \
       python3 -u 05Optimization.py Runs/trial_scratch.csv

# afterwards, move or delete everything named *ZZ_Trial*
```

- Choose the `type` so that the scratch name resolves to the **same morphology** as
  the source star (e.g. `EB/SD` for a semi-detached one). Overrides in
  `morphology.py` do not carry over to a new name.
- With the wrapper, pass the same command through `RUN_CMD="mpirun … -x …"`.

**Smoke test after every code change.** `smoke_05.sh` does all of the above in one
step for the scratch star `ZZ Smoke`. It copies the bundle, runs NM 20 + 30 MCMC
steps, and prints the preflight / post-NM / `[health]` / park lines. It is required
after any change to `05Optimization.py`, `earlywarn.py` or `morphology.py`, before
a queue is relaunched. It costs about 20–40 min at `-n 8` beside a live queue.

```bash
bash smoke_05.sh <bundle.phoebe> <gcvs_type> [label] [scratch name]
bash smoke_05.sh data/bundles/bundle_V358_Cep.phoebe EB/SD sd-check
IGNORE_PARK= bash smoke_05.sh <bundle> EB park-check   # let a park happen (default overrides it)
Q_FROM="TY UMa" bash smoke_05.sh data/bundles/bundle_TY_UMa.phoebe EW ty "ZZ Smoke T"
SCRIPT=05Optimization_staged.py bash smoke_05.sh <bundle> <type>   # test a staged copy
```

A **contact** scratch star needs an adopted q: `Q_FROM="<star>"` copies that
star's `q_adopted` file to the scratch name. Several smokes can run side by side
if each has its own scratch name (it must start with `ZZ `).

The previous `ZZ_Smoke` products are moved into `Runs/smoke/` rather than deleted.

### 6.9 Throughput facts (measured on an 8-core / 16-thread desktop)

- **One star uses at most 8 MPI processes.** emcee's stretch move evaluates half the
  walkers at a time: 7 models for 14 walkers, plus the master process. Extra
  processes sit idle **and busy-spin**, slowing the working ones.
- **Same contact star, same settings:**

  | Processes | s/step |
  |---|---|
  | `-n 14` | 20.0 |
  | `-n 8` | 13.6 |

- **Two queues at `-n 8`** slow each star down. The throughput gain over one `-n 8`
  queue was measured twice:
  - contact scratch star (Phase D): 1.1–1.2×;
  - V358 Cep, 600 s cadence (Phase E): 103 s/step alone vs 144 s/step beside a
    second queue, i.e. ~1.4×.

  The machine is fully loaded either way.
- **Exposure integration ×5** (600 s cadence) costs about 3.3× per step. Such a
  star needs ~1 week at the 6000-step ceiling.
- **Typical star lengths:** contact stars take hours to ~1 day; detached and
  semi-detached stars take ~1–3 days.

### 6.10 Files 05 writes

| File | Meaning |
|---|---|
| `data/results/results_<star>.csv` | Parameters (`Parameter,Symbol,Value,Sigma_minus,Sigma_plus,Unit`) plus rows `chi2_red`, `chi2_red_map`, `err_scale`, `morphology*`, `run_id`, `code_sha`, `input_bundle_sha`, `steps`, `burn_in`, `stop_reason`, `flux_source`, `flags`. |
| `data/bundles/bundle_<star>_final.phoebe` | The fitted bundle with the full chain. Written **before** the results file. |
| `data/plots/models_final/phoebe_final_<star>.png` | Data, model, residuals. |
| `data/plots/mcmc_diag/mcmc_chains_<star>.png` | Walker traces and chain-health verdict. |
| `bundle_<star>_mcmc.phoebe` + `.input_sha`, `.nm_start`, `.sigma_prescale`, `.mesh` | Checkpoint and its notes: input checksum, winning start, error-bar factor, mesh chosen (all deleted on success). |
| `bundle_<star>.lock` | `host:pid` of the queue fitting the star. |
| `data/results/parked_<star>.txt` | Why the star was parked, values at that point, the fix, how to un-park (§6.11). |
| `Runs/run_stats/05Optimization_health.csv` | One row per early-warning event (WARN / PARK / OVERRIDDEN). |
| `…_superseded_<date>…`, `…_mcmc_stale_<date>.phoebe`, `parked_…_resolved_<date>.txt` | Retired versions (§6.6). Never deleted automatically. |

`flags` lists:
- `<parameter>@low` / `@high` (railed at a prior bound: report as a limit);
- `interval_clipped_at_bound`;
- `r2_constrained_to_roche_lobe`;
- `blended_with_variable_neighbour`;
- `SAP_flux_PDC_distorted`;
- `dead_walkers_excluded:w0,…` (walkers stuck at an invalid model, left out of
  τ, R̂ and the posterior);
- `park_overridden` (an early-warning park was overridden with `IGNORE_PARK`);
- `q_gt_1_lobe_filler_more_massive` (semi-detached fit with q > 1);
- `primary_near_roche_lobe:<fraction>` (non-contact: the hot star is within 3 %
  of its Roche lobe, i.e. near-contact).

- `q_from_literature` / `q_from_qsearch`, `q_weak_photometric` (contact: where
  the fixed q came from; every q-search value is weak);
- `third_light_fitted` / `third_light_fixed`;
- `mesh_raised:<triangles>` (the mesh gate chose a finer mesh);
- `mesh_noise_high:<value>` (no mesh brought the noise under the limit).

Further rows:
- `nm_start`, `nm_scouts`: which Nelder–Mead start won and the log-probability
  each scout reached;
- `q_source`, `q_reference`, `q_search_min`, `q_search_interval` (contact);
- `sigma_prescale`, `sigma_total_scale`: the factor applied to the uncertainties
  before sampling, and the total inflation;
- `gravb_bol_<component>`, `albedo_<component>`, `gd_rule`;
- `ntriangles`, `mesh_noise_scaled`, `mesh_noise_history`;
- `l3_frac` for third-light stars.

### 6.11 Early warning and parked stars

05 checks every star three times, so that a wrong setup costs at most one
Nelder–Mead run or one chunk, never days. A **PARK** finding stops only that star:
- 05 writes `data/results/parked_<star>.txt`;
- any checkpoint is kept;
- the queue moves on to the next star.

A **WARN** finding is printed and logged but changes nothing.

| When | Cost | PARK if | WARN if |
|---|---|---|---|
| **Preflight**, before NM; the file and geometry part also on resume | seconds | bundle geometry ≠ routing (`morphology.py` changed, 04 not re-run); light curve or ephemeris newer than the bundle; bundle period ≠ Script 03 `P_new`; times not increasing; NaN data or σ ≤ 0; exposure ≥ 300 s without `fti_method = oversample`; wrong irradiation method | — |
| **Post-NM**, before the emcee ball | none | **detached** star with r / `requiv_max` ≥ 0.97 after Nelder–Mead (a lobe-filling secondary: likely semi-detached; a lobe-filling primary: needs a decision) | an NM value within 2 % of a prior bound; **semi-detached** primary ≥ 0.97 of its lobe (near-contact; RZ Dra's correct solution has 0.98) |
| **Each chunk** (`[health]` line) | none | fewer than 7 live walkers; **detached**: Roche-overflow failures ≥ 25 % of the chunk's evaluations, or a median walker ≥ 0.985 of its Eggleton lobe | walkers at −∞ (but ≥ 7 live); a walker frozen for the whole chunk; acceptance < 0.05; a walker median at a prior bound; Roche-overflow failures ≥ 25 % for a semi-detached or contact star |

The thresholds and their calibration are in `earlywarn.py`. They are set so that
both known lobe-filling cases (V358 Cep and OT Cep, run detached) park, while
semi-detached and contact stars and the 38-star queue's preflight do not.

**When a star is parked:**
1. read `data/results/parked_<star>.txt`: the reason, the parameter values and the
   suggested fix;
2. usually: add the star to `MORPHOLOGY_OVERRIDE`, re-run 04 for it (one-star
   catalogue), and relaunch. The rebuilt bundle clears the park automatically;
3. if it is a false positive: run it anyway with `IGNORE_PARK="<star>"`. Forward
   it under `mpirun` with `-x IGNORE_PARK`, e.g. through the wrapper's `RUN_CMD`.
   Its results then carry `park_overridden`. A park at a chunk resumes from the
   kept checkpoint.

---

## 7. Audits

All audits run from `Code/` and write their CSV next to themselves in `Audits/`.

### 7.1 Data-quality chain (light; run in this order)

| Script | Checks | Run |
|---|---|---|
| `audit_period_fold.py` | Does `P_new` fold the TESS data? Compares `P_new`, Kreiner and a phase-dispersion `P_best`; reports drift, per-segment minimum phases and the **amplitude lost** to smearing. **Pass: drift < 0.01 cycle.** | `python Audits/audit_period_fold.py ["Star A" "Star B"]` (names: also write fold figures to `data/plots/audit_period/`) |
| `audit_tom_estimator.py` | Replica of the historical (defective) timing code vs the corrected estimator. Needs `period_fold_audit.csv`. With star names, the CSV is rewritten with those stars only. | `python Audits/audit_tom_estimator.py ["Star"]` |
| `audit_crowding_pdc.py` | Per cached sector: contamination (a periodic signal left after removing the fold, matched against other stars' periods) and PDCSAP distortion (PDC vs SAP cycle-amplitude scatter). Reads the lightkurve cache only; never downloads. Needs `period_fold_audit.csv`. | `python Audits/audit_crowding_pdc.py` |
| `fetch_alt_sectors.py` | Lists, or with `--download` downloads into the cache, a star's other SPOC/TESS-SPOC sectors, so the PDC audit can score them. | `python Audits/fetch_alt_sectors.py [--download] "Star"` |
| `audit_results_ledger.py` | One row per result: code era, stop reason, railing, input defects, keep/re-run decision. Writes `results_ledger.csv` and `kept_results_caveats.csv`. | `python Audits/audit_results_ledger.py` |

### 7.2 Chain and model audits

- **`audit_mcmc_chains.py`** checks per-walker chain health. Verdicts:
  `DEAD_WALKER`, `ACCEPTANCE_COLLAPSE`, `MULTIMODAL`, `BURNIN_MASKED`.
  ```bash
  python Audits/audit_mcmc_chains.py Targets/neglected_targets_batch*.csv [--per-walker] [--plot | --plot-all] [--burnin N] [--skip-missing]
  ```
  It loads every final bundle (moderate cost) and writes `mcmc_chain_audit.csv` and
  optionally `mcmc_chain_walkers.csv`. Stage 05 already runs the same check per
  star.
- **`run_health.py <catalogue.csv> [--all-warnings]`** is the run health board
  (§6.7). It is read-only and light, so it can run at any time.
- **`audit_bundle_phase_alignment.py <catalogue.csv> …`** checks that each bundle's
  own period and t0 put the primary eclipse at phase 0. Prints verdicts.
- **`audit_fillout_identifiability.py ["Star" …]`** profiles χ² against the contact
  fill-out with ~40 forward models per star. This is **real compute: do not run it
  beside a live MCMC.**
- **`audit_model_tests.py`** fits one star under one chosen setting, so two fits
  that differ in a single setting can be compared: geometry (`contact`, `sd2`,
  `sd1`, `detached`), limb darkening (`--ld legacy|table|quad|free`), gravity
  darkening (`--gd step8000|ramp|step6600`), third light, one spot. `--check`
  proves the rebuilt bundle equals the stage-04 bundle; `--start-from` continues
  an earlier fit; `--skip-existing` resumes a batch. About 150–500 models per
  fit: **heavy**.
  ```bash
  python3 Audits/audit_model_tests.py fit --star "RZ Dra" --geometry sd2 --free-q --gd ramp --label test
  python3 Audits/audit_model_tests.py collect --label test
  ```
- **`audit_mesh_noise.py`** measures the model's numerical noise for a star (the
  same measurement as stage 05's mesh gate): `probe --star S --kind contact
  [--ntriangles N]`, then `collect` → `Audits/mesh_noise.csv`. 23 models per
  probe.
- **`audit_oconnell.py`** measures the difference between the two light maxima,
  cycle by cycle, on the clean light curves → `Audits/oconnell.csv`. Light.

### 7.3 Legacy audits

`audit_lc.py`, `audit_lc_benchmark.py`, `audit_oc.py`, `audit_oc_benchmark.py`,
`audit_modeling.py`, `audit_modeling_benchmark.py` and `audit_phase_offset.py`
predate the current folder layout. They hard-code early test stars and look for
files in the current directory, so run as-is they find nothing. Their functions can
still be imported and pointed at the right folder, e.g.
`audit_oc_outputs("<star>", data_dir=paths.TIMING_DIR)`. `audit_phase_offset.py`
runs on import and is superseded by `audit_bundle_phase_alignment.py`.

---

## 8. Run statistics and provenance

Every script records itself in `Runs/run_stats/` (via `runstats.py`):

| File | Contents |
|---|---|
| `runs.csv` | One row per invocation, written at **start**: `run_id`, script, catalogue, host, Python, `code_sha` (hash of the code files), command, settings (05: walkers, gate, priors, PHOEBE/emcee versions, process count). |
| `<script>_stars.csv` | One row per star: status, wall time, plus script-specific columns (05: steps, s/step, stop reason, χ²ν, acceptance, failed models). |
| `05Optimization_chunks.csv` | One row per 05 chunk: step range, s/step, acceptance (from the chain), failed models, ETA. |
| `05Optimization_health.csv` | One row per early-warning event: time, `run_id`, star, stage (`preflight`, `post-NM`, `chunk <step>`), level (WARN / PARK / OVERRIDDEN), code, detail (§6.11). |
| `run_summary.txt` | A summary block per run (also printed at exit). |

A results file's `run_id` and `code_sha` join to `runs.csv`, which answers "which
code and settings produced this number". If a script's columns change, a new
`<script>_stars_v2.csv` is started rather than corrupting the old file.

---

## 9. Publishing and the published layout

### 9.1 The sync tool

The public repository is a separate git repo, updated **only** through:

```bash
python neglected-eclipsing-binaries/tools/sync_from_working_tree.py
```

- It copies an **explicit** list of scripts plus `Audits/*.py`, `Notes/*.txt` and the
  target catalogues.
- A guard refuses the private running log, any `*.phoebe`, `*.fits` and
  `results_*` file.
- A new top-level module must be added to its list, or the published code fails at
  import.
- Review `git status` / `git diff` in the repo before committing.

### 9.2 Name mapping

| Working tree | Published repository |
|---|---|
| `01TargetSelection.py` … `06Visualization.py` | `src/TargetSelection.py` … `src/Visualization.py` |
| `GenerateThesisTables.py`, `Schematic*.py`, `paths.py`, `runstats.py`, `morphology.py`, `earlywarn.py`, `limbdark.py`, `gravdark.py`, `ContactQSearch.py` | `src/` (same names) |
| `run_05_autorestart.sh`, `smoke_05.sh` | `src/` (both call `Optimization.py`) |
| `Audits/*.py` | `audits/*.py` |
| `Notes/*.txt` | `docs/*.txt` |
| `Targets/neglected_targets*.csv`, `triage_*.csv`, `contact_q_literature.csv`, `third_light.csv` | `data/targets/` |
| `HOW_TO.md` | `HOW_TO.md` |

### 9.3 Running from a clone

The scripts expect `Targets/` and `data/` in the **current directory** (§3.1).
Create a working folder:

```bash
git clone <repository-url> repo && mkdir run && cd run
cp ../repo/src/*.py ../repo/src/run_05_autorestart.sh .
mkdir -p Audits Targets && cp ../repo/audits/*.py Audits/ && cp ../repo/data/targets/*.csv Targets/
python DataAcquisition.py Targets/neglected_targets_batch1.csv        # and so on
```

In a clone, substitute the published names in every command of this manual
(`05Optimization.py` → `Optimization.py`, etc.). Light curves, bundles and results
are not published. Stage 02 regenerates the light curves and 03–05 regenerate the
rest.

---

## 10. Safety rules (checklist)

- [ ] **Run from `Code/`** and **always pass a catalogue.**
- [ ] Run 05 **only under `mpirun`**, in Linux/WSL, preferably via the wrapper.
      Bare `python` can trigger an exponential process storm on Windows; the
      script has no `__main__` guard.
- [ ] **Never test 05 on a real star**; use a scratch star (§6.8). Never use
      `Targets/trial_*.csv`.
- [ ] Rebuild with 04 **only** the bundles you want refitted (§6.6).
- [ ] **Never launch between a code change and the rebuild it needs.** A new 05
      against old bundles fits the wrong model. (The preflight now parks a
      morphology mismatch or a stale bundle, but do not rely on it.)
- [ ] After any change to `05Optimization.py`, `earlywarn.py`, `morphology.py` or
      `gravdark.py`: a **smoke pass** (`smoke_05.sh`, §6.8) before relaunching a
      queue. A file that only compiles has not been tested: the first run of a
      staged 05 crashed at import.
- [ ] **Contact stars need an adopted q** (§5.4b) and a bundle rebuilt with it.
- [ ] Do not edit `ContactQSearch.py` while its jobs run.
- [ ] **Pause Windows and Microsoft Store updates** before a campaign: a WSL
      self-update kills every run in it (§2.4).
- [ ] Check the health board (`python Audits/run_health.py Targets/<catalogue>.csv`)
      when you check in. A parked star waits for you; the queue does not.
- [ ] After any change to 03, run `Audits/audit_period_fold.py` (drift < 0.01).
- [ ] Keep the launching terminal open (or use tmux). Disable Windows sleep and
      update restarts during runs.
- [ ] One log per queue. A second queue needs `ALLOW_CONCURRENT=1`. Never run an
      old copy of 05 (without per-star locks) beside a queue.
- [ ] Never put a launch behind `&&`. A precondition such as `grep -c` exiting 1
      silently cancels it.
- [ ] Back up products inside the project (e.g. `Runs/<purpose>/`), never in a
      temporary folder.
- [ ] Never import `05Optimization.py` from another script; it would run the
      pipeline.
- [ ] Publish only through the sync tool, and review the diff first.

---

## 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `05Optimization.py is already running; not starting a second copy.` | A queue is running. Intended second queue: add `ALLOW_CONCURRENT=1`. Otherwise check `pgrep -fa 05Optimization.py`. |
| `Another wrapper is already writing to <log>` | Use a different log file for each queue. |
| `[skip] <star> is being fitted by another queue (host:pid)` | Normal with two queues. If no other queue runs, the lock is stale and is taken over on the next pass. |
| `Segmentation fault` inside `solve_radiosity_equation_Horvat_nbody` | A rare crash in PHOEBE's reflection solver, seen near the Roche lobe. The wrapper restarts from the last checkpoint. |
| A burst of `target_volume … slightly over critical value, likely due to numerics` | Semi-detached stars: rounding puts the lobe-filling volume ~1e-16 over the limit and PHOEBE clamps it. Harmless; shown once per process, then suppressed. |
| Many `secondary is overflowing at periastron` / `lnprobability=-inf` | Proposals beyond the Roche lobe are rejected. A few are normal near the lobe. If they dominate a **detached** star, the star wants to be semi-detached; the chunk check parks it (§6.11). |
| `[parked] <star> at <stage>: <codes> — see …parked_<star>.txt` | An early-warning check stopped that star; the queue went on. Follow the fix in the file (§6.11). |
| `[parked] … was parked … for this bundle — skipped` | The star is still parked; rebuild its bundle or set `IGNORE_PARK`. |
| `[health] chunk check unavailable: …` | The per-chunk check itself failed (it never stops the run). Report it; sampling continues normally. |
| `Nelder-Mead did NOT converge (Maximum number of iterations…)` | The pre-fit reached `NM_MAXITER`. The MCMC continues from there; usually harmless. |
| `[refit] … was fitted from an unrecorded bundle` | Expected after 04 rebuilt the bundle (§6.6). Old files are kept as `_superseded_<date>`. |
| `[resume] checkpoint was started from … set aside` | The bundle changed since the checkpoint; the star restarts fresh. |
| A run dies when a terminal or WSL session closes | `mpirun` aborts on hang-up. Keep the terminal open or use tmux (§6.4); then §6.3. |
| Output redirected on Windows shows `UnicodeEncodeError` | Run with `PYTHONIOENCODING=utf-8` (or `python -X utf8`). |
| Very slow first steps on an offline host | Passband checks. Ensure `PHOEBE_ENABLE_ONLINE_PASSBANDS=FALSE` (§2.3). |
