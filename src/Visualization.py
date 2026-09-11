"""
================================================================================
SCRIPT 06 — THREE-DIMENSIONAL MESH VISUALISATION (Mayavi)
================================================================================
Summary:
    Renders the PHOEBE 2 triangular surface mesh of each target system as a
    three-dimensional scene: a static image for every star that has a bundle,
    and, for systems with a converged posterior solution, an animated GIF
    spanning one full orbital cycle. Every triangle is colour-mapped by its
    local effective temperature, so the Roche geometry and the gravity
    darkening implied by the fit are directly legible.

    Two-tier rendering policy:

        "init" tier (bundle_<star>.phoebe, from Script 04 — seed geometry,
        not yet optimised): one low-resolution static image, computed with
        irrad_method="none" to match Script 04's own initial forward model.
        Both filename and figure title are marked "preliminary".

        "final" tier (bundle_<star>_final.phoebe, from Script 05 — converged
        MCMC solution): a print-resolution static image AND an orbital
        animation, computed with irrad_method="wilson" (contact) or "horvat"
        (detached) and distortion_method="roche", matching Script 05's own
        final compute. Animations are restricted to this tier because a seed
        geometry is superseded the moment Script 05 converges it, making the
        frame sequence wasted work.

    Compute optimisation:
        The bundles carry an enabled lc01 dataset under the shared "phoebe01"
        compute label. Adding a mesh dataset and calling run_compute() without
        further action therefore also forward-models the full light curve at
        every mesh time, which this script never reads. Disabling the non-mesh
        datasets in memory first reduces the cost from 12.5 s to 1.2 s per mesh
        time (measured, bundle_VW_Cep_final.phoebe).

    Orbital period:
        The animation spans the period stored in the bundle (the refined value
        from Script 03), not the catalogue period, so one loop corresponds to
        exactly one cycle of the model being drawn.

    Idempotency:
        Each output is compared against the modification time of its source
        bundle rather than merely tested for existence, so a star whose bundle
        is regenerated is re-rendered on the next run while completed stars are
        skipped. FORCE_RERENDER=1 overrides the staleness test.

    Bundles are treated as read-only: each is loaded, given an in-memory mesh
    dataset, computed, read, and discarded. This script never calls save() on
    a pipeline bundle.

Inputs:
    neglected_targets.csv      — Target list from Script 01.
    bundle_<star>.phoebe       — Initialised bundles from Script 04.
    bundle_<star>_final.phoebe — Converged bundles from Script 05 (optional).

Outputs:
    plots/mesh_init_<star>.png      — Preliminary render (stars without a solution).
    plots/mesh_final_<star>.png     — Print-resolution render (converged stars).
    plots/mesh_final_<star>.gif     — One-cycle orbital animation (converged stars).
    plots/appendix_mesh_gallery.png — Contact sheet of all converged stars, emitted
                                      once FINAL_STAR_GALLERY_THRESHOLD is reached
                                      and also copied to Text/Figures/ if reachable.

Dependencies:
    pip install phoebe mayavi vtk pyqt5 pillow numpy

References:
    Ramachandran & Varoquaux (2011), Comput. Sci. Eng. 13, 40 — Mayavi.
    Prša & Zwitter (2005), ApJ 628, 426 — PHOEBE computational framework.
    Conroy et al. (2020), ApJS 250, 34 — PHOEBE 2 inverse-problem interface.
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import csv
import gc
import os
import sys
import warnings
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8")

# Force PHOEBE to use only locally-installed passbands; see Script 05 for the
# full rationale (online lookups block for minutes on a firewalled host).
# MUST be set before `import phoebe`.
os.environ.setdefault("PHOEBE_ENABLE_ONLINE_PASSBANDS", "FALSE")

# ── Scientific libraries ──────────────────────────────────────────────────────
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mayavi import mlab

# Render through the offscreen buffer: the batch runs headless and must never
# open an interactive window. Set before any figure is created.
mlab.options.offscreen = True

import phoebe

warnings.filterwarnings("ignore")
phoebe.logger(clevel="WARNING")


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

import paths
paths.ensure_dirs()

# Curated figures for the thesis text live here. This script writes only the
# generated contact sheet into it; per-star renders are promoted by hand.
# Path defined in paths.py; resolved relative to the run directory (Code/).
THESIS_FIGURES_DIR: str = paths.THESIS_FIGURES_DIR

# Frame count for one orbital cycle. Thirty-six frames at 70 ms gives a ~2.5 s
# loop, smooth enough at figure scale without inflating render time or GIF size.
N_FRAMES: int = 36
GIF_FRAME_DURATION_MS: int = 70
GIF_LOOP: int = 0                      # 0 = loop indefinitely

# Preliminary renders and animation frames use the small canvas; the
# final-tier stills are the thesis-candidate images and use the large one.
LORES_SIZE: tuple[int, int] = (700, 500)
HIRES_SIZE: tuple[int, int] = (1600, 1200)

# White canvas with black annotation, chosen for print reproduction rather than
# Mayavi's default black scene.
FIGURE_BACKGROUND: tuple[float, float, float] = (1.0, 1.0, 1.0)
FIGURE_FOREGROUND: tuple[float, float, float] = (0.0, 0.0, 0.0)

MESH_COLORMAP: str = "inferno"
CAMERA_AZIMUTH: float = 60.0
CAMERA_ELEVATION: float = 70.0

# Labels used for the mesh dataset and model attached in memory. Distinct from
# the "lc01"/"init_model"/"solution_model" labels written by Scripts 04 and 05
# so that a stray save could never collide with pipeline state.
MESH_DATASET: str = "mesh01"
MESH_MODEL: str = "vizmesh"
COMPUTE_LABEL: str = "phoebe01"

# The contact sheet is emitted once this many stars have converged. Counted
# from the bundles present on disk, so it grows as batches 3-8 complete.
FINAL_STAR_GALLERY_THRESHOLD: int = 10
GALLERY_COLUMNS: int = 4
GALLERY_THUMB_SIZE: tuple[int, int] = (360, 270)
GALLERY_CAPTION_HEIGHT: int = 28

FORCE_RERENDER: bool = os.environ.get("FORCE_RERENDER", "0") == "1"


# =============================================================================
# SECTION 2 — HELPER FUNCTIONS
# =============================================================================

def is_contact_morphology(gcvs_type: str) -> bool:
    """
    Decide whether a GCVS variability type denotes a contact (overcontact)
    binary. Qualifier suffixes after a slash or colon are stripped and only
    the leading code is tested, so "EW/KW", "EW/DW" and "EW:" all classify as
    contact. Identical to the classifier in Scripts 04 and 05.
    """
    leading_code = gcvs_type.split("/")[0].split(":")[0].strip().upper()
    return leading_code.startswith("EW")


def resolve_bundle_state(safe_name: str) -> dict:
    """
    Identify which bundle represents a star's current best geometry.

    The converged bundle from Script 05 is preferred over the seed bundle from
    Script 04. A mid-MCMC checkpoint (bundle_<star>_mcmc.phoebe) is transient
    state that Script 05 deletes on completion, and is never rendered.
    """
    final_path = os.path.join(paths.BUNDLE_DIR, f"bundle_{safe_name}_final.phoebe")
    init_path  = os.path.join(paths.BUNDLE_DIR, f"bundle_{safe_name}.phoebe")
    if os.path.exists(final_path):
        return {"tier": "final", "bundle_path": final_path}
    if os.path.exists(init_path):
        return {"tier": "init", "bundle_path": init_path}
    return {"tier": None, "bundle_path": None}


def outputs_stale(output_path: str, bundle_path: str) -> bool:
    """
    Report whether an output needs rebuilding: missing, older than the bundle
    it derives from, or forced. The modification-time test matters because
    bundles are regenerated in place when an upstream fix lands, which leaves
    an output that exists but no longer matches its source.
    """
    if FORCE_RERENDER:
        return True
    if not os.path.exists(output_path):
        return True
    return os.path.getmtime(bundle_path) > os.path.getmtime(output_path)


def disable_non_mesh_datasets(bundle, compute: str = COMPUTE_LABEL,
                              keep_dataset: str = MESH_DATASET) -> list[str]:
    """
    Disable every enabled dataset except the mesh, in memory only, and return
    the labels switched off.

    Scripts 04 and 05 attach an lc01 dataset to the same compute label, so an
    untouched run_compute() would forward-model the light curve at every mesh
    time. Suppressing it reduces the cost from 12.5 s to 1.2 s per mesh time
    (measured on bundle_VW_Cep_final.phoebe).
    """
    disabled: list[str] = []
    for param in bundle.filter(qualifier="enabled", compute=compute,
                               context="compute").to_list():
        if param.dataset != keep_dataset and param.get_value() is True:
            param.set_value(False)
            disabled.append(param.dataset)
    return disabled


def orbital_period_of(bundle) -> float:
    """
    Return the orbital period the bundle itself models, i.e. the refined value
    from Script 03 rather than the Kreiner catalogue value in the target CSV.
    An animation spanning this period closes on exactly one cycle.
    """
    return bundle.get_value(qualifier="period", component="binary",
                            context="component")


# =============================================================================
# SECTION 3 — MESH EXTRACTION & RENDERING
# =============================================================================

def compute_mesh_dataset(bundle, times: list[float],
                         irrad_method: Optional[str]) -> list[str]:
    """
    Attach a mesh dataset covering `times`, suppress the non-mesh datasets, and
    compute in a single pass. Batching every requested time into one
    run_compute() call is markedly cheaper than one call per time. Returns the
    star components present, which is ['primary', 'secondary'] for both contact
    and detached configurations.
    """
    bundle.add_dataset("mesh", times=list(times), dataset=MESH_DATASET,
                       columns=["teffs"])

    disabled = disable_non_mesh_datasets(bundle)
    if disabled:
        print(f"    [optimise] non-mesh datasets disabled for compute: {disabled}")

    compute_kwargs = dict(compute=COMPUTE_LABEL, model=MESH_MODEL,
                          overwrite=True, progressbar=False,
                          distortion_method="roche")
    if irrad_method is not None:
        compute_kwargs["irrad_method"] = irrad_method
    bundle.run_compute(**compute_kwargs)

    return bundle.hierarchy.get_stars()


def extract_mesh_frame(bundle, time: float, components: list[str]) -> dict:
    """
    Read one time's surface mesh out of a computed bundle.

    PHOEBE returns `uvw_elements` as (N, 3, 3): the three vertex coordinates of
    each of N triangles, already in the plane-of-sky frame. Flattening to a
    vertex list with disjoint triangle indices is what Mayavi's
    triangular_mesh() consumes, and keeping the triangles disjoint preserves the
    per-element temperature as a flat facet colour instead of smoothing across
    shared vertices.

    Returns {component: (points (3N, 3), triangles (N, 3), teffs (N,))}.
    """
    frame: dict = {}
    for component in components:
        vertices = np.asarray(bundle.get_value(
            qualifier="uvw_elements", component=component,
            dataset=MESH_DATASET, model=MESH_MODEL, time=time))
        teffs = np.asarray(bundle.get_value(
            qualifier="teffs", component=component,
            dataset=MESH_DATASET, model=MESH_MODEL, time=time))
        n_triangles = vertices.shape[0]
        points    = vertices.reshape(-1, 3)
        triangles = np.arange(3 * n_triangles).reshape(n_triangles, 3)
        frame[component] = (points, triangles, teffs)
    return frame


def global_teff_range(bundle, times: list[float],
                      components: list[str]) -> tuple[float, float]:
    """
    Find the temperature extremes over every requested time and component, so
    one colour scale can be fixed for a whole sequence. Without this the
    mapping would rescale frame to frame and identical surfaces would change
    colour as the system rotates.
    """
    tmin, tmax = np.inf, -np.inf
    for time in times:
        for component in components:
            teffs = np.asarray(bundle.get_value(
                qualifier="teffs", component=component,
                dataset=MESH_DATASET, model=MESH_MODEL, time=time))
            tmin = min(tmin, teffs.min())
            tmax = max(tmax, teffs.max())
    return float(tmin), float(tmax)


def draw_mesh_frame(figure, frame: dict, tmin: float, tmax: float):
    """
    Clear the figure and draw every component's surface, flat-shaded by
    effective temperature against the supplied fixed colour limits. Returns the
    last surface drawn, which the colour bar attaches to.
    """
    mlab.clf(figure=figure)
    surface = None
    for points, triangles, teffs in frame.values():
        surface = mlab.triangular_mesh(
            points[:, 0], points[:, 1], points[:, 2], triangles,
            scalars=np.repeat(teffs, 3), colormap=MESH_COLORMAP,
            vmin=tmin, vmax=tmax, figure=figure)
        # Flat per-triangle colour: the temperature is an element property, and
        # interpolating it across vertices would misrepresent the mesh.
        surface.actor.mapper.interpolate_scalars_before_mapping = False
    return surface


def annotate_scene(figure, surface, title: str) -> None:
    """
    Apply the camera, title and colour bar shared by every render. The colour
    bar is bound to `surface` explicitly rather than to Mayavi's current scene,
    which is ambiguous while more than one figure exists. Text colours are set
    per object because Mayavi defaults to white, invisible on the white canvas.
    """
    mlab.view(azimuth=CAMERA_AZIMUTH, elevation=CAMERA_ELEVATION,
              distance="auto", focalpoint="auto", figure=figure)

    title_actor = mlab.title(title, figure=figure, size=0.35, height=0.95)
    title_actor.property.color = FIGURE_FOREGROUND

    colour_bar = mlab.colorbar(surface, title="T_eff (K)", orientation="vertical")
    colour_bar.title_text_property.color = FIGURE_FOREGROUND
    colour_bar.label_text_property.color = FIGURE_FOREGROUND


def render_static_png(bundle_path: str, safe_name: str, star_name: str,
                      tier: str, is_contact_binary: bool, figure) -> str:
    """
    Render a single-phase still for one star and return the path written.
    """
    out_path = os.path.join(paths.PLOT_MESH_DIR, f"mesh_{tier}_{safe_name}.png")
    label    = "preliminary" if tier == "init" else "final"

    # Script 04 computes its seed model without irradiation; Script 05 enables
    # the scheme matching the morphology. Mirroring both keeps each tier's
    # render physically consistent with the bundle it came from.
    irrad_method = None if tier == "init" else (
        "wilson" if is_contact_binary else "horvat")

    bundle = phoebe.load(bundle_path)
    try:
        components  = compute_mesh_dataset(bundle, [0.0], irrad_method)
        tmin, tmax  = global_teff_range(bundle, [0.0], components)
        frame       = extract_mesh_frame(bundle, 0.0, components)
        surface     = draw_mesh_frame(figure, frame, tmin, tmax)
        annotate_scene(figure, surface, f"{star_name} ({label})")
        mlab.savefig(out_path, figure=figure)
    finally:
        # Bundles run to tens of megabytes and a batch walks sixty of them, so
        # the reference is dropped explicitly rather than left to the loop.
        del bundle
        gc.collect()

    return out_path


def render_orbital_gif(bundle_path: str, safe_name: str, star_name: str,
                       is_contact_binary: bool, figure) -> str:
    """
    Render one full orbital cycle as an animated GIF and return the path
    written. The camera is fixed after the first frame so that only the system
    rotates, and the colour scale is fixed across the whole sequence.
    """
    out_path     = os.path.join(paths.PLOT_MESH_DIR, f"mesh_final_{safe_name}.gif")
    irrad_method = "wilson" if is_contact_binary else "horvat"
    frame_paths: list[str] = []

    try:
        bundle = phoebe.load(bundle_path)
        try:
            period     = orbital_period_of(bundle)
            times      = list(np.linspace(0.0, period, N_FRAMES, endpoint=False))
            components = compute_mesh_dataset(bundle, times, irrad_method)
            tmin, tmax = global_teff_range(bundle, times, components)

            fixed_view = None
            for index, time in enumerate(times):
                frame   = extract_mesh_frame(bundle, time, components)
                surface = draw_mesh_frame(figure, frame, tmin, tmax)
                annotate_scene(figure, surface, star_name)

                # annotate_scene() re-derives the camera from the current
                # geometry; the first frame's view is captured and replayed so
                # the framing cannot drift as the stars move.
                if fixed_view is None:
                    fixed_view = mlab.view(figure=figure)
                else:
                    mlab.view(*fixed_view, figure=figure)

                frame_path = os.path.join(
                    paths.PLOT_MESH_DIR, f"_tmp_mesh_{safe_name}_{index:03d}.png")
                mlab.savefig(frame_path, figure=figure)
                frame_paths.append(frame_path)
        finally:
            del bundle
            gc.collect()

        frames = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as raw:
                frames.append(raw.convert("P", palette=Image.ADAPTIVE, colors=256))
        frames[0].save(out_path, save_all=True, append_images=frames[1:],
                       duration=GIF_FRAME_DURATION_MS, loop=GIF_LOOP)
    finally:
        # Remove the intermediate frames even if the render or assembly failed,
        # so a partial run cannot leave hundreds of stray files in plots/.
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                os.remove(frame_path)

    return out_path


# =============================================================================
# SECTION 4 — APPENDIX GALLERY MONTAGE
# =============================================================================

def find_converged_stars() -> list[str]:
    """
    List every star with a converged bundle on disk.

    The gallery must reflect the whole sample, but the pipeline is driven one
    batch CSV at a time, so a single invocation sees at most a handful of
    converged stars in its own target list — never enough to reach the
    threshold. Counting bundles on disk instead makes the contact sheet
    independent of which batch triggered it.
    """
    return [
        fname[len("bundle_"):-len("_final.phoebe")]
        for fname in os.listdir(paths.BUNDLE_DIR)
        if fname.startswith("bundle_") and fname.endswith("_final.phoebe")
    ]


def make_gallery_montage(star_names: list[str]) -> str:
    """
    Assemble a captioned contact sheet from the converged stars' stills and
    return the path written under plots/. A copy is also placed in
    Text/Figures/ when that directory is reachable; failure to do so is
    reported but not fatal, since the script may run where the thesis text is
    not present.
    """
    thumb_width, thumb_height = GALLERY_THUMB_SIZE
    tile_width  = thumb_width
    tile_height = thumb_height + GALLERY_CAPTION_HEIGHT
    rows = (len(star_names) + GALLERY_COLUMNS - 1) // GALLERY_COLUMNS

    sheet = Image.new("RGB", (GALLERY_COLUMNS * tile_width, rows * tile_height),
                      "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for index, safe_name in enumerate(sorted(star_names)):
        still_path = os.path.join(paths.PLOT_MESH_DIR, f"mesh_final_{safe_name}.png")
        if not os.path.exists(still_path):
            continue
        with Image.open(still_path) as raw:
            thumb = raw.convert("RGB")
            thumb.thumbnail(GALLERY_THUMB_SIZE)
        row, column = divmod(index, GALLERY_COLUMNS)
        sheet.paste(thumb, (column * tile_width + (thumb_width - thumb.width) // 2,
                            row * tile_height))
        draw.text((column * tile_width + 8,
                   row * tile_height + thumb_height + 4),
                  safe_name.replace("_", " "), fill="black", font=font)

    out_path = os.path.join(paths.PLOT_THESIS_DIR, "appendix_mesh_gallery.png")
    sheet.save(out_path)

    try:
        os.makedirs(THESIS_FIGURES_DIR, exist_ok=True)
        sheet.save(os.path.join(THESIS_FIGURES_DIR, "appendix_mesh_gallery.png"))
    except OSError as error:
        print(f"  WARNING: could not copy gallery into Text/Figures/: {error}")

    return out_path


# =============================================================================
# SECTION 5 — TARGET LIST
# =============================================================================

# Optional positional argument selects the target catalogue, so a single batch
# (e.g. Targets/neglected_targets_batch1.csv) can be rendered in isolation.
INPUT_CATALOG: str = sys.argv[1] if len(sys.argv) > 1 else "neglected_targets.csv"

target_list: list[dict] = []
try:
    with open(INPUT_CATALOG, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # The period is parsed to validate the catalogue row, but the
            # animation uses the refined period stored in the bundle instead.
            target_list.append({
                "name":   row["name"],
                "type":   row["type"],
                "period": float(row["period"]),
            })
except FileNotFoundError:
    print(f"ERROR: {INPUT_CATALOG} not found. Run Script 01 first or check the path.")
    raise SystemExit(1)


# =============================================================================
# SECTION 6 — MAIN VISUALISATION LOOP
# =============================================================================

# Two figures are created once and reused for the whole batch: building an
# offscreen rendering context is the dominant fixed cost, and recreating one
# per star would repeat it sixty times.
figure_lores = mlab.figure(size=LORES_SIZE, bgcolor=FIGURE_BACKGROUND,
                           fgcolor=FIGURE_FOREGROUND)
figure_hires = mlab.figure(size=HIRES_SIZE, bgcolor=FIGURE_BACKGROUND,
                           fgcolor=FIGURE_FOREGROUND)

n_rendered, n_skipped, n_missing = 0, 0, 0

for target in target_list:
    star_name         = target["name"]
    variable_type     = target["type"]
    safe_name         = star_name.replace(" ", "_")
    is_contact_binary = is_contact_morphology(variable_type)

    print("\n" + "=" * 70)
    print(f"MESH VISUALISATION: {star_name}  (type: {variable_type})")
    print("=" * 70)

    state = resolve_bundle_state(safe_name)
    if state["tier"] is None:
        print(f"  WARNING: no bundle found for {safe_name}. Skipping.")
        n_missing += 1
        continue

    tier        = state["tier"]
    bundle_path = state["bundle_path"]

    try:
        still_path   = os.path.join(paths.PLOT_MESH_DIR, f"mesh_{tier}_{safe_name}.png")
        still_figure = figure_hires if tier == "final" else figure_lores

        if outputs_stale(still_path, bundle_path):
            print(f"  [render] {still_path}  (tier: {tier})")
            render_static_png(bundle_path, safe_name, star_name, tier,
                              is_contact_binary, still_figure)
            n_rendered += 1
        else:
            print(f"  [skip] {still_path} already current.")
            n_skipped += 1

        if tier == "final":
            # A star promoted from seed to converged leaves a preliminary
            # render that the final one supersedes. Script 05 discards its
            # checkpoint on convergence for the same reason.
            superseded = os.path.join(paths.PLOT_MESH_DIR, f"mesh_init_{safe_name}.png")
            if os.path.exists(superseded):
                os.remove(superseded)
                print(f"  [clean] removed superseded {superseded}")

            gif_path = os.path.join(paths.PLOT_MESH_DIR, f"mesh_final_{safe_name}.gif")
            if outputs_stale(gif_path, bundle_path):
                print(f"  [render] {gif_path}  ({N_FRAMES} frames, one orbital cycle)")
                render_orbital_gif(bundle_path, safe_name, star_name,
                                   is_contact_binary, figure_lores)
                n_rendered += 1
            else:
                print(f"  [skip] {gif_path} already current.")
                n_skipped += 1

    except Exception as error:
        # A malformed or partially-written bundle must not abandon the rest of
        # the batch, matching the per-target tolerance of Scripts 02-05.
        print(f"  WARNING: rendering failed for {star_name}: {error}. Skipping.")
        continue


# ── Appendix contact sheet ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("APPENDIX GALLERY")
print("=" * 70)

converged_stars = find_converged_stars()
if len(converged_stars) >= FINAL_STAR_GALLERY_THRESHOLD:
    gallery_path = make_gallery_montage(converged_stars)
    print(f"  {len(converged_stars)} converged stars on disk "
          f"(threshold {FINAL_STAR_GALLERY_THRESHOLD}) → {gallery_path}")
else:
    print(f"  {len(converged_stars)} converged stars on disk — fewer than the "
          f"threshold of {FINAL_STAR_GALLERY_THRESHOLD}; gallery not generated.")

print("\n" + "=" * 70)
print("SCRIPT 06 COMPLETE — Mesh visualisations rendered for all targets.")
print(f"  Rendered: {n_rendered}   Already current: {n_skipped}   "
      f"No bundle: {n_missing}")
print("=" * 70)
