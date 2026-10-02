"""
================================================================================
SCHEMATIC — GENERIC ROCHE-GEOMETRY ILLUSTRATION (NOT A PIPELINE STAGE)
================================================================================
Summary:
    Renders two idealised, non-data-fitted Mayavi mesh illustrations of Roche
    geometry: a detached binary at a representative mass ratio and
    inclination (visible tidal elongation toward the companion, but the two
    surfaces separate), and a contact (overcontact) binary (a shared
    envelope with the characteristic "peanut" neck). Both are built from
    phoebe.default_binary() with hand-picked, illustrative parameter values
    — NOT fit to any real star's light curve, unlike Scripts 04/05/06.

    Intended as a candidate Section 2.1 (Theoretical Background) figure
    explaining Roche-lobe geometry conceptually, before Section 4.4's
    data-driven example (currently VW Cep, via Script 06). Deliberately
    NOT wired into thesis_v2.tex yet — see thesis_v2_progress.md
    "Post-completion additions (figures)". Run manually when needed; it
    does not iterate a target catalogue and is not part of the numbered
    01-06 pipeline sequence.

Outputs:
    plots/roche_schematic_detached.png
    plots/roche_schematic_contact.png

Dependencies:
    pip install phoebe mayavi vtk pyqt5 numpy

References:
    Prša & Zwitter (2005), ApJ 628, 426 — PHOEBE computational framework.
================================================================================
"""

import os
import warnings

os.environ.setdefault("PHOEBE_ENABLE_ONLINE_PASSBANDS", "FALSE")

import numpy as np
from mayavi import mlab

mlab.options.offscreen = True

import phoebe

warnings.filterwarnings("ignore")
phoebe.logger(clevel="WARNING")

import paths
paths.ensure_dirs()

MESH_TRIANGLE_COUNT: int = 2000   # matches the pipeline's own constant (Script 04)
FIGURE_SIZE = (1000, 800)
MESH_COLORMAP: str = "bone"       # neutral, non-data colormap — this is a geometry
                                   # illustration, not a temperature measurement
CAMERA_AZIMUTH: float = 55.0
CAMERA_ELEVATION: float = 70.0


def render_schematic(contact: bool, out_name: str, title: str) -> str:
    bundle = phoebe.default_binary(contact_binary=contact)
    bundle.set_value("q@binary", 0.5)
    if contact:
        bundle.set_value("incl@binary", 80.0)
    else:
        bundle.set_value("incl@binary", 85.0)
        # Tighten the orbit from PHOEBE's default sma=5.3 so each star sits
        # close enough to its Roche lobe to show a visible tidal bulge
        # toward the companion, while remaining clearly detached (not
        # touching) -- the whole point of this schematic.
        bundle.set_value("sma@binary", 3.0)
        bundle.set_value("requiv@primary", 1.2)
        bundle.set_value("requiv@secondary", 0.85)
    bundle.set_value_all("ntriangles", MESH_TRIANGLE_COUNT)

    bundle.add_dataset("mesh", times=[0.0], dataset="mesh01", columns=["teffs"])
    bundle.run_compute(compute="phoebe01", model="schematic", overwrite=True,
                        progressbar=False, distortion_method="roche")

    fig = mlab.figure(size=FIGURE_SIZE, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    for comp in bundle.hierarchy.get_stars():
        verts = np.asarray(bundle.get_value(
            qualifier="uvw_elements", component=comp, dataset="mesh01",
            model="schematic", time=0.0))
        n = verts.shape[0]
        points = verts.reshape(-1, 3)
        triangles = np.arange(3 * n).reshape(n, 3)
        mesh = mlab.triangular_mesh(
            points[:, 0], points[:, 1], points[:, 2], triangles,
            color=(0.75, 0.75, 0.78), figure=fig)
        mesh.actor.property.specular = 0.3

    mlab.view(azimuth=CAMERA_AZIMUTH, elevation=CAMERA_ELEVATION,
              distance="auto", focalpoint="auto", figure=fig)
    title_obj = mlab.title(title, figure=fig, size=0.4, height=0.92)
    title_obj.property.color = (0, 0, 0)

    out_path = os.path.join(paths.PLOT_THESIS_DIR, out_name)
    mlab.savefig(out_path, figure=fig)
    mlab.close(fig)
    del bundle
    return out_path


if __name__ == "__main__":
    # Run timing → Runs/run_stats/run_summary.txt (written at exit). Inside the
    # __main__ guard so that importing this module never starts a tracker.
    import sys as _rs_sys, os as _rs_os
    _rs_sys.path.insert(0, _rs_os.path.dirname(_rs_os.path.abspath(__file__)))
    import runstats
    _run_tracker = runstats.RunTracker("SchematicRocheGeometry", catalogue=_rs_sys.argv[1:],
                                       per_star=False)
    detached_path = render_schematic(
        contact=False, out_name="roche_schematic_detached.png",
        title="Detached binary (q=0.5, i=85 deg)")
    print(f"wrote {detached_path}")

    contact_path = render_schematic(
        contact=True, out_name="roche_schematic_contact.png",
        title="Contact (overcontact) binary (q=0.5, i=80 deg)")
    print(f"wrote {contact_path}")
