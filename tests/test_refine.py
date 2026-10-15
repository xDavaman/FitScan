import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from fitscan.body import BodyParams
from fitscan.camera import Intrinsics, render_mask
from fitscan.fit import VIEW_BONES, FitFrame, View, _mesh_edges, _soft_gate, view_vertices
from fitscan.measure import measure_params
from fitscan.refine import IGNORE_BONES, outlines, refine_shape

TRUTH = BodyParams(phenotype={"gender": 0.0, "age": 0.5, "muscle": 0.95, "weight": 0.45, "height": 0.5,
                              "proportions": 0.6},
                   local={"torso-vshape-incr": 0.3, "measure-calf-circ-incr": 0.3, "buttocks-volume-incr": -0.2})


def _still() -> View:
    return View(0.0, np.zeros(3, np.float32), np.zeros((len(VIEW_BONES), 3), np.float32))


def _film(model, params: BodyParams, n: int, seed: int = 0):
    """Umrisse einer sich drehenden Person und leicht verrauschte Ansichten (wie nach fit_views)."""
    rng = np.random.default_rng(seed)
    intr = Intrinsics.phone_default(720, 1280)  # wie die Analysebilder
    verts = view_vertices(model, params, _still())
    zc = (verts[:, 2].max() + verts[:, 2].min()) / 2
    frames, views = [], []
    for i in range(n):
        yaw = 2 * math.pi * i / n
        trans = np.array([0.0, zc, 2.6], np.float32)
        frames.append(FitFrame(mask=render_mask(verts, model.faces, yaw, trans, intr).astype(np.float32) / 255,
                               intr=intr))
        views.append(View(yaw + rng.normal(0, 0.01), trans + rng.normal(0, 0.004, 3).astype(np.float32),
                          np.zeros((len(VIEW_BONES), 3), np.float32)))
    return frames, views


def test_outlines_are_signed_distances():
    mask = np.zeros((200, 100), np.float32)
    mask[50:150, 30:70] = 1.0
    out = outlines([FitFrame(mask=mask, intr=Intrinsics.phone_default(100, 200))], scale=2.0)
    sdf = out.sdf[0, 0].numpy()
    m = out.masks[0]
    assert sdf[m].max() < 0 < sdf[~m].min()
    ys, xs = np.nonzero(m)
    # mitten in der Person tief innen, Abstand in Pixeln der doppelten Auflösung
    assert sdf[(ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2] < -35


@pytest.mark.slow
def test_refine_moves_shape_towards_true_muscles(model):
    start = BodyParams(phenotype={**TRUTH.phenotype, "muscle": 0.5, "weight": 0.55})
    frames, views = _film(model, TRUTH, 24)
    waist = measure_params(model, TRUTH)["waist"]  # Maßband: Start liegt rund 3 cm daneben
    out, new_views, rep = refine_shape(model, start, frames, views, measures={"waist": waist})
    assert len(new_views) == len(views)
    assert rep.rim_after_mm < 0.4 * rep.rim_before_mm
    assert rep.iou_after > rep.iou_before

    # Abstand zur wahren Oberfläche (Rumpf, Arme, Beine; ohne Kopf, Hände, Füße)
    truth = model.rest_numpy(TRUTH)["vertices"]
    normals = np.cross(truth[model.faces[:, 1]] - truth[model.faces[:, 0]], truth[model.faces[:, 2]] - truth[model.faces[:, 0]])
    vn = np.zeros_like(truth)
    for i in range(3):
        np.add.at(vn, model.faces[:, i], normals)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True)
    body = (_soft_gate(model, _mesh_edges(model.faces)) > 0.99) & ~model.dominant_bone_mask(
        lambda n: n.startswith(IGNORE_BONES))
    tree = cKDTree(truth)

    def error(p):
        v = model.rest_numpy(p)["vertices"][body]
        _, nn = tree.query(v)
        return float(np.abs(((v - truth[nn]) * vn[nn]).sum(1)).mean())

    assert error(out) < 0.9 * error(start)
    # Kopf und Füße bleiben: Körpergröße unverändert; eingegebener Taillenumfang wird erreicht
    heights = [float(np.ptp(model.rest_numpy(p)["vertices"][:, 2])) for p in (start, out)]
    assert abs(heights[1] - heights[0]) < 0.001
    assert abs(measure_params(model, out)["waist"] - waist) < 1.0
