import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from fitscan import hands as HS
from fitscan.body import BodyParams
from fitscan.camera import Intrinsics


@pytest.fixture(scope="module")
def rig(model):
    return HS.HandRig(model, BodyParams(), "R")


def test_rig_rest_and_bend(rig):
    zero = torch.zeros(1, 15, 3, dtype=torch.float64)
    pts, skin = rig.forward(zero, torch.zeros(1, dtype=torch.float64))
    assert torch.allclose(pts[0], rig.p0) and torch.allclose(skin[0], rig.v0)
    bend = zero.clone()
    bend[0, 3:6, 0] = 0.6  # Zeigefinger in allen drei Gelenken beugen
    moved, skin2 = rig.forward(bend, torch.zeros(1, dtype=torch.float64))
    tip = (moved[0, 8] - rig.p0[8]).numpy()
    assert tip @ rig.hand.palm > 0.02  # Spitze wandert zur Handfläche
    assert torch.allclose(moved[0, 12], rig.p0[12])  # Mittelfinger bleibt
    longer, _ = rig.forward(zero, torch.tensor([1.0], dtype=torch.float64))
    assert (longer[0, 12] - longer[0, 9]).norm() > (rig.p0[12] - rig.p0[9]).norm()


def test_fit_recovers_finger_length(rig):
    """Künstliche Handpunkte aus bekannter Haltung, Lage und Fingerlänge: die Anpassung findet sie wieder."""
    rng = np.random.default_rng(3)
    intr = Intrinsics.phone_default(720, 1280)
    true = torch.tensor([0.5], dtype=torch.float64)
    angles = torch.zeros(14, 15, 3, dtype=torch.float64)
    angles[..., 0] = torch.tensor(rng.uniform(0.0, 0.6, (14, 15))) * torch.tensor(HS.MAX_FLEX.reshape(15))
    pts, _ = rig.forward(angles, true, with_skin=False)
    tracks = []
    for k in range(14):
        r = Rotation.from_euler("xyz", rng.uniform(-np.pi, np.pi, 3)).as_matrix()
        cam = pts[k].numpy() @ r.T + np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 0.35])
        uv = intr.focal * cam[:, :2] / cam[:, 2:] + [intr.cx, intr.cy] + rng.normal(0, 1.0, (21, 2))
        tracks.append(HS.HandTrack(10 * k, uv.astype(np.float32), (cam - cam.mean(0)).astype(np.float32), 1.0))
    fit = HS.fit_hand(rig, tracks, intr, steps=600)
    assert np.median(fit.errors) < 1.5, fit.errors
    assert abs(fit.sliders[0] - 0.5) < 0.15, fit.sliders
    assert abs(fit.intr.focal / intr.focal - 1) < 0.1


def test_hand_region_excludes_wrist(model):
    region = HS.hand_region(model, "L")
    wrist = model.dominant_bone_mask(lambda n: n == "wrist.L")
    finger = model.dominant_bone_mask(lambda n: n == "finger3-2.L")
    assert region[finger].min() > 0.9 and region[wrist].mean() < 0.3
    assert region[model.dominant_bone_mask(lambda n: n.endswith(".R"))].max() == 0


def test_skin_mask_keeps_hand_colour():
    img = np.zeros((200, 200, 3), np.uint8)
    img[:] = (60, 140, 70)  # grüner Teppich
    img[50:150, 50:150] = (220, 170, 150)  # Haut
    img[50:150, 140:150] = (60, 140, 70)  # Teppich, den das Modell noch überdeckt
    outline = np.zeros((200, 200), bool)
    outline[50:150, 50:150] = True
    m = HS.skin_mask(img, outline)
    assert m[100, 100] > 0.9 and m[100, 145] < 0.1 and m[10, 10] == 0
