"""Die Anny-Daten und Prüfwerte der Swift-App passen zum PC (scripts/export_swift_data.py)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fitscan import swiftdata
from fitscan.body import BodyParams

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "apple/FitScanCore/Sources/FitScanCore/Resources/anny_v1.fsan"
REFERENCE = ROOT / "apple/FitScanCore/Tests/FitScanCoreTests/Resources/referenz.json"
STALE = "Swift-Daten sind veraltet: python scripts/export_swift_data.py"


@pytest.fixture(scope="module")
def data():
    return swiftdata.AnnyData(DATA)


def test_labels_match_anny(model, data):
    assert data.phenotype_labels == model.phenotype_labels, STALE
    assert data.local_labels == model.local_labels, STALE
    assert data.bone_labels == model.bone_labels, STALE
    assert data.meta["num_vertices"] == model.num_vertices


def test_rest_matches_anny(model, data):
    """Der Rechenweg der App (numpy-Nachbau) trifft Anny auf weit unter 1 mm."""
    for params in [*swiftdata.REFERENCE_BODIES.values(),
                   BodyParams(phenotype={"gender": 0.3, "age": 0.9, "muscle": 0.1, "weight": 0.95,
                                         "height": 0.05, "proportions": 0.9},
                              local={n: 0.5 * np.sin(i) for i, n in enumerate(model.local_labels[:40])})]:
        ref = model.rest_numpy(params)
        mine = data.rest(params)
        assert np.abs(ref["vertices"] - mine["vertices"]).max() < 1e-4, STALE
        assert np.abs(ref["bone_heads"] - mine["bone_heads"]).max() < 1e-5, STALE
        assert np.abs(ref["bone_poses"][:, :3, :3] - mine["bone_poses"][:, :3, :3]).max() < 1e-5


def test_interpolation_like_anny():
    from anny.utils.interpolation import linear_interpolation_coefficients

    anchors = [-1 / 3, 0.0, 1 / 3, 2 / 3, 1.0]
    values = [-0.5, -1 / 3, 0.0, 0.1, 1 / 3, 0.5, 2 / 3, 0.99, 1.0, 1.2]
    ref = linear_interpolation_coefficients(torch.tensor(values, dtype=torch.float64),
                                            torch.tensor(anchors, dtype=torch.float64))
    for v, row in zip(values, ref.numpy(), strict=True):
        assert np.allclose(swiftdata.interpolation_weights(v, anchors), row)


def test_reference_is_current(model):
    """referenz.json (Prüfwerte der Swift-Tests) stimmt noch mit dem PC überein."""
    from fitscan.measure import measure_params

    ref = json.loads(REFERENCE.read_text("utf-8"))
    for name, body in ref["bodies"].items():
        p = body["params"]
        params = BodyParams(phenotype=p["phenotype"], local=p["local"], scale=p["scale"])
        rest = model.rest_numpy(params)
        assert np.abs(rest["vertices"][body["sample_vertices"]] - np.array(body["vertices"])).max() < 1e-5, STALE
        mine = measure_params(model, params)
        for key, value in body["measures"].items():
            assert abs(mine[key] - value) < 0.11, f"{name} {key}: {STALE}"


def test_photo_settings_are_current():
    """fotoscan.json (Einstellungen des Fotoscans in der App) passt zu `fitscan.photofit`."""
    path = ROOT / "apple/FitScanCore/Sources/FitScanCore/Resources/fotoscan.json"
    assert json.loads(path.read_text("utf-8")) == swiftdata.photo_settings(), STALE


def test_photo_reference_is_current():
    ref = json.loads(REFERENCE.read_text("utf-8"))["fotoscan"]
    assert ref["body"] == swiftdata.PHOTO_BODY, STALE
    assert len(ref["frames"]) == 4
    f = ref["frames"][0]
    mask = swiftdata.mask_from_runs(f["runs"], f["width"], f["height"])
    assert mask.shape == (f["height"], f["width"])
    assert mask.sum() >= ref["prepared"]["pixels"] > 0


def test_mask_runs_round_trip():
    rng = np.random.default_rng(4)
    for first in (False, True):
        mask = rng.random((7, 9)) < 0.4
        mask[0, 0] = first
        runs = swiftdata.mask_runs(mask)
        assert sum(runs) == mask.size
        assert np.array_equal(swiftdata.mask_from_runs(runs, 9, 7), mask)
