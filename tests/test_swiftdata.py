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


CLIPS = ROOT / "apple/FitScanCore/Sources/FitScanCore/Resources/bewegungen.json"
FACE = ROOT / "apple/FitScanCore/Sources/FitScanCore/Resources/mimik.json"
SCAN_RULES = ROOT / "apple/FitScanCore/Sources/FitScanCore/Resources/scanregeln.json"


def _face_deltas(face: dict) -> dict[str, dict[int, np.ndarray]]:
    return {a["name"]: dict(zip(a["index"], np.reshape(a["delta"], (-1, 3)), strict=True)) for a in face["actions"]}


def test_clip_files_are_current(model):
    """bewegungen.json und mimik.json passen zu den Clips und Annys Gesichtsreglern. Die Mimik rechnet
    Anny in float32; auf anderen Rechnern rundet sie um 1 µm anders, das zählt nicht als veraltet."""
    assert json.loads(CLIPS.read_text("utf-8")) == swiftdata.clip_library(), STALE
    stored, fresh = json.loads(FACE.read_text("utf-8")), swiftdata.face_rig_data(model)
    plain = ("format", "version", "unit", "landmarks")
    assert {k: stored[k] for k in plain} == {k: fresh[k] for k in plain}, STALE
    assert np.allclose(stored["base_face_size"], fresh["base_face_size"], rtol=0, atol=1e-7), STALE
    old, new = _face_deltas(stored), _face_deltas(fresh)
    assert old.keys() == new.keys(), STALE
    zero = np.zeros(3, np.int64)
    for name in new:
        for i in old[name].keys() | new[name].keys():
            assert np.abs(old[name].get(i, zero) - new[name].get(i, zero)).max() <= 1, STALE


def test_motion_rig_rest_pose(data):
    """Die Ruhepose als Haltung ergibt die unbewegte Haut (Übergang Ruhepose → Bewegung in der App)."""
    rig = swiftdata.MotionRig(data, swiftdata.REFERENCE_BODIES["frau_60"])
    rotations, offset = rig.rest_pose()
    _, moves = rig.frame(rotations, offset)
    assert np.abs(rig.skin(moves) - rig.vertices).max() < 1e-5


def test_motion_rig_matches_play(model, data):
    """Der Rechenweg der App für Bewegungen trifft `motion.play`, `gltf_channels` und `face_morphs`."""
    from fitscan.clips import DATA as CLIP_DATA
    from fitscan.face_rig import face_morphs
    from fitscan.motion import Motion, gltf_channels, play

    library = json.loads(CLIPS.read_text("utf-8"))
    face = json.loads(FACE.read_text("utf-8"))
    parents = [-1] + [p + 1 if p >= 0 else 0 for p in data.meta["bone_parents"]]
    params = swiftdata.REFERENCE_BODIES["frau_60"]
    rig = swiftdata.MotionRig(data, params)
    for clip in library["clips"]:
        pb = play(model, params, Motion.load(CLIP_DATA / f"{clip['key']}.npz"))
        rotations, root = gltf_channels(pb, parents)
        assert abs(rig.hip_height - pb.hip_height) < 1e-5
        for f in (0, len(clip["root"]) // 2):
            poses, moves = rig.clip_frame(clip, f)
            assert np.abs(poses - pb.bones[f]).max() < 1e-4
            assert np.abs(rig.skin(moves) - pb.vertices[f]).max() < 1e-4
            q, r = swiftdata.gltf_channels_np(poses, data.meta["bone_parents"])
            assert np.minimum(np.abs(q - rotations[f]).max(1), np.abs(q + rotations[f]).max(1)).max() < 1e-5
            assert np.abs(r - root[f]).max() < 1e-5
    mine = swiftdata.face_morphs_np(face, data.rest(params)["vertices"], np.asarray(model.faces))
    theirs = face_morphs(model, model.rest_numpy(params)["vertices"])
    assert sorted(mine) == sorted(theirs)
    assert max(np.abs(mine[k] - theirs[k]).max() for k in theirs) < 1e-5


def test_motion_reference_is_current(model):
    from fitscan.clips import DATA as CLIP_DATA
    from fitscan.motion import Motion, play

    ref = json.loads(REFERENCE.read_text("utf-8"))["bewegungen"]
    body = ref["bodies"]["standard"]
    clip = body["clips"]["winken"]
    pb = play(model, BodyParams(), Motion.load(CLIP_DATA / "winken.npz"))
    assert abs(clip["hip_height"] - pb.hip_height) < 1e-6, STALE
    f = clip["frames"][1]
    assert np.abs(np.array(f["vertices"]) - pb.vertices[f["frame"]][ref["sample"]]).max() < 1e-5, STALE


def test_scan_rules_are_current():
    """scanregeln.json hat die Grenzwerte und Hinweise von `quality.py`."""
    assert json.loads(SCAN_RULES.read_text("utf-8")) == swiftdata.scan_rules(), STALE


def test_scan_reference_is_current():
    """Die künstlichen Kamerabilder und ihre Hinweise in referenz.json passen zu `quality.check_frame`."""
    import base64

    import cv2

    ref = json.loads(REFERENCE.read_text("utf-8"))["scan"]
    now = swiftdata.scan_reference()
    assert ref.keys() == now.keys(), STALE
    for name, case in now.items():
        old = ref[name]
        if name.startswith("_"):
            assert old == case, STALE
            continue
        assert old["hints"] == case["hints"], STALE
        assert old["keypoints"] == case["keypoints"] and old["mask_runs"] == case["mask_runs"], STALE
        decode = [cv2.imdecode(np.frombuffer(base64.b64decode(c["luma_png"]), np.uint8), cv2.IMREAD_UNCHANGED)
                  for c in (old, case)]
        assert np.array_equal(*decode), STALE
        assert abs(old["sharpness"] - case["sharpness"]) < 1e-6, STALE
