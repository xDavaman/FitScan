import numpy as np
import pytest

from fitscan import face_rig as R
from fitscan.animation import add_face_controls, glb_to_anny, read_glb
from fitscan.body import BodyParams
from fitscan.export import build_rig, write_glb
from fitscan.head import eye_vertex_masks, load_correspondence, surface_points


@pytest.fixture(scope="module")
def rest(model):
    return model.rest_numpy(BodyParams())["vertices"].astype(np.float64)


@pytest.fixture(scope="module")
def morphs(model, rest):
    return R.face_morphs(model, rest)


def _lm(model, verts, idx):
    tri, bary = load_correspondence()
    return surface_points(verts, model.faces, tri, bary)[idx]


def _left_right(model, rest, a, b):
    """MediaPipe-Paar -> (linke Seite der Person, rechte Seite); in Anny ist links +X."""
    return (a, b) if _lm(model, rest, a)[0] > _lm(model, rest, b)[0] else (b, a)


def test_all_arkit_controls_local_to_head(model, morphs):
    assert len(morphs) == 52
    assert {"jawOpen", "eyeBlinkLeft", "mouthSmileRight", "tongueOut", "eyeLookUpLeft"} <= set(morphs)
    head = model.dominant_bone_mask(lambda n: n.startswith(("head", "neck", "eye", "jaw", "tongue")))
    for name, d in morphs.items():
        moved = np.linalg.norm(d, axis=1)
        assert np.isfinite(d).all(), name
        assert moved.max() > 0.001, f"{name} bewegt nichts"
        assert moved[~head].max() < 1e-4, f"{name} bewegt Punkte außerhalb des Kopfes"


def test_scaled_to_face_size(model, rest):
    big = rest * 1.1
    a = R.face_morphs(model, rest)["jawOpen"]
    b = R.face_morphs(model, big)["jawOpen"]
    assert np.allclose(b, a * 1.1, atol=1e-6)


def test_jaw_opens_mouth(model, rest, morphs):
    before = _lm(model, rest, [13, 14, 152])
    after = _lm(model, rest + morphs["jawOpen"], [13, 14, 152])
    assert (after[0, 2] - after[1, 2]) - (before[0, 2] - before[1, 2]) > 0.01
    assert after[2, 2] < before[2, 2] - 0.01


def test_sides_follow_arkit(model, rest, morphs):
    """mouthSmileLeft hebt den linken Mundwinkel der Person, eyeBlinkLeft schließt ihr linkes Auge."""
    left, right = _left_right(model, rest, 291, 61)
    d = morphs["mouthSmileLeft"]
    up = _lm(model, rest + d, [left, right])[:, 2] - _lm(model, rest, [left, right])[:, 2]
    assert up[0] > 0.002 and up[0] > 3 * abs(up[1])
    lu, ru = _left_right(model, rest, 386, 159)
    ll = 374 if lu == 386 else 145
    before = _lm(model, rest, [lu, ll])
    after = _lm(model, rest + morphs["eyeBlinkLeft"], [lu, ll, ru])
    assert before[0, 2] - before[1, 2] > 0.004
    assert after[0, 2] - after[1, 2] < 0.3 * (before[0, 2] - before[1, 2])
    assert np.allclose(after[2], _lm(model, rest, ru), atol=1e-4)


def test_gaze_moves_the_eyeball(model, morphs):
    eyes = eye_vertex_masks(model)
    d = np.linalg.norm(morphs["eyeLookUpLeft"], axis=1)
    assert d[eyes["L"]].max() > 0.001
    assert d[eyes["R"]].max() < 1e-4


def test_glb_roundtrip_adds_controls(model, tmp_path):
    rig = build_rig(model, BodyParams(), preset="unreal", skin=np.full((64, 64, 3), 180, np.uint8),
                    eyes=np.full((64, 64, 3), 200, np.uint8))
    src = write_glb(rig, tmp_path / "koerper_unreal.glb")
    gltf, blob = read_glb(src)
    index, anny = glb_to_anny(gltf, blob, model)
    rest = model.rest_numpy(BodyParams())["vertices"]
    assert np.allclose(anny - anny[:, 2].min(), rest - rest[:, 2].min(), atol=1e-5)
    names = add_face_controls(src, tmp_path / "mimik.glb", model=model)
    out, _ = read_glb(tmp_path / "mimik.glb")
    targets = out["meshes"][0]["extras"]["targetNames"]
    assert targets[:4] == ["weight_up", "weight_down", "muscle_up", "muscle_down"]
    assert targets[4:] == names and len(names) == 52
    assert all(len(p["targets"]) == len(targets) for p in out["meshes"][0]["primitives"])
