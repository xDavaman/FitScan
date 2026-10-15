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
    return R.morphs_for_model(model, rest)


@pytest.fixture(scope="module")
def frame(model, rest):
    tri, bary = load_correspondence()
    return R.face_frame(surface_points(rest, model.faces, tri, bary))


def _lm(model, verts, idx):
    tri, bary = load_correspondence()
    return surface_points(verts, model.faces, tri, bary)[idx]


def test_face_frame_matches_anny(frame):
    """Anny: linke Seite der Person +X, oben +Z, Blick -Y."""
    assert frame.axes[0] @ [1, 0, 0] > 0.95
    assert frame.axes[1] @ [0, 0, 1] > 0.95
    assert frame.axes[2] @ [0, -1, 0] > 0.95
    assert 0.8 < frame.scale < 1.2


def test_all_controls_present_and_local(model, morphs):
    assert len(morphs) == 37
    head = model.dominant_bone_mask(lambda n: n.startswith(("head", "neck03")))
    for name, d in morphs.items():
        moved = np.linalg.norm(d, axis=1)
        assert np.isfinite(d).all(), name
        assert moved.max() > 0.002, f"{name} bewegt nichts"
        assert moved[~head].max() < 1e-9, f"{name} bewegt Punkte außerhalb des Kopfes"
        assert moved.max() < 0.03, f"{name} bewegt zu weit"


def test_jaw_opens_mouth(model, rest, morphs):
    before = _lm(model, rest, [13, 14, 152])
    after = _lm(model, rest + morphs["jawOpen"], [13, 14, 152])
    gap0 = before[0, 2] - before[1, 2]
    gap1 = after[0, 2] - after[1, 2]
    assert gap1 - gap0 > 0.01  # Lippen mehr als 1 cm auseinander
    assert after[2, 2] < before[2, 2] - 0.01  # Kinn geht nach unten
    assert abs(after[0, 2] - before[0, 2]) < 0.002  # Oberlippe bleibt


def test_sides_follow_arkit(model, rest, morphs):
    """mouthSmileLeft hebt den linken Mundwinkel der Person (+X), nicht den rechten."""
    corners = [291, 61] if _lm(model, rest, 291)[0] > 0 else [61, 291]  # [links, rechts]
    up = _lm(model, rest + morphs["mouthSmileLeft"], corners)[:, 2] - _lm(model, rest, corners)[:, 2]
    assert up[0] > 0.003 and abs(up[1]) < 0.001


def test_blink_closes_lid(model, rest, morphs):
    left_upper, left_lower = (386, 374) if _lm(model, rest, 386)[0] > 0 else (159, 145)
    before = _lm(model, rest, [left_upper, left_lower])
    after = _lm(model, rest + morphs["eyeBlinkLeft"], [left_upper, left_lower])
    assert before[0, 2] - before[1, 2] > 0.004
    assert abs(after[0, 2] - after[1, 2]) < 0.0015
    right_upper = 159 if left_upper == 386 else 386
    assert np.allclose(_lm(model, rest + morphs["eyeBlinkLeft"], right_upper), _lm(model, rest, right_upper))


def test_gaze_turns_only_the_eyeball(model, morphs):
    eyes = eye_vertex_masks(model)
    d = np.linalg.norm(morphs["eyeLookUpLeft"], axis=1)
    left = eyes["L"]
    assert d[left].max() > 0.003 and d[~left].max() == 0.0
    up = morphs["eyeLookUpLeft"][left]
    assert up[:, 2].max() > 0.003  # Vorderseite des Augapfels geht nach oben


def test_glb_roundtrip_adds_controls(model, tmp_path):
    rig = build_rig(model, BodyParams(), preset="unreal", skin=np.full((64, 64, 3), 180, np.uint8),
                    eyes=np.full((64, 64, 3), 200, np.uint8))
    src = write_glb(rig, tmp_path / "koerper_unreal.glb")
    gltf, blob = read_glb(src)
    index, anny = glb_to_anny(gltf, blob, model)
    assert np.allclose(anny - anny[:, 2].min(), model.rest_numpy(BodyParams())["vertices"]
                       - model.rest_numpy(BodyParams())["vertices"][:, 2].min(), atol=1e-5)
    names = add_face_controls(src, tmp_path / "mimik.glb", model=model)
    out, _ = read_glb(tmp_path / "mimik.glb")
    targets = out["meshes"][0]["extras"]["targetNames"]
    assert targets[:4] == ["weight_up", "weight_down", "muscle_up", "muscle_down"]
    assert targets[4:] == names and len(names) == 37
    assert all(len(p["targets"]) == len(targets) for p in out["meshes"][0]["primitives"])
