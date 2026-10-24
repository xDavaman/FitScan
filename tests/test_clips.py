import json
import struct

import numpy as np
import pytest

from fitscan import clips as C
from fitscan import motion as M
from fitscan.body import BodyParams
from fitscan.face_rig import anny_facial_actions

SLIM = BodyParams(phenotype={"height": 0.3, "weight": 0.3})
TALL = BodyParams(phenotype={"height": 0.9, "weight": 0.7, "muscle": 0.7})


def _read_glb(path):
    data = path.read_bytes()
    n = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + n]), data[20 + n + 8:]


def test_face_curves_survive_save_and_resample(tmp_path):
    walk = M.demo_walk(seconds=1.0)
    jaw = np.linspace(0, 1, walk.frames).astype(np.float32)
    mo = M.Motion(walk.name, walk.fps, walk.bones, walk.rotations, walk.root, face={"jawOpen": jaw})
    back = M.Motion.load(mo.save(tmp_path / "sprechen.npz"))
    assert set(back.face) == {"jawOpen"} and np.allclose(back.face["jawOpen"], jaw)
    r = back.resampled(60.0)
    assert len(r.face["jawOpen"]) == r.frames and np.allclose(r.face["jawOpen"][::2], jaw, atol=1e-6)
    # ältere Bewegungen ohne Mimik laden weiter
    assert M.Motion.load(walk.save(tmp_path / "gang.npz")).face == {}


def test_library_clips_are_clean_loops():
    labels = set(anny_facial_actions()[0])
    for name in C.NAMES:
        mo = C.load(name)
        assert mo.fps == C.FPS and mo.loop and mo.frames > 20, name
        assert np.allclose(np.abs((mo.rotations[0] * mo.rotations[-1]).sum(-1)), 1, atol=1e-4), name
        assert set(mo.face) <= labels, name
        for curve in mo.face.values():
            assert curve.min() >= 0 and curve.max() <= 1 and abs(curve[0] - curve[-1]) < 1e-6
        assert all(mo.contacts[f"foot.{s}"].any() for s in "LR"), name
    assert {"jawOpen", "mouthFunnel"} <= set(C.load("sprechen").face)
    assert C.load("rufen").face["jawOpen"].max() > 0.5
    walk = C.load("gehen")
    assert walk.root[-1, 1] < -0.8 and np.allclose(walk.root[-1, [0, 2]], walk.root[0, [0, 2]], atol=1e-3)


@pytest.mark.parametrize(("params", "limit"), [(BodyParams(), 0.003), (SLIM, 0.01), (TALL, 0.01)],
                         ids=["grundkoerper", "schlank", "gross"])
def test_walk_feet_do_not_slide_on_any_body(model, params, limit):
    """Auf dem Grundkörper stehen die Füße genau; andere Körper halten sie mit fix_ground fest."""
    mo = C.load("gehen")
    if params is not None and params.phenotype != BodyParams().phenotype:
        mo = M.fix_ground(model, params, mo)
    pb = M.play(model, params, mo)
    for foot in M.FEET:
        sole = M._sole_vertices(model, foot)
        on = mo.contacts[foot][:-1]  # letztes Bild = erstes einen Doppelschritt weiter
        centre = pb.vertices[:-1, sole].mean(1)
        strike = int(np.flatnonzero(on & ~np.roll(on, 1))[0])  # Ferse setzt auf
        on, centre = np.roll(on, -strike), np.roll(centre, -strike, 0)
        stance = int(np.argmin(on))
        # flacher Teil der Standphase (nach dem Abrollen über die Ferse, vor dem Anheben)
        flat = centre[int(0.2 * stance):int(0.65 * stance), :2]
        drift = np.linalg.norm(flat - flat[0], axis=1).max()
        assert drift < limit, (foot, drift)
    assert pb.vertices[..., 2].min() > -0.015


def test_glb_carries_face_animation(model, tmp_path):
    mo = C.load("rufen").resampled(10.0)
    path = M.export_glb(model, BodyParams(), [mo], tmp_path / "rufen.glb")
    js, _ = _read_glb(path)
    names = js["meshes"][0]["extras"]["targetNames"]
    assert {"jawOpen", "eyeBlinkLeft"} <= set(names)
    anim = js["animations"][0]
    weights = [c for c in anim["channels"] if c["target"]["path"] == "weights"]
    assert len(weights) == 1 and js["nodes"][weights[0]["target"]["node"]].get("mesh") == 0
    out = js["accessors"][anim["samplers"][weights[0]["sampler"]]["output"]]
    assert out["count"] == mo.frames * len(names)
