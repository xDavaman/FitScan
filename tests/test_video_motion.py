import json
import math
import struct

import numpy as np
import roma
import torch

from fitscan import motion as M
from fitscan import video_motion as V
from fitscan.body import BodyParams

SLIM = BodyParams(phenotype={"height": 0.3, "weight": 0.3})


def _person(x: float, y: float, size: float = 600.0) -> np.ndarray:
    """MediaPipe-Punkte einer stehenden Person, Hüftmitte bei (x, y)."""
    kp = np.zeros((33, 3), np.float32)
    kp[:, 2] = 0.9
    kp[:, 0] = x
    kp[:, 1] = np.linspace(y - size / 2, y + size / 2, 33)
    kp[[23, 24], 1] = y
    kp[[25, 26], 1] = y + size / 4
    kp[[27, 28], 1] = y + size / 2 - 10
    return kp


def _track(kps: list, cut_at: tuple[int, ...] = ()) -> V.Track:
    cut = np.zeros(len(kps), np.float32)
    cut[list(cut_at)] = 0.8
    return V.Track(30.0, 720, 1280, cut, np.ones(len(kps), int), kps, [None] * len(kps))


def test_find_shots_splits_at_cuts_and_jumps():
    kps = [_person(360, 640)] * 60  # 2 s
    kps += [_person(360, 640)] * 20  # 0,67 s: allein zu kurz
    kps += [None] * 10  # kurz nicht erkannt: gehört noch zur Szene, wird aufgefüllt
    kps += [_person(360, 640)] * 50 + [_person(600, 640, 200)] * 45  # Sprung ohne Farbwechsel
    kps += [_person(360, 640)] * 20  # nach dem Schnitt nur 0,67 s
    tr = _track(kps, cut_at=(60, 185))
    shots = V.find_shots(tr)
    assert [(s.start, s.end) for s in shots] == [(0, 60), (60, 140), (140, 185)]
    # abgeschnittene Beine zählen nicht als ganze Person
    cropped = [_person(360, 1200)] * 60
    assert V.find_shots(_track(cropped)) == []


def _motion(rv: np.ndarray, bones: list[str]) -> M.Motion:
    root = np.tile([0.0, 0.0, 1.0], (len(rv), 1))
    return M.from_rotvecs("Test", 30.0, bones, rv, root)


def test_remove_outliers_replaces_single_spike():
    rv = np.zeros((30, 1, 3), np.float32)
    rv[:, 0, 0] = np.linspace(0, 0.5, 30)
    rv[12, 0, 0] = 2.0  # MediaPipe verwechselt das Bein in einem Bild
    mo, n = V.remove_outliers(_motion(rv, ["upperleg01.L"]))
    assert n == 1
    back = roma.unitquat_to_rotvec(torch.as_tensor(mo.rotations)).numpy()[:, 0, 0]
    assert abs(back[12] - rv[12 - 1, 0, 0] / 2 - rv[12 + 1, 0, 0] / 2) < 0.01
    assert np.allclose(np.delete(back, 12), np.delete(rv[:, 0, 0], 12), atol=1e-5)


def test_steady_heading_removes_slow_turn_keeps_fast():
    n = 150
    yaw = np.linspace(0, math.radians(60), n)  # Kamerafahrt: 60° in 5 s
    yaw[70:80] += math.radians(40) * np.sin(np.linspace(0, math.pi, 10))  # schnelle Drehung, kommt zurück
    rv = np.zeros((n, 1, 3), np.float32)
    rv[:, 0, 2] = yaw
    mo, turned = V.steady_heading(_motion(rv, ["root"]))
    rot = roma.unitquat_to_rotmat(torch.as_tensor(mo.rotations[:, 0])).double().numpy()
    after = V._heading(rot)
    assert turned > 45
    assert np.abs(after[:60]).max() < math.radians(5) and np.abs(after[90:]).max() < math.radians(5)
    assert after[75] > math.radians(25)  # die schnelle Drehung bleibt


def test_limit_joints_keeps_knee_and_elbow_human(model):
    from fitscan import mocap

    pts = mocap._Points(model, SLIM)
    axes, signs = mocap._hinge_axes(model, pts)
    k = mocap.BONES.index("lowerleg01.L")
    e = mocap.BONES.index("lowerarm01.R")
    bones = ["lowerleg01.L", "lowerarm01.R", "neck01"]
    rv = np.zeros((3, 3, 3), np.float32)
    rv[0, 0] = -signs[k].item() * math.radians(40) * axes[k].numpy()  # Knie nach vorn durchgedrückt
    rv[1, 1] = signs[e].item() * math.radians(175) * axes[e].numpy()  # Ellbogen weiter als möglich
    rv[2, 2] = [math.radians(120), 0, 0]  # Kopf weit in den Nacken
    rv[2, 0] = signs[k].item() * math.radians(90) * axes[k].numpy()  # normal gebeugt: bleibt
    mo, limited = V.limit_joints(model, SLIM, _motion(rv, bones))
    out = roma.unitquat_to_rotvec(torch.as_tensor(mo.rotations)).numpy()
    knee = out[:, 0] @ axes[k].numpy() * signs[k].item()
    elbow = out[:, 1] @ axes[e].numpy() * signs[e].item()
    assert knee[0] > -math.radians(V.HINGE_SLACK) - 1e-4
    assert abs(knee[2] - math.radians(90)) < 1e-3
    assert elbow[1] <= math.radians(V.HINGE_MAX["lowerarm01"]) + 1e-4
    assert np.linalg.norm(out[2, 2]) <= math.radians(V.LIMITS["neck01"]) + 1e-4
    assert set(limited) == {"lowerleg01.L", "lowerarm01.R", "neck01"}


def _read_glb(path):
    data = path.read_bytes()
    js_len = struct.unpack_from("<I", data, 12)[0]
    js = json.loads(data[20:20 + js_len])
    return js, data[28 + js_len:]


def test_animate_glb_matches_figure(model, tmp_path):
    """Bewegung in ein fertiges GLB schreiben und das Skinning aus der Datei nachrechnen."""
    from fitscan.posing import Figure

    plain = M.export_glb(model, SLIM, [], tmp_path / "koerper.glb")
    mo = M.fix_ground(model, SLIM, M.demo_walk(seconds=1.0))
    out = V.animate_glb(model, plain, [mo], tmp_path / "bewegt.glb")
    js, blob = _read_glb(out)
    assert [a["name"] for a in js["animations"]] == ["Testgang"]

    def acc(i):
        a = js["accessors"][i]
        v = js["bufferViews"][a["bufferView"]]
        dt = {5126: np.float32, 5123: np.uint16, 5121: np.uint8, 5125: np.uint32}[a["componentType"]]
        k = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
        return np.frombuffer(blob, dt, a["count"] * k, v.get("byteOffset", 0)).reshape(a["count"], k)

    prim = js["meshes"][0]["primitives"][0]
    pos = acc(prim["attributes"]["POSITION"]).astype(np.float64)
    joints4 = acc(prim["attributes"]["JOINTS_0"]).astype(int)
    weights4 = acc(prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
    joints = js["skins"][0]["joints"]
    ibm = acc(js["skins"][0]["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    nodes = js["nodes"]

    def trs(n):
        m = np.eye(4)
        m[:3, :3] = roma.unitquat_to_rotmat(torch.tensor(n.get("rotation", [0, 0, 0, 1.0]))).numpy()
        m[:3, 3] = n.get("translation", [0, 0, 0])
        return m

    anim = js["animations"][0]
    local = np.stack([np.stack([trs(nodes[j]) for j in joints])] * mo.frames)
    index = {node: k for k, node in enumerate(joints)}
    for ch in anim["channels"]:
        vals = acc(anim["samplers"][ch["sampler"]]["output"]).astype(np.float64)
        k = index[ch["target"]["node"]]
        if ch["target"]["path"] == "rotation":
            local[:, k, :3, :3] = roma.unitquat_to_rotmat(torch.tensor(vals)).numpy()
        else:
            local[:, k, :3, 3] = vals
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    world = np.empty_like(local)
    for k, node in enumerate(joints):
        p = parent.get(node)
        world[:, k] = local[:, k] if p not in index else world[:, index[p]] @ local[:, k]
    skinning = world @ ibm[None]
    _, verts = V.figure_playback(Figure.from_glb(model, plain), mo)
    ph = np.concatenate([pos, np.ones((len(pos), 1))], 1)
    for f in (0, mo.frames // 2, mo.frames - 1):
        v = np.einsum("vk,vkab,vb->va", weights4, skinning[f][joints4], ph)[:, :3]
        target = verts[f] @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]]).T  # Anny -> glTF
        d = np.linalg.norm(v - target, axis=1)
        assert d.mean() < 1e-3 and d.max() < 0.02, (f, d.mean(), d.max())
    # gleicher Körper, gleiche Bewegung: wie Annys eigenes Abspielen (Boden, Hüfthöhe)
    pb = M.play(model, SLIM, mo)
    assert np.abs(verts - pb.vertices).max() < 0.01


def test_mannequin_has_smooth_head_and_exports(model, tmp_path):
    """Puppe: gleiche Größe, Gesicht ausgestrichen (nur Kopf verändert), GLB mit Bewegung."""
    pp = V.mannequin_params(model, SLIM)
    moved = np.linalg.norm(pp.offsets, axis=1) > 0.002
    rest = M._rest(model, BodyParams(phenotype=pp.phenotype, scale=pp.scale))["_unscaled_vertices"][0].numpy()
    head_z = float(M._rest(model, pp)["bone_heads"][0, model.bone("head"), 2]) / pp.scale
    assert moved.sum() > 1000 and rest[moved, 2].min() > head_z - 0.1  # nur Kopf und Kinn
    path = V.export_puppet(model, pp, [M.demo_walk(seconds=0.5)], tmp_path / "puppe.glb")
    js, _ = _read_glb(path)
    assert [a["name"] for a in js["animations"]] == ["Testgang"]
