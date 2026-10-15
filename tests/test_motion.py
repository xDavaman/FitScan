import json
import struct

import numpy as np
import pytest
import roma
import torch

from fitscan import motion as M
from fitscan.body import BodyParams
from fitscan.export import _TO_Y_UP, build_rig

SLIM = BodyParams(phenotype={"height": 0.3, "weight": 0.3})
TALL = BodyParams(phenotype={"height": 0.9, "weight": 0.7, "muscle": 0.7})


@pytest.fixture(scope="module")
def walk():
    return M.turned(M.demo_walk(seconds=2.0), 90.0)


def test_save_load_keeps_everything(tmp_path, walk):
    offset = np.eye(4)
    offset[:3, 3] = [0.0, -0.05, 0.02]
    contacts = {"foot.L": np.arange(walk.frames) % 2 == 0, "hand.R": np.zeros(walk.frames, bool)}
    mo = M.Motion(walk.name, walk.fps, walk.bones, walk.rotations, walk.root, "künstlich", True, contacts,
                  [M.Attachment("Hantel", "wrist.R", 5, 40, offset, "gegenstaende/hantel.glb")])
    back = M.Motion.load(mo.save(tmp_path / "gang.npz"))
    assert back.name == mo.name and back.bones == mo.bones and back.loop and back.fps == mo.fps
    assert np.allclose(back.rotations, mo.rotations) and np.allclose(back.root, mo.root)
    assert set(back.contacts) == set(contacts)
    assert np.array_equal(back.contacts["foot.L"], contacts["foot.L"])
    a = back.attachments[0]
    assert (a.name, a.bone, a.start, a.end, a.asset) == ("Hantel", "wrist.R", 5, 40, "gegenstaende/hantel.glb")
    assert np.allclose(a.offset, offset)


def test_resample_keeps_duration(walk):
    mo = M.Motion(walk.name, walk.fps, walk.bones, walk.rotations, walk.root,
                  attachments=[M.Attachment("Ball", "wrist.L", 0, walk.frames - 1, np.eye(4))])
    r = mo.resampled(60.0)
    assert abs(r.duration - mo.duration) < 1e-6
    assert r.frames == 2 * (mo.frames - 1) + 1
    assert np.allclose(np.linalg.norm(r.rotations, axis=-1), 1.0, atol=1e-5)
    assert np.allclose(r.rotations[::2], mo.rotations, atol=1e-5)
    assert r.attachments[0].end == r.frames - 1


def _squat(seconds: float = 2.0, fps: float = 30.0) -> M.Motion:
    """Zweimal in die Hocke, Füße bleiben stehen."""
    t = np.arange(int(seconds * fps) + 1) / fps
    bend = np.sin(np.pi * t / seconds * 2) ** 2
    bones = ["upperleg01.L", "upperleg01.R", "lowerleg01.L", "lowerleg01.R", "foot.L", "foot.R"]
    rv = np.zeros((len(t), len(bones), 3), np.float32)
    rv[:, 0:2, 0] = -0.9 * bend[:, None]  # Hüfte beugen
    rv[:, 2:4, 0] = 1.5 * bend[:, None]  # Knie beugen
    rv[:, 4:6, 0] = -0.5 * bend[:, None]  # Fuß gegen das Schienbein
    root = np.stack([np.zeros_like(t), np.zeros_like(t), 1.0 - 0.3 * bend], 1)
    return M.from_rotvecs("Hocke", fps, bones, rv, root, source="künstlich")


def test_same_motion_fits_every_body(model):
    """Hüfte skaliert mit der Hüfthöhe: Füße stehen bei großen und kleinen Körpern auf dem Boden."""
    hips = []
    source = M.fix_ground(model, SLIM, _squat())  # Kontakte werden einmal an der Aufnahme erkannt
    for params in (SLIM, TALL):
        mo = M.fix_ground(model, params, source)
        pb = M.play(model, params, mo)
        low = pb.vertices[..., 2].min(1)
        hips.append(pb.hip_height)
        assert all(mo.contacts[f].all() for f in M.FEET)
        assert np.abs(low).max() < 3e-3, np.abs(low).max()  # weder im Boden noch schwebend
    assert hips[1] > hips[0] + 0.05


def test_standing_foot_does_not_slide(model):
    """Hüfte wandert während der Hocke zur Seite: fix_ground hält die stehenden Füße fest."""
    mo = _squat()
    mo.root[:, 0] += np.linspace(0.0, 0.4, mo.frames)
    free = M.fix_ground(model, SLIM, mo, contacts={f: np.zeros(mo.frames, bool) for f in M.FEET})
    fixed = M.fix_ground(model, SLIM, mo)

    def slide(motion):
        pb = M.play(model, SLIM, motion)
        sole = M._sole_vertices(model, "foot.L")
        xy = pb.vertices[:, sole, :2].mean(1)
        return float(np.linalg.norm(xy - xy[0], axis=1).max())

    assert slide(free) > 0.2
    assert slide(fixed) < 0.02, slide(fixed)


def _read_glb(path):
    data = path.read_bytes()
    n = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + n]), data[20 + n + 8:]


def test_glb_animation_reproduces_anny(model, tmp_path):
    """Skinning aus der GLB-Datei nachrechnen und mit Annys eigener Verformung vergleichen."""
    mo = M.demo_walk(seconds=1.0)
    path = M.export_glb(model, SLIM, [mo, M.turned(mo, 90.0)], tmp_path / "gang.glb")
    js, blob = _read_glb(path)
    assert [a["name"] for a in js["animations"]] == ["Testgang", "Testgang"]

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
    skin = js["skins"][0]
    joints = skin["joints"]
    ibm = acc(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    nodes = js["nodes"]

    def trs(n):
        m = np.eye(4)
        q = torch.tensor(n.get("rotation", [0, 0, 0, 1]), dtype=torch.float64)
        m[:3, :3] = roma.unitquat_to_rotmat(q).numpy()
        m[:3, 3] = n.get("translation", [0, 0, 0])
        return m

    anim = js["animations"][0]
    times = acc(anim["samplers"][0]["input"])[:, 0]
    assert len(times) == mo.frames and abs(times[-1] - mo.duration) < 1e-5
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

    rig = build_rig(model, SLIM)
    assert len(pos) == len(rig.vertices)
    pb = M.play(model, SLIM, mo)
    ph = np.concatenate([pos, np.ones((len(pos), 1))], 1)
    for f in (0, mo.frames // 2, mo.frames - 1):
        v = np.einsum("vk,vkab,vb->va", weights4, skinning[f][joints4], ph)[:, :3]
        target = pb.vertices[f] @ _TO_Y_UP[:3, :3].T
        d = np.linalg.norm(v - target, axis=1)
        assert d.mean() < 1e-3 and d.max() < 0.02, (f, d.mean(), d.max())


@pytest.mark.slow
def test_motion_from_video(model, tmp_path):
    """Bekannten Gang künstlich filmen und mit mocap zurückgewinnen."""
    from fitscan import mocap, synthetic
    from fitscan.video import read_frames
    from fitscan.vision import PoseAnalyzer

    params = BodyParams(phenotype={"height": 0.6, "weight": 0.5, "muscle": 0.6})
    h = M.hip_height(model, params)
    mo = M.turned(M.demo_walk(seconds=1.5), 90.0)
    mo.root[:, 0] -= 0.8 / h
    truth = M.fix_ground(model, params, mo)
    skin = np.zeros((512, 512, 3), np.uint8)
    skin[:] = (200, 150, 125)
    skin[256:] = (60, 70, 110)
    gt = synthetic.motion_video(model, params, truth, tmp_path / "gang.mp4", skin, width=640, height=360,
                                pitch_deg=5.0, distance=3.5)
    analyzer = PoseAnalyzer(video_mode=True)
    obs = []
    for idx, ts, rgb in read_frames(tmp_path / "gang.mp4", fps=30.0):
        o = analyzer.analyze(rgb, idx, ts)
        obs.append(mocap.Observation(o.keypoints, o.world))
    analyzer.close()
    tilt = mocap._M.numpy() @ gt["to_cam"].T  # Neigung bekannt, wie aus level.estimate oder dem Lagesensor
    cap = mocap.fit_motion(model, params, obs, gt["intr"], 30.0, tilt=tilt)
    assert cap.frames_used == cap.frames_total == truth.frames
    assert cap.reprojection_px < 3.0
    assert abs(cap.camera_height - 1.0) < 0.15
    rec = M.play(model, params, cap.motion, vertices=False).bones[:, :, :3, 3]
    tru = gt["playback"].bones[:, :, :3, 3] - np.array([gt["eye"][0], gt["eye"][1], 0.0])
    walked = rec[-1, 0, 0] - rec[0, 0, 0]
    assert abs(walked - (tru[-1, 0, 0] - tru[0, 0, 0])) < 0.15
    rel = np.linalg.norm((rec - rec[:, :1]) - (tru - tru[:, :1]), axis=-1)
    torso = [model.bone(n) for n in ("spine01", "head", "upperleg01.L", "upperleg01.R", "upperarm01.R")]
    limbs = [model.bone(n) for n in ("lowerleg01.L", "lowerleg01.R", "foot.L", "foot.R", "wrist.R")]
    # Einfarbige Figur von der Seite: Beine und der verdeckte Arm sind schwer; echte Haut hilft MediaPipe
    assert rel[:, torso].mean() < 0.06, rel[:, torso].mean()
    assert rel[:, limbs].mean() < 0.2, rel[:, limbs].mean()
