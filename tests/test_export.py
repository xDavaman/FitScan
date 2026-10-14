import json
import struct

import numpy as np
import pytest

from fitscan.body import BodyParams
from fitscan.export import bone_name, build_rig, cascadeur_check, export_body, write_glb
from fitscan.mesh_check import check_mesh


@pytest.fixture(scope="module")
def rig(model):
    return build_rig(model, BodyParams(phenotype={"weight": 0.6, "muscle": 0.6}))


def _read_glb(path):
    data = path.read_bytes()
    magic, version, total = struct.unpack("<4sII", data[:12])
    assert magic == b"glTF" and version == 2 and total == len(data)
    length, kind = struct.unpack("<I4s", data[12:20])
    assert kind == b"JSON"
    return json.loads(data[20:20 + length])


def test_mesh_is_watertight(model, rig):
    assert rig.report.ok, rig.report.summary()
    assert rig.report.components == 4


def test_y_up_feet_on_ground_real_size(rig):
    assert abs(rig.vertices[:, 1].min()) < 1e-6
    assert 1.4 < rig.vertices[:, 1].max() < 2.1
    assert rig.vertices[:, 2].mean() > 0  # Blick nach +Z


def test_weights_and_joints(rig):
    assert np.allclose(rig.weights4.sum(1), 1.0, atol=1e-5)
    assert rig.joints4.max() < len(rig.joint_names)
    assert np.all(rig.joints4[rig.weights4 == 0] == 0)


def test_joint_hierarchy_reproduces_bind_pose(rig, tmp_path):
    gltf = _read_glb(write_glb(rig, tmp_path / "k.glb"))
    nodes = gltf["nodes"]
    skin = gltf["skins"][0]

    def local(n):
        x, y, z, w = n.get("rotation", [0, 0, 0, 1])
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        m = np.eye(4)
        m[:3, :3] = r
        m[:3, 3] = n.get("translation", [0, 0, 0])
        return m

    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}

    def world(i):
        m = local(nodes[i])
        while i in parent:
            i = parent[i]
            m = local(nodes[i]) @ m
        return m

    for j, node_index in enumerate(skin["joints"]):
        assert np.allclose(world(node_index), rig.joint_global[j], atol=1e-4)


def test_morph_targets_present(rig):
    assert set(rig.morphs) == {"weight_up", "weight_down", "muscle_up", "muscle_down"}
    assert np.abs(rig.morphs["weight_up"]).max() > 0.005


def test_cascadeur_rules(rig):
    assert cascadeur_check(rig) == []


def test_unreal_names():
    assert bone_name("upperarm01.L", "unreal") == "upperarm_l"
    assert bone_name("wrist.R", "unreal") == "hand_r"
    assert bone_name("finger1-1.L", "unreal") == "thumb_01_l"
    assert bone_name("spine05", "unreal") == "spine_01"


def test_export_body_files(model, tmp_path):
    out = export_body(model, BodyParams(), tmp_path, name="test")
    assert out["glb"].exists() and out["glb_unreal"].exists() and out["obj"].exists()
    names = [n["name"] for n in _read_glb(out["glb_unreal"])["nodes"]]
    assert len(set(names)) == len(names) and "upperarm_l" in names and "root" in names
    lines = out["obj"].read_text().splitlines()
    assert sum(1 for x in lines if x.startswith("f ")) == len(model.faces)


def test_mesh_check_finds_holes(model):
    v = BodyParams()
    verts = model.rest_numpy(v)["vertices"]
    assert check_mesh(verts, model.faces).ok
    holed = check_mesh(verts, model.faces[10:])
    assert not holed.ok and not holed.watertight


def test_mesh_check_finds_intersections_from_fine_tuning(model):
    verts = model.rest_numpy(BodyParams())["vertices"]
    assert check_mesh(verts, model.faces, template=verts).intersections == 0
    pushed = verts.copy()
    arm = model.arm_vertex_mask() & (verts[:, 0] > 0)
    pushed[arm, 0] -= 0.12  # linker Arm in den Oberkörper geschoben
    report = check_mesh(pushed, model.faces, template=verts)
    assert report.intersections > 0 and not report.ok


def test_repair_removes_flips(model):
    from fitscan.fit import repair_offsets

    p = BodyParams()
    p.offsets = np.random.default_rng(0).normal(0, 0.003, (model.num_vertices, 3)).astype(np.float32)
    repaired, touched = repair_offsets(model, p)
    assert touched > 0
    rig = build_rig(model, repaired, morphs=False)
    assert rig.report.ok, rig.report.summary()
