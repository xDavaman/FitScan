import numpy as np
import pytest
import torch

from fitscan import contact as C
from fitscan import grip as G
from fitscan.body import BodyParams


@pytest.fixture(scope="module")
def ref(model):
    return G.reference_pose(model, BodyParams())


@pytest.fixture(scope="module")
def checker(model, ref):
    return C.Checker(model, ref["vertices"])


def _on_floor(v):
    v = v.copy()
    v[:, 2] -= v[:, 2].min()
    return v


def test_primitives():
    cyl = C.Cylinder("Stange", np.zeros(3), np.array([1.0, 0, 0]), 0.014, 1.1)
    pts = np.array([[0, 0, 0], [0, 0.01, 0], [0, 0.02, 0], [1.2, 0, 0]])
    assert np.allclose(cyl.depth(pts), [0.014, 0.004, -0.006, -0.1])
    move = np.eye(4)
    move[:3, 3] = [0, 0, 1]
    assert cyl.moved(move).depth(np.array([[0, 0, 1.0]]))[0] == pytest.approx(0.014)
    box = C.Box("Bank", np.zeros(3), np.eye(3), np.array([0.2, 0.6, 0.22]))
    assert box.depth(np.array([[0, 0, 0.2], [0, 0, 0.3]])) == pytest.approx([0.02, -0.08])


def test_segment_triangle():
    a, b, c = np.array([[0.0, 0, 0]]), np.array([[1.0, 0, 0]]), np.array([[0.0, 1, 0]])
    through = C._segment_hits_triangle(np.array([[0.2, 0.2, -1]]), np.array([[0.2, 0.2, 1]]), a, b, c)
    beside = C._segment_hits_triangle(np.array([[0.8, 0.8, -1]]), np.array([[0.8, 0.8, 1]]), a, b, c)
    short = C._segment_hits_triangle(np.array([[0.2, 0.2, 0.1]]), np.array([[0.2, 0.2, 1]]), a, b, c)
    assert through[0] and not beside[0] and not short[0]


def test_reference_pose_is_clean(checker, ref):
    assert checker.check_frame(_on_floor(ref["vertices"])) == []


def test_finds_floor_object_and_body(model, checker):
    rest = model.rest(model.phenotype_tensor(BodyParams()), model.local_tensor(BodyParams()))
    pose = model._identity_pose.clone()
    for s in "LR":  # tiefe Hocke ohne Korrektur: Wade drückt in den Oberschenkel
        pose[0, model.bone(f"lowerleg01.{s}"), :3, :3] = torch.tensor(G.rotation([1, 0, 0], 150))
    with torch.no_grad():
        v = model.posed(rest, pose)["vertices"][0].double().numpy()
    v = _on_floor(v)
    v[:, 2] -= 0.01
    bar = C.Cylinder("Stange", np.array([0, 0, 0.3]), np.array([1.0, 0, 0]), 0.5, 0.05)
    found = checker.check_frame(v, [bar])
    kinds = {(f.kind, f.where) for f in found}
    assert ("koerper", "Oberschenkel links / Unterschenkel links") in kinds
    assert ("koerper", "Oberschenkel rechts / Unterschenkel rechts") in kinds
    assert any(k == "boden" for k, _ in kinds)
    floor = [f for f in found if f.kind == "boden"]
    assert max(f.depth for f in floor) == pytest.approx(0.01, abs=1e-6)
    assert any(f.kind == "geraet" and f.detail == "Stange" for f in found)
    report = C.summarize([found, [], found])
    assert any(line.startswith("boden:") for line in report)


def test_body_parts():
    assert C.body_part("finger2-3.L") == "Hand links"
    assert C.body_part("upperleg02.R") == "Oberschenkel rechts"
    assert C.body_part("spine03") == "Bauch"
