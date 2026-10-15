import numpy as np
import pytest
import torch

from fitscan import grip as G
from fitscan.body import BodyParams


@pytest.fixture(scope="module")
def ref(model):
    return G.reference_pose(model, BodyParams())


@pytest.fixture(scope="module")
def hands(model, ref):
    return {s: G.Hand(model, BodyParams(), s, ref) for s in ("L", "R")}


def _posed(model, grips):
    pose = model._identity_pose.clone()
    for g in grips:
        for name, r in g.rotations.items():
            pose[0, model.bone(name), :3, :3] = torch.tensor(r, dtype=torch.float32)
    rest = model.rest(model.phenotype_tensor(BodyParams()), model.local_tensor(BodyParams()))
    with torch.no_grad():
        out = model.posed(rest, pose)
        _, poses = model.anny.get_bone_transforms(pose, rest["_unscaled_bone_poses"])
    return out["vertices"][0].double().numpy(), poses[0].double().numpy()


def test_hand_frame(hands):
    for side, sign in (("L", 1), ("R", -1)):
        h = hands[side]
        for v in (h.across, h.along, h.palm):
            assert abs(np.linalg.norm(v) - 1) < 1e-9
        assert abs(h.across @ h.along) < 1e-9 and abs(h.palm @ h.along) < 1e-9
        # Die Finger zeigen in der Bezugshaltung nach unten und zur eigenen Seite
        assert h.along[2] < -0.3 and sign * h.along[0] > 0


def test_own_skinning_matches_anny(model, hands):
    g = G.solve_grip(hands["L"], 0.014)
    verts, _ = _posed(model, [g])
    world = {}
    for f in (G.THUMB, *G.FINGERS):
        names, _ = hands["L"].chain(f)
        world.update(zip(names, G._world([g.rotations[n] for n in names]), strict=True))
    assert np.abs(hands["L"].skin(world) - verts[hands["L"].ids]).max() < 0.0005


@pytest.mark.parametrize("radius", [0.0125, 0.014, 0.016, 0.02])
def test_fingers_close_around_bar_without_penetration(model, hands, radius):
    grips = [G.solve_grip(hands[s], radius) for s in ("L", "R")]
    verts, poses = _posed(model, grips)
    for g, hand in zip(grips, (hands["L"], hands["R"]), strict=True):
        bar = poses[model.bone(hand.wrist_name)] @ g.offset
        centre, axis = bar[:3, 3], bar[:3, 0]
        q = verts - centre
        along = q @ axis
        d = np.linalg.norm(q - np.outer(along, axis), axis=1) - radius
        near = np.abs(along) < 0.08
        assert d[near].min() > -0.001, "Haut steckt in der Stange"
        for finger in (*G.FINGERS, G.THUMB):
            assert abs(g.gap[finger]) < 0.004, f"{finger} liegt nicht an ({g.gap[finger] * 1000:.1f} mm)"
        # Ein echter Faustschluss: alle Finger deutlich gebeugt, Gelenkgrenzen eingehalten
        for finger in G.FINGERS:
            assert sum(g.angles[finger]) > 80
            assert all(a <= m + 1e-6 for a, m in zip(g.angles[finger], G.MAX_BEND, strict=True))


def test_sides_mirror(hands):
    a, b = (G.solve_grip(hands[s], 0.014) for s in ("L", "R"))
    for finger in G.FINGERS:
        assert np.allclose(a.angles[finger], b.angles[finger], atol=2)


def test_blend_and_quaternions(hands):
    g = G.solve_grip(hands["L"], 0.014)
    half = g.blend(0.5)
    for name, r in g.rotations.items():
        assert abs(G.angle_of(half[name]) - G.angle_of(r) / 2) < 0.5
        q = G.to_quaternion(r)
        assert abs(np.linalg.norm(q) - 1) < 1e-9
        x, y, z, w = q
        back = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        assert np.allclose(back, r, atol=1e-6)
