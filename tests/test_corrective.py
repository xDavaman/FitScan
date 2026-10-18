import dataclasses

import numpy as np
import pytest

from fitscan import contact as C
from fitscan import corrective as K
from fitscan import posing as P
from fitscan.body import BodyParams


@pytest.fixture(scope="module")
def fig(model):
    return P.Figure.from_params(model, BodyParams())


def _depth(checker, vertices, where):
    return max([f.depth for f in checker.check_frame(vertices) if f.kind == "koerper" and f.where == where],
               default=0.0)


def test_winding_inside_and_outside(model, fig):
    v = fig.reference_vertices
    spine = fig.reference_poses[model.bone("spine03"), :3, 3]
    w = C.winding(np.array([spine, [3.0, 3.0, 3.0]]), v, model.faces)
    assert w[0] == pytest.approx(1, abs=0.05) and abs(w[1]) < 0.05


def test_elbow_shape_removes_biceps_contact(model, fig):
    joint = next(j for j in K.joints() if j.name == "ellbogen" and j.side == "L")
    joint = dataclasses.replace(joint, angles=(135.0,))
    shapes = K.build(fig, joint)
    (angle, delta), = shapes.items()
    assert angle == pytest.approx(135, abs=1)
    frame = [fig.poser.rotations(joint.pose(135))]
    checker = C.Checker(model, fig.reference_vertices)
    before = _depth(checker, fig.pose(frame)["vertices"][0], "Oberarm links / Unterarm links")
    after = _depth(checker, fig.pose(frame, offsets=delta[None])["vertices"][0], "Oberarm links / Unterarm links")
    assert before > 0.015 and after < before / 3
    # nur Arm und Umgebung bewegen sich, der Rest bleibt
    moved = np.linalg.norm(delta, axis=1) > 1e-5
    parts = checker.vertex_part[moved]
    assert set(parts) <= {"Oberarm links", "Unterarm links"}


def test_weights_interpolate_and_clamp(model, fig):
    n = len(fig.reference_vertices)
    shapes = {"ellbogen.L": (np.array([60.0, 120.0]), np.stack([np.full((n, 3), 1.0), np.full((n, 3), 3.0)]))}
    cs = K.Correctives(model, fig.reference_poses, shapes)
    elbows = [0, 30, 90, 160]
    poses = fig.pose([fig.poser.rotations(P.pose(None, arm_L=dict(elbow=e))) for e in elbows])["poses"]
    off = cs.offsets(poses)[:, 0, 0]
    assert off == pytest.approx([0, 0.5, 2, 3], abs=0.1)


def test_shoulder_angle_counts_only_arm_to_body(model, fig):
    joint = next(j for j in K.joints() if j.name == "schulter" and j.side == "L")
    poses = [P.pose(None, arm_L=dict(down=d, fwd=f)) for d, f in ((0, 0), (32, 5), (40, 0), (0, 90))]
    a = joint.angle(model, fig.pose([fig.poser.rotations(p) for p in poses])["poses"], fig.reference_poses)
    assert abs(a[0]) < 2
    assert 25 < a[1] < a[2] < 45
    assert a[3] <= 0  # nach vorn gehoben: keine Schulterform
