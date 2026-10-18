import numpy as np
import pytest

from fitscan import grip as G
from fitscan import gym
from fitscan import posing as P
from fitscan.body import BodyParams


@pytest.fixture(scope="module")
def fig(model):
    return P.Figure.from_params(model, BodyParams())


def _angle(a, b):
    return np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def test_reference_matches_grip_reference(model, fig):
    ref = G.reference_pose(model, BodyParams())
    assert np.abs(fig.reference_vertices - ref["vertices"]).max() < 1e-5


def test_elbow_and_knee_are_absolute(model, fig):
    for elbow in (0, 90):
        p = P.pose(None, arm_L=dict(elbow=elbow))
        heads = fig.pose([fig.poser.rotations(p)])["poses"][0][:, :3, 3]
        upper = heads[model.bone("lowerarm01.L")] - heads[model.bone("upperarm01.L")]
        fore = heads[model.bone("wrist.L")] - heads[model.bone("lowerarm01.L")]
        assert _angle(upper, fore) == pytest.approx(elbow, abs=1.5)
    p = P.pose(None, leg_R=dict(knee=60))
    heads = fig.pose([fig.poser.rotations(p)])["poses"][0][:, :3, 3]
    thigh = heads[model.bone("lowerleg01.R")] - heads[model.bone("upperleg01.R")]
    shin = heads[model.bone("foot.R")] - heads[model.bone("lowerleg01.R")]
    assert _angle(thigh, shin) == pytest.approx(60, abs=1.5)
    assert shin[1] > 0  # Unterschenkel geht nach hinten (Person blickt nach -Y)


def test_directions(model, fig):
    rest = fig.pose([fig.poser.rotations(P.pose(None))])["poses"][0]
    bent = fig.pose([fig.poser.rotations(P.pose(None, body=dict(pitch=40)))])["poses"][0]
    head = model.bone("head")
    assert bent[head, 1, 3] < rest[head, 1, 3] - 0.2  # Oberkörper kippt nach vorn (-Y)
    up = fig.pose([fig.poser.rotations(P.pose(None, arm_L=dict(fwd=90)))])["poses"][0]
    wrist = model.bone("wrist.L")
    assert up[wrist, 1, 3] < rest[wrist, 1, 3] - 0.3  # Arm nach vorn gehoben


def test_turn_is_mirrored(model, fig):
    """Gleiche Werte drehen links und rechts spiegelbildlich (Fußspitzen beide nach außen)."""
    po = fig.pose([fig.poser.rotations(P.pose(None, legs=dict(turn=-30), arms=dict(forearm=60)))])["poses"][0]
    for s in P.SIDES:
        foot = po[model.bone(f"toe3-1.{s}"), :3, 3] - po[model.bone(f"foot.{s}"), :3, 3]
        assert foot[0] * (1 if s == "L" else -1) > 0.3 * np.linalg.norm(foot)
    a, b = (po[model.bone(f"finger2-1.{s}"), :3, 3] for s in P.SIDES)
    assert a[0] == pytest.approx(-b[0], abs=0.01) and a[1:] == pytest.approx(b[1:], abs=0.01)


def test_quaternions_roundtrip(fig):
    rots = fig.poser.rotations(P.pose(None, body=dict(pitch=20), arm_R=dict(elbow=70)))
    q = fig.poser.quaternions(rots)
    assert q.shape == (len(fig.poser.names), 4)
    assert np.allclose(np.linalg.norm(q, axis=1), 1)


def test_sample_hits_keys():
    a, b = P.pose(None, arm_L=dict(elbow=0)), P.pose(None, arm_L=dict(elbow=100))
    out = P.sample([(0.0, a), (1.0, b)], np.array([0.0, 0.5, 1.0]))
    values = [o[("lowerarm01.L", "beugen")] for o in out]
    assert values[0] == pytest.approx(0) and values[2] == pytest.approx(100)
    assert 0 < values[1] < 100


def test_orient_grip_lays_bar_across(model, fig):
    grips = gym._grips(fig, gym.BAR_RADIUS)
    palms = {s: np.array([0, 1.0, 0]) for s in P.SIDES}
    frames, wrist = gym.hold(fig, [fig.poser.rotations(gym.standing())], grips, palms)
    poses = fig.pose(frames)["poses"][0]
    for s, g in grips.items():
        bar = poses[model.bone(f"wrist.{s}")] @ g.offset
        assert abs(bar[0, 0]) > 0.999  # Stange quer (x)
        assert bar[1, 2] > 0.999  # Handfläche zum Körper
    assert wrist.max() < 45
