import math

import cv2
import numpy as np
import pytest

from fitscan import level
from fitscan.camera import Intrinsics


def _rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _tilted(pitch: float, roll: float) -> np.ndarray:
    """Waagerechte Kamera -> gekippte Kamera (+pitch = nach oben)."""
    return _rot_z(roll) @ _rot_x(-pitch)


def _project(points: np.ndarray, rot: np.ndarray, intr: Intrinsics) -> np.ndarray:
    c = points @ rot.T
    return np.c_[intr.focal * c[:, 0] / c[:, 2] + intr.cx, intr.focal * c[:, 1] / c[:, 2] + intr.cy]


def _room(rot: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Türrahmen und Schrankkanten als senkrechte Linien im Raum, gezeichnet von der gekippten Kamera."""
    img = np.full((intr.height, intr.width, 3), 40, np.uint8)
    for x in (-1.2, -0.8, -0.5, 0.6, 0.9, 1.3):
        for z in (3.0, 3.6):
            top, bottom = _project(np.array([[x, -1.5, z], [x, 0.9, z]]), rot, intr)
            cv2.line(img, tuple(np.round(top).astype(int)), tuple(np.round(bottom).astype(int)),
                     (230, 230, 230), 4, cv2.LINE_AA)
    return img


@pytest.mark.parametrize("pitch,roll", [(8.0, 0.0), (-12.0, 2.0), (0.0, 0.0)])
def test_estimate_recovers_tilt(pitch, roll):
    intr = Intrinsics.phone_default(720, 1280)
    lev = level.estimate([_room(_tilted(pitch, roll), intr)], [None], intr)
    assert lev is not None
    assert lev.pitch_deg == pytest.approx(pitch, abs=0.5)
    assert lev.roll_deg == pytest.approx(roll, abs=0.5)


def test_apply_matches_level_camera():
    intr = Intrinsics.phone_default(720, 1280)
    rot = _tilted(9.0, 1.5)
    lev = level.estimate([_room(rot, intr)], [None], intr)
    pts = np.array([[0.1, 0.8, 2.5], [-0.2, -0.9, 2.4], [0.3, 0.0, 2.6]])  # Füße, Kopf, Hüfte
    tilted = _project(pts, rot, intr)
    mask = np.zeros((intr.height, intr.width), np.float32)
    cv2.fillConvexPoly(mask, np.round(tilted).astype(np.int32), 1.0)
    dst = level.leveled_intrinsics(lev, intr, [mask])
    kp = np.c_[tilted, np.ones(3)].astype(np.float32)
    warped_mask, warped_kp = level.apply(lev, intr, dst, mask, kp)
    # Nach dem Ausgleich liegen die Punkte dort, wo eine waagerechte Kamera sie sehen würde
    np.testing.assert_allclose(warped_kp[:, :2], _project(pts, np.eye(3), dst), atol=3.0)
    assert warped_mask.sum() > 0.8 * mask.sum()
    assert warped_mask.shape == (dst.height, dst.width)


def test_no_lines_no_estimate():
    intr = Intrinsics.phone_default(720, 1280)
    assert level.estimate([np.full((1280, 720, 3), 128, np.uint8)], [None], intr) is None
