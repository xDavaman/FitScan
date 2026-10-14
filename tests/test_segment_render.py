import cv2
import numpy as np

from fitscan.body import BodyParams
from fitscan.camera import Intrinsics, render_mask
from fitscan.segment import refine_mask


def test_rendered_silhouette_has_no_holes(model):
    verts = model.rest_numpy(BodyParams())["vertices"]
    intr = Intrinsics.phone_default(360, 640)
    zc = (verts[:, 2].max() + verts[:, 2].min()) / 2
    mask = render_mask(verts, model.faces, 0.3, np.array([0.0, zc, 2.5]), intr) > 0
    n, _ = cv2.connectedComponents((~mask).astype(np.uint8))
    assert n - 1 <= 3  # Hintergrund plus höchstens kleine Lücken zwischen Arm und Körper


def test_refine_mask_sharpens_blurry_edges():
    rng = np.random.default_rng(0)
    img = (rng.normal(0, 20, (480, 320, 3)) + 60).clip(0, 255).astype(np.uint8)
    truth = np.zeros((480, 320), np.uint8)
    cv2.ellipse(truth, (160, 240), (70, 200), 0, 0, 360, 1, -1)
    img[truth > 0] = (210, 170, 150)
    soft = cv2.GaussianBlur(cv2.dilate(truth.astype(np.float32), np.ones((9, 9))), (31, 31), 0)
    before = ((soft > 0.5) & (truth > 0)).sum() / ((soft > 0.5) | (truth > 0)).sum()
    refined = refine_mask(img, soft) > 0.5
    after = (refined & (truth > 0)).sum() / (refined | (truth > 0)).sum()
    assert after > before
    assert after > 0.97


def test_keypoint_calibration_loaded():
    from fitscan.camera import KEYPOINT_OFFSETS, MEDIAPIPE_TO_BONE

    assert set(KEYPOINT_OFFSETS) == set(MEDIAPIPE_TO_BONE)
    # Schulterpunkt von MediaPipe liegt über Annys Schultergelenk, links/rechts spiegelgleich
    assert KEYPOINT_OFFSETS[11][2] > 0.005
    assert np.allclose(KEYPOINT_OFFSETS[11] * [-1, 1, 1], KEYPOINT_OFFSETS[12])
