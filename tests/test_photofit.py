"""Foto-/Videoscan der iPhone-App: der von Hand abgeleitete Rechenweg (numpy) trifft PyTorch."""

import math

import numpy as np
import pytest
import torch

from fitscan import photofit, synthetic
from fitscan.body import BodyParams


@pytest.fixture(scope="module")
def scene(model):
    params, extra = synthetic.example_body(3)
    frames = synthetic.film(model, params, arm_deg=extra["arm_deg"], views=4, seed=3)
    rest = model.rest_numpy(BodyParams())  # bewusst ein anderer Körper als im Bild
    rig = photofit.Rig(model)
    rng = np.random.default_rng(0)
    pose = rng.normal(0, 0.15, (len(rig.pose_bones), 3))
    view = photofit.prepare_view(model, frames[1])
    return model, rig, rest, pose, view


def test_prepare_view(scene):
    model, _, _, _, view = scene
    assert max(view.mask.shape) == photofit.WORK_SIDE
    assert len(view.contour) == photofit.CONTOUR_POINTS
    assert view.dt[view.mask].max() == 0 and view.dt[~view.mask].min() >= 1
    # Randpunkte liegen auf der Maske, jeweils mit einem Nachbarn außen
    x, y = view.contour[:, 0].astype(int), view.contour[:, 1].astype(int)
    assert view.mask[y, x].all()
    assert len(view.kp_bones) == 12


def test_resize_mask_like_area():
    import cv2

    rng = np.random.default_rng(1)
    mask = np.zeros((960, 540), np.float32)
    cv2.ellipse(mask, (270, 480), (120, 400), 10, 0, 360, 1.0, -1)
    mask += (rng.random(mask.shape) < 0.01)
    ours = photofit.resize_mask(np.clip(mask, 0, 1), 360, 640)
    ref = cv2.resize(np.clip(mask, 0, 1), (360, 640), interpolation=cv2.INTER_AREA) > 0.5
    assert (ours != ref).mean() < 0.005


def test_view_gradient_matches_torch(scene):
    model, rig, rest, pose, view = scene
    verts = rest["vertices"].astype(np.float64)
    heads = rest["bone_heads"].astype(np.float64)
    rs, ts = photofit.transforms_np(rig, heads, pose)
    posed = photofit.Rig.skin(rig, torch.as_tensor(verts), torch.as_tensor(rs), torch.as_tensor(ts)).numpy()
    b = view.kp_bones
    kp = np.einsum("pij,pj->pi", rs[b], heads[b]) + ts[b]
    yaw, trans = 0.4 + math.pi / 2, np.array([0.02, 0.85, 2.9])

    tv = torch.tensor(posed, requires_grad=True)
    tk = torch.tensor(kp, requires_grad=True)
    ty = torch.tensor(yaw, dtype=torch.float64, requires_grad=True)
    tt = torch.tensor(trans, requires_grad=True)
    loss = photofit.view_loss(tv, tk, ty, tt, view, 0.5)
    loss.backward()

    mine, g_v, g_k, g_y, g_t = photofit.view_loss_np(posed, kp, yaw, trans, view, 0.5)
    assert mine == pytest.approx(float(loss.detach()), rel=1e-9)
    assert np.allclose(g_v, tv.grad.numpy(), atol=1e-12, rtol=1e-6)
    assert np.allclose(g_k, tk.grad.numpy(), atol=1e-12, rtol=1e-6)
    assert g_y == pytest.approx(float(ty.grad), rel=1e-6)
    assert np.allclose(g_t, tt.grad.numpy(), rtol=1e-6)


def test_skin_gradient_matches_torch(scene):
    model, rig, rest, pose, view = scene
    verts = rest["vertices"].astype(np.float64)
    heads = rest["bone_heads"].astype(np.float64)
    rng = np.random.default_rng(2)
    g_posed = rng.normal(0, 1, verts.shape)
    b = view.kp_bones
    g_kp = rng.normal(0, 1, (len(b), 3))

    tv = torch.tensor(verts, requires_grad=True)
    tp = torch.tensor(pose, requires_grad=True)
    th = torch.tensor(heads)
    rs, ts = rig.transforms(th, tp)
    posed = rig.skin(tv, rs, ts)
    kp = (rs[b] @ th[b][..., None])[..., 0] + ts[b]
    ((posed * torch.as_tensor(g_posed)).sum() + (kp * torch.as_tensor(g_kp)).sum()).backward()

    g_rest, g_pose = photofit.skin_gradient_np(rig, verts, heads, pose, g_posed, b, g_kp)
    assert np.allclose(g_rest, tv.grad.numpy(), atol=1e-10)
    assert np.allclose(g_pose, tp.grad.numpy(), rtol=1e-6, atol=1e-8)
