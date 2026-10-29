"""Foto- und Videoscan der iPhone-App: Körperform aus wenigen Umrissen, ohne automatisches Ableiten.

Derselbe Ablauf wie `fit.fit_to_frames`, aber so eingeschränkt, dass FitScanCore (Swift) jeden
Schritt mit von Hand abgeleiteten Gradienten nachrechnen kann:

- Form: Phänotyp, lokale Regler und Maßstab wie `fit_to_measurements`, keine Feinform-Offsets.
- Haltung: Drehung einiger Gelenke um ihren Kopf im Raum der Ruhepose (eigene kleine Kinematik
  statt Annys Pose-Parametrisierung). Sie dient nur dem Ausrichten und wird nicht exportiert.
- Bildterme wie am PC: Abstand der Modellpunkte zur Maske, Randpunkte der Maske zum nächsten
  Modellpunkt, Körperpunkte.
- Was Swift nicht ableitet, ist hier abgekoppelt (`detach`): Knochenköpfe in den Gelenk-
  transformationen und in den Körperpunkten hängen nur über die Haltung, nicht über die Form ab.

Die Funktionen mit `_np` sind der Rechenweg der App in numpy; `tests/test_photofit.py` prüft ihre
Gradienten gegen PyTorch, `swiftdata.reference` schreibt Prüfwerte für die Swift-Tests.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import cv2
import numpy as np
import torch

from .body import BodyModel, BodyParams, axis_angle_to_matrix
from .camera import KEYPOINT_OFFSETS, MEDIAPIPE_TO_BONE, Intrinsics, render_mask
from .fit import (
    KEYPOINT_WEIGHT_SHAPE,
    LOCAL_REG,
    MUSCLE_STARTS,
    POSE_BONES,
    SILHOUETTE_WEIGHT,
    VIDEO_LOCALS,
    FitFrame,
    FrameFitResult,
    Person,
    ShapeVariables,
    exact_height,
    fit_to_measurements,
    measurement_loss,
)
from .measure import MeasureProbe, measure_params

Log = Callable[[str], None]

WORK_SIDE = 640  # lange Seite der Arbeitsmaske in Pixeln (wie am PC)
CONTOUR_POINTS = 400  # Randpunkte der Maske je Bild
POSE_REG = 0.01  # Haltung sanft zur Ruhepose ziehen
LR = 0.02
PROBE_EVERY = 50  # Maßband-Schnitte neu festlegen
STEPS = (80, 160)  # Schritte: Lage und Haltung, dann Körperform
DTYPE = torch.float64


# --------------------------------------------------------------------------------------
# Bild vorbereiten (in Swift genauso)
# --------------------------------------------------------------------------------------

def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Flächenmittel über ganzzahlige Quellbereiche, dann Schwelle 0,5 (wie cv2.INTER_AREA beim
    Verkleinern; beim Vergrößern nächster Nachbar)."""
    m = np.asarray(mask, np.float64)
    h, w = m.shape
    ii = np.zeros((h + 1, w + 1))
    ii[1:, 1:] = m.cumsum(0).cumsum(1)
    y0 = (np.arange(height) * h) // height
    y1 = np.maximum(y0 + 1, ((np.arange(height) + 1) * h) // height)
    x0 = (np.arange(width) * w) // width
    x1 = np.maximum(x0 + 1, ((np.arange(width) + 1) * w) // width)
    s = ii[y1][:, x1] - ii[y0][:, x1] - ii[y1][:, x0] + ii[y0][:, x0]
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return s / area > 0.5


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Größte zusammenhängende Fläche (8er-Nachbarschaft); bei Gleichstand die zuerst gefundene."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def distance_outside(mask: np.ndarray) -> np.ndarray:
    """Exakter euklidischer Abstand jedes Pixels zum nächsten Maskenpixel (innen 0)."""
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(np.float64)


def boundary_points(mask: np.ndarray, count: int = CONTOUR_POINTS) -> np.ndarray:
    """Randpixel der Maske (ein 4er-Nachbar liegt außen oder am Bildrand), gleichmäßig ausgedünnt."""
    pad = np.pad(mask, 1, constant_values=False)
    inner = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    ys, xs = np.nonzero(mask & ~inner)
    if len(xs) == 0:
        return np.zeros((0, 2))
    idx = np.linspace(0, len(xs) - 1, min(count, len(xs))).astype(int)
    return np.stack([xs[idx], ys[idx]], -1).astype(np.float64)


@dataclasses.dataclass
class PreparedView:
    intr: Intrinsics
    mask: np.ndarray  # (h, w) bool, Arbeitsauflösung
    dt: np.ndarray  # (h, w) Abstand außerhalb der Maske in Pixeln
    contour: np.ndarray  # (C, 2) Randpunkte x, y
    px_height: float
    bbox_center: tuple[float, float]
    kp_bones: list[int]  # Knochenindex je Körperpunkt
    kp_xy: np.ndarray  # (P, 2)
    kp_w: np.ndarray  # (P,)
    kp_off: np.ndarray  # (P, 3) Versatz Punkt -> Gelenk, Anteil der Körpergröße
    yaw_hint: float | None


def prepare_view(model: BodyModel, frame: FitFrame) -> PreparedView:
    factor = WORK_SIDE / max(frame.mask.shape)
    intr = frame.intr.scaled(factor)
    mask = largest_component(resize_mask(frame.mask, intr.width, intr.height))
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:
        raise ValueError("Auf einem Bild ist keine Person zu erkennen.")
    bones, xy, w, off = [], [], [], []
    if frame.keypoints is not None:
        for mp_idx, bone in MEDIAPIPE_TO_BONE.items():
            x, y, vis = frame.keypoints[mp_idx]
            if vis > 0.5:
                bones.append(model.bone(bone))
                xy.append([x * factor, y * factor])
                w.append(float(vis))
                off.append(KEYPOINT_OFFSETS[mp_idx] if frame.calibrated else np.zeros(3))
    return PreparedView(
        intr=intr, mask=mask, dt=distance_outside(mask), contour=boundary_points(mask),
        px_height=float(ys.max() - ys.min()),
        bbox_center=((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2),
        kp_bones=bones, kp_xy=np.array(xy, np.float64).reshape(-1, 2),
        kp_w=np.array(w, np.float64), kp_off=np.array(off, np.float64).reshape(-1, 3),
        yaw_hint=frame.yaw_init,
    )


# --------------------------------------------------------------------------------------
# Kinematik: Drehung ausgewählter Gelenke um ihren Kopf, an die Kinder weitergegeben
# --------------------------------------------------------------------------------------

class Rig:
    def __init__(self, model: BodyModel, bones: list[str] = POSE_BONES):
        self.parents = list(model.bone_parents)
        self.pose_bones = [model.bone(b) for b in bones]
        self.slot = {b: i for i, b in enumerate(self.pose_bones)}
        self.indices = model.vertex_bone_indices.astype(np.int64)
        self.weights = model.vertex_bone_weights.astype(np.float64)
        # Für jedes Haltungsgelenk: welche Knochen es mitdreht (es selbst und alle Nachfahren)
        n = len(self.parents)
        self.moves = np.zeros((len(self.pose_bones), n), bool)
        for b in range(n):
            a = b
            while a >= 0:
                if a in self.slot:
                    self.moves[self.slot[a], b] = True
                a = self.parents[a]

    def transforms(self, heads: torch.Tensor, pose_aa: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Je Knochen Drehung R und Verschiebung t: Punkt im Ruheraum -> gestellter Punkt."""
        rot = axis_angle_to_matrix(pose_aa)[..., :3, :3]
        eye = torch.eye(3, dtype=pose_aa.dtype, device=pose_aa.device)
        zero = torch.zeros(3, dtype=pose_aa.dtype, device=pose_aa.device)
        rs, ts = [], []
        for b, p in enumerate(self.parents):
            r, t = (eye, zero) if p < 0 else (rs[p], ts[p])
            if b in self.slot:
                ra, h = rot[self.slot[b]], heads[b]
                t = r @ (h - ra @ h) + t
                r = r @ ra
            rs.append(r)
            ts.append(t)
        return torch.stack(rs), torch.stack(ts)

    def skin(self, verts: torch.Tensor, rs: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        idx = torch.as_tensor(self.indices, device=verts.device)
        w = torch.as_tensor(self.weights, dtype=verts.dtype, device=verts.device)
        a = (w[..., None, None] * rs[idx]).sum(1)  # (V, 3, 3)
        c = (w[..., None] * ts[idx]).sum(1)
        return (a @ verts[..., None])[..., 0] + c


def transforms_np(rig: Rig, heads: np.ndarray, pose_aa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, t = rig.transforms(torch.as_tensor(heads, dtype=DTYPE), torch.as_tensor(pose_aa, dtype=DTYPE))
    return r.numpy(), t.numpy()


def left_jacobian_np(a: np.ndarray) -> np.ndarray:
    """J mit exp(a + δ) ≈ exp(J δ) exp(a)."""
    theta = float(np.linalg.norm(a))
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    if theta < 1e-6:
        return np.eye(3) + 0.5 * k
    return np.eye(3) + (1 - math.cos(theta)) / theta**2 * k + (theta - math.sin(theta)) / theta**3 * (k @ k)


def skin_gradient_np(rig: Rig, rest: np.ndarray, heads: np.ndarray, pose_aa: np.ndarray,
                     g_verts: np.ndarray, kp_bones: list[int], g_kp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradient nach den Ruhepunkten und nach der Haltung, aus dem Gradienten nach den gestellten
    Punkten (g_verts) und den gestellten Körperpunkt-Gelenken (g_kp)."""
    rs, ts = transforms_np(rig, heads, pose_aa)
    w, idx = rig.weights, rig.indices
    a = np.einsum("vk,vkij->vij", w, rs[idx])
    g_rest = np.einsum("vji,vj->vi", a, g_verts)  # Aᵀ g
    per_bone = np.einsum("vkij,vj->vki", rs[idx], rest) + ts[idx]  # R_b v + t_b
    g_pose = np.zeros_like(pose_aa)
    kp_pos = np.einsum("pij,pj->pi", rs[kp_bones], heads[kp_bones]) + ts[kp_bones] if kp_bones else None
    for j, b in enumerate(rig.pose_bones):
        p = rig.parents[b]
        rp, tp = (np.eye(3), np.zeros(3)) if p < 0 else (rs[p], ts[p])
        hj = rp @ heads[b] + tp  # gestellter Gelenkkopf
        sel = rig.moves[j][idx]  # (V, K)
        ws = np.where(sel, w, 0.0)
        s = np.einsum("vk,vki->vi", ws, per_bone) - ws.sum(1)[:, None] * hj
        torque = np.cross(s, g_verts).sum(0)
        if kp_bones:
            m = rig.moves[j][kp_bones]
            torque += np.cross(kp_pos[m] - hj, g_kp[m]).sum(0)
        g_pose[j] = left_jacobian_np(pose_aa[j]).T @ (rp.T @ torque)
    return g_rest, g_pose


# --------------------------------------------------------------------------------------
# Bildterme
# --------------------------------------------------------------------------------------

def _yaw(yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([c, -s, z, s, c, z, z, z, o]).reshape(3, 3)


def _camera(points: torch.Tensor, yaw: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    r = points @ _yaw(yaw).T
    return torch.stack([r[:, 0], -r[:, 2], r[:, 1]], -1) + trans


def _project(cam: torch.Tensor, intr: Intrinsics) -> torch.Tensor:
    z = cam[:, 2].clamp_min(1e-3)
    return torch.stack([intr.focal * cam[:, 0] / z + intr.cx, intr.focal * cam[:, 1] / z + intr.cy], -1)


def _bilinear(img: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinear, Koordinaten auf den Bildrand begrenzt (wie grid_sample mit padding 'border')."""
    h, w = img.shape
    x = uv[:, 0].clamp(0, w - 1)
    y = uv[:, 1].clamp(0, h - 1)
    x0 = x.detach().floor().clamp(max=w - 2)
    y0 = y.detach().floor().clamp(max=h - 2)
    fx, fy = x - x0, y - y0
    xi, yi = x0.long(), y0.long()
    return ((1 - fx) * (1 - fy) * img[yi, xi] + fx * (1 - fy) * img[yi, xi + 1]
            + (1 - fx) * fy * img[yi + 1, xi] + fx * fy * img[yi + 1, xi + 1])


def view_loss(verts: torch.Tensor, kp: torch.Tensor, yaw: torch.Tensor, trans: torch.Tensor,
              view: PreparedView, kp_weight: float) -> torch.Tensor:
    """verts: gestellte Modellpunkte, kp: gestellte Körperpunkt-Gelenke inkl. Versatz."""
    intr = view.intr
    uv = _project(_camera(torch.cat([verts, kp], 0), yaw, trans), intr)
    uv_v, uv_k = uv[: len(verts)], uv[len(verts):]
    h = view.px_height
    dt = torch.as_tensor(view.dt, dtype=verts.dtype)
    d = _bilinear(dt, uv_v)
    l_in = (d / h).pow(2).mean()
    contour = torch.as_tensor(view.contour, dtype=verts.dtype)
    l_cov = (torch.cdist(contour, uv_v).min(1).values / h).pow(2).mean()
    loss = 4.0 * l_in + 4.0 * l_cov
    if len(kp):
        diff = (uv_k - torch.as_tensor(view.kp_xy, dtype=verts.dtype)) / h
        loss = loss + kp_weight * (torch.as_tensor(view.kp_w, dtype=verts.dtype) * diff.pow(2).sum(-1)).mean()
    return loss


def view_loss_np(verts: np.ndarray, kp: np.ndarray, yaw: float, trans: np.ndarray, view: PreparedView,
                 kp_weight: float) -> tuple[float, np.ndarray, np.ndarray, float, np.ndarray]:
    """Rechenweg der App: Fehler und Gradienten nach Punkten, Körperpunkten, Drehung, Verschiebung."""
    intr = view.intr
    pts = np.concatenate([verts, kp], 0)
    c, s = math.cos(yaw), math.sin(yaw)
    rx, ry, rz = c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1], pts[:, 2]
    cam = np.stack([rx, -rz, ry], -1) + trans
    zc = np.maximum(cam[:, 2], 1e-3)
    f = intr.focal
    u = f * cam[:, 0] / zc + intr.cx
    v = f * cam[:, 1] / zc + intr.cy
    nv = len(verts)
    h = view.px_height
    gu = np.zeros(len(pts))
    gv = np.zeros(len(pts))

    # Modellpunkte außerhalb der Maske
    hh, ww = view.dt.shape
    x = np.clip(u[:nv], 0, ww - 1)
    y = np.clip(v[:nv], 0, hh - 1)
    inside_x = (u[:nv] >= 0) & (u[:nv] <= ww - 1)
    inside_y = (v[:nv] >= 0) & (v[:nv] <= hh - 1)
    x0 = np.minimum(np.floor(x), ww - 2).astype(int)
    y0 = np.minimum(np.floor(y), hh - 2).astype(int)
    fx, fy = x - x0, y - y0
    d00, d01 = view.dt[y0, x0], view.dt[y0, x0 + 1]
    d10, d11 = view.dt[y0 + 1, x0], view.dt[y0 + 1, x0 + 1]
    d = (1 - fx) * (1 - fy) * d00 + fx * (1 - fy) * d01 + (1 - fx) * fy * d10 + fx * fy * d11
    l_in = float(((d / h) ** 2).mean())
    dd = 4.0 * 2 * d / (h * h * nv)
    gu[:nv] += dd * ((1 - fy) * (d01 - d00) + fy * (d11 - d10)) * inside_x
    gv[:nv] += dd * ((1 - fx) * (d10 - d00) + fx * (d11 - d01)) * inside_y

    # Randpunkte der Maske zum nächsten Modellpunkt
    uv = np.stack([u[:nv], v[:nv]], -1)
    diff = uv[None, :, :] - view.contour[:, None, :]
    dist2 = (diff ** 2).sum(-1)
    nearest = dist2.argmin(1)
    best = dist2[np.arange(len(view.contour)), nearest]
    l_cov = float((best / (h * h)).mean())
    gc = 4.0 * 2 / (h * h * len(view.contour))
    np.add.at(gu, nearest, gc * (uv[nearest, 0] - view.contour[:, 0]))
    np.add.at(gv, nearest, gc * (uv[nearest, 1] - view.contour[:, 1]))
    loss = 4.0 * l_in + 4.0 * l_cov

    # Körperpunkte
    if len(kp):
        du = u[nv:] - view.kp_xy[:, 0]
        dv = v[nv:] - view.kp_xy[:, 1]
        loss += kp_weight * float((view.kp_w * (du ** 2 + dv ** 2)).mean() / (h * h))
        gk = kp_weight * 2 * view.kp_w / (h * h * len(kp))
        gu[nv:] += gk * du
        gv[nv:] += gk * dv

    # Projektion -> Kamera -> Welt
    clamped = cam[:, 2] < 1e-3
    g_cam = np.stack([gu * f / zc, gv * f / zc,
                      np.where(clamped, 0.0, -(gu * f * cam[:, 0] + gv * f * cam[:, 1]) / zc ** 2)], -1)
    g_trans = g_cam.sum(0)
    # cam = P(Rz p) + t mit P(x, y, z) = (x, −z, y);  d(Rz p)/dyaw = (−s x − c y, c x − s y, 0)
    g_yaw = float((g_cam[:, 0] * (-s * pts[:, 0] - c * pts[:, 1]) + g_cam[:, 2] * (c * pts[:, 0] - s * pts[:, 1])).sum())
    g_rot = np.stack([g_cam[:, 0], g_cam[:, 2], -g_cam[:, 1]], -1)  # Pᵀ g
    g_pts = np.stack([c * g_rot[:, 0] + s * g_rot[:, 1], -s * g_rot[:, 0] + c * g_rot[:, 1], g_rot[:, 2]], -1)  # Rzᵀ
    return loss, g_pts[:nv], g_pts[nv:], g_yaw, g_trans


# --------------------------------------------------------------------------------------
# Anpassung
# --------------------------------------------------------------------------------------

def _align_translation(verts: torch.Tensor, yaw: torch.Tensor, view: PreparedView) -> torch.Tensor:
    """Verschiebung, bei der das Modell bei dieser Drehung so groß ist und so liegt wie im Bild."""
    intr = view.intr
    cam = _camera(verts, yaw, torch.zeros(3, dtype=verts.dtype))
    cx, cy = view.bbox_center
    zmean = float(cam[:, 2].mean())
    span = float(cam[:, 1].max() - cam[:, 1].min())
    t = torch.tensor([0.0, 0.0, intr.focal * span / view.px_height - zmean], dtype=verts.dtype)
    for _ in range(3):
        uv = _project(cam + t, intr)
        z = float((cam[:, 2] + t[2]).mean())
        px_h = float(uv[:, 1].max() - uv[:, 1].min())
        t[2] = (t[2] + zmean) * px_h / view.px_height - zmean
        t[0] += (cx - float(uv[:, 0].max() + uv[:, 0].min()) / 2) * z / intr.focal
        t[1] += (cy - float(uv[:, 1].max() + uv[:, 1].min()) / 2) * z / intr.focal
    return t


def _initial_view(verts, kp, view: PreparedView, refine_steps: int = 25) -> tuple[float, list[float]]:
    """Drehung und Lage eines Bildes: 12 Kandidaten (plus Hinweis) prüfen, die besten 4 nachschärfen."""
    cands = [i * math.pi / 6 for i in range(12)]
    if view.yaw_hint is not None:
        y = view.yaw_hint
        cands += [y, -y, y + math.pi, math.pi - y]
    scored = []
    with torch.no_grad():
        for c in cands:
            yaw = torch.tensor(c, dtype=verts.dtype)
            t = _align_translation(verts, yaw, view)
            scored.append((float(view_loss(verts, kp, yaw, t, view, 1.0)), c, t))
    scored.sort(key=lambda s: s[0])
    best = None
    for _, c, t in scored[:4]:
        yaw = torch.nn.Parameter(torch.tensor(c, dtype=verts.dtype))
        tt = torch.nn.Parameter(t.clone())
        opt = torch.optim.Adam([yaw, tt], lr=LR)
        for _ in range(refine_steps):
            opt.zero_grad()
            view_loss(verts, kp, yaw, tt, view, 1.0).backward()
            opt.step()
        with torch.no_grad():
            final = float(view_loss(verts, kp, yaw, tt, view, 1.0))
        if best is None or final < best[0]:
            best = (final, float(yaw.detach()), tt.detach().tolist())
    return best[1], best[2]


def fit_photos(
    model: BodyModel,
    person: Person,
    frames: list[FitFrame],
    init: BodyParams | None = None,
    steps: tuple[int, int] = STEPS,
    log: Log | None = None,
) -> tuple[BodyParams, dict[str, float], list[FrameFitResult]]:
    """Passt Form und Haltung an Umrisse und Körperpunkte aus einem oder mehreren Bildern an."""
    if not frames:
        raise ValueError("Keine brauchbaren Bilder für die Anpassung.")
    problems = person.validate()
    if problems:
        raise ValueError(" ".join(problems))
    targets = {k: float(v) for k, v in person.measures.items() if v}
    if init is None:
        init, _ = fit_to_measurements(model, person, log=log)
    init = dataclasses.replace(init, offsets=None)
    var = ShapeVariables(model, person, init, extra_locals=VIDEO_LOCALS)
    views = [prepare_view(model, f) for f in frames]
    rig = Rig(model)
    k = len(views)
    pose_aa = torch.nn.Parameter(torch.zeros(len(rig.pose_bones), 3, dtype=DTYPE))

    def body():
        rest = var.rest()
        verts = rest["vertices"][0].to(DTYPE)
        heads = rest["bone_heads"][0].to(DTYPE).detach()
        rs, ts = rig.transforms(heads, pose_aa)
        posed = rig.skin(verts, rs, ts)
        height = float((verts[:, 2].max() - verts[:, 2].min()).detach())
        return rest, posed, heads, rs, ts, height

    def keypoints(view: PreparedView, heads, rs, ts, height) -> torch.Tensor:
        b = view.kp_bones
        if not b:
            return heads[:0]
        off = torch.as_tensor(view.kp_off, dtype=DTYPE) * height
        return (rs[b] @ heads[b][..., None])[..., 0] + ts[b] + off

    with torch.no_grad():
        _, posed, heads, rs, ts, height = body()
    yaw0, trans0 = [], []
    for i, view in enumerate(views):
        y, t = _initial_view(posed, keypoints(view, heads, rs, ts, height), view)
        yaw0.append(y)
        trans0.append(t)
        if log:
            log(f"Bild {i + 1}/{k}: Startdrehung {math.degrees(y) % 360:.0f}°")
    yaws = torch.nn.Parameter(torch.tensor(yaw0, dtype=DTYPE))
    trans = torch.nn.Parameter(torch.tensor(trans0, dtype=DTYPE))

    def run_stage(name: str, n_steps: int, params: list, kp_w: float) -> float:
        opt = torch.optim.Adam(params, lr=LR)
        probe, last = None, float("inf")
        for step in range(n_steps):
            if step % PROBE_EVERY == 0:
                with torch.no_grad():
                    rest = var.rest()
                    probe = MeasureProbe(model, rest["vertices"][0].cpu().numpy(),
                                         rest["bone_heads"][0].cpu().numpy())
            opt.zero_grad()
            rest, posed, heads, rs, ts, height = body()
            loss = sum(view_loss(posed, keypoints(v, heads, rs, ts, height), yaws[i], trans[i], v, kp_w)
                       for i, v in enumerate(views)) * (SILHOUETTE_WEIGHT / k)
            loss = loss + 0.5 * measurement_loss(probe.evaluate(rest["vertices"][0]), targets)
            loss = loss + LOCAL_REG * (var.local_values() ** 2).sum()
            loss = loss + POSE_REG * (pose_aa ** 2).sum()
            loss.backward()
            opt.step()
            last = loss.item()
            if log and step % 40 == 0:
                log(f"{name}: Schritt {step}/{n_steps}, Fehler {last:.5f}")
        return last

    view_params = [yaws, trans, pose_aa]
    shape_params = [*view_params, *var.parameters()]
    run_stage("Lage und Haltung", steps[0], view_params, 1.0)

    # Muskeln und Fett sehen bei gleichem Gewicht ähnlich aus: drei Startwerte kurz anrechnen
    tracked = [yaws, trans, pose_aa, *var.parameters()]
    snapshot = [t.detach().clone() for t in tracked]
    muscle_i = var.free.index("muscle")
    start_muscle = float(torch.sigmoid(var.pheno_raw[muscle_i].detach()))
    trial = max(1, steps[1] // 3)
    best = None
    for m0 in dict.fromkeys([start_muscle, *MUSCLE_STARTS]):
        with torch.no_grad():
            for t, v in zip(tracked, snapshot, strict=True):
                t.copy_(v)
            var.pheno_raw[muscle_i] = torch.logit(torch.tensor(m0))
        score = run_stage(f"Körperform (Start Muskeln {m0:.2f})", trial, shape_params, KEYPOINT_WEIGHT_SHAPE)
        if best is None or score < best[0]:
            best = (score, [t.detach().clone() for t in tracked])
    with torch.no_grad():
        for t, v in zip(tracked, best[1], strict=True):
            t.copy_(v)
    run_stage("Körperform", steps[1] - trial, shape_params, KEYPOINT_WEIGHT_SHAPE)

    params = exact_height(model, var.to_params(), targets["height"])

    results = []
    with torch.no_grad():
        _, posed, *_ = body()
        verts = posed.numpy()
    for i, view in enumerate(views):
        rendered = render_mask(verts, model.faces, float(yaws[i].detach()), trans[i].detach().numpy(),
                               view.intr) > 0
        inter = (rendered & view.mask).sum()
        union = (rendered | view.mask).sum()
        overlay = np.full((*rendered.shape, 3), 245, np.uint8)
        overlay[rendered & view.mask] = (110, 180, 110)
        overlay[view.mask & ~rendered] = (220, 60, 60)
        overlay[rendered & ~view.mask] = (60, 100, 230)
        results.append(FrameFitResult(math.degrees(float(yaws[i].detach())) % 360, float(inter / max(union, 1)), overlay))
    return params, measure_params(model, params), results
