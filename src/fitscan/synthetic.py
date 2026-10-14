"""Synthetische Testkörper und -aufnahmen: eigene, lizenzfreie Testdaten.

Ein Anny-Körper mit bekannten Maßen wird aus mehreren Blickwinkeln als Silhouette mit
Körperpunkten "gefilmt". Damit lässt sich die Anpassung automatisch prüfen, und später
entstehen daraus Trainingsdaten für eigene KI-Modelle.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from .body import BodyModel, BodyParams, age_to_param, axis_angle_to_matrix
from .camera import MEDIAPIPE_TO_BONE, Intrinsics, project, project_points, render_mask, to_camera
from .fit import FitFrame


def example_body(seed: int = 0, variation: float = 1.0) -> tuple[BodyParams, dict[str, float]]:
    """Zufälliger Testkörper. variation > 1 weicht stärker vom Anny-Durchschnitt ab."""
    rng = np.random.default_rng(seed)
    v = variation
    params = BodyParams(
        phenotype={
            "gender": float(rng.choice([0.0, 1.0])),
            "age": age_to_param(float(rng.uniform(22, 55))),
            "muscle": float(rng.uniform(0.25, 0.8)),
            "weight": float(rng.uniform(0.3, 0.85)),
            "height": float(rng.uniform(0.3, 0.7)),
            "proportions": float(rng.uniform(0.3, 0.7)),
        },
        local={
            "stomach-pregnant-incr": float(np.clip(rng.uniform(0.0, 0.4 * v), 0, 1)),
            "buttocks-volume-incr": float(np.clip(rng.uniform(-0.3, 0.3) * v, -1, 1)),
            "torso-scale-depth-incr": float(np.clip(rng.uniform(-0.3, 0.3) * v, -1, 1)),
            "hip-scale-horiz-incr": float(np.clip(rng.uniform(-0.2, 0.2) * v, -1, 1)),
            "measure-shoulder-dist-incr": float(np.clip(rng.uniform(-0.2, 0.2) * v, -1, 1)),
            "measure-thigh-circ-incr": float(np.clip(rng.uniform(-0.2, 0.2) * v, -1, 1)),
        },
    )
    return params, {"arm_deg": float(rng.uniform(-12, 8))}


def film(model: BodyModel, params: BodyParams, arm_deg: float = 0.0, views: int = 12,
         width: int = 540, height: int = 960, distance: float = 2.8,
         seed: int = 0) -> list[FitFrame]:
    """Rendert Silhouetten und Körperpunkte einer sich drehenden Person."""
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        rest = model.rest(model.phenotype_tensor(params), model.local_tensor(params), None, params.scale)
        pose = torch.eye(4)[None, None].repeat(1, len(model.bone_labels), 1, 1)
        for bone, sign in (("upperarm01.L", 1.0), ("upperarm01.R", -1.0)):
            aa = torch.tensor([[0.0, sign * math.radians(arm_deg), 0.0]])
            pose[0, model.bone(bone)] = axis_angle_to_matrix(aa)[0]
        posed = model.posed(rest, pose.to(model.device), params.scale)
        verts = posed["vertices"][0].cpu().numpy()
        heads = posed["bone_heads"][0].cpu().numpy()
    intr = Intrinsics.phone_default(width, height)
    zc = (verts[:, 2].max() + verts[:, 2].min()) / 2
    frames = []
    for i in range(views):
        yaw = 2 * math.pi * i / views + rng.normal(0, 0.03)
        trans = np.array([rng.normal(0, 0.03), zc + rng.normal(0, 0.02), distance + rng.normal(0, 0.05)])
        mask = render_mask(verts, model.faces, yaw, trans, intr).astype(np.float32) / 255
        kp = np.zeros((33, 3), np.float32)
        for mp_idx, bone in MEDIAPIPE_TO_BONE.items():
            kp[mp_idx, :2] = project_points(heads[[model.bone(bone)]], yaw, trans, intr)[0]
            kp[mp_idx, :2] += rng.normal(0, 3.0, 2)
            kp[mp_idx, 2] = 0.9
        yaw_noisy = yaw + rng.normal(0, 0.2)
        frames.append(FitFrame(mask=mask, intr=intr, keypoints=kp, calibrated=False,
                               yaw_init=math.atan2(math.sin(yaw_noisy), math.cos(yaw_noisy))))
    return frames


def render_shaded(vertices: np.ndarray, faces: np.ndarray, yaw: float, trans: np.ndarray,
                  intr: Intrinsics, skin=(205, 160, 135), background=(235, 235, 235)) -> np.ndarray:
    """Einfaches schattiertes Bild (Maler-Algorithmus), z. B. als künstliches Testvideo."""
    cam = to_camera(torch.as_tensor(vertices, dtype=torch.float32), torch.tensor([yaw]),
                    torch.as_tensor(trans, dtype=torch.float32)[None])[0].numpy()
    uv = project(torch.as_tensor(cam), intr).numpy()
    tri = cam[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    center = tri.mean(1)
    visible = (normal * center).sum(1) < 0  # Vorderseiten
    light = np.array([0.3, -0.5, -0.8])
    light /= np.linalg.norm(light)
    shade = np.clip(0.25 + 0.75 * (normal @ light), 0.15, 1.0)
    img = np.empty((intr.height, intr.width, 3), np.uint8)
    img[:] = background
    order = np.argsort(-center[:, 2])
    pts = np.round(uv[faces] * 4).astype(np.int32)
    for i in order:
        if not visible[i]:
            continue
        c = tuple(int(v * shade[i]) for v in skin)
        cv2.fillPoly(img, [pts[i]], c, lineType=cv2.LINE_AA, shift=2)
    return img


def film_video(model: BodyModel, params: BodyParams, path, arm_deg: float = 0.0, seconds: float = 12.0,
               fps: float = 10.0, width: int = 540, height: int = 960, distance: float = 2.2,
               skin: np.ndarray | None = None):
    """Künstliches Drehvideo eines Testkörpers (eine volle Umdrehung) als MP4.

    skin: Hautbild auf Annys UV-Karte (RGB); ohne wird einfarbig schattiert."""
    with torch.no_grad():
        rest = model.rest(model.phenotype_tensor(params), model.local_tensor(params), None, params.scale)
        pose = torch.eye(4)[None, None].repeat(1, len(model.bone_labels), 1, 1)
        for bone, sign in (("upperarm01.L", 1.0), ("upperarm01.R", -1.0)):
            aa = torch.tensor([[0.0, sign * math.radians(arm_deg), 0.0]])
            pose[0, model.bone(bone)] = axis_angle_to_matrix(aa)[0]
        verts = model.posed(rest, pose.to(model.device), params.scale)["vertices"][0].cpu().numpy()
    intr = Intrinsics.phone_default(width, height)
    zc = (verts[:, 2].max() + verts[:, 2].min()) / 2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    n = int(seconds * fps)
    for i in range(n):
        yaw = 2 * math.pi * i / n
        trans = np.array([0.0, zc, distance])
        if skin is None:
            rgb = render_shaded(verts, model.faces, yaw, trans, intr)
        else:
            from .texture import render_textured

            rgb = render_textured(verts, model.faces, model.anny.texture_coordinates.cpu().numpy(),
                                  model.anny.face_texture_coordinate_indices.cpu().numpy(), skin, yaw, trans, intr)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    return path
