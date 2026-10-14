"""Kamera: Lochkamera vor einer Person, die sich auf der Stelle dreht.

Anny-Koordinaten (Z oben, Blick -Y) werden so in Kamerakoordinaten (x rechts, y unten,
z vom Objektiv weg) gebracht, dass eine Person mit Drehung 0 direkt in die Kamera schaut.
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np
import torch

# Typische Hauptkamera eines Handys: etwa 69° Blickwinkel über die lange Bildseite
DEFAULT_FOV_LONG_SIDE_DEG = 69.0

# MediaPipe-Pose-Punkt -> Anny-Bone, dessen Kopf ungefähr dort liegt
MEDIAPIPE_TO_BONE = {
    11: "upperarm01.L", 12: "upperarm01.R",
    13: "lowerarm01.L", 14: "lowerarm01.R",
    15: "wrist.L", 16: "wrist.R",
    23: "upperleg01.L", 24: "upperleg01.R",
    25: "lowerleg01.L", 26: "lowerleg01.R",
    27: "foot.L", 28: "foot.R",
}


@dataclasses.dataclass
class Intrinsics:
    width: int
    height: int
    focal: float

    @classmethod
    def phone_default(cls, width: int, height: int) -> Intrinsics:
        f = 0.5 * max(width, height) / math.tan(math.radians(DEFAULT_FOV_LONG_SIDE_DEG) / 2)
        return cls(width, height, f)

    def scaled(self, factor: float) -> Intrinsics:
        return Intrinsics(round(self.width * factor), round(self.height * factor), self.focal * factor)


def yaw_matrix(yaw: torch.Tensor) -> torch.Tensor:
    """Drehung um die Hochachse (Z), (..., 3, 3)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([c, -s, zero, s, c, zero, zero, zero, one], -1).reshape(*yaw.shape, 3, 3)


def to_camera(points: torch.Tensor, yaw: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """points (N, 3) Anny -> (K, N, 3) Kamera, für K Bilder mit eigener Drehung/Verschiebung."""
    rotated = points[None] @ yaw_matrix(yaw).transpose(-1, -2)  # (K, N, 3)
    cam = torch.stack([rotated[..., 0], -rotated[..., 2], rotated[..., 1]], -1)
    return cam + trans[:, None, :]


def project(cam: torch.Tensor, intr: Intrinsics) -> torch.Tensor:
    z = cam[..., 2].clamp_min(1e-3)
    u = intr.focal * cam[..., 0] / z + intr.width / 2
    v = intr.focal * cam[..., 1] / z + intr.height / 2
    return torch.stack([u, v], -1)


def render_mask(vertices: np.ndarray, faces: np.ndarray, yaw: float, trans: np.ndarray,
                intr: Intrinsics) -> np.ndarray:
    """Einfache Silhouette (für Tests und synthetische Trainingsdaten)."""
    cam = to_camera(torch.as_tensor(vertices, dtype=torch.float32),
                    torch.tensor([yaw], dtype=torch.float32),
                    torch.as_tensor(trans, dtype=torch.float32)[None])
    uv = project(cam, intr)[0].numpy()
    mask = np.zeros((intr.height, intr.width), np.uint8)
    tris = np.round(uv[faces] * 4).astype(np.int32)  # 2 Bit Subpixel
    cv2.fillPoly(mask, list(tris), 255, lineType=cv2.LINE_8, shift=2)
    return mask


def project_points(points: np.ndarray, yaw: float, trans: np.ndarray, intr: Intrinsics) -> np.ndarray:
    cam = to_camera(torch.as_tensor(points, dtype=torch.float32),
                    torch.tensor([yaw], dtype=torch.float32),
                    torch.as_tensor(trans, dtype=torch.float32)[None])
    return project(cam, intr)[0].numpy()
