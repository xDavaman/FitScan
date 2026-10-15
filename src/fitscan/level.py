"""Gekippte Kamera ausgleichen: Bilder so umrechnen, als wäre das Handy waagerecht gewesen.

Steht das Handy tief und ist nach oben gekippt (oder hoch und nach unten), laufen senkrechte
Kanten im Hintergrund (Türrahmen, Schränke) auf einen Fluchtpunkt zu. Aus diesem Fluchtpunkt
folgt die Neigung. Eine reine Drehung der Kamera verschiebt Bildpunkte unabhängig von ihrer
Tiefe, daher lassen sich Maske und Körperpunkte exakt auf eine waagerechte Kamera umrechnen.
Die Anpassung selbst rechnet weiter mit waagerechter Kamera.
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np

from .camera import Intrinsics

MIN_SEGMENT = 0.04  # Mindestlänge einer Kante relativ zur langen Bildseite
MAX_LEAN_DEG = 15.0  # stärker geneigte Kanten zählen nicht als senkrecht
MIN_SEGMENTS = 8
MAX_TILT_DEG = 40.0
MIN_TILT_DEG = 1.0  # darunter lohnt die Umrechnung nicht


@dataclasses.dataclass
class Leveling:
    rotation: np.ndarray  # (3, 3) Kamera -> waagerechte Kamera
    pitch_deg: float  # + = nach oben gekippt
    roll_deg: float
    segments: int

    def homography(self, src: Intrinsics, dst: Intrinsics) -> np.ndarray:
        return _K(dst) @ self.rotation @ np.linalg.inv(_K(src))


def _K(intr: Intrinsics) -> np.ndarray:
    return np.array([[intr.focal, 0, intr.cx], [0, intr.focal, intr.cy], [0, 0, 1]])


def _vertical_segments(rgb: np.ndarray, person: np.ndarray | None) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lines = cv2.createLineSegmentDetector().detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4), np.float32)
    lines = lines.reshape(-1, 4)
    d = lines[:, 2:] - lines[:, :2]
    length = np.hypot(d[:, 0], d[:, 1])
    lean = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1])))
    keep = (length > MIN_SEGMENT * max(gray.shape)) & (lean < MAX_LEAN_DEG)
    lines = lines[keep]
    if person is not None and len(lines):
        # Kanten am Körper bewegen sich mit der Person und sind nicht senkrecht
        r = max(3, int(0.02 * max(gray.shape)))
        near = cv2.dilate((person > 0.5).astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8))
        mid = ((lines[:, :2] + lines[:, 2:]) / 2).astype(int)
        mid[:, 0] = mid[:, 0].clip(0, gray.shape[1] - 1)
        mid[:, 1] = mid[:, 1].clip(0, gray.shape[0] - 1)
        lines = lines[near[mid[:, 1], mid[:, 0]] == 0]
    return lines


def estimate(images: list[np.ndarray], masks: list[np.ndarray | None], intr: Intrinsics) -> Leveling | None:
    """Neigung aus senkrechten Hintergrundkanten mehrerer Bilder; None, wenn unsicher."""
    segs = [s for img, m in zip(images, masks, strict=True) for s in _vertical_segments(img, m)]
    if len(segs) < MIN_SEGMENTS:
        return None
    segs = np.array(segs, np.float64)
    p, q = segs[:, :2], segs[:, 2:]
    # Jede Kante als Gerade n·x = c; gesucht ist die Richtung, auf die alle zulaufen. Als Strahl
    # (homogen) gerechnet, damit auch „parallel“ (Fluchtpunkt im Unendlichen) sauber geht.
    K_inv = np.linalg.inv(_K(intr))
    ph = np.c_[p, np.ones(len(p))] @ K_inv.T
    qh = np.c_[q, np.ones(len(q))] @ K_inv.T
    normals = np.cross(ph, qh)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    weight = np.hypot(*(q - p).T)
    keep = np.ones(len(segs), bool)
    for _ in range(4):
        w = np.sqrt(weight * keep)[:, None]
        _, _, vt = np.linalg.svd(normals * w)
        d = vt[-1]
        err = np.abs(normals @ d)
        if keep.sum() <= MIN_SEGMENTS:
            break
        keep = err <= np.quantile(err[keep], 0.75) * 1.5 + 1e-4
    if keep.sum() < MIN_SEGMENTS:
        return None
    if d[1] < 0:
        d = -d  # Kamera-y zeigt nach unten, die Senkrechte also auch
    pitch = math.degrees(math.atan2(-d[2], d[1]))  # Blickachse über der Waagerechten: positiv
    roll = math.degrees(math.atan2(-d[0], d[1]))  # Drehung um die Blickachse
    if abs(pitch) > MAX_TILT_DEG or abs(roll) > MAX_TILT_DEG:
        return None
    # kleinste Drehung, die die gemessene Senkrechte auf die Kamera-y-Achse legt
    target = np.array([0.0, 1.0, 0.0])
    axis = np.cross(d, target)
    s, c = np.linalg.norm(axis), float(d @ target)
    if s < 1e-9:
        rot = np.eye(3)
    else:
        k = axis / s
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        rot = np.eye(3) + s * kx + (1 - c) * kx @ kx
    return Leveling(rot, pitch, roll, int(keep.sum()))


def leveled_intrinsics(lev: Leveling, intr: Intrinsics, masks: list[np.ndarray],
                       margin: float = 0.08) -> Intrinsics:
    """Bildausschnitt der waagerechten Kamera: gleiche Brennweite, so groß wie das Original und
    so verschoben, dass alle Personen mit Rand hineinpassen."""
    H = lev.homography(intr, intr)
    boxes = []
    for m in masks:
        ys, xs = np.where(m > 0.5)
        if len(xs):
            pts = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.min(), ys.max()],
                            [xs.max(), ys.max()]], np.float64)
            boxes.append(cv2.perspectiveTransform(pts[None], H)[0])
    if not boxes:
        return intr
    pts = np.concatenate(boxes)
    lo, hi = pts.min(0), pts.max(0)
    pad = margin * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    width, height = max(intr.width, int(math.ceil(hi[0] - lo[0]))), max(intr.height, int(math.ceil(hi[1] - lo[1])))
    # Ausschnitt um die Personen zentrieren; die Brennweite bleibt, der Bildmittelpunkt wandert mit
    x0 = (lo[0] + hi[0]) / 2 - width / 2
    y0 = (lo[1] + hi[1]) / 2 - height / 2
    return Intrinsics(width, height, intr.focal, intr.cx - x0, intr.cy - y0)


def apply(lev: Leveling, src: Intrinsics, dst: Intrinsics, mask: np.ndarray,
          keypoints: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    """Maske und Körperpunkte (Pixel) in die waagerechte Kamera umrechnen."""
    H = lev.homography(src, dst)
    out = cv2.warpPerspective(mask.astype(np.float32), H, (dst.width, dst.height), flags=cv2.INTER_LINEAR)
    if keypoints is None:
        return out, None
    kp = keypoints.copy()
    kp[:, :2] = cv2.perspectiveTransform(keypoints[None, :, :2].astype(np.float64), H)[0]
    return out, kp
