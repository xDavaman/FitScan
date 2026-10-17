"""Farbe: sRGB <-> linear und Farbstich des Lichts an hellen, farblosen Flächen im Hintergrund.

Ohne torch, damit auch der Scan-Assistent (Webcam-Ampel, später iPhone) es nutzen kann.
"""

from __future__ import annotations

import cv2
import numpy as np

WHITE_LIMIT = 1.35  # stärkere Korrektur je Kanal wäre eher eine farbige Wand als ein Farbstich
WHITE_MIN_SHARE = 0.02  # so viel heller, farbloser Hintergrund muss im Bild sein


def to_linear(img: np.ndarray) -> np.ndarray:
    c = np.asarray(img, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0, 1)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(c * 255 + 0.5, 0, 255).astype(np.uint8)


def white_samples(rgb: np.ndarray, person: np.ndarray | None = None, side: int = 480) -> np.ndarray:
    """Helle, fast farblose, nicht überstrahlte Stellen (weiße Wand, Decke, Türen) als lineare
    Farbe je Stelle, Grün = 1. person: Maske im selben Bild, diese Stellen zählen nicht."""
    k = side / max(rgb.shape[:2])
    small = cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_AREA).astype(np.float32)
    mx, mn = small.max(-1), small.min(-1)
    ok = (mx < 248) & (mn > 120) & ((mx - mn) / np.maximum(mx, 1) < 0.25)
    if person is not None:
        p = cv2.resize(person.astype(np.uint8), small.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        ok &= cv2.dilate(p, np.ones((9, 9), np.uint8)) == 0
    if ok.mean() < WHITE_MIN_SHARE:
        return np.zeros((0, 3), np.float32)
    lin = to_linear(small[ok])
    return lin / np.maximum(lin[:, 1:2], 1e-6)


def white_point(samples: list[np.ndarray]) -> np.ndarray | None:
    """Farbe von Weiß in diesen Aufnahmen (linear, Grün = 1) oder None, wenn unsicher."""
    s = [x for x in samples if len(x)]
    if len(s) < max(1, len(samples) // 3):
        return None
    white = np.median(np.concatenate(s), axis=0)
    if (white > WHITE_LIMIT).any() or (white < 1 / WHITE_LIMIT).any():
        return None
    return white.astype(np.float32)


def neutralize(img: np.ndarray, white: np.ndarray | None) -> np.ndarray:
    """Farbstich entfernen: was in der Aufnahme die Farbe white hatte, wird neutral grau."""
    if white is None:
        return img
    return to_srgb(to_linear(img) / np.asarray(white, np.float32))
