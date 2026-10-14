"""Person freistellen: MediaPipe-Maske an den Rändern mit GrabCut (OpenCV) nachschärfen.

MediaPipe liefert eine weiche Maske mit unscharfen Kanten. Für die Silhouettenanpassung zählen
genau diese Kanten, daher wird nur ein schmaler Saum um die Kante neu entschieden; der Rest der
Maske bleibt, wie er ist. Ohne zusätzliches KI-Modell, komplett offline.
"""

from __future__ import annotations

import cv2
import numpy as np

WORK_SIDE = 720  # Arbeitsauflösung für GrabCut
BAND = 0.015  # halbe Saumbreite relativ zur Bildhöhe


def refine_mask(rgb: np.ndarray, soft_mask: np.ndarray, iterations: int = 3) -> np.ndarray:
    """rgb (H, W, 3), soft_mask (H, W) 0..1 -> verbesserte Maske (H, W) float32 0/1."""
    h, w = soft_mask.shape
    hard = (soft_mask > 0.5).astype(np.uint8)
    if hard.sum() < 500:
        return hard.astype(np.float32)
    factor = min(1.0, WORK_SIDE / max(h, w))
    small = cv2.resize(rgb, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA) if factor < 1 else rgb
    m = cv2.resize(hard, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    r = max(2, int(BAND * small.shape[0]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    sure_fg = cv2.erode(m, kernel)
    maybe = cv2.dilate(m, kernel)
    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    gc[maybe > 0] = cv2.GC_PR_BGD
    gc[m > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, iterations,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return hard.astype(np.float32)
    out = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    # nur die größte zusammenhängende Fläche behalten, kleine Löcher schließen
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out)
    if n > 1:
        out = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # Sicherheitsnetz: weicht GrabCut stark ab (z. B. Hintergrund in Hautfarbe), MediaPipe behalten
    inter = (out & m).sum()
    union = (out | m).sum()
    if union == 0 or inter / union < 0.85:
        return hard.astype(np.float32)
    if factor < 1:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out.astype(np.float32)
