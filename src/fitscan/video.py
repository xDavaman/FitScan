"""Video einlesen, Bilder prüfen und die besten Bilder pro Blickwinkel auswählen."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np

from .quality import SECTORS, FrameQuality, ScanReport, check_frame, sector_of, summarize
from .vision import FrameObservation, PoseAnalyzer

MAX_SIDE = 1280  # größere Videos werden für die Analyse verkleinert


@dataclasses.dataclass
class AnalyzedFrame:
    rgb: np.ndarray
    obs: FrameObservation
    quality: FrameQuality


def read_frames(path: Path, fps: float = 4.0) -> Iterator[tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Video lässt sich nicht öffnen: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / fps))
    index = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if index % step == 0:
                h, w = bgr.shape[:2]
                factor = MAX_SIDE / max(h, w)
                if factor < 1:
                    bgr = cv2.resize(bgr, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
                yield index, index / src_fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        cap.release()


def read_images(paths: list[Path]) -> Iterator[tuple[int, float, np.ndarray]]:
    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise ValueError(f"Bild lässt sich nicht öffnen: {p}")
        h, w = bgr.shape[:2]
        factor = MAX_SIDE / max(h, w)
        if factor < 1:
            bgr = cv2.resize(bgr, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        yield i, float(i), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def analyze(
    frames: Iterator[tuple[int, float, np.ndarray]],
    analyzer: PoseAnalyzer | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[list[AnalyzedFrame], ScanReport]:
    analyzer = analyzer or PoseAnalyzer()
    result: list[AnalyzedFrame] = []
    prev = None
    for n, (index, t, rgb) in enumerate(frames):
        obs = analyzer.analyze(rgb, index, t)
        q = check_frame(rgb, obs, prev)
        result.append(AnalyzedFrame(rgb, obs, q))
        prev = obs if obs.keypoints is not None else prev
        if log and n % 10 == 0:
            log(f"Bild {n + 1} geprüft: {q.level}")
    return result, summarize([a.quality for a in result])


def select_for_fit(frames: list[AnalyzedFrame], per_sector: int = 1) -> list[AnalyzedFrame]:
    """Pro Blickwinkel das schärfste brauchbare Bild."""
    buckets: dict[int, list[AnalyzedFrame]] = {}
    for f in frames:
        if not f.quality.usable or f.obs.mask is None or f.quality.yaw is None:
            continue
        buckets.setdefault(sector_of(f.quality.yaw), []).append(f)
    chosen = []
    for sector in range(SECTORS):
        items = sorted(buckets.get(sector, []), key=lambda f: -f.quality.metrics.get("sharpness", 0.0))
        chosen.extend(items[:per_sector])
    return chosen
