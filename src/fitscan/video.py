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


VIDEO_HINT = ("Falls es ein iPhone-Video ist: am iPhone unter Einstellungen › Kamera › Formate "
              "„Maximale Kompatibilität“ wählen und neu aufnehmen, oder das Video in MP4 (H.264) umwandeln.")
HEIC_HINT = ("HEIC-Fotos (iPhone-Standard) kann FitScan noch nicht lesen. Bitte als JPG speichern: "
             "am iPhone unter Einstellungen › Kamera › Formate „Maximale Kompatibilität“ wählen "
             "oder die Fotos am PC als JPG exportieren.")


# ------------------------------------------------------------------------------------- HDR-Videos
# iPhones filmen meist in HDR (HLG, 10 Bit, BT.2020-Farben). OpenCV liest solche Videos wie normale:
# Haut und Wände überstrahlen (Rot fast überall 255), Hautdetails, Augenbrauen und Farben gehen
# verloren. Solche Videos werden daher mit PyAV (FFmpeg) roh gelesen und hier selbst in normale
# sRGB-Bilder umgerechnet; eine Tonkurve legt den hellsten Wert auf Weiß, statt abzuschneiden.

HLG, PQ = 18, 16  # FFmpeg-Kennungen der HDR-Übertragungskurven (ARIB STD-B67, SMPTE ST 2084)
HDR_EXPOSURE = 1.0  # Helligkeit von Referenzweiß der Aufnahme vor der Tonkurve
_HLG_A, _HLG_B, _HLG_C = 0.17883277, 0.28466892, 0.55991073
_HLG_WHITE = (np.exp((0.75 - _HLG_C) / _HLG_A) + _HLG_B) / 12  # Szenenlicht von 75 % Signal (BT.2408)
_PQ_WHITE = 203.0  # Nits von Referenzweiß bei PQ (BT.2408)
HDR_PEAK = {HLG: 1 / _HLG_WHITE, PQ: 1000 / _PQ_WHITE}  # hellster Wert der Aufnahme, relativ zu Referenzweiß
BT2020_TO_709 = np.array([[1.6605, -0.5876, -0.0728],
                          [-0.1246, 1.1329, -0.0083],
                          [-0.0182, -0.1006, 1.1187]], np.float32)
_LUT = 16384


def _hlg_linear(e: np.ndarray) -> np.ndarray:
    """HLG-Signal (0..1) -> Szenenlicht (0..1), ITU-R BT.2100."""
    e = np.clip(e, 0.0, 1.0)
    return np.where(e <= 0.5, e * e / 3, (np.exp((e - _HLG_C) / _HLG_A) + _HLG_B) / 12)


def _pq_linear(e: np.ndarray) -> np.ndarray:
    """PQ-Signal (0..1) -> Nits, SMPTE ST 2084."""
    m1, m2, c1, c2, c3 = 0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875
    p = np.power(np.clip(e, 0.0, 1.0), 1 / m2)
    return 10000 * np.power(np.maximum(p - c1, 0) / (c2 - c3 * p), 1 / m1)


def _srgb_lut() -> np.ndarray:
    x = np.linspace(0.0, 1.0, _LUT)
    return np.round(255 * np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)).astype(np.uint8)


def hdr_to_srgb(y: np.ndarray, u: np.ndarray, v: np.ndarray, transfer: int = HLG, bits: int = 10,
                full_range: bool = False, size: tuple[int, int] | None = None) -> np.ndarray:
    """10-Bit-YUV-Ebenen eines HDR-Bildes (BT.2020) -> sRGB uint8 (H,W,3).
    u, v dürfen halb so groß sein (4:2:0); size=(Breite, Höhe) verkleinert vor der Umrechnung."""
    h, w = y.shape
    out_w, out_h = size or (w, h)
    scale = float(1 << (bits - 8))
    y = y.astype(np.float32)
    u, v = u.astype(np.float32), v.astype(np.float32)
    if (out_w, out_h) != (w, h):
        y = cv2.resize(y, (out_w, out_h), interpolation=cv2.INTER_AREA)
    interp = cv2.INTER_AREA if u.shape[1] > out_w else cv2.INTER_LINEAR
    u = cv2.resize(u, (out_w, out_h), interpolation=interp)
    v = cv2.resize(v, (out_w, out_h), interpolation=interp)
    if full_range:
        top = 255 * scale
        y, cb, cr = y / top, (u - 128 * scale) / top, (v - 128 * scale) / top
    else:
        y, cb, cr = (y - 16 * scale) / (219 * scale), (u - 128 * scale) / (224 * scale), (v - 128 * scale) / (224 * scale)
    # BT.2020 (nicht konstante Luminanz): Kr = 0.2627, Kb = 0.0593
    rgb = np.stack([y + 1.4746 * cr, y - 0.16455 * cb - 0.57135 * cr, y + 1.8814 * cb], -1)
    if transfer == PQ:
        lin = _pq_linear(rgb) / _PQ_WHITE
    else:
        lut = _hlg_linear(np.linspace(0.0, 1.0, _LUT)).astype(np.float32) / np.float32(_HLG_WHITE)
        lin = lut[np.clip(rgb * (_LUT - 1) + 0.5, 0, _LUT - 1).astype(np.int32)]
    lin = cv2.transform(lin.astype(np.float32), BT2020_TO_709 * np.float32(HDR_EXPOSURE))
    np.maximum(lin, 0, out=lin)
    # Tonkurve (erweitertes Reinhard) auf den hellsten Kanal, damit der Farbton bleibt: der hellste
    # Wert der Aufnahme wird Weiß, helle Haut behält ihre Zeichnung, statt abzuschneiden
    white = np.float32(HDR_PEAK.get(transfer, HDR_PEAK[HLG]) * HDR_EXPOSURE)
    peak = lin.max(-1, keepdims=True)
    lin *= (1 + peak / (white * white)) / (1 + peak)
    return _srgb_lut()[np.clip(lin * (_LUT - 1) + 0.5, 0, _LUT - 1).astype(np.int32)]


def _open_hdr(path: Path):
    """(Container, Videostrom, Kurve), falls das Video HDR ist und PyAV es lesen kann, sonst None."""
    try:
        import av
    except ImportError:
        return None
    try:
        container = av.open(str(path))
    except Exception:
        return None
    stream = container.streams.video[0] if container.streams.video else None
    trc = stream.codec_context.color_trc if stream is not None else None
    if trc not in (HLG, PQ):
        container.close()
        return None
    stream.thread_type = "AUTO"
    return container, stream, trc


def _hdr_frames(opened, wanted: Callable[[int], bool], max_side: int | None) -> Iterator[tuple[int, np.ndarray]]:
    container, stream, trc = opened
    try:
        for index, frame in enumerate(container.decode(stream)):
            if not wanted(index):
                continue
            fmt = frame.format.name
            if not fmt.startswith(("yuv420p", "yuvj420p")):
                frame = frame.reformat(format="yuv420p10le")
                fmt = "yuv420p10le"
            bits = 10 if "10" in fmt else 12 if "12" in fmt else 8
            planes = frame.to_ndarray()
            h, w = frame.height, frame.width
            y, uv = planes[:h], planes[h:].reshape(2, h // 2, w // 2)
            factor = min(1.0, max_side / max(h, w)) if max_side else 1.0
            size = (round(w * factor), round(h * factor))
            rgb = hdr_to_srgb(y, uv[0], uv[1], trc, bits, frame.color_range == 2 or fmt.startswith("yuvj"), size)
            turn = round((frame.rotation or 0) / 90) % 4  # gegen den Uhrzeigersinn drehen
            yield index, np.ascontiguousarray(np.rot90(rgb, turn)) if turn else rgb
    finally:
        container.close()


def read_frames(path: Path, fps: float = 4.0) -> Iterator[tuple[int, float, np.ndarray]]:
    hdr = _open_hdr(path)
    if hdr is not None:
        src_fps = float(hdr[1].average_rate or 30.0)
        step = max(1, round(src_fps / fps))
        for index, rgb in _hdr_frames(hdr, lambda i: i % step == 0, MAX_SIDE):
            yield index, index / src_fps, rgb
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Video lässt sich nicht öffnen: {Path(path).name}. {VIDEO_HINT}")
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
    if index == 0:
        # Container ließ sich öffnen, aber kein Bild dekodieren (z. B. fehlender HEVC-Decoder)
        raise ValueError(f"Video enthält keine lesbaren Bilder: {Path(path).name}. {VIDEO_HINT}")


def read_full_frames(path: Path, indices: set[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Ausgewählte Bilder eines Videos in voller Auflösung (für das Hautbild), der Reihe nach."""
    hdr = _open_hdr(path)
    if hdr is not None:
        last = max(indices, default=-1)
        for index, rgb in _hdr_frames(hdr, lambda i: i in indices, None):
            yield index, rgb
            if index >= last:
                return
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Video lässt sich nicht öffnen: {Path(path).name}. {VIDEO_HINT}")
    last = max(indices, default=-1)
    index = 0
    try:
        while index <= last:
            if index in indices:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif not cap.grab():
                break
            index += 1
    finally:
        cap.release()


def read_images(paths: list[Path], max_side: int | None = MAX_SIDE) -> Iterator[tuple[int, float, np.ndarray]]:
    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p))
        if bgr is None:
            if Path(p).suffix.lower() in (".heic", ".heif"):
                raise ValueError(f"Foto lässt sich nicht öffnen: {Path(p).name}. {HEIC_HINT}")
            raise ValueError(f"Foto lässt sich nicht öffnen: {Path(p).name}. Erwartet wird ein Foto als JPG oder PNG.")
        h, w = bgr.shape[:2]
        factor = max_side / max(h, w) if max_side else 1.0
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


def select_for_texture(frames: list[AnalyzedFrame], max_frames: int = 48) -> list[AnalyzedFrame]:
    """Viele brauchbare Bilder rundherum fürs Hautbild, gleichmäßig über die Aufnahme verteilt."""
    ok = [f for f in frames if f.quality.usable and f.obs.mask is not None and f.obs.keypoints is not None]
    if len(ok) <= max_frames:
        return ok
    pick = np.linspace(0, len(ok) - 1, max_frames).round().astype(int)
    return [ok[i] for i in dict.fromkeys(pick.tolist())]


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
