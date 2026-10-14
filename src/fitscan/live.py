"""Live-Scan-Assistent mit der Webcam: Ampel, Hinweise und Rundum-Anzeige, Aufnahme per Leertaste.

Auf dem iPhone übernimmt die App diese Rolle; am PC dient das Fenster zum Üben und Testen
der KI-Prüfung (Licht, Abstand, Position, Haltung, Drehung).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .quality import CHECKLIST, GELB, GRUEN, ROT, SECTORS, check_frame, sector_of
from .vision import PoseAnalyzer, fitscan_home

COLORS = {GRUEN: (60, 200, 80), GELB: (255, 200, 0), ROT: (230, 50, 50)}  # RGB


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def draw_overlay(rgb: np.ndarray, level: str, hints: list[str], covered: list[bool],
                 recording: bool, seconds: float) -> np.ndarray:
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    color = COLORS.get(level, (200, 200, 200))
    d.rectangle([0, 0, w - 1, h - 1], outline=color + (255,), width=max(6, w // 120))
    small, big = _font(max(16, h // 36)), _font(max(20, h // 26))
    # Ampel oben links
    r = max(14, h // 40)
    d.ellipse([20, 20, 20 + 2 * r, 20 + 2 * r], fill=color + (255,))
    status = {GRUEN: "Alles gut", GELB: "Fast gut", ROT: "So nicht verwendbar"}.get(level, "")
    d.text((30 + 2 * r, 20 + r), status, font=big, fill=(255, 255, 255, 255), anchor="lm",
           stroke_width=2, stroke_fill=(0, 0, 0, 255))
    # Hinweise
    y = 40 + 2 * r
    for text in hints[:3]:
        d.text((20, y), text, font=small, fill=(255, 255, 255, 255), stroke_width=2,
               stroke_fill=(0, 0, 0, 255))
        y += int(small.size * 1.4) if hasattr(small, "size") else 24
    # Rundum-Anzeige unten rechts: welche Blickwinkel schon sauber erfasst sind
    cx, cy, rad = w - 90, h - 90, 60
    for i in range(SECTORS):
        a0 = 360 * i / SECTORS - 90 - 180 / SECTORS
        fill = (60, 200, 80, 230) if covered[i] else (80, 80, 80, 160)
        d.pieslice([cx - rad, cy - rad, cx + rad, cy + rad], a0, a0 + 360 / SECTORS - 2, fill=fill)
    d.ellipse([cx - rad // 2, cy - rad // 2, cx + rad // 2, cy + rad // 2], fill=(0, 0, 0, 200))
    d.text((cx, cy), f"{sum(covered)}/{SECTORS}", font=small, fill=(255, 255, 255, 255), anchor="mm")
    # Aufnahme
    foot = "LEERTASTE: Aufnahme starten/stoppen   ESC: beenden"
    if recording:
        d.ellipse([w - 160, 20, w - 136, 44], fill=(230, 30, 30, 255))
        d.text((w - 128, 32), f"REC {seconds:4.0f} s", font=small, fill=(255, 255, 255, 255),
               anchor="lm", stroke_width=2, stroke_fill=(0, 0, 0, 255))
    d.text((20, h - 20), foot, font=small, fill=(255, 255, 255, 255), anchor="ls", stroke_width=2,
           stroke_fill=(0, 0, 0, 255))
    return np.asarray(img)


def run(camera: int = 0, out_dir: Path | None = None, width: int = 1280, height: int = 720) -> Path | None:
    """Öffnet die Kamera. Gibt den Pfad der letzten Aufnahme zurück (oder None)."""
    print("Vor dem Scan:")
    for line in CHECKLIST:
        print("  •", line)
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {camera} lässt sich nicht öffnen.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_dir = out_dir or fitscan_home() / "aufnahmen"
    out_dir.mkdir(parents=True, exist_ok=True)
    analyzer = PoseAnalyzer(video_mode=True)
    covered = [False] * SECTORS
    writer, last_path, rec_start, prev = None, None, 0.0, None
    t0 = time.time()
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            obs = analyzer.analyze(rgb, 0, time.time() - t0)
            q = check_frame(rgb, obs, prev)
            prev = obs if obs.keypoints is not None else prev
            if writer is not None:
                writer.write(bgr)
                if q.usable and q.yaw is not None:
                    covered[sector_of(q.yaw)] = True
            shown = draw_overlay(rgb, q.level, [hint.text for hint in q.hints], covered,
                                 writer is not None, time.time() - rec_start)
            cv2.imshow("FitScan Live-Assistent", cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 32:
                if writer is None:
                    last_path = out_dir / time.strftime("scan_%Y-%m-%d_%H-%M-%S.mp4")
                    h, w = bgr.shape[:2]
                    writer = cv2.VideoWriter(str(last_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    covered = [False] * SECTORS
                    rec_start = time.time()
                else:
                    writer.release()
                    writer = None
                    print(f"Aufnahme gespeichert: {last_path} ({sum(covered)}/{SECTORS} Blickwinkel)")
    finally:
        if writer is not None:
            writer.release()
        analyzer.close()
        cap.release()
        cv2.destroyAllWindows()
    return last_path
