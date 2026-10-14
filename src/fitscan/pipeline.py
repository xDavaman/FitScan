"""Der ganze Ablauf in einem Aufruf: Aufnahme prüfen, Körper anpassen, exportieren, speichern."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

from . import level
from .body import BodyModel, BodyParams
from .camera import Intrinsics
from .export import PRESETS, cascadeur_check, export_body
from .fit import FitFrame, FrameFitResult, Person, fit_to_frames, fit_to_measurements
from .quality import ScanReport
from .segment import refine_mask
from .store import ScanRecord, Store

Log = Callable[[str], None]

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}


@dataclasses.dataclass
class ScanResult:
    record: ScanRecord
    params: BodyParams
    measures: dict[str, float]
    report: ScanReport | None
    frames: list[FrameFitResult]
    files: dict[str, str]
    hints: list[str]


def _noop(_: str) -> None:
    pass


def _export(model: BodyModel, params: BodyParams, folder: Path, presets) -> tuple[dict[str, str], list[str]]:
    out = export_body(model, params, folder, presets=presets)
    hints = list(out.pop("hinweise", []))
    return {k: str(v) for k, v in out.items()}, hints


def body_from_measurements(person: Person, name: str, store: Store | None = None,
                           presets: tuple[str, ...] = PRESETS, device: str | None = None,
                           log: Log | None = None) -> ScanResult:
    """Körper nur aus den eingegebenen Maßen (ohne Video), z. B. als Startpunkt oder Test."""
    log = log or _noop
    store = store or Store()
    model = BodyModel.get(device)
    rec = store.save_person(name, person.gender, person.age_years, person.measures)
    previous = store.latest(rec.id)
    init = previous.params() if previous else None
    log("Körper wird an die Maße angepasst …")
    params, measures = fit_to_measurements(model, person, init=init, log=log)
    folder = store.new_scan_folder(rec.id)
    files, hints = _export(model, params, folder, presets)
    record = store.save_scan(rec.id, folder, params, person.measures, measures, {}, "masse", files)
    log(f"Fertig: {folder}")
    return ScanResult(record, params, measures, None, [], files, hints)


def scan(source: Path | list[Path], person: Person, name: str, store: Store | None = None,
         presets: tuple[str, ...] = PRESETS, device: str | None = None, fps: float = 4.0,
         refine: bool = False, log: Log | None = None) -> ScanResult:
    """Video oder Fotos -> angepasster, riggbarer Körper als GLB/OBJ, gespeichert beim Namen.

    refine: Maskenkanten mit GrabCut nachschärfen. Hilft bei unruhigem Hintergrund, schadet bei
    ruhigem Hintergrund leicht (Testvideo: Brust 2,5 statt 3,2 cm Abweichung ohne Nachschärfen).
    """
    from .video import analyze, read_frames, read_images, select_for_fit
    from .vision import PoseAnalyzer

    log = log or _noop
    problems = person.validate()
    if problems:
        raise ValueError(" ".join(problems))
    store = store or Store()
    paths = [Path(p) for p in (source if isinstance(source, list) else [source])]
    is_video = len(paths) == 1 and paths[0].suffix.lower() in VIDEO_SUFFIXES
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {p}")

    log("Bilder werden geprüft (Licht, Position, Haltung) …")
    analyzer = PoseAnalyzer(video_mode=is_video)
    try:
        frames = read_frames(paths[0], fps=fps) if is_video else read_images(paths)
        analyzed, report = analyze(frames, analyzer, log)
    finally:
        analyzer.close()
    for line in report.summary:
        log(line)
    chosen = select_for_fit(analyzed)
    if not chosen:
        raise ValueError("Kein Bild war für die Anpassung brauchbar. " + " ".join(report.summary))
    log(f"{len(chosen)} Bilder aus verschiedenen Blickwinkeln werden verwendet.")

    if refine:
        log("Kanten der Personenmaske werden nachgeschärft (unruhiger Hintergrund) …")
    masks = [refine_mask(a.rgb, a.obs.mask) if refine else a.obs.mask for a in chosen]
    keypoints = [a.obs.keypoints for a in chosen]
    h, w = chosen[0].rgb.shape[:2]
    intr = Intrinsics.phone_default(w, h)
    tilt = level.estimate([a.rgb for a in chosen], masks, intr)
    if tilt is not None and max(abs(tilt.pitch_deg), abs(tilt.roll_deg)) >= level.MIN_TILT_DEG:
        direction = "oben" if tilt.pitch_deg > 0 else "unten"
        log(f"Handy war um {abs(tilt.pitch_deg):.0f}° nach {direction} gekippt "
            f"(seitlich {tilt.roll_deg:+.0f}°), wird ausgeglichen.")
        src, intr = intr, level.leveled_intrinsics(tilt, intr, masks)
        warped = [level.apply(tilt, src, intr, m, k) for m, k in zip(masks, keypoints, strict=True)]
        masks, keypoints = [m for m, _ in warped], [k for _, k in warped]
    fit_frames = [FitFrame(mask=m, intr=intr, keypoints=k, yaw_init=a.quality.yaw)
                  for a, m, k in zip(chosen, masks, keypoints, strict=True)]

    model = BodyModel.get(device)
    rec = store.save_person(name, person.gender, person.age_years, person.measures)
    previous = store.latest(rec.id)
    if previous:
        log(f"Re-Scan: Start vom letzten Scan ({previous.id}).")
        init = previous.params()
    else:
        log("Erster Scan: Startform aus den Maßen …")
        init, _ = fit_to_measurements(model, person, log=log)
    log("Körper wird an die Bilder angepasst …")
    params, measures, frame_results = fit_to_frames(model, person, fit_frames, init=init, log=log)

    folder = store.new_scan_folder(rec.id)
    files, hints = _export(model, params, folder, presets)
    quality = {
        "bericht": report.to_json(),
        "bilder": [dataclasses.asdict(r) for r in frame_results],
        "quelle": [str(p) for p in paths],
    }
    record = store.save_scan(rec.id, folder, params, person.measures, measures, quality,
                             "video" if is_video else "bilder", files)
    (folder / "bilder.json").write_text(json.dumps(quality["bilder"], indent=2), "utf-8")
    log(f"Fertig: {folder}")
    return ScanResult(record, params, measures, report, frame_results, files, hints)


def reexport(record: ScanRecord, presets: tuple[str, ...] = PRESETS, device: str | None = None,
             changes: dict[str, float] | None = None, log: Log | None = None) -> dict[str, str]:
    """Gespeicherten Scan neu exportieren, optional mit geänderter Statur (z. B. weight=0.7)."""
    log = log or _noop
    model = BodyModel.get(device)
    params = record.params()
    suffix = ""
    if changes:
        for key, value in changes.items():
            params.phenotype[key] = float(value)
        suffix = "_" + "_".join(f"{k}{v:.2f}" for k, v in changes.items())
    files, hints = _export(model, params, record.folder / f"export{suffix}", presets)
    for h in hints:
        log(h)
    return files


__all__ = ["ScanResult", "body_from_measurements", "scan", "reexport", "cascadeur_check"]
