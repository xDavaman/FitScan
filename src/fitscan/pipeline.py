"""Der ganze Ablauf in einem Aufruf: Aufnahme prüfen, Körper anpassen, exportieren, speichern."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from . import level
from .body import BodyModel, BodyParams
from .camera import Intrinsics
from .export import PRESETS, cascadeur_check, export_body
from .fit import FitFrame, FrameFitResult, Person, fit_to_frames, fit_to_measurements
from .measure import measure_params
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


def source_groups(paths: list[Path]) -> list[list[Path]]:
    """Ausgewählte Dateien in Aufnahmen aufteilen: jedes Video einzeln, alle Fotos zusammen."""
    paths = list(dict.fromkeys(Path(p) for p in paths))  # doppelt ausgewählte Dateien nur einmal
    videos = [p for p in paths if p.suffix.lower() in VIDEO_SUFFIXES]
    images = [p for p in paths if p.suffix.lower() in IMAGE_SUFFIXES]
    other = [p for p in paths if p not in videos and p not in images]
    if other:
        raise ValueError("Dateiformat wird nicht unterstützt: " + ", ".join(p.name for p in other)
                         + ". Möglich sind Videos (" + ", ".join(sorted(VIDEO_SUFFIXES))
                         + ") und Fotos (" + ", ".join(sorted(IMAGE_SUFFIXES)) + ").")
    if not videos and not images:
        raise ValueError("Keine Datei ausgewählt.")
    return [[v] for v in videos] + ([images] if images else [])


@dataclasses.dataclass
class Recording:
    paths: list[Path]
    is_video: bool
    label: str
    frames: list  # AnalyzedFrame
    report: ScanReport | None
    error: str | None = None


def _score(rec: Recording) -> tuple[int, int]:
    return (sum(rec.report.covered), rec.report.usable) if rec.report is not None else (-1, -1)


def check_recordings(paths: list[Path], fps: float = 4.0, log: Log | None = None,
                     keep_frames: bool = True) -> list[Recording]:
    """Jede Aufnahme prüfen: jedes Video einzeln (eigene Kamera, eigene Neigung), Fotos zusammen.

    keep_frames: die Bilder der bisher besten Aufnahme behalten (für die Anpassung), alle anderen
    sofort verwerfen, damit mehrere 4K-Videos nicht gleichzeitig im Speicher liegen."""
    from .video import analyze, read_frames, read_images
    from .vision import PoseAnalyzer

    log = log or _noop
    groups = source_groups(paths)
    out = []
    for group in groups:
        is_video = group[0].suffix.lower() in VIDEO_SUFFIXES
        label = group[0].name if is_video else f"{len(group)} Fotos"
        log(f"Bilder werden geprüft (Licht, Position, Haltung): {label} …")
        analyzer = PoseAnalyzer(video_mode=is_video)
        try:
            frames = read_frames(group[0], fps=fps) if is_video else read_images(group)
            analyzed, report = analyze(frames, analyzer, log)
            out.append(Recording(group, is_video, label, analyzed, report))
        except ValueError as exc:
            if len(groups) == 1:
                raise
            log(f"{label}: {exc}")
            out.append(Recording(group, is_video, label, [], None, str(exc)))
        finally:
            analyzer.close()
        if len(groups) > 1 and out[-1].report is not None:
            r = out[-1].report
            log(f"{label}: {r.usable} von {r.frames} Bildern brauchbar, "
                f"{sum(r.covered)} von {len(r.covered)} Blickwinkeln.")
        best = max(out, key=_score)
        for rec in out:
            if rec is not best or not keep_frames:
                rec.frames = []
    return out


def best_recording(recordings: list[Recording], log: Log | None = None) -> Recording:
    """Die Aufnahme mit den meisten erfassten Blickwinkeln (dann den meisten brauchbaren Bildern)."""
    log = log or _noop
    readable = [r for r in recordings if r.report is not None]
    if not readable:
        raise ValueError(" ".join(r.error or "" for r in recordings).strip())
    best = max(readable, key=_score)
    if len(recordings) > 1:
        log(f"Verwendet wird {best.label}.")
    return best


MAP_FILES = ("haut_farbe.jpg", "haut_relief.jpg", "haut_glanz.jpg")


def skin_maps(model: BodyModel, params: BodyParams, skin: np.ndarray, folder: Path,
              log: Log | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hautmaterial zum Hautbild (material.py): Farbe ohne Raumlicht, Relief, Glanz. Liegen die
    Bilder schon im Ordner und sind neuer als haut.jpg, werden sie wiederverwendet."""
    from . import head as H
    from .export import build_rig
    from .material import skin_material

    folder = Path(folder)
    paths = [folder / f for f in MAP_FILES]
    source = folder / "haut.jpg"
    if all(p.exists() for p in paths) and (not source.exists() or
                                           min(p.stat().st_mtime for p in paths) >= source.stat().st_mtime):
        albedo, relief = _read_rgb(paths[0]), _read_rgb(paths[1])
        return albedo, relief, cv2.imread(str(paths[2]), cv2.IMREAD_GRAYSCALE)
    (log or _noop)("Hautmaterial: Raumlicht herausrechnen, Relief und Glanz")
    rig = build_rig(model, params, morphs=False)
    mat = skin_material(model, rig.vertices, rig.faces, skin, head=H.head_region(model), log=log)
    folder.mkdir(parents=True, exist_ok=True)
    for path, img in zip(paths, (mat.albedo, mat.normal), strict=False):
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(paths[2]), mat.roughness, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (folder / "hautmaterial.json").write_text(json.dumps(mat.report, indent=2), "utf-8")
    return mat.albedo, mat.normal, mat.roughness


def _export(model: BodyModel, params: BodyParams, folder: Path, presets,
            skin: np.ndarray | None = None, eyes: np.ndarray | None = None, maps: Path | None = None,
            log: Log | None = None) -> tuple[dict[str, str], list[str]]:
    """maps: Ordner mit haut.jpg, in dem das Hautmaterial liegt bzw. abgelegt wird (Standard: folder)."""
    relief = roughness = None
    if skin is not None:
        skin, relief, roughness = skin_maps(model, params, skin, maps or folder, log=log)
    out = export_body(model, params, folder, presets=presets, skin=skin, eyes=eyes, relief=relief,
                      roughness=roughness)
    hints = list(out.pop("hinweise", []))
    return {k: str(v) for k, v in out.items()}, hints


def _read_rgb(path: Path) -> np.ndarray | None:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) if Path(path).exists() else None


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
         refine: bool = False, texture: bool = True, head_video: Path | None = None, detail: bool = True,
         hand_videos: list[Path] | None = None, log: Log | None = None) -> ScanResult:
    """Video oder Fotos -> angepasster, riggbarer Körper als GLB/OBJ, gespeichert beim Namen.

    texture: Hautbild (Farben, Tattoos, Muttermale) aus den Bildern aufs Modell übertragen.

    detail: Feinform aus allen Blickwinkeln: Schultern, Arme, Brust, Rücken, Bauch, Gesäß und Waden
    folgen in jedem Bild dem Umriss (Muskeln als Form, nicht nur als Schatten im Hautbild).

    head_video: zusätzliche Nahaufnahme des Kopfes (Kopf langsam in alle Richtungen drehen). Daraus
    kommen Gesichtsform, Schädel, Ohren und das scharfe Hautbild von Gesicht und Augen.

    hand_videos: je Hand ein Video aus der Nähe (Hand langsam drehen, offen, gespreizt, Faust). Daraus
    kommen die Fingerlänge und das scharfe Hautbild von Handrücken, Handfläche und Fingern.

    refine: Maskenkanten mit GrabCut nachschärfen. Hilft bei unruhigem Hintergrund, schadet bei
    ruhigem Hintergrund leicht (Testvideo: Brust 2,5 statt 3,2 cm Abweichung ohne Nachschärfen).
    """
    from .video import select_for_fit

    log = log or _noop
    problems = person.validate()
    if problems:
        raise ValueError(" ".join(problems))
    store = store or Store()
    paths = [Path(p) for p in (source if isinstance(source, list) else [source])]
    for p in paths + ([Path(head_video)] if head_video else []) + [Path(v) for v in hand_videos or []]:
        if not p.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {p}")
    recording = best_recording(check_recordings(paths, fps, log), log)
    paths, analyzed, report, is_video = recording.paths, recording.frames, recording.report, recording.is_video
    for line in report.summary:
        log(line)
    chosen = select_for_fit(analyzed)
    if not chosen:
        raise ValueError("Kein Bild war für die Anpassung brauchbar. " + " ".join(report.summary))
    log(f"{len(chosen)} Bilder aus verschiedenen Blickwinkeln werden verwendet.")

    if refine:
        log("Kanten der Personenmaske werden nachgeschärft (unruhiger Hintergrund) …")
    masks = [refine_mask(a.rgb, a.obs.mask) if refine else a.obs.mask for a in chosen]
    cam = level_camera(chosen, masks, log)
    fit_frames = cam.fit_frames(chosen, masks)

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
    quality = {
        "bericht": report.to_json(),
        "bilder": [r.to_json() for r in frame_results],
        "quelle": [str(p) for p in paths],
    }
    if detail:
        refined = detail_stage(model, params, recording, cam, person.measures, folder, log=log)
        if refined is not None:
            params, quality["feinform"] = refined
            measures = measure_params(model, params)
    skin = None
    if texture:
        skin = bake_skin(model, params, recording, cam, folder, log=log)
    eyes = None
    if head_video:
        params, skin, eyes, quality["kopf"] = head_stage(model, params, Path(head_video), skin, folder,
                                                         texture=texture, log=log)
    if hand_videos:
        params, skin, quality["haende"] = hand_stage(model, params, hand_videos, skin, folder, texture=texture,
                                                     log=log)
    files, hints = _export(model, params, folder, presets, skin, eyes, log=log)
    record = store.save_scan(rec.id, folder, params, person.measures, measures, quality,
                             ("video" if is_video else "bilder") + ("+kopf" if head_video else "")
                             + ("+haende" if hand_videos else ""), files)
    (folder / "bilder.json").write_text(json.dumps(quality["bilder"], indent=2), "utf-8")
    sheet = fit_sheet(frame_results)
    if sheet is not None:
        cv2.imwrite(str(folder / "passung.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    log(f"Fertig: {folder}")
    return ScanResult(record, params, measures, report, frame_results, files, hints)


# ------------------------------------------------------------------------- Kamera und Hautbild


@dataclasses.dataclass
class LevelCamera:
    """Kamera der Analysebilder und, falls das Handy gekippt war, die gerade ausgerichtete."""

    source: Intrinsics  # Kamera der Analysebilder
    intr: Intrinsics  # gerade Kamera, in der angepasst wird
    tilt: level.Leveling | None

    def warp(self, mask: np.ndarray, keypoints: np.ndarray | None):
        if self.tilt is None:
            return mask, keypoints
        return level.apply(self.tilt, self.source, self.intr, mask, keypoints)

    def fit_frames(self, frames: list, masks: list[np.ndarray] | None = None) -> list[FitFrame]:
        masks = masks or [a.obs.mask for a in frames]
        out = []
        for a, m in zip(frames, masks, strict=True):
            mask, kp = self.warp(m, a.obs.keypoints)
            out.append(FitFrame(mask=mask, intr=self.intr, keypoints=kp, yaw_init=a.quality.yaw))
        return out

    def to_image(self, full_width: int) -> np.ndarray:
        """3x3: Pixel der geraden Kamera -> Pixel im Originalbild der Breite full_width."""
        f = full_width / self.source.width
        scale = np.diag([f, f, 1.0])
        if self.tilt is None:
            return scale
        return scale @ np.linalg.inv(self.tilt.homography(self.source, self.intr))


def level_camera(frames: list, masks: list[np.ndarray], log: Log | None = None) -> LevelCamera:
    log = log or _noop
    h, w = frames[0].rgb.shape[:2]
    intr = Intrinsics.phone_default(w, h)
    tilt = level.estimate([a.rgb for a in frames], masks, intr)
    if tilt is None or max(abs(tilt.pitch_deg), abs(tilt.roll_deg)) < level.MIN_TILT_DEG:
        return LevelCamera(intr, intr, None)
    direction = "oben" if tilt.pitch_deg > 0 else "unten"
    log(f"Handy war um {abs(tilt.pitch_deg):.0f}° nach {direction} gekippt "
        f"(seitlich {tilt.roll_deg:+.0f}°), wird ausgeglichen.")
    return LevelCamera(intr, level.leveled_intrinsics(tilt, intr, masks), tilt)


DETAIL_FRAMES = 48  # Bilder für Feinform und Hautbild


def detail_stage(model: BodyModel, params: BodyParams, recording: Recording, cam: LevelCamera,
                 measures: dict[str, float] | None = None, folder: Path | None = None,
                 max_frames: int = DETAIL_FRAMES, log: Log | None = None):
    """Feinform aus den Umrissen vieler Bilder rundherum. Gibt (Parameter, Kennzahlen) zurück oder
    None, wenn zu wenige Bilder brauchbar sind.

    Die dabei mitgeführten Haltungen je Bild bleiben hier: Sie treffen den Umriss, legen aber das
    Hautbild weniger deckungsgleich (Rumpf und Becken kippen je Bild ein paar Grad). bake_skin
    richtet die Bilder deshalb auf der neuen Form noch einmal mit fit_views aus."""
    from .fit import fit_views
    from .refine import refine_shape
    from .video import select_for_texture

    log = log or _noop
    frames = select_for_texture(recording.frames, max_frames)
    if len(frames) < 12:
        log("Zu wenige Blickwinkel für die Feinform, sie wird übersprungen.")
        return None
    fit = cam.fit_frames(frames)
    log(f"Feinform: {len(frames)} Bilder rundherum werden genau ausgerichtet …")
    views = fit_views(model, params, fit, log=log)
    params, _, report = refine_shape(model, params, fit, views, measures=measures, log=log)
    info = report.to_json()
    if folder is not None:
        (Path(folder) / "feinform.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), "utf-8")
    return params, info


def bake_skin(model: BodyModel, params: BodyParams, recording: Recording, cam: LevelCamera,
              folder: Path | None = None, max_frames: int = DETAIL_FRAMES, size: int | None = None,
              views: list | None = None, fine_align: bool = True, white_balance: bool = True,
              log: Log | None = None) -> np.ndarray | None:
    """Hautbild aus den Bildern einer Aufnahme. Speichert haut.jpg und ansichten.json im Ordner.

    Erst werden die Bilder über Silhouette und Körperpunkte ausgerichtet (Lage und Haltung je
    Bild). fine_align: danach ein zweiter Durchgang, in dem das vorläufige Hautbild in jedes
    Videobild gerendert und per optischem Fluss nachgeschoben wird, damit Tattoos, Schrift und
    Kanten aus allen Bildern genau übereinander liegen. white_balance: Farbstich des Lichts an
    hellen, farblosen Flächen im Hintergrund (weiße Wand) messen und herausrechnen."""
    from .fit import fit_views, view_vertices
    from .texture import (
        TEXTURE_SIZE,
        Shot,
        TextureBaker,
        alignment_flow,
        neutralize,
        render_textured,
        white_point,
        white_samples,
    )
    from .video import read_full_frames, read_images, select_for_texture

    log = log or _noop
    frames = select_for_texture(recording.frames, max_frames)
    if views is not None and len(views) != len(frames):
        views = None  # andere Bildauswahl: neu ausrichten
    if len(frames) < 4:
        log("Zu wenige brauchbare Bilder für das Hautbild, Modell bleibt einfarbig.")
        return None
    if views is None:
        log(f"Hautbild: {len(frames)} Bilder werden genau ausgerichtet …")
        views = fit_views(model, params, cam.fit_frames(frames), log=log)
    if folder is not None:
        (Path(folder) / "ansichten.json").write_text(json.dumps(
            [{"bild": a.obs.index, **v.to_json()} for a, v in zip(frames, views, strict=True)], indent=1), "utf-8")
    sharp = np.array([a.quality.metrics.get("sharpness", 1.0) for a in frames])
    weights = np.sqrt(sharp / max(sharp.max(), 1e-6))
    by_index = {a.obs.index: i for i, a in enumerate(frames) if views[i].iou >= 0.7}
    if not by_index:
        log("Kein Bild passte gut genug für das Hautbild, Modell bleibt einfarbig.")
        return None
    uv = model.anny.texture_coordinates.detach().cpu().numpy()
    face_uv = model.anny.face_texture_coordinate_indices.detach().cpu().numpy()

    def shots():
        if recording.is_video:
            full = read_full_frames(recording.paths[0], set(by_index))
        else:
            full = ((i, rgb) for i, _, rgb in read_images(recording.paths, max_side=None) if i in by_index)
        for index, rgb in full:
            if index not in by_index:
                continue
            i = by_index[index]
            view = views[i]
            mask, _ = cam.warp(frames[i].obs.mask, None)
            yield Shot(rgb=rgb, mask=mask, intr=cam.intr, to_image=cam.to_image(rgb.shape[1]),
                       vertices=view_vertices(model, params, view), yaw=view.yaw, trans=view.trans,
                       weight=float(weights[i]))

    size = size or TEXTURE_SIZE
    baker = TextureBaker(model, size)
    whites = []
    for n, shot in enumerate(shots()):
        baker.add(shot)
        if white_balance:
            person = cv2.warpPerspective((shot.mask > 0.5).astype(np.uint8), shot.to_image,
                                         shot.rgb.shape[1::-1], flags=cv2.INTER_NEAREST)
            whites.append(white_samples(shot.rgb, person))
        if n % 8 == 0:
            log(f"Hautbild: Bild {n + 1}/{len(by_index)} übertragen")
    skin = baker.finish()
    if fine_align:
        # Raster für den Vergleich: etwa 1920 Pixel lange Seite
        k = 1920 / max(cam.intr.width, cam.intr.height)
        cintr = cam.intr.scaled(k)
        refined = TextureBaker(model, size, reference=skin)
        shifts = []
        for n, shot in enumerate(shots()):
            to_cmp = np.diag([k, k, 1.0]) @ np.linalg.inv(shot.to_image)  # Originalbild -> Vergleichsraster
            photo = cv2.warpPerspective(shot.rgb, to_cmp, (cintr.width, cintr.height), flags=cv2.INTER_AREA)
            render, body = render_textured(shot.vertices, model.faces, uv, face_uv, skin, shot.yaw, shot.trans,
                                           cintr, background=photo, with_mask=True)
            flow = alignment_flow(render, photo, body)
            shifts.append(float(np.median(np.linalg.norm(flow[body], axis=-1))) / k if body.any() else 0.0)
            shot.flow, shot.flow_scale = flow, k
            refined.add(shot)
            if n % 8 == 0:
                log(f"Hautbild fein ausgerichtet: Bild {n + 1}/{len(by_index)}")
        log(f"Feinausrichtung: Bilder im Mittel um {np.median(shifts):.1f} Pixel nachgeschoben.")
        baker, skin = refined, refined.finish()
    log(f"Hautbild aus {baker.shots} Bildern, {baker.coverage() * 100:.0f} % der Haut direkt gesehen, "
        "der Rest ist aus der Umgebung ergänzt.")
    white = white_point(whites) if white_balance else None
    if white is not None:
        skin = neutralize(skin, white)
        log("Weißabgleich an hellen Flächen im Hintergrund (Licht hatte Rot/Grün/Blau "
            + "/".join(f"{w:.2f}" for w in white) + ").")
    if folder is not None:
        (Path(folder) / "weiss.json").write_text(json.dumps(
            {"weisspunkt": None if white is None else [round(float(w), 4) for w in white]}), "utf-8")
        cv2.imwrite(str(Path(folder) / "haut.jpg"), cv2.cvtColor(skin, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
    return skin


def fit_sheet(results: list[FrameFitResult], height: int = 360, per_row: int = 6) -> np.ndarray | None:
    """Übersicht der Passung je Bild (grün passt, rot nur im Video, blau nur im Modell)."""
    tiles = []
    for r in results:
        if r.overlay is None:
            continue
        h, w = r.overlay.shape[:2]
        tile = cv2.resize(r.overlay, (max(1, round(w * height / h)), height), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"{r.yaw_deg:.0f} Grad  {r.iou:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (30, 30, 30), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return None
    width = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, width - t.shape[1], cv2.BORDER_CONSTANT, value=(245, 245, 245))
             for t in tiles]
    while len(tiles) % per_row:
        tiles.append(np.full_like(tiles[0], 245))
    rows = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
    return np.vstack(rows)


# ------------------------------------------------------------------------- Kopf


def head_stage(model: BodyModel, params: BodyParams, video: Path, body_skin: np.ndarray | None,
               folder: Path | None = None, body_white: np.ndarray | None = None, texture: bool = True,
               size: int | None = None, log: Log | None = None
               ) -> tuple[BodyParams, np.ndarray | None, np.ndarray | None, dict]:
    """Kopf aus einer Kopfaufnahme: Gesichtsform, Schädel und Ohren als Feinform, Gesicht als
    Hautbild (eingesetzt ins Körper-Hautbild), Augen als eigenes Bild. Gibt (Parameter, Hautbild,
    Augenbild, Kennzahlen) zurück; speichert kopf.json, kopfvergleich.jpg, haut.jpg, augen.jpg."""
    from . import head as H
    from .fit import keep_height, repair_offsets

    log = log or _noop
    log("Kopfvideo: Gesichtspunkte werden in jedem Bild gesucht …")
    tracks, intr = H.track_video(video, log=log)
    verts = model.rest_numpy(params)["vertices"].astype(np.float64)
    tri, bary = H.load_correspondence()
    sol = H.solve_face(tracks, intr, H.surface_points(verts, model.faces, tri, bary), log=log)
    log(f"Gesichtsform aus {len(tracks)} Bildern, mittlere Abweichung {sol.error_px:.1f} Pixel.")
    place = H.place_head(model, verts, sol)
    first = H.fit_head_shape(model, verts, sol, place, log=log)
    log("Kopfumriss wird in jedem Bild freigestellt …")
    masks = H.head_masks(video, sol, place, model, verts + first, log=log)
    disp = H.fit_head_shape(model, verts, sol, place, masks, log=log)
    out = params.copy()
    base = np.zeros_like(verts, dtype=np.float32) if params.offsets is None else params.offsets
    out.offsets = (base + disp / params.scale).astype(np.float32)
    out, repaired = repair_offsets(model, out)
    if repaired:
        log(f"Kopf: an {repaired} Punkten Feinform zurückgenommen (Flächen schoben sich ineinander).")
    new = model.rest_numpy(out)["vertices"].astype(np.float64)
    pts = place.to_anny(sol.points)
    info = {
        "bilder": len(tracks), "umrisse": len(masks), "fehler_px": round(sol.error_px, 2),
        "augenabstand_mm": round(float(np.linalg.norm(pts[468] - pts[473])) * 1000, 1),
        "gesichtsbreite_mm": round(float(np.linalg.norm(pts[234] - pts[454])) * 1000, 1),
        "nase_kinn_mm": round(float(np.linalg.norm(pts[H.NOSE_TIP] - pts[152])) * 1000, 1),
        "verschiebung_max_mm": round(float(np.linalg.norm(disp, axis=1).max()) * 1000, 1),
    }
    skin, eyes = body_skin, None
    if texture:
        log("Hautbild des Kopfes aus der Kopfaufnahme …")
        hs = H.bake_head(model, new, sol, place, video, masks, tracks, size=size, log=log)
        skin, eyes = H.merge_head_skin(hs.skin if body_skin is None else body_skin, hs.skin, hs.alpha,
                                       body_white, hs.white, log=log, eyes=hs.eyes)
        info["weisspunkt_kopf"] = None if hs.white is None else [round(float(w), 4) for w in hs.white]
        if folder is not None:
            cv2.imwrite(str(Path(folder) / "haut.jpg"), cv2.cvtColor(skin, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 94])
            if eyes is not None:
                cv2.imwrite(str(Path(folder) / "augen.jpg"), cv2.cvtColor(eyes, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 94])
            sheet = H.compare_head(model, new, sol, place, video, skin, eyes=eyes)
            if sheet is not None:
                cv2.imwrite(str(Path(folder) / "kopfvergleich.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
    # Zeigt die Kopfaufnahme einen flacheren (höheren) Schädel als Annys, gleicht der Hals aus: die
    # eingegebene Körpergröße und die Schultern aus dem Körpervideo bleiben (erst nach dem Hautbild,
    # das noch zur Lage des Kopfes im Kopfvideo passen muss; die UV-Karte bleibt gleich)
    target, now = (float(np.ptp(model.rest_numpy(p)["vertices"][:, 2])) for p in (params, out))
    if abs(now - target) > 0.001:
        out, _ = keep_height(model, out, target)
        info["hals_ausgleich_mm"] = round((target - now) * 1000, 1)
        log(f"Schädel {abs(target - now) * 1000:.0f} mm {'flacher' if now < target else 'höher'} als bei Anny: "
            f"Hals gleicht aus, Körpergröße bleibt {target * 100:.1f} cm.")
    if folder is not None:
        (Path(folder) / "kopf.json").write_text(json.dumps(info, indent=1), "utf-8")
    return out, skin, eyes, info


MIN_HAND_FRAMES = 8


def hand_stage(model: BodyModel, params: BodyParams, videos: list[Path], body_skin: np.ndarray | None,
               folder: Path | None = None, texture: bool = True, log: Log | None = None
               ) -> tuple[BodyParams, np.ndarray | None, list[dict]]:
    """Hände aus Handvideos (eine Hand je Video): Fingerlänge als Annys Regler,
    Hautbild von Mittelhand und Fingern (eingesetzt ins Körper-Hautbild). Gibt (Parameter, Hautbild,
    Kennzahlen je Hand) zurück; speichert haende.json, handvergleich_links/rechts.jpg und haut.jpg."""
    from . import hands as HS

    log = log or _noop
    out = params.copy()
    best: dict[str, tuple[Path, object]] = {}
    for video in videos:
        video = Path(video)
        side, tracks, intr = HS.track_hands(video, log=log)
        if len(tracks) < MIN_HAND_FRAMES:
            log(f"Handvideo {video.name}: zu wenige Bilder mit Hand, wird übergangen.")
            continue
        fit = HS.fit_hand(HS.HandRig(model, out, side), tracks, intr, log=log)
        hand = "Linke" if side == "L" else "Rechte"
        log(f"{hand} Hand: Handpunkte im Mittel {np.median(fit.errors):.1f} Pixel neben MediaPipes Punkten.")
        good = int((fit.errors < HS.GOOD_FRAME_PX).sum())
        if side not in best or good > int((best[side][1].errors < HS.GOOD_FRAME_PX).sum()):
            best[side] = (video, fit)
    for _, fit in best.values():
        for name, value in zip(fit.rig.names, fit.sliders, strict=True):
            out.local[name] = float(value)
    info = [HS.hand_info(fit) for _, fit in best.values()]
    skin = body_skin
    if texture and body_skin is not None and best:
        rest = model.rest_numpy(out)["vertices"].astype(np.float64)
        for video, fit in best.values():
            log(f"Hautbild der {'linken' if fit.side == 'L' else 'rechten'} Hand aus dem Handvideo …")
            hs = HS.bake_hand(model, rest, fit, video, size=body_skin.shape[0], log=log)
            skin = HS.merge_hand_skin(skin, hs, log=log)
            if folder is not None:
                sheet = HS.compare_hand(model, rest, fit, video, skin)
                if sheet is not None:
                    name = "handvergleich_" + ("links" if fit.side == "L" else "rechts") + ".jpg"
                    cv2.imwrite(str(Path(folder) / name), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
        if folder is not None:
            cv2.imwrite(str(Path(folder) / "haut.jpg"), cv2.cvtColor(skin, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 94])
    if folder is not None:
        HS.save_sides(info, folder)
    return out, skin, info


def video_white(path: Path, samples: int = 8) -> np.ndarray | None:
    """Farbstich einer Aufnahme an hellen, farblosen Flächen (für ältere Scans ohne weiss.json)."""
    from .texture import white_point, white_samples
    from .video import read_frames

    frames = list(read_frames(path, fps=1.0))
    pick = frames[:: max(1, len(frames) // samples)]
    return white_point([white_samples(rgb) for _, _, rgb in pick])


def add_head(record: ScanRecord, video: Path, store: Store | None = None, presets: tuple[str, ...] = PRESETS,
             device: str | None = None, log: Log | None = None) -> ScanResult:
    """Kopfvideo zu einem gespeicherten Scan hinzunehmen; ergibt einen neuen Scan derselben Person."""
    log = log or _noop
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {video}")
    store = store or Store()
    model = BodyModel.get(device)
    params = record.params()
    skin, white = None, None
    if (record.folder / "haut.jpg").exists():
        skin = cv2.cvtColor(cv2.imread(str(record.folder / "haut.jpg")), cv2.COLOR_BGR2RGB)
        if not (record.folder / "weiss.json").exists():
            sources = [Path(p) for p in record.quality.get("quelle", [])]
            if sources and sources[0].exists() and sources[0].suffix.lower() not in (".jpg", ".jpeg", ".png"):
                white = video_white(sources[0])
    folder = store.new_scan_folder(record.person_id)
    params, skin, eyes, info = head_stage(model, params, video, skin, folder, body_white=white, log=log)
    files, hints = _export(model, params, folder, presets, skin, eyes, log=log)
    quality = {**record.quality, "kopf": info, "kopfvideo": str(video), "aus_scan": record.id}
    new = store.save_scan(record.person_id, folder, params, record.inputs, record.measures, quality,
                          record.source.split("+")[0] + "+kopf", files)
    log(f"Fertig: {folder}")
    return ScanResult(new, params, record.measures, None, [], files, hints)


def add_hands(record: ScanRecord, videos: list[Path], store: Store | None = None,
              presets: tuple[str, ...] = PRESETS, device: str | None = None, log: Log | None = None) -> ScanResult:
    """Handvideos zu einem gespeicherten Scan hinzunehmen; ergibt einen neuen Scan derselben Person."""
    log = log or _noop
    videos = [Path(v) for v in videos]
    for v in videos:
        if not v.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {v}")
    store = store or Store()
    model = BodyModel.get(device)
    skin, eyes = (_read_rgb(record.folder / f) for f in ("haut.jpg", "augen.jpg"))
    folder = store.new_scan_folder(record.person_id)
    params, skin, info = hand_stage(model, record.params(), videos, skin, folder, log=log)
    files, hints = _export(model, params, folder, presets, skin, eyes if skin is not None else None, log=log)
    if eyes is not None:
        cv2.imwrite(str(folder / "augen.jpg"), cv2.cvtColor(eyes, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])
    quality = {**record.quality, "haende": info, "handvideos": [str(v) for v in videos], "aus_scan": record.id}
    source = record.source if "+haende" in record.source else record.source + "+haende"
    new = store.save_scan(record.person_id, folder, params, record.inputs, record.measures, quality, source, files)
    log(f"Fertig: {folder}")
    return ScanResult(new, params, record.measures, None, [], files, hints)


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
    skin, eyes = (_read_rgb(record.folder / f) for f in ("haut.jpg", "augen.jpg"))
    files, hints = _export(model, params, record.folder / f"export{suffix}", presets, skin,
                           eyes if skin is not None else None, maps=record.folder, log=log)
    for h in hints:
        log(h)
    return files


__all__ = ["ScanResult", "add_head", "body_from_measurements", "scan", "reexport", "cascadeur_check"]
