"""Hände aus Handvideos: Fingerlängen und Handhaut.

1. MediaPipe Hand Landmarker (Apache 2.0) findet in jedem Bild 21 Handpunkte (Handgelenk, Fingergelenke,
   Fingerspitzen) und welche Hand es ist.
2. Anpassung: Annys Hand (grip.Hand) bekommt je Bild eine eigene Lage und Fingerbeugung; für alle
   Bilder gemeinsam werden die Fingerlänge (Annys Regler) und die Brennweite gesucht.
   Die Größe der Hand bleibt wie vom Körper (aus einem Handvideo allein ist sie nicht messbar).
3. Hautbild der Hand aus den scharfen Nahbildern (wie beim Kopf: erst grob, dann per optischem Fluss
   nachgeschoben), in das Körper-Hautbild eingesetzt und an dessen Farbe angeglichen. Die Uhr am
   Handgelenk bleibt draußen: nur Mittelhand und Finger kommen aus dem Handvideo.
"""

from __future__ import annotations

import dataclasses
import json
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

from .body import BodyModel, BodyParams
from .camera import Intrinsics
from .grip import FINGERS, THUMB, Hand, reference_pose

Log = Callable[[str], None]

HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")
LANDMARKS = 21
ALL_FINGERS = (THUMB, *FINGERS)
# MediaPipe: 0 Handgelenk, dann je Finger (Daumen zuerst) Grund-, Mittel-, Endgelenk und Spitze;
# beim Daumen Sattel-, Grund-, Endgelenk und Spitze. Annys finger1 ist der Daumen.
SEGMENTS = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)] + [
    (4 * k + j, 4 * k + j + 1) for k in range(5) for j in (1, 2, 3)]
PALM = [0, 1, 5, 9, 13, 17]  # bewegen sich mit der Mittelhand (für den Start)
# Annys Regler "{l,r}-hand-...-incr". Fingerabstand, Fingerdicke und Handgröße verschieben die Gelenke
# kaum bzw. sind aus einem Handvideo allein nicht messbar: sie bleiben wie vom Körper.
SLIDERS = ("fingers-length",)
SLIDER_STEP = 0.5
MAX_FLEX = np.radians([[25, 70, 60], [90, 110, 80], [90, 110, 80], [90, 110, 80], [90, 110, 80]])
HYPEREXTEND = np.radians(15.0)  # so weit dürfen Gelenke nach hinten durchgebogen sein
HUBER_PX = 6.0  # Pixel (Analysebilder, lange Seite 1280)
# Handgelenk und Daumensattel liegen unter dicker Haut und wandern beim Beugen: zählen weniger
POINT_WEIGHT = np.array([0.3, 0.5] + [1.0] * 19)
GOOD_FRAME_PX = 7.0  # mittlere Abweichung, bis zu der ein Bild für das Hautbild zählt
HAND_FEATHER = 6.0  # Texel: weicher Übergang zum Hautbild aus dem Körpervideo
HAND_BEST_VIEW = 0.15


def ensure_hand_model() -> Path:
    from .vision import fitscan_home

    target = fitscan_home() / "modelle" / "hand_landmarker.task"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(HAND_MODEL_URL, tmp)
        tmp.replace(target)
    return target


@dataclasses.dataclass
class HandTrack:
    index: int  # Bildnummer im Video
    points: np.ndarray  # (21, 2) Pixel im Analysebild
    world: np.ndarray  # (21, 3) Meter, um die Handmitte, Achsen wie die Kamera
    score: float


def track_hands(path: Path, fps: float = 6.0, log: Log | None = None
                ) -> tuple[str, list[HandTrack], Intrinsics]:
    """Handpunkte in allen Bildern eines Handvideos. Gibt (Seite "L"/"R", Bilder, Kamera) zurück.
    Zeigt das Video beide Hände, zählt die Hand, die öfter zu sehen ist."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    from .video import read_frames

    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ensure_hand_model())),
        running_mode=vision.RunningMode.VIDEO, num_hands=2, min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3, min_tracking_confidence=0.3)
    found: dict[str, list[HandTrack]] = {"L": [], "R": []}
    size, seen = None, 0
    with vision.HandLandmarker.create_from_options(opts) as lm:
        for index, t, rgb in read_frames(path, fps=fps):
            size, seen = rgb.shape[:2], seen + 1
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
                                      int(t * 1000))
            h, w = rgb.shape[:2]
            for k, pts in enumerate(res.hand_landmarks):
                side = "L" if res.handedness[k][0].category_name == "Left" else "R"
                found[side].append(HandTrack(
                    index, np.array([[p.x * w, p.y * h] for p in pts], np.float32),
                    np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[k]], np.float32),
                    float(res.handedness[k][0].score)))
    if size is None:
        raise ValueError(f"Handvideo enthält keine lesbaren Bilder: {Path(path).name}")
    side = max(found, key=lambda s: len(found[s]))
    # je Bild höchstens eine Hand dieser Seite (die sicherere)
    best: dict[int, HandTrack] = {}
    for tr in found[side]:
        if tr.index not in best or tr.score > best[tr.index].score:
            best[tr.index] = tr
    tracks = [best[i] for i in sorted(best)]
    if log:
        log(f"Handvideo {Path(path).name}: {'linke' if side == 'L' else 'rechte'} Hand in {len(tracks)} "
            f"von {seen} Bildern.")
    return side, tracks, Intrinsics.phone_default(size[1], size[0])


# ------------------------------------------------------------------------- Handmodell


def _side_name(side: str) -> str:
    return "l" if side == "L" else "r"


def slider_names(side: str) -> list[str]:
    return [f"{_side_name(side)}-hand-{s}-incr" for s in SLIDERS]


def _rodrigues(aa: torch.Tensor) -> torch.Tensor:
    theta = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / theta
    kx, ky, kz = k.unbind(-1)
    zero = torch.zeros_like(kx)
    km = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], -1).reshape(*aa.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=aa.dtype).expand_as(km)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return eye + s * km + (1 - c) * km @ km


class HandRig:
    """Annys Hand als kleines, differenzierbares Modell: Fingerbeugung und Regler -> Handpunkte und Haut.

    Je Fingerknochen drei Winkel um feste Achsen der Bezugshaltung: Beugen (zur Handfläche hin
    positiv), Spreizen, Verdrehen. Mittelhand und Handgelenk bewegen sich nicht gegeneinander."""

    def __init__(self, model: BodyModel, params: BodyParams, side: str):
        self.side = side
        self.names = slider_names(side)
        base = [params.local.get(n, 0.0) for n in self.names]
        hands = [Hand(model, params, side)]
        for k, name in enumerate(self.names):
            p = params.copy()
            p.local[name] = base[k] + SLIDER_STEP
            hands.append(Hand(model, p, side, reference_pose(model, p)))
        h0 = hands[0]
        self.hand, self.base = h0, np.array(base)
        self.ids, self.faces = h0.ids, h0.faces

        def points(h: Hand) -> np.ndarray:
            out = [h.head[h.wrist_name]]
            for f in ALL_FINGERS:
                out += [h.head[f"{f}-{k}.{side}"] for k in (1, 2, 3)] + [h.tip[f]]
            return np.array(out)

        p0, v0 = points(h0), h0.rest
        self.origin = p0[0].copy()
        dt = torch.float64
        self.p0 = torch.tensor(p0 - self.origin, dtype=dt)
        self.dp = torch.tensor(np.stack([(points(h) - p0) / SLIDER_STEP for h in hands[1:]]), dtype=dt)
        self.v0 = torch.tensor(v0 - self.origin, dtype=dt)
        self.dv = torch.tensor(np.stack([(h.rest - v0) / SLIDER_STEP for h in hands[1:]]), dtype=dt)
        # Skinning-Gewichte der 15 Fingerknochen; Rest (Mittelhand, Handgelenk, Unterarm) steht
        bones = [f"{f}-{k}.{side}" for f in ALL_FINGERS for k in (1, 2, 3)]
        cols = [h0.names.index(b) for b in bones]
        self.w = torch.tensor(h0.weights[:, cols], dtype=dt)
        # Achsen je Knochen: Richtung d, Beugeachse h (positiv = zur Handfläche), Spreizachse n
        axes = []
        for k in range(15):
            a, b = p0[1 + k + k // 3], p0[2 + k + k // 3]
            d = (b - a) / np.linalg.norm(b - a)
            h = np.cross(d, h0.palm)
            h /= np.linalg.norm(h)
            axes.append(np.stack([h, np.cross(h, d), d]))
        self.axes = torch.tensor(np.array(axes), dtype=dt)  # (15, 3, 3): Zeilen Beugen, Spreizen, Drehen

    def forward(self, angles: torch.Tensor, sliders: torch.Tensor, with_skin: bool = True):
        """angles (B, 15, 3) Beugen/Spreizen/Drehen; sliders (2,) Änderung der Regler.
        Gibt Handpunkte (B, 21, 3) und Hautpunkte (B, n, 3) relativ zum Handgelenk zurück."""
        b = angles.shape[0]
        p = self.p0 + torch.einsum("s,sjc->jc", sliders, self.dp)
        aa = torch.einsum("bka,kac->bkc", angles, self.axes)
        rot = _rodrigues(aa)  # (B, 15, 3, 3) je Knochen um seine Achsen der Bezugshaltung
        pts = [p[0].expand(b, 3)]
        world, moved = [], []
        for f in range(5):
            chain = p[1 + 4 * f: 5 + 4 * f]
            w = torch.eye(3, dtype=p.dtype).expand(b, 3, 3)
            j = chain[0].expand(b, 3)
            pts.append(j)
            for k in range(3):
                w = w @ rot[:, 3 * f + k]
                world.append(w)
                moved.append(j)
                j = j + (w @ (chain[k + 1] - chain[k]))
                pts.append(j)
        pts = torch.stack(pts, 1)
        if not with_skin:
            return pts, None
        v = self.v0 + torch.einsum("s,snc->nc", sliders, self.dv)
        heads = torch.stack([p[1 + 4 * f + k] for f in range(5) for k in range(3)])  # (15, 3)
        world_t, moved_t = torch.stack(world, 1), torch.stack(moved, 1)  # (B, 15, 3, 3), (B, 15, 3)
        rel = v[None, None] - heads[None, :, None]  # (1, 15, n, 3)
        posed = torch.einsum("bkij,bknj->bkni", world_t, rel.expand(b, -1, -1, -1)) + moved_t[:, :, None]
        stay = (1 - self.w.sum(1))[None, :, None] * v[None]
        skin = stay + torch.einsum("nk,bkni->bni", self.w, posed)
        return pts, skin


# ------------------------------------------------------------------------- Anpassung


@dataclasses.dataclass
class HandFit:
    side: str
    frames: list[int]  # Bildnummern im Video
    rotations: np.ndarray  # (B, 3, 3) Hand (Anny, um das Handgelenk) -> Kamera
    translations: np.ndarray  # (B, 3)
    angles: np.ndarray  # (B, 15, 3)
    sliders: np.ndarray  # (2,) neue Reglerwerte (absolut)
    intr: Intrinsics
    errors: np.ndarray  # (B,) mittlere Abweichung der Handpunkte in Pixeln
    rig: HandRig
    anchors: np.ndarray  # (6, 3) wo MediaPipes Punkte der Mittelhand (PALM) gegenüber Annys Gelenken liegen

    def camera_points(self, k: int, model: BodyModel, rest: np.ndarray) -> np.ndarray:
        """Ganzer Körper (Ruhepose) mit gebeugten Fingern in Kamerakoordinaten von Bild k."""
        rig = self.rig
        with torch.no_grad():
            _, skin = rig.forward(torch.tensor(self.angles[k:k + 1]), torch.tensor(self.sliders - rig.base))
        pts = rest - rig.origin
        pts[rig.ids] = skin[0].numpy()
        return pts @ self.rotations[k].T + self.translations[k]


def _kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a, b = src - src.mean(0), dst - dst.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1, 1, d]) @ u.T


def fit_hand(rig: HandRig, tracks: list[HandTrack], intr: Intrinsics, steps: int = 500,
             log: Log | None = None) -> HandFit:
    """Lage, Fingerbeugung je Bild und gemeinsame Regler und Brennweite aus den Handpunkten."""
    dt = torch.float64
    pts2 = torch.tensor(np.stack([t.points for t in tracks]), dtype=dt)
    world = np.stack([t.world for t in tracks]).astype(np.float64)
    b = len(tracks)
    # Start: Drehung aus MediaPipes 3D-Punkten der Mittelhand, Abstand aus der Größe im Bild
    p0 = rig.p0.numpy()
    rots, trans = [], []
    for k in range(b):
        r = _kabsch(p0[PALM], world[k, PALM])
        size3 = np.linalg.norm(p0[PALM] - p0[PALM].mean(0), axis=1).mean()
        size2 = np.linalg.norm(tracks[k].points[PALM] - tracks[k].points[PALM].mean(0), axis=1).mean()
        z = intr.focal * size3 / max(size2, 1.0)
        c = tracks[k].points[PALM].mean(0)
        centre = np.array([(c[0] - intr.cx) * z / intr.focal, (c[1] - intr.cy) * z / intr.focal, z])
        trans.append(centre - r @ p0[PALM].mean(0))
        rots.append(cv2.Rodrigues(r)[0][:, 0])
    rvec = torch.nn.Parameter(torch.tensor(np.array(rots), dtype=dt))
    tvec = torch.nn.Parameter(torch.tensor(np.array(trans), dtype=dt))
    angles = torch.nn.Parameter(torch.zeros((b, 15, 3), dtype=dt))
    sliders = torch.nn.Parameter(torch.zeros(len(SLIDERS), dtype=dt))
    log_f = torch.nn.Parameter(torch.zeros((), dtype=dt))
    # MediaPipes Handgelenk liegt an der Handwurzel, Annys Handgelenk weiter im Unterarm; auch Daumen-
    # sattel und Fingergrundgelenke liegen etwas anders. Handgelenk und Daumensattel bekommen je einen
    # festen Versatz, die vier Grundgelenke einen gemeinsamen (sonst ginge der Fingerabstand darin auf).
    shift = torch.nn.Parameter(torch.zeros((3, 3), dtype=dt))
    spread_to_palm = torch.tensor([0, 1, 2, 2, 2, 2])
    palm = torch.tensor(PALM)
    point_w = torch.tensor(POINT_WEIGHT, dtype=dt)
    seg = torch.tensor(SEGMENTS)
    wdir = torch.tensor(world[:, seg[:, 1]] - world[:, seg[:, 0]], dtype=dt)
    wdir = wdir / wdir.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    max_flex = torch.tensor(MAX_FLEX.reshape(15), dtype=dt)
    # Nachbarbilder (höchstens zwei Analyseschritte auseinander) sollen sich ähnlich bewegen
    idx = np.array([t.index for t in tracks])
    gaps = np.diff(idx)
    near = torch.tensor(np.flatnonzero(gaps <= 2 * max(1, int(np.median(gaps)))))
    distal = torch.tensor([k for k in range(15) if k % 3 and k >= 3])  # Mittel- und Endglieder der Finger

    def project(rv, tv, ang, sl, lf):
        pts, _ = rig.forward(ang, sl, with_skin=False)
        pts = pts.index_add(1, palm, shift[spread_to_palm][None].expand(len(pts), -1, -1))
        cam = torch.einsum("bij,bkj->bki", _rodrigues(rv), pts) + tv[:, None]
        f = intr.focal * torch.exp(lf)
        z = cam[..., 2].clamp_min(0.03)
        uv = torch.stack([f * cam[..., 0] / z + intr.cx, f * cam[..., 1] / z + intr.cy], -1)
        return uv, pts

    def loss_fn(stage: int):
        uv, pts = project(rvec, tvec, angles, sliders, log_f)
        err = (uv - pts2).norm(dim=-1)
        huber = torch.where(err < HUBER_PX, 0.5 * err ** 2 / HUBER_PX, err - 0.5 * HUBER_PX)
        loss = (huber * point_w).mean()
        mdir = torch.einsum("bij,bkj->bki", _rodrigues(rvec), pts[:, seg[:, 1]] - pts[:, seg[:, 0]])
        mdir = mdir / mdir.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        loss = loss + 20.0 * (1 - (mdir * wdir).sum(-1)).mean()
        if stage:
            flex, spread, twist = angles.unbind(-1)
            loss = loss + 200 * (torch.relu(-flex - HYPEREXTEND) ** 2 + torch.relu(flex - max_flex) ** 2).mean()
            loss = loss + 20 * (twist ** 2).mean() + 50 * (spread[:, distal] ** 2).mean() + 2 * (spread ** 2).mean()
            if len(near):
                loss = loss + 5 * ((angles[near + 1] - angles[near]) ** 2).mean()
            loss = loss + (sliders ** 2).sum() + 2 * log_f ** 2 + 300 * (shift ** 2).sum()
        return loss, err

    # Schrittweiten je Größe: Winkel in Bogenmaß, Lage und Versatz in Metern
    groups = [(rvec, 0.02), (tvec, 0.002), (angles, 0.02), (sliders, 0.01), (log_f, 0.005), (shift, 0.001)]
    for stage, n, k in ((0, steps // 3, 2), (1, steps, len(groups))):
        opt = torch.optim.Adam([{"params": [v], "lr": lr} for v, lr in groups[:k]])
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.1 ** (1 / n))  # am Ende ein Zehntel
        for i in range(n):
            opt.zero_grad()
            loss, err = loss_fn(stage)
            loss.backward()
            opt.step()
            sched.step()
            if log and stage and i % 150 == 0:
                log(f"Hand: Schritt {i}/{n}, mittlere Abweichung {float(err.detach().mean()):.1f} px")
    with torch.no_grad():
        _, err = loss_fn(1)
    f = intr.focal * float(torch.exp(log_f.detach()))
    return HandFit(rig.side, [t.index for t in tracks],
                   np.array([cv2.Rodrigues(r)[0] for r in rvec.detach().numpy()]), tvec.detach().numpy(),
                   angles.detach().numpy(), rig.base + sliders.detach().numpy(),
                   Intrinsics(intr.width, intr.height, f, intr.cx, intr.cy), err.mean(1).numpy(), rig,
                   shift.detach()[spread_to_palm].numpy())


# ------------------------------------------------------------------------- Hautbild


def hand_region(model: BodyModel, side: str) -> np.ndarray:
    """(V,) 0..1: Anteil von Mittelhand und Fingern am Skinning, ohne Handgelenk (dort sitzt oft eine Uhr)."""
    names = {f"metacarpal{i}.{side}" for i in range(1, 5)}
    names |= {f"{f}-{k}.{side}" for f in ALL_FINGERS for k in (1, 2, 3)}
    sel = np.array([n in names for n in model.bone_labels])
    idx, w = model.vertex_bone_indices, model.vertex_bone_weights
    share = np.where(sel[idx], w, 0).sum(1) / np.maximum(w.sum(1), 1e-9)
    return np.clip((share - 0.5) / 0.4, 0, 1).astype(np.float32)


SKIN_CHROMA = 12.0  # Lab-Einheiten: so weit darf die Farbe eines Pixels von der Hautfarbe abweichen
NEAR_HAND = 0.03  # Anteil der langen Bildseite um den Umriss des Modells


def skin_mask(rgb: np.ndarray, outline: np.ndarray) -> np.ndarray:
    """Wo im Bild Haut der Hand liegt: nahe am Umriss des Modells und in der Farbe der Hand (nur der
    Farbton zählt, nicht die Helligkeit). Hält Boden, Teppich und Kleidung von den Fingerrändern fern,
    wenn das Modell dort nicht genau auf dem Finger liegt."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    r = max(3, int(NEAR_HAND * max(rgb.shape[:2])))
    near = cv2.dilate(outline.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    core = cv2.erode(outline.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r + 1, r + 1)))
    if core.sum() < 200:
        return near.astype(np.float32)
    ref = np.median(lab[core > 0][:, 1:], 0)
    d = np.linalg.norm(lab[..., 1:] - ref, axis=-1)
    return (near * np.exp(-((d / SKIN_CHROMA) ** 2))).astype(np.float32)


@dataclasses.dataclass
class HandSkin:
    skin: np.ndarray
    alpha: np.ndarray
    forearm: np.ndarray  # (size, size) bool: Unterarm derselben Seite, Bezug für die Farbe


def bake_hand(model: BodyModel, rest: np.ndarray, fit: HandFit, video: Path, size: int | None = None,
              fine_align: bool = True, log: Log | None = None) -> HandSkin:
    """Hautbild einer Hand aus dem Handvideo (volle Auflösung), wie beim Kopf in zwei Durchgängen."""
    from .texture import (
        TEXTURE_SIZE,
        Shot,
        TextureBaker,
        _pad,
        alignment_flow,
        face_buffer,
        render_textured,
    )
    from .video import read_full_frames

    size = size or TEXTURE_SIZE
    region = hand_region(model, fit.side)
    good = [k for k in range(len(fit.frames)) if fit.errors[k] < GOOD_FRAME_PX]
    where = {fit.frames[k]: k for k in good}
    intr, faces = fit.intr, model.faces
    used = region[faces].max(1) > 0
    part = faces[used]
    uv = model.anny.texture_coordinates.detach().cpu().numpy()
    face_uv = model.anny.face_texture_coordinate_indices.detach().cpu().numpy()[used]
    masks: dict[int, np.ndarray] = {}

    def shots():
        for index, rgb in read_full_frames(video, set(where)):
            k = where[index]
            f = rgb.shape[1] / intr.width
            cam = fit.camera_points(k, model, rest.copy())
            if index not in masks:
                small = cv2.resize(rgb, (intr.width, intr.height), interpolation=cv2.INTER_AREA)
                masks[index] = skin_mask(small, face_buffer(cam, part, intr) > 0)
            yield Shot(rgb=rgb, mask=masks[index], intr=intr, to_image=np.diag([f, f, 1.0]), vertices=rest,
                       yaw=0.0, trans=np.zeros(3), camera=cam, region=region,
                       weight=float(np.clip(2.0 - fit.errors[k] / GOOD_FRAME_PX, 0.2, 1.0)))

    baker = TextureBaker(model, size, best_full=HAND_BEST_VIEW)
    for shot in shots():
        baker.add(shot)
    skin = baker.finish()
    if fine_align:
        k_c = 1920 / max(intr.width, intr.height)
        cintr = intr.scaled(k_c)
        refined = TextureBaker(model, size, reference=skin, best_full=HAND_BEST_VIEW)
        shifts = []
        for shot in shots():
            to_cmp = np.diag([k_c, k_c, 1.0]) @ np.linalg.inv(shot.to_image)
            photo = cv2.warpPerspective(shot.rgb, to_cmp, (cintr.width, cintr.height), flags=cv2.INTER_AREA)
            render, body = render_textured(rest, part, uv, face_uv, skin, 0.0, np.zeros(3), cintr,
                                           background=photo, with_mask=True, camera=shot.camera)
            flow = alignment_flow(render, photo, body)
            shifts.append(float(np.median(np.linalg.norm(flow[body], axis=-1))) / k_c if body.any() else 0.0)
            shot.flow, shot.flow_scale = flow, k_c
            refined.add(shot)
        if log and shifts:
            log(f"Hand: Bilder im Mittel um {np.median(shifts):.1f} Pixel nachgeschoben.")
        baker, skin = refined, refined.finish()
    m = baker.map
    alpha = np.zeros(size * size, np.float32)
    alpha[m.texel] = (region[faces[m.face]] * m.bary).sum(1)
    filled = np.zeros(size * size, bool)
    filled[m.texel] = True
    alpha = _pad(np.repeat(alpha.reshape(size, size, 1), 3, -1), filled.reshape(size, size), 12)[..., 0]
    fore = model.dominant_bone_mask(lambda n: n.startswith("lowerarm") and n.endswith("." + fit.side))
    forearm = np.zeros(size * size, bool)
    forearm[m.texel] = fore[faces[m.face]].all(1)
    if log:
        log(f"Hautbild der {'linken' if fit.side == 'L' else 'rechten'} Hand aus {len(good)} Bildern.")
    return HandSkin(skin, alpha, forearm.reshape(size, size))


def merge_hand_skin(body: np.ndarray, hand: HandSkin, log: Log | None = None) -> np.ndarray:
    """Hand-Hautbild einsetzen; Farbe und Helligkeit an den Unterarm im Körper-Hautbild angleichen
    (anderes Licht). Bezug ist der Unterarm, nicht die Hand im Körpervideo: dort ist die Hand klein,
    oft im Schatten und mit Hintergrund vermischt, die Hand wirkte sonst dunkler als der Arm."""
    from .texture import to_linear, to_srgb

    alpha, head, fore = hand.alpha, hand.skin, hand.forearm
    if head.shape != body.shape:
        head = cv2.resize(head, body.shape[1::-1], interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, body.shape[1::-1], interpolation=cv2.INTER_AREA)
        fore = cv2.resize(fore.astype(np.uint8), body.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
    lb, lh = to_linear(body), to_linear(head)
    gain = np.ones(3, np.float32)
    ref, own = _typical(lb[fore]), _typical(lh[alpha > 0.8])
    if ref is not None and own is not None:
        gain = (ref / np.maximum(own, 1e-4)).astype(np.float32)
    if log:
        log("Hand an den Körper angeglichen (Rot/Grün/Blau " + "/".join(f"{g:.2f}" for g in gain) + ").")
    a = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), HAND_FEATHER)[..., None]
    a = np.minimum(a, alpha[..., None])  # nur innerhalb der Hand, nicht auf das Handgelenk hinaus
    return to_srgb(a * lh * gain + (1 - a) * lb)


def _typical(px: np.ndarray) -> np.ndarray | None:
    """Typische Hautfarbe: Median ohne die dunkelsten und hellsten Stellen (Tattoos, Uhr, Glanzlichter)."""
    if len(px) < 500:
        return None
    luma = px @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lo, hi = np.percentile(luma, [25, 85])
    keep = (luma >= lo) & (luma <= hi)
    return np.median(px[keep], 0)


def compare_hand(model: BodyModel, rest: np.ndarray, fit: HandFit, video: Path, skin: np.ndarray,
                 count: int = 4) -> np.ndarray | None:
    """Foto und Modell (mit Hautbild) nebeneinander für einige Bilder des Handvideos."""
    from .texture import render_textured
    from .video import read_full_frames

    good = [k for k in np.argsort(fit.errors) if fit.errors[k] < GOOD_FRAME_PX]
    if not good:
        return None
    pick = sorted(good[:: max(1, len(good) // count)][:count])
    where = {fit.frames[k]: k for k in pick}
    region = hand_region(model, fit.side)
    faces = model.faces
    used = (region[faces].max(1) > 0) | (np.isin(faces, fit.rig.ids).all(1))
    uv = model.anny.texture_coordinates.detach().cpu().numpy()
    face_uv = model.anny.face_texture_coordinate_indices.detach().cpu().numpy()[used]
    intr = fit.intr.scaled(640 / max(fit.intr.width, fit.intr.height))
    tiles = []
    for index, rgb in read_full_frames(video, set(where)):
        k = where[index]
        photo = cv2.resize(rgb, (intr.width, intr.height), interpolation=cv2.INTER_AREA)
        model_img = render_textured(rest, faces[used], uv, face_uv, skin, 0.0, np.zeros(3), intr,
                                    camera=fit.camera_points(k, model, rest.copy()))
        tiles.append(np.concatenate([photo, model_img], 1))
    return np.concatenate(tiles, 0) if tiles else None


# ------------------------------------------------------------------------- Ablauf


def hand_info(fit: HandFit) -> dict:
    good = fit.errors < GOOD_FRAME_PX
    return {
        "seite": "links" if fit.side == "L" else "rechts", "bilder": len(fit.frames),
        "bilder_hautbild": int(good.sum()), "fehler_px": round(float(np.median(fit.errors)), 2),
        "brennweite_px": round(fit.intr.focal, 1),
        "regler": {n: round(float(v), 3) for n, v in zip(fit.rig.names, fit.sliders, strict=True)},
    }


def save_sides(info: list[dict], folder: Path) -> None:
    (Path(folder) / "haende.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), "utf-8")


def sides_summary(sides: list[str]) -> str:
    c = Counter(sides)
    return ", ".join(f"{n}× {'links' if s == 'L' else 'rechts'}" for s, n in c.items())
