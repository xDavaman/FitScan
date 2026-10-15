"""Hautbild: Farben aus den Videobildern auf die Körperoberfläche übertragen.

Jede Stelle der Oberfläche hat einen festen Platz im Texturbild (UV-Karte aus Anny/MakeHuman,
CC0). Für jedes Videobild wird berechnet, welche Stellen die Kamera sieht und wie gut: frontal
statt streifend, nicht verdeckt, nicht am Rand der Silhouette (dort stimmt die Ausrichtung am
wenigsten). Die Farben werden in zwei Lagen gemischt:

- grobe Farbe und Helligkeit aus vielen Bildern (gleicht Belichtung und kleine Versätze aus),
- feine Einzelheiten wie Tattoos, Muttermale und Hautstruktur aus den besten Bildern (bleiben scharf).

Was nie zu sehen war (Fußsohlen, Achseln, Schritt), wird über die Oberfläche aus der Umgebung
aufgefüllt, damit keine Lücken bleiben.
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np
import torch

from .body import BodyModel
from .camera import Intrinsics, to_camera

TEXTURE_SIZE = 4096  # passt zur Auflösung eines 4K-Videos (etwa 2 Texel pro Millimeter Haut)
DEPTH_SCALE = 2.0  # Tiefenpuffer in doppelter Maskenauflösung
VISIBLE_TOLERANCE = 0.012  # Meter: so weit darf ein Punkt hinter der sichtbaren Fläche liegen
DEPTH_EDGE = 0.04  # Meter: Tiefensprung, ab dem eine Verdeckungskante beginnt
EDGE_RAMP = 0.03  # Anteil der Personenhöhe im Bild: Abstand zur Kante bis zum vollen Gewicht
DETAIL_SIGMA = 0.004  # Anteil der Personenhöhe im Bild: Grenze zwischen grober Farbe und Feinheiten
MIN_COS = 0.15  # streifender Blick zählt nicht
BASE_POWER = 2.0  # grobe Farbe: bessere Bilder zählen deutlich mehr
DETAIL_POWER = 6.0  # Feinheiten fast nur aus den besten Bildern
MAX_FLOW = 0.02  # Anteil der Personenhöhe: so weit darf die Feinausrichtung ein Bild verschieben
ROBUST_SIGMA = 40.0  # Farbabstand (0..255) zum vorläufigen Hautbild, ab dem ein Bild kaum noch zählt
SEEN_FULL = 0.05  # Gewichtssumme, ab der eine Stelle als sicher gesehen gilt
CLIPPED_RAMP = 8.0  # so nah an Weiß (255) zählt ein Pixel immer weniger (überstrahlt)
PADDING = 12  # Texel: Farbe über die Ränder der UV-Inseln hinaus (gegen Nähte beim Verkleinern)


# ------------------------------------------------------------------------- UV-Karte


@dataclasses.dataclass
class UVMap:
    size: int
    uv: np.ndarray  # (U, 2) Texturkoordinaten wie in OBJ (v nach oben)
    face_uv: np.ndarray  # (F, 3) Index in uv je Dreiecksecke
    texel: np.ndarray  # (N,) flacher Index ins Texturbild (Zeile * size + Spalte)
    face: np.ndarray  # (N,) Dreieck des Texels
    bary: np.ndarray  # (N, 3) baryzentrische Koordinaten des Texelmittelpunkts


_UV_CACHE: dict[int, UVMap] = {}


def uv_pixels(uv: np.ndarray, size: int) -> np.ndarray:
    """UV (v nach oben) -> Pixel im Texturbild (Zeile 0 oben, ganzzahlig = Texelmitte)."""
    return np.stack([uv[:, 0] * size - 0.5, (1.0 - uv[:, 1]) * size - 0.5], -1)


def uv_map(model: BodyModel, size: int = TEXTURE_SIZE) -> UVMap:
    if size in _UV_CACHE:
        return _UV_CACHE[size]
    uv = model.anny.texture_coordinates.detach().cpu().numpy().astype(np.float64)
    face_uv = model.anny.face_texture_coordinate_indices.detach().cpu().numpy().astype(np.int64)
    out = rasterize_uv(uv, face_uv, size)
    _UV_CACHE[size] = out
    return out


def rasterize_uv(uv: np.ndarray, face_uv: np.ndarray, size: int, faces: np.ndarray | None = None) -> UVMap:
    """Welches Dreieck (und wo darin) jeder Texel zeigt; faces: (F,) bool, nur diese Dreiecke."""
    px = uv_pixels(uv, size)
    ids = np.zeros((size, size), np.int32)
    tris = np.round(px[face_uv] * 16).astype(np.int32)
    for f in range(len(face_uv)) if faces is None else np.flatnonzero(faces):
        cv2.fillConvexPoly(ids, tris[f], int(f) + 1, lineType=cv2.LINE_8, shift=4)
    texel = np.flatnonzero(ids.ravel())
    face = ids.ravel()[texel] - 1
    rows, cols = np.divmod(texel, size)
    p = np.stack([cols, rows], -1).astype(np.float64)
    a, b, c = (px[face_uv[face, i]] for i in range(3))
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    w1 = (d11 * d20 - d01 * d21) / den
    w2 = (d00 * d21 - d01 * d20) / den
    bary = np.clip(np.stack([1 - w1 - w2, w1, w2], -1), 0, None)
    bary /= np.maximum(bary.sum(1, keepdims=True), 1e-9)
    return UVMap(size, uv.astype(np.float32), face_uv, texel, face, bary.astype(np.float32))


# Die Augäpfel haben in Annys UV-Karte nur etwa einen Texel pro Millimeter (Iris gut 10 Texel breit).
# Sie bekommen deshalb ein eigenes Bild: linkes Auge (der Person) in der linken, rechtes in der
# rechten Hälfte, jeweils so groß wie möglich. Netz und Skelett bleiben gleich.
EYE_TEXTURE = 2048


def eye_faces(model: BodyModel) -> dict[str, np.ndarray]:
    """(F,) bool je Augapfel ("L", "R"): Dreiecke, deren Ecken alle am Augenknochen hängen."""
    return {side: model.dominant_bone_mask(lambda n, s=side: n == f"eye.{s}")[model.faces].all(1)
            for side in ("L", "R")}


def eye_layout(model: BodyModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(uv, face_uv, eye): Texturkoordinaten mit zusätzlichen Einträgen für die Augäpfel im eigenen
    Augenbild, die Ecken-Indizes aller Dreiecke dazu und welche Dreiecke Augäpfel sind."""
    uv = model.anny.texture_coordinates.detach().cpu().numpy().astype(np.float64)
    face_uv = model.anny.face_texture_coordinate_indices.detach().cpu().numpy().astype(np.int64).copy()
    eye = np.zeros(len(face_uv), bool)
    extra = [uv]
    count = len(uv)
    for k, sel in enumerate(eye_faces(model).values()):
        if not sel.any():
            continue
        idx, inverse = np.unique(face_uv[sel], return_inverse=True)
        pts = uv[idx]
        lo, hi = pts.min(0), pts.max(0)
        scale = 0.47 / max(float((hi - lo).max()), 1e-9)  # Insel füllt ihre Hälfte, mit Rand
        pts = (pts - (lo + hi) / 2) * scale + [0.25 + 0.5 * k, 0.5]
        face_uv[sel] = count + inverse.reshape(-1, 3)
        extra.append(pts)
        count += len(pts)
        eye |= sel
    return np.concatenate(extra), face_uv, eye


_EYE_UV_CACHE: dict[int, UVMap] = {}


def eye_uv_map(model: BodyModel, size: int = EYE_TEXTURE) -> UVMap:
    if size not in _EYE_UV_CACHE:
        uv, face_uv, eye = eye_layout(model)
        _EYE_UV_CACHE[size] = rasterize_uv(uv, face_uv, size, eye)
    return _EYE_UV_CACHE[size]


# ------------------------------------------------------------------------- Sichtbarkeit


def _vertex_normals(v: np.ndarray, faces: np.ndarray) -> np.ndarray:
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    vn = np.zeros_like(v)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def face_buffer(cam: np.ndarray, faces: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Vorderstes Dreieck je Pixel (+1, 0 = Hintergrund). Maleralgorithmus über die zur Kamera
    zeigenden Dreiecke (bei einem geschlossenen Körper reicht das)."""
    z = np.maximum(cam[:, 2], 1e-3)
    uv = np.stack([intr.focal * cam[:, 0] / z + intr.cx, intr.focal * cam[:, 1] / z + intr.cy], -1)
    tri = cam[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    front = (fn * tri.mean(1)).sum(1) < 0
    idx = np.flatnonzero(front)
    idx = idx[np.argsort(-tri[idx, :, 2].mean(1))]
    ids = np.zeros((intr.height, intr.width), np.int32)
    pts = np.round(uv[faces[idx]] * 4).astype(np.int32)
    for f, p in zip(idx, pts, strict=True):
        cv2.fillConvexPoly(ids, p, int(f) + 1, lineType=cv2.LINE_8, shift=2)
    return ids


def _hit(cam: np.ndarray, faces: np.ndarray, intr: Intrinsics, ids: np.ndarray):
    """Pixel mit Körper: Lage (Zeile, Spalte), Dreieck und Schnittpunkt des Sehstrahls."""
    ys, xs = np.nonzero(ids)
    f = ids[ys, xs] - 1
    tri = cam[faces[f]]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ray = np.stack([(xs - intr.cx) / intr.focal, (ys - intr.cy) / intr.focal, np.ones(len(xs))], -1)
    denom = (n * ray).sum(1)
    t = (n * tri[:, 0]).sum(1) / np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return ys, xs, f, tri, n, ray * t[:, None]


def render_textured(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, face_uv: np.ndarray,
                    skin: np.ndarray, yaw: float, trans: np.ndarray, intr: Intrinsics,
                    background: np.ndarray | None = None, with_mask: bool = False,
                    camera: np.ndarray | None = None):
    """Modell mit Hautbild aus Sicht einer Kamera (ohne zusätzliches Licht: die Helligkeit steckt
    schon im Hautbild). Zum Vergleich mit dem Videobild. camera: Punkte schon in Kamerakoordinaten."""
    if camera is not None:
        cam = np.asarray(camera, np.float64)
    else:
        cam = to_camera(torch.as_tensor(vertices, dtype=torch.float32), torch.tensor([float(yaw)]),
                        torch.as_tensor(trans, dtype=torch.float32)[None])[0].numpy().astype(np.float64)
    ids = face_buffer(cam, faces, intr)
    out = (np.full((intr.height, intr.width, 3), 235, np.uint8) if background is None
           else background.copy())
    ys, xs, f, tri, n, p = _hit(cam, faces, intr, ids)
    nn = np.maximum((n * n).sum(1), 1e-18)
    bary = np.stack([(np.cross(tri[:, 2] - tri[:, 1], p - tri[:, 1]) * n).sum(1),
                     (np.cross(tri[:, 0] - tri[:, 2], p - tri[:, 2]) * n).sum(1),
                     (np.cross(tri[:, 1] - tri[:, 0], p - tri[:, 0]) * n).sum(1)], -1) / nn[:, None]
    bary = np.clip(bary, 0, None)
    bary /= np.maximum(bary.sum(1, keepdims=True), 1e-9)
    t = (uv_pixels(uv, skin.shape[0])[face_uv[f]] * bary[..., None]).sum(1)
    out[ys, xs] = np.clip(_sample(skin.astype(np.float32), t[:, 0], t[:, 1]), 0, 255).astype(np.uint8)
    return (out, ids > 0) if with_mask else out


def alignment_flow(rendered: np.ndarray, photo: np.ndarray, body: np.ndarray) -> np.ndarray:
    """Feinausrichtung: wohin sich jeder Pixel des gerenderten Modells verschieben muss, damit
    Tattoos, Bund und Kanten genau auf dem Videobild liegen (optischer Fluss, DIS aus OpenCV).
    Beide Bilder werden vorher auf ihre Struktur reduziert, damit Belichtung keine Rolle spielt."""

    def structure(rgb):
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hp = g - cv2.GaussianBlur(g, (0, 0), 6)
        sd = np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), 12)) + 2.0
        return np.clip(128 + 40 * hp / sd, 0, 255).astype(np.uint8)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(structure(rendered), structure(photo), None)
    ys = np.nonzero(body.any(1))[0]
    height = float(ys.max() - ys.min()) if len(ys) else float(body.shape[0])
    # nur am Körper, geglättet und begrenzt (einzelne Fehlzuordnungen dürfen nichts verziehen)
    m = cv2.dilate(body.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
    sigma = 0.004 * height
    num = cv2.GaussianBlur(flow * m[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)[..., None]
    flow = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)
    size = np.linalg.norm(flow, axis=-1, keepdims=True)
    limit = MAX_FLOW * height
    return (flow * np.minimum(1.0, limit / np.maximum(size, 1e-6))).astype(np.float32)


def depth_buffer(cam: np.ndarray, faces: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Tiefe (Meter) der vordersten Fläche je Pixel, inf = Hintergrund."""
    ids = face_buffer(cam, faces, intr)
    depth = np.full(ids.shape, np.inf, np.float32)
    ys, xs, _, _, _, p = _hit(cam, faces, intr, ids)
    depth[ys, xs] = np.where(p[:, 2] > 0, p[:, 2], np.inf)
    return depth


def _edge_distance(depth: np.ndarray) -> np.ndarray:
    """Abstand in Pixeln zur nächsten Silhouetten- oder Verdeckungskante."""
    fg = np.isfinite(depth)
    d = np.where(fg, depth, 1e3).astype(np.float32)
    jump = np.zeros(depth.shape, bool)
    for dy, dx in ((0, 1), (1, 0)):
        a = d[: d.shape[0] - dy, : d.shape[1] - dx]
        b = d[dy:, dx:]
        e = np.abs(a - b) > DEPTH_EDGE
        jump[: d.shape[0] - dy, : d.shape[1] - dx] |= e
        jump[dy:, dx:] |= e
    return cv2.distanceTransform((~jump).astype(np.uint8), cv2.DIST_L2, 5)


def _sample(img: np.ndarray, x: np.ndarray, y: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
    """Bild an vielen Punkten abtasten (cv2.remap verträgt keine sehr langen Einzelzeilen)."""
    n = len(x)
    width = 4096
    rows = max(1, math.ceil(n / width))
    pad = rows * width - n
    mx = np.pad(x.astype(np.float32), (0, pad)).reshape(rows, width)
    my = np.pad(y.astype(np.float32), (0, pad)).reshape(rows, width)
    out = cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.reshape(rows * width, *img.shape[2:])[:n]


# ------------------------------------------------------------------------- Weißabgleich

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


# ------------------------------------------------------------------------- Hautbild


@dataclasses.dataclass
class Shot:
    """Ein Bild für das Hautbild."""

    rgb: np.ndarray  # Originalbild, gern in voller Auflösung (uint8 RGB)
    mask: np.ndarray  # Personenmaske 0..1 in der Kamera intr (gerade ausgerichtet)
    intr: Intrinsics  # Kamera, zu der yaw/trans gehören (Maskenauflösung)
    to_image: np.ndarray  # (3, 3) Pixel in intr -> Pixel im Originalbild rgb
    vertices: np.ndarray  # (V, 3) Körper in der Haltung dieses Bildes (Anny-Koordinaten)
    yaw: float
    trans: np.ndarray
    weight: float = 1.0  # z. B. Schärfe des Bildes
    flow: np.ndarray | None = None  # Feinausrichtung (H, W, 2) auf dem Raster intr.scaled(flow_scale)
    flow_scale: float = 1.0
    camera: np.ndarray | None = None  # (V, 3) Körper direkt in Kamerakoordinaten (statt vertices/yaw/trans)
    region: np.ndarray | None = None  # (V,) 0..1: nur dieser Teil zählt (z. B. Kopf aus einer Kopfaufnahme)

    def camera_points(self) -> np.ndarray:
        if self.camera is not None:
            return np.asarray(self.camera, np.float64)
        return to_camera(torch.as_tensor(self.vertices, dtype=torch.float32), torch.tensor([float(self.yaw)]),
                         torch.as_tensor(self.trans, dtype=torch.float32)[None])[0].numpy().astype(np.float64)


@dataclasses.dataclass
class Observation:
    """Was ein Bild von der Oberfläche zeigt: Texel, Gewicht und Lage im Originalbild."""

    sel: np.ndarray  # (M,) Index in UVMap.texel
    weight: np.ndarray  # (M,) 0..1 mal Bildgewicht
    x: np.ndarray  # (M,) Pixel im Originalbild
    y: np.ndarray
    person_px: float  # Personenhöhe im Originalbild in Pixeln
    cam: np.ndarray  # (V, 3) Körper in Kamerakoordinaten


def observe(model: BodyModel, m: UVMap, shot: Shot) -> Observation | None:
    """Welche Texel zeigt dieses Bild wie gut (frontal, unverdeckt, nicht am Rand)?"""
    faces = model.faces
    cam = shot.camera_points()
    normals = _vertex_normals(cam, faces)
    # Nur Dreiecke, die zur Kamera zeigen (und zum gewünschten Teil gehören)
    tri = cam[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    facing = (fn * tri.mean(1)).sum(1) < 0
    used = faces
    if shot.region is not None:
        part = shot.region[faces].max(1) > 0
        facing &= part
        used = faces[part]
    sel = np.flatnonzero(facing[m.face])
    f, bary = m.face[sel], m.bary[sel].astype(np.float64)
    corners = faces[f]
    p = (cam[corners] * bary[..., None]).sum(1)
    nrm = (normals[corners] * bary[..., None]).sum(1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    cos = -(nrm * p).sum(1) / np.linalg.norm(p, axis=1)
    intr = shot.intr
    u = intr.focal * p[:, 0] / p[:, 2] + intr.cx
    v = intr.focal * p[:, 1] / p[:, 2] + intr.cy

    # Verdeckung über einen Tiefenpuffer in höherer Auflösung
    dintr = intr.scaled(DEPTH_SCALE)
    depth = depth_buffer(cam, used, dintr)
    du, dv = u * DEPTH_SCALE, v * DEPTH_SCALE
    front = _sample(depth, du, dv, cv2.INTER_NEAREST)
    visible = (p[:, 2] <= front + VISIBLE_TOLERANCE) & (cos > MIN_COS)
    # Abstand zu Kanten: im Modell (Silhouette, Arm vor dem Körper) und in der Videomaske
    fg = np.isfinite(depth)
    ys = np.nonzero(fg.any(1))[0]
    person_px = float(ys.max() - ys.min()) / DEPTH_SCALE if len(ys) else float(intr.height)
    ramp = EDGE_RAMP * person_px
    edge = _sample(_edge_distance(depth), du, dv, cv2.INTER_LINEAR) / DEPTH_SCALE
    mask = (shot.mask > 0.5).astype(np.uint8)
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    inside_px = _sample(inside, u, v, cv2.INTER_LINEAR)
    closeness = np.clip(np.minimum(edge, inside_px) / ramp, 0, 1)
    w = shot.weight * visible * closeness ** 2 * np.clip((cos - MIN_COS) / (1 - MIN_COS), 0, 1) ** 2
    if shot.region is not None:
        w = w * (shot.region[corners] * bary).sum(1)
    keep = w > 1e-4
    if not keep.any():
        return None
    u, v = u[keep], v[keep]
    if shot.flow is not None:
        k = shot.flow_scale
        shift = _sample(shot.flow, u * k, v * k) / k
        u, v = u + shift[:, 0], v + shift[:, 1]
    # Lage im Originalbild (über die Ausgleichsdrehung zurückgerechnet)
    h = shot.to_image
    s = h[2, 0] * u + h[2, 1] * v + h[2, 2]
    x = (h[0, 0] * u + h[0, 1] * v + h[0, 2]) / s
    y = (h[1, 0] * u + h[1, 1] * v + h[1, 2]) / s
    return Observation(sel[keep], w[keep].astype(np.float32), x, y,
                       math.hypot(h[0, 0], h[1, 0]) * person_px, cam)


class TextureBaker:
    """Sammelt Farben aus beliebig vielen Bildern (eins nach dem anderen, spart Speicher)."""

    def __init__(self, model: BodyModel, size: int = TEXTURE_SIZE, reference: np.ndarray | None = None,
                 best_full: float | None = None):
        """reference: vorläufiges Hautbild; Bilder, die an einer Stelle stark davon abweichen
        (verrutscht, Schatten, Hintergrund), zählen dort kaum. best_full: eine Stelle gilt erst als
        sicher gesehen, wenn ein Bild sie mindestens mit diesem Gewicht zeigt; nur streifend oder am
        Umriss gesehene Stellen (dort mischt sich leicht Hintergrund hinein) werden aus der Umgebung
        ergänzt."""
        self.model = model
        self.map = uv_map(model, size)
        self.reference = None
        if reference is not None:
            ref = cv2.GaussianBlur(reference.astype(np.float32), (0, 0), 2.0)
            self.reference = ref.reshape(-1, 3)[self.map.texel]
        n = len(self.map.texel)
        self.base = np.zeros((n, 3), np.float32)
        self.base_w = np.zeros(n, np.float32)
        self.detail = np.zeros((n, 3), np.float32)
        self.detail_w = np.zeros(n, np.float32)
        self.best_full = best_full
        self.best = np.zeros(n, np.float32)
        self.shots = 0

    def add(self, shot: Shot) -> float:
        """Ein Bild einrechnen. Gibt den Anteil der Oberfläche zurück, den es gut zeigt."""
        obs = observe(self.model, self.map, shot)
        if obs is None:
            return 0.0
        sel, w = obs.sel, obs.weight
        img = shot.rgb.astype(np.float32)
        blur = cv2.GaussianBlur(img, (0, 0), max(1.0, DETAIL_SIGMA * obs.person_px))
        col = _sample(img, obs.x, obs.y)
        base = _sample(blur, obs.x, obs.y)
        # überstrahlte Stellen (Glanzlicht auf Glatze oder Stirn) zeigen nicht die Hautfarbe
        w = w * np.clip((256.0 - col.max(1)) / CLIPPED_RAMP, 0.05, 1.0)
        if self.reference is not None:
            d = np.linalg.norm(base - self.reference[sel], axis=1)
            w = w * np.exp(-((d / ROBUST_SIGMA) ** 2))
        wb = w ** BASE_POWER
        self.best[sel] = np.maximum(self.best[sel], w)
        self.base[sel] += wb[:, None] * base
        self.base_w[sel] += wb
        wd = w ** DETAIL_POWER
        self.detail[sel] += wd[:, None] * (col - base)
        self.detail_w[sel] += wd
        self.shots += 1
        return float((w > 0.25).sum() / len(self.map.texel))

    def coverage(self) -> float:
        """Anteil der Oberfläche, der in mindestens einem Bild sicher zu sehen war."""
        return float((self.base_w >= SEEN_FULL).mean())

    def finish(self) -> np.ndarray:
        """Fertiges Texturbild (size, size, 3) uint8 RGB, Lücken aufgefüllt."""
        m, faces = self.map, self.model.faces
        seen = self.base_w > 1e-6
        col = np.zeros_like(self.base)
        col[seen] = self.base[seen] / self.base_w[seen, None]
        has_detail = self.detail_w > 1e-12
        col[has_detail] += self.detail[has_detail] / self.detail_w[has_detail, None]

        # Farbe je Körperpunkt aus den sicher gesehenen Texeln, dann über Kanten auffüllen
        nv = self.model.num_vertices
        corners = faces[m.face]
        alpha = np.clip(self.base_w / SEEN_FULL, 0, 1)
        if self.best_full:
            alpha *= np.clip(self.best / self.best_full, 0, 1)
        wv = (m.bary * alpha[:, None]).ravel()
        idx = corners.ravel()
        vw = np.bincount(idx, weights=wv, minlength=nv)
        vcol = np.stack([np.bincount(idx, weights=wv * np.repeat(col[:, c], 3), minlength=nv)
                         for c in range(3)], -1)
        known = vw > 1e-3
        vcol[known] /= vw[known, None]
        vcol = _diffuse(vcol, known, faces)
        fill = (vcol[corners] * m.bary[..., None]).sum(1)
        out = alpha[:, None] * col + (1 - alpha[:, None]) * fill

        img = np.zeros((m.size * m.size, 3), np.float32)
        img[m.texel] = out
        filled = np.zeros(m.size * m.size, bool)
        filled[m.texel] = True
        img = _pad(img.reshape(m.size, m.size, 3), filled.reshape(m.size, m.size), PADDING)
        return np.clip(img + 0.5, 0, 255).astype(np.uint8)


def _diffuse(values: np.ndarray, known: np.ndarray, faces: np.ndarray, rounds: int = 400) -> np.ndarray:
    """Unbekannte Punkte bekommen den Mittelwert ihrer Nachbarn (bekannte bleiben fest)."""
    if known.all():
        return values
    if not known.any():
        return np.full_like(values, 160.0)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a, b = np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])
    count = np.bincount(a, minlength=len(values)).astype(np.float64)
    vals = values.copy()
    vals[~known] = values[known].mean(0)
    for _ in range(rounds):
        acc = np.stack([np.bincount(a, weights=vals[b, c], minlength=len(values)) for c in range(3)], -1)
        new = acc / np.maximum(count, 1)[:, None]
        vals[~known] = new[~known]
    return vals


def _pad(img: np.ndarray, filled: np.ndarray, rounds: int) -> np.ndarray:
    """Farbe Schritt für Schritt in leere Texel neben den UV-Inseln schieben."""
    img = img * filled[..., None]
    mask = filled.astype(np.float32)
    for _ in range(rounds):
        acc = cv2.blur(img, (3, 3))
        wm = cv2.blur(mask, (3, 3))
        grow = (wm > 0) & (mask == 0)
        img[grow] = acc[grow] / wm[grow, None]
        mask[grow] = 1.0
    return img
