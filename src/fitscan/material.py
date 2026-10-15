"""Hautmaterial: Hautbild ohne Raumlicht, Relief für das Licht (Normal Map) und Glanz (Rauheit).

Das Hautbild aus dem Video enthält das Licht des Raums: Schultern und Kopf sind von oben hell,
Achseln, Kinnunterseite und Innenseiten der Beine dunkel. In Blender kommt dann noch einmal Licht
dazu, und der Körper wirkt flach und angemalt. Deshalb wird hier das Raumlicht herausgerechnet
(übrig bleibt die Hautfarbe, „Albedo“) und die Form der Haut als Normal Map beigelegt, damit
Muskeln, Falten und Poren auf das Licht in Blender reagieren.

Raumlicht: Die Person dreht sich vor der Kamera, das Licht im Raum steht. Jede Stelle wird vor
allem in den Bildern gesehen, in denen sie zur Kamera zeigt, also immer etwa gleich von der Seite
beleuchtet. Was übrig bleibt, hängt von zwei Dingen ab: wie weit die Stelle nach oben oder unten
zeigt (Deckenlicht, Boden) und wie viel vom Raum sie überhaupt sieht (Umgebungsverdeckung, in
Achseln und zwischen den Beinen wenig). Beides wird über die ganze Haut gemeinsam geschätzt und
herausgeteilt; die Hautfarbe selbst ändert sich nicht mit diesen Größen.

Relief: Mittelfeine Helligkeitsunterschiede ohne Farbänderung (Muskelränder, Falten, Knöchel)
werden als Höhe gedeutet; Tattoos, Muttermale und Schrift ändern die Farbe und bleiben flach.
Dazu kommt eine feine, gleichmäßige Hautstruktur (Poren), die im Video nicht sichtbar ist.
Normal Map im Tangentenraum wie in glTF und Blender: x entlang u, y entlang v (im Bild nach oben).
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from .texture import UVMap, _vertex_normals, to_linear, to_srgb, uv_map

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
AO_DIRECTIONS = 128
AO_RESOLUTION = 512
SHADING_RANGE = (0.55, 1.8)  # stärker wird nicht korrigiert (Fehler der Schätzung sollen nicht auffallen)
RELIEF_BAND_MM = (1.5, 40.0)  # Helligkeitsunterschiede dieser Größe werden zu Relief (Falten bis Bauchmuskeln)
RELIEF_DEPTH_MM = 4.0  # Tiefe für einen Helligkeitsunterschied von 100 %
PORE_MM = 0.35
PORE_DEPTH_MM = 0.04
SKIN_ROUGHNESS = 0.62
FACE_ROUGHNESS = 0.5
CLOTH_ROUGHNESS = 0.85


@dataclasses.dataclass
class SkinMaterial:
    albedo: np.ndarray  # (S, S, 3) uint8 sRGB, Hautfarbe ohne Raumlicht
    normal: np.ndarray  # (S, S, 3) uint8, Normal Map (Tangentenraum, OpenGL/glTF)
    roughness: np.ndarray  # (S, S) uint8, 0 = spiegelnd, 255 = matt
    shading: np.ndarray  # (V,) herausgerechnetes Raumlicht je Punkt (1 = Mittel)
    report: dict


# ------------------------------------------------------------ Umgebungsverdeckung


def _directions(n: int) -> np.ndarray:
    """Gleichmäßig verteilte Richtungen auf der Kugel (Fibonacci)."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    phi = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)


def ambient_occlusion(vertices: np.ndarray, faces: np.ndarray, directions: int = AO_DIRECTIONS,
                      resolution: int = AO_RESOLUTION) -> np.ndarray:
    """Anteil des Himmels, den jeder Punkt sieht (cos-gewichtet, 1 = frei, 0 = ganz verdeckt).

    Je Richtung ein Bild aus großer Entfernung (Parallelprojektion), das je Pixel das vorderste
    Dreieck zeigt. Ein Punkt sieht in diese Richtung, wenn an seiner Stelle eines seiner eigenen
    oder benachbarten Dreiecke vorn liegt."""
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    n = len(v)
    normals = _vertex_normals(v, f)
    # Nachbarschaft bis zu zwei Kanten weit, als sortierte Zahlen Punkt * n + Nachbar
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = np.unique(np.concatenate([edges, edges[:, ::-1], np.stack([np.arange(n)] * 2, 1)]), axis=0)
    starts = np.searchsorted(edges[:, 0], np.arange(n + 1))
    ring1 = [edges[starts[i]:starts[i + 1], 1] for i in range(n)]
    ring2 = np.unique(np.concatenate([i * n + np.concatenate([ring1[j] for j in ring1[i]]) for i in range(n)]))
    center = v.mean(0)
    radius = np.linalg.norm(v - center, axis=1).max() * 1.02
    scale = (resolution - 1) / (2 * radius)
    seen = np.zeros(n)
    total = np.zeros(n)
    for d in _directions(directions):
        cos = normals @ d
        facing = cos > 0
        if not facing.any():
            continue
        a = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        p = v - center
        px = np.stack([(p @ a + radius) * scale, (p @ b + radius) * scale], -1)
        depth = -(p @ d)  # Abstand von der Lichtseite, klein = vorn
        tri_order = np.argsort(-depth[f].mean(1))  # hinten zuerst, vorn übermalt
        ids = np.zeros((resolution, resolution), np.int32)
        pts = np.round(px[f[tri_order]] * 4).astype(np.int32)
        for k, poly in zip(tri_order, pts, strict=True):
            cv2.fillConvexPoly(ids, poly, int(k) + 1, lineType=cv2.LINE_8, shift=2)
        col = np.round(px[:, 0]).astype(int)
        row = np.round(px[:, 1]).astype(int)
        visible = np.zeros(n, bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                tri = ids[np.clip(row + dy, 0, resolution - 1), np.clip(col + dx, 0, resolution - 1)] - 1
                hit = tri >= 0
                for c in range(3):
                    code = np.arange(n) * n + f[np.maximum(tri, 0), c]
                    pos = np.minimum(np.searchsorted(ring2, code), len(ring2) - 1)
                    visible |= hit & (ring2[pos] == code)
        seen += np.where(facing & visible, cos, 0.0)
        total += np.where(facing, cos, 0.0)
    return (seen / np.maximum(total, 1e-9)).astype(np.float32)


# ---------------------------------------------------------------- Raumlicht


def _texel_values(m: UVMap, faces: np.ndarray, per_vertex: np.ndarray) -> np.ndarray:
    corners = faces[m.face]
    return (per_vertex[corners] * m.bary[..., None]).sum(1) if per_vertex.ndim == 2 else \
        (per_vertex[corners] * m.bary).sum(1)


def _features(normal: np.ndarray, ao: np.ndarray) -> np.ndarray:
    """Lichtmodell: glatte Funktion der Richtung (glTF-Achsen: x zur Seite, y oben, z vorn) und der
    Umgebungsverdeckung. Links und rechts gleich, damit die Hautfarbe nicht hineinrutscht."""
    side, up, front = normal[:, 0], normal[:, 1], normal[:, 2]
    a = np.log(np.clip(ao, 0.02, 1.0))
    return np.stack([np.ones_like(up), up, up ** 2, up ** 3, side ** 2, front, front ** 2, up * front,
                     a, a ** 2, up * a], -1)


def _skin_like(lin: np.ndarray) -> np.ndarray:
    """Texel, die nach Haut aussehen (nicht Hose, Socken, Tattoo-Tinte, Augen)."""
    s = to_srgb(lin).reshape(-1, 1, 3)
    lab = cv2.cvtColor(s, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    L, a, b = lab[:, 0] * 100 / 255, lab[:, 1] - 128, lab[:, 2] - 128
    return (L > 25) & (L < 92) & (a > 4) & (b > 6) & (a < 40) & (b < 45)


def estimate_shading(lin_texels: np.ndarray, normal: np.ndarray, ao: np.ndarray,
                     mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Koeffizienten des Lichtmodells (robuste Ausgleichsrechnung auf log-Helligkeit)."""
    lum = np.maximum(lin_texels @ LUMA, 1e-4)
    y = np.log(lum[mask])
    x = _features(normal[mask], ao[mask])
    w = np.ones(len(y))
    for _ in range(6):
        sw = np.sqrt(w)[:, None]
        beta, *_ = np.linalg.lstsq(x * sw, y * sw[:, 0], rcond=None)
        r = y - x @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        w = 1.0 / np.maximum(1.0, np.abs(r) / (1.5 * s))  # Huber: Tattoos, Muttermale zählen wenig
    explained = 1 - np.var(y - x @ beta) / max(np.var(y), 1e-9)
    return beta, {"texel": int(mask.sum()), "erklaerter_anteil": round(float(explained), 3)}


def _shading(beta: np.ndarray, normal: np.ndarray, ao: np.ndarray, ref: float) -> np.ndarray:
    s = np.exp(_features(normal, ao) @ beta - ref)
    return np.clip(s, *SHADING_RANGE)


# ---------------------------------------------------------------- Relief


def _texel_size_mm(m: UVMap, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Kantenlänge eines Texels auf der Haut (mm) je Dreieck."""
    from .texture import uv_pixels

    px = uv_pixels(m.uv.astype(np.float64), m.size)[m.face_uv]
    e1, e2 = px[:, 1] - px[:, 0], px[:, 2] - px[:, 0]
    uv_area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return np.sqrt(area / np.maximum(uv_area, 1e-12)) * 1000


def _fill(img: np.ndarray, known: np.ndarray, rounds: int = 24) -> np.ndarray:
    """Außerhalb der UV-Inseln mit den Nachbarn auffüllen (sonst entstehen Kanten an den Nähten)."""
    out = img.copy()
    k = known.astype(np.float32)
    for _ in range(rounds):
        num = cv2.blur(out * k[..., None] if out.ndim == 3 else out * k, (3, 3))
        den = cv2.blur(k, (3, 3))
        grow = (den > 1e-6) & (k < 0.5)
        val = num / np.maximum(den, 1e-6)[..., None] if out.ndim == 3 else num / np.maximum(den, 1e-6)
        out[grow] = val[grow]
        k = np.where(grow, 1.0, k).astype(np.float32)
    return out


def relief_height(albedo_lin: np.ndarray, m: UVMap, texel_mm: np.ndarray, skin: np.ndarray,
                  seed: int = 0) -> np.ndarray:
    """Höhe (mm) je Texel aus Helligkeit ohne Farbänderung plus feine Hautstruktur."""
    size = m.size
    img = np.zeros((size * size, 3), np.float32)
    img[m.texel] = albedo_lin
    img = img.reshape(size, size, 3)
    known = np.zeros(size * size, np.float32)
    known[m.texel] = 1.0
    known = known.reshape(size, size)

    skin_img = np.zeros(size * size, np.float32)
    skin_img[m.texel] = skin.astype(np.float32)
    skin_img = skin_img.reshape(size, size)
    mm = float(np.median(texel_mm))  # Texel sind auf dem Körper etwa gleich groß

    def blur(x: np.ndarray, sigma: float, w: np.ndarray) -> np.ndarray:
        """Unschärfe nur über Texel mit Gewicht w (normierte Faltung: Hose, Hintergrund und die
        Ränder der UV-Inseln ziehen nichts hinein)."""
        k = cv2.GaussianBlur(w, (0, 0), sigma)
        if x.ndim == 3:
            return cv2.GaussianBlur(x * w[..., None], (0, 0), sigma) / np.maximum(k, 1e-4)[..., None]
        return cv2.GaussianBlur(x * w, (0, 0), sigma) / np.maximum(k, 1e-4)

    w_skin = known * skin_img
    lum = np.log(np.maximum(img @ LUMA, 1e-4)) * known
    band = blur(lum, RELIEF_BAND_MM[0] / mm, w_skin) - blur(lum, RELIEF_BAND_MM[1] / mm, w_skin)
    # Farbe: weicht der Farbton vom Umfeld ab (Tinte, Muttermal, Schrift), ist es kein Relief
    chroma = img / np.maximum(img.sum(-1, keepdims=True), 1e-4)
    dc = np.linalg.norm(chroma - blur(chroma, RELIEF_BAND_MM[1] / mm, w_skin), axis=-1)
    keep = cv2.GaussianBlur(np.clip(1 - dc / 0.025, 0, 1) * w_skin, (0, 0), 1.5)
    band = np.clip(band, -0.5, 0.5)
    # wo die Farbe wechselt, die Höhe aus der Umgebung übernehmen (sonst wären Tattoos geprägt);
    # zum Rand der Haut hin (Hose, Socken) weich auslaufen lassen
    fill = blur(band, 4.0 / mm, keep)
    reach = int(6 * 3.0 / mm) | 1  # Haut über die UV-Nähte hinaus fortsetzen, sonst fehlt das Relief dort
    beyond = cv2.dilate(w_skin, np.ones((reach, reach), np.uint8))
    fade = np.clip(cv2.GaussianBlur(np.where(known > 0, w_skin, beyond), (0, 0), 3.0 / mm) * 2 - 0.5, 0, 1)
    height = (keep * band + (1 - keep) * fill) * fade * RELIEF_DEPTH_MM
    skin_img = fade
    rng = np.random.default_rng(seed)
    pores = rng.standard_normal((size, size)).astype(np.float32)
    pores = cv2.GaussianBlur(pores, (0, 0), max(0.6, PORE_MM / mm / 2))
    pores *= PORE_DEPTH_MM / max(float(pores.std()), 1e-6)
    return (height + pores * np.clip(skin_img, 0, 1)).astype(np.float32)


def normal_map(height_mm: np.ndarray, texel_mm: float) -> np.ndarray:
    """Höhe (mm) -> Normal Map (uint8). Bildzeilen laufen nach unten, v nach oben."""
    gx = cv2.Sobel(height_mm, cv2.CV_32F, 1, 0, ksize=3) / (8 * texel_mm)
    gy_down = cv2.Sobel(height_mm, cv2.CV_32F, 0, 1, ksize=3) / (8 * texel_mm)
    n = np.stack([-gx, gy_down, np.ones_like(gx)], -1)  # y nach oben: -(dh/dv) = +dh/dZeile
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.clip((n * 0.5 + 0.5) * 255 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- Rauheit


def roughness_map(m: UVMap, skin: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Haut matt-seidig, Gesicht etwas glänzender (Talg), Stoff matt. skin, head: je Texel."""
    rough = np.where(skin, SKIN_ROUGHNESS, CLOTH_ROUGHNESS).astype(np.float32)
    rough = np.where(skin, rough + (FACE_ROUGHNESS - SKIN_ROUGHNESS) * np.clip(head, 0, 1), rough)
    img = np.full(m.size * m.size, SKIN_ROUGHNESS, np.float32)
    img[m.texel] = rough
    img = cv2.GaussianBlur(img.reshape(m.size, m.size), (0, 0), 4.0)
    return np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- Gesamt


def skin_material(model, vertices: np.ndarray, faces: np.ndarray, skin_img: np.ndarray,
                  head: np.ndarray | None = None, log=None) -> SkinMaterial:
    """vertices: Ruhepose des Modells wie im glTF-Export (y oben, Blick +z). head: (V,) Anteil
    der Kopfaufnahme an der Farbe je Punkt (head.head_region); dort war das Licht ein anderes."""
    size = skin_img.shape[0]
    m = uv_map(model, size)
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    if log:
        log("Umgebungsverdeckung berechnen")
    ao = ambient_occlusion(v, f)
    vn = _vertex_normals(v, f)
    lin = to_linear(skin_img).reshape(-1, 3)[m.texel]
    nt = _texel_values(m, f, vn)
    nt /= np.maximum(np.linalg.norm(nt, axis=1, keepdims=True), 1e-9)
    ao_t = _texel_values(m, f, ao.astype(np.float64))
    skin = _skin_like(lin)
    w_head = np.zeros(len(v)) if head is None else np.clip(np.asarray(head, np.float64), 0, 1)
    texel_head = _texel_values(m, f, w_head)
    report = {}
    log_shading = {}
    for name, sel_t in (("koerper", texel_head < 0.1), ("kopf", texel_head > 0.9)):
        sel_t = sel_t & skin
        if sel_t.sum() < 1000:
            continue
        beta, rep = estimate_shading(lin, nt, ao_t, sel_t)
        # die Haut bleibt im Mittel gleich hell: Bezug ist das mittlere Licht dieser Gruppe
        ref = float(np.median(_features(nt[sel_t], ao_t[sel_t]) @ beta))
        log_shading[name] = np.log(_shading(beta, vn, ao, ref))
        report[name] = rep
    body = log_shading.get("koerper", np.zeros(len(v)))
    shading_v = np.exp((1 - w_head) * body + w_head * log_shading.get("kopf", body))
    shading_t = _texel_values(m, f, shading_v)
    albedo = lin / shading_t[:, None]
    report["licht_min"] = round(float(shading_v.min()), 2)
    report["licht_max"] = round(float(shading_v.max()), 2)
    if log:
        log("Relief und Glanz berechnen")
    texel_mm = _texel_size_mm(m, v, f)
    height = relief_height(albedo, m, texel_mm[m.face], skin)
    nmap = normal_map(height, float(np.median(texel_mm[m.face])))
    out = np.zeros((size * size, 3), np.float32)
    out[m.texel] = albedo
    known = np.zeros(size * size, bool)
    known[m.texel] = True
    out = _fill(out.reshape(size, size, 3), known.reshape(size, size), rounds=8)
    rough = roughness_map(m, skin, texel_head)
    return SkinMaterial(to_srgb(out), nmap, rough, shading_v.astype(np.float32), report)
