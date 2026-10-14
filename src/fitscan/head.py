"""Kopf aus einer Kopfaufnahme: Gesichtsform und Gesichtshaut.

1. MediaPipe Face Landmarker (Apache 2.0) liefert je Bild 478 Gesichtspunkte.
2. Bündelausgleich: Der Kopf ist starr. Aus allen Bildern zusammen folgen die 3D-Lage der Punkte
   (die Gesichtsform) und die Lage des Kopfes in jedem Bild.
3. Jeder Gesichtspunkt hat eine feste Stelle auf Annys Kopf (einmal berechnet mit
   scripts/calibrate_face.py, data/face_landmarks_anny.json).
4. Annys Kopf wird glatt so verformt, dass diese Stellen auf der gemessenen Gesichtsform liegen;
   die Augäpfel werden als Ganzes mitgeschoben. Skelett und Netz bleiben.
5. Das Hautbild des Kopfes kommt aus den Kopfbildern (scharf, nah) und wird an die Farben des
   Körpervideos angeglichen.
"""

from __future__ import annotations

import dataclasses
import json
import math
import urllib.request
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

from .body import BodyModel
from .camera import Intrinsics
from .texture import _hit, _vertex_normals, face_buffer

Log = Callable[[str], None]

FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/latest/face_landmarker.task")
LANDMARKS = 478
NOSE_TIP = 1
CORRESPONDENCE = Path(__file__).with_name("data") / "face_landmarks_anny.json"
# Bilder mit deutlicher Mimik verfälschen die Form (Mund offen, Lächeln, Stirnrunzeln)
EXPRESSION_LIMITS = {"jawOpen": 0.15, "mouthSmileLeft": 0.35, "mouthSmileRight": 0.35,
                     "mouthPucker": 0.4, "browInnerUp": 0.5}
HEAD_BONES = ("head", "eye", "neck02")


def ensure_face_model() -> Path:
    from .vision import fitscan_home

    target = fitscan_home() / "modelle" / "face_landmarker.task"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(FACE_MODEL_URL, tmp)
        tmp.replace(target)
    return target


class FaceAnalyzer:
    def __init__(self, video_mode: bool = True):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._video = video_mode
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_face_model())),
            running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
            num_faces=1, output_face_blendshapes=True, min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3, min_tracking_confidence=0.3)
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def analyze(self, rgb: np.ndarray, time_s: float = 0.0) -> tuple[np.ndarray | None, dict[str, float]]:
        """478 Punkte (x, y in Pixeln) und Mimik-Werte, oder None ohne Gesicht."""
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._lm.detect_for_video(image, int(time_s * 1000)) if self._video else self._lm.detect(image)
        if not res.face_landmarks:
            return None, {}
        h, w = rgb.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], np.float32)
        shapes = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
        return pts, shapes

    def close(self) -> None:
        self._lm.close()


# ------------------------------------------------------------------------- Zuordnung zu Anny


def head_vertex_mask(model: BodyModel) -> np.ndarray:
    return model.dominant_bone_mask(lambda n: n.startswith(HEAD_BONES))


def eye_vertex_masks(model: BodyModel) -> dict[str, np.ndarray]:
    return {side: model.dominant_bone_mask(lambda n, s=side: n == f"eye.{s}") for side in ("L", "R")}


def render_head(model: BodyModel, verts: np.ndarray, yaw_deg: float, intr: Intrinsics,
                distance: float = 0.45) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Schattierter Kopf (Haut, Augen mit Iris) von vorn, zur Zuordnung der Gesichtspunkte.
    Gibt Bild, Dreieck je Pixel (+1) und Kamerakoordinaten der Punkte zurück."""
    faces = model.faces
    head = head_vertex_mask(model)
    center = verts[head].mean(0)
    a = math.radians(yaw_deg)
    rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    p = (verts - center) @ rot.T
    cam = np.stack([p[:, 0], -p[:, 2], p[:, 1] + distance], -1)  # wie camera.to_camera
    ids = face_buffer(cam, faces, intr)
    ys, xs, f, tri, n, hit = _hit(cam, faces, intr, ids)
    bary = _bary(tri, n, hit)
    vn = _vertex_normals(cam, faces)
    nrm = (vn[faces[f]] * bary[..., None]).sum(1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    light = np.array([-0.2, -0.4, -1.0])
    light /= np.linalg.norm(light)
    shade = np.clip(0.3 + 0.7 * (nrm @ light), 0.0, 1.0)
    col = np.array([215.0, 170.0, 145.0])[None] * shade[:, None]
    eyes = np.zeros(model.num_vertices, bool)
    for m in eye_vertex_masks(model).values():
        eyes |= m
    on_eye = eyes[faces[f]].all(1)
    iris = -nrm[:, 2] > 0.93
    col[on_eye] = np.where(iris[on_eye, None], [60.0, 70.0, 90.0], [235.0, 235.0, 235.0])
    img = np.full((intr.height, intr.width, 3), 200, np.uint8)
    img[ys, xs] = col.astype(np.uint8)
    return img, ids, cam


def _bary(tri: np.ndarray, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    nn = np.maximum((n * n).sum(1), 1e-18)
    b = np.stack([(np.cross(tri[:, 2] - tri[:, 1], p - tri[:, 1]) * n).sum(1),
                  (np.cross(tri[:, 0] - tri[:, 2], p - tri[:, 2]) * n).sum(1),
                  (np.cross(tri[:, 1] - tri[:, 0], p - tri[:, 0]) * n).sum(1)], -1) / nn[:, None]
    b = np.clip(b, 0, None)
    return b / np.maximum(b.sum(1, keepdims=True), 1e-12)


def closest_surface_points(verts: np.ndarray, faces: np.ndarray, points: np.ndarray, near: int = 12):
    """Nächster Punkt auf der Oberfläche: Dreieck, baryzentrische Koordinaten, Abstand."""
    from scipy.spatial import cKDTree

    _, nearest = cKDTree(verts).query(points, k=near)
    incident: dict[int, list[int]] = {}
    for f, tri in enumerate(faces):
        for v in tri:
            incident.setdefault(int(v), []).append(f)
    out_tri, out_bary, out_dist = [], [], []
    for p, cand_v in zip(points, nearest, strict=True):
        cand = np.unique([f for v in cand_v for f in incident.get(int(v), [])])
        best = None
        for f in cand:
            q, b = _closest_on_triangle(p, *verts[faces[f]])
            d = float(np.linalg.norm(p - q))
            if best is None or d < best[2]:
                best = (int(f), b, d)
        out_tri.append(best[0])
        out_bary.append(best[1])
        out_dist.append(best[2])
    return np.array(out_tri), np.array(out_bary), np.array(out_dist)


def _closest_on_triangle(p, a, b, c):
    """Nächster Punkt auf dem Dreieck abc (Ericson, Real-Time Collision Detection) und seine
    baryzentrischen Koordinaten."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a, np.array([1.0, 0, 0])
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b, np.array([0, 1.0, 0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.array([1 - v, v, 0])
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c, np.array([0, 0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.array([1 - w, 0, w])
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.array([0, 1 - w, w])
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return a + ab * v + ac * w, np.array([1 - v - w, v, w])


def load_correspondence() -> tuple[np.ndarray, np.ndarray]:
    """Je Gesichtspunkt: Dreieck auf Annys Netz und baryzentrische Koordinaten."""
    data = json.loads(CORRESPONDENCE.read_text("utf-8"))
    return np.array(data["dreieck"], np.int64), np.array(data["bary"], np.float64)


def surface_points(verts: np.ndarray, faces: np.ndarray, tri: np.ndarray, bary: np.ndarray) -> np.ndarray:
    return (verts[faces[tri]] * bary[..., None]).sum(1)


# ------------------------------------------------------------------------- Kopfaufnahme auswerten


@dataclasses.dataclass
class FaceTrack:
    index: int  # Bildnummer im Video
    points: np.ndarray  # (478, 2) Pixel in der Analyseauflösung
    shapes: dict[str, float] = dataclasses.field(default_factory=dict)  # Mimik (Blick, Lider …)


@dataclasses.dataclass
class FaceSolution:
    points: np.ndarray  # (478, 3) Gesichtsform im Kopf-Koordinatensystem (Meter, nach Anpassung)
    rotations: np.ndarray  # (K, 3, 3) Kopf -> Kamera je Bild
    translations: np.ndarray  # (K, 3)
    intr: Intrinsics  # Kamera der Analysebilder
    frames: list[int]  # Bildnummern
    error_px: float  # mittlerer Reprojektionsfehler


def track_video(path: Path, fps: float = 6.0, log: Log | None = None) -> tuple[list[FaceTrack], Intrinsics]:
    """Gesichtspunkte in allen Bildern einer Kopfaufnahme; Bilder mit Mimik werden übergangen."""
    from .video import read_frames

    analyzer = FaceAnalyzer(video_mode=True)
    tracks, size, seen, expressive = [], None, 0, 0
    try:
        for index, t, rgb in read_frames(path, fps=fps):
            size = rgb.shape[:2]
            seen += 1
            pts, shapes = analyzer.analyze(rgb, t)
            if pts is None:
                continue
            if any(shapes.get(k, 0.0) > v for k, v in EXPRESSION_LIMITS.items()):
                expressive += 1
                continue
            tracks.append(FaceTrack(index, pts, shapes))
    finally:
        analyzer.close()
    if size is None:
        raise ValueError(f"Kopfvideo enthält keine lesbaren Bilder: {Path(path).name}")
    if log:
        log(f"Kopfvideo: Gesicht in {len(tracks) + expressive} von {seen} Bildern, "
            f"{expressive} wegen Mimik übergangen.")
    return tracks, Intrinsics.phone_default(size[1], size[0])


def face_mesh_edges() -> np.ndarray:
    """Kanten des MediaPipe-Gesichtsnetzes (Nachbarschaft der 478 Punkte)."""
    from mediapipe.tasks.python.vision import FaceLandmarksConnections

    return np.array([[c.start, c.end] for c in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION], np.int64)


def _rodrigues(aa: torch.Tensor) -> torch.Tensor:
    theta = aa.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = aa / theta
    kx, ky, kz = k.unbind(-1)
    zero = torch.zeros_like(kx)
    kmat = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], -1).reshape(*aa.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=aa.dtype).expand_as(kmat)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return eye + s * kmat + (1 - c) * kmat @ kmat


def solve_face(tracks: list[FaceTrack], intr: Intrinsics, template: np.ndarray, steps: int = 600,
               log: Log | None = None) -> FaceSolution:
    """Bündelausgleich: Gesichtsform und Kopflage je Bild aus den 2D-Punkten aller Bilder.

    template: (478, 3) Startform im Kopf-Koordinatensystem (z. B. Annys Gesichtspunkte, Meter):
    legt auch den Maßstab fest (eine einzelne Kamera kann Größe und Abstand nicht trennen)."""
    if len(tracks) < 8:
        raise ValueError("Zu wenige Bilder mit Gesicht im Kopfvideo (mindestens 8 nötig).")
    obs = torch.tensor(np.stack([t.points for t in tracks]), dtype=torch.float64)
    tmpl = template - template.mean(0)
    # Startlage je Bild mit solvePnP
    cam_k = np.array([[intr.focal, 0, intr.cx], [0, intr.focal, intr.cy], [0, 0, 1.0]])
    aa0, t0 = [], []
    for tr in tracks:
        ok, rvec, tvec = cv2.solvePnP(tmpl.astype(np.float64), tr.points.astype(np.float64), cam_k, None,
                                      flags=cv2.SOLVEPNP_EPNP)
        if ok:
            ok, rvec, tvec = cv2.solvePnP(tmpl.astype(np.float64), tr.points.astype(np.float64), cam_k, None,
                                          rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        aa0.append(rvec.ravel() if ok else np.zeros(3))
        t0.append(tvec.ravel() if ok else np.array([0, 0, 0.4]))
    aa = torch.nn.Parameter(torch.tensor(np.array(aa0), dtype=torch.float64))
    tt = torch.nn.Parameter(torch.tensor(np.array(t0), dtype=torch.float64))
    base = torch.tensor(tmpl, dtype=torch.float64)
    delta = torch.nn.Parameter(torch.zeros_like(base))
    log_f = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    scale = float(np.linalg.norm(tmpl, axis=1).mean())
    # Normalen der Startform: Richtung von einem Punkt hinter dem Gesicht (gegenüber der Nasenspitze),
    # gut genug, um abgewandte Punkte zu erkennen
    forward = tmpl[NOSE_TIP] / np.linalg.norm(tmpl[NOSE_TIP])
    center = torch.tensor(-0.6 * scale * forward, dtype=torch.float64)

    def residuals():
        pts = base + delta
        rot = _rodrigues(aa)
        cam = pts[None] @ rot.transpose(-1, -2) + tt[:, None]
        f = intr.focal * torch.exp(log_f)
        uv = torch.stack([f * cam[..., 0] / cam[..., 2] + intr.cx, f * cam[..., 1] / cam[..., 2] + intr.cy], -1)
        normal = torch.nn.functional.normalize(pts - center, dim=-1)
        view = torch.nn.functional.normalize(-cam, dim=-1)
        facing = ((normal[None] @ rot.transpose(-1, -2)) * view).sum(-1)
        w = torch.clamp((facing - 0.1) / 0.4, 0, 1).detach()
        return (uv - obs).norm(dim=-1), w

    edges = torch.tensor(face_mesh_edges())
    opt = torch.optim.Adam([aa, tt, delta, log_f], lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    px = float(np.median([np.ptp(t.points[:, 1]) for t in tracks]))  # Gesichtshöhe in Pixeln
    for step in range(steps):
        opt.zero_grad()
        err, w = residuals()
        robust = torch.where(err < 3.0, 0.5 * err ** 2, 3.0 * (err - 1.5))  # Huber, 3 px
        loss = (w * robust).sum() / w.sum().clamp_min(1)
        # Form glatt halten (Nachbarpunkte bewegen sich ähnlich), Lage festhalten
        d = delta / scale * px
        loss = loss + 1e-2 * (d[edges[:, 0]] - d[edges[:, 1]]).pow(2).mean() + 1e-5 * d.pow(2).mean()
        loss = loss + 10.0 * d.mean(0).pow(2).sum()
        loss.backward()
        opt.step()
        sched.step()
        if log and step % 150 == 0:
            log(f"Gesichtsform: Schritt {step}/{steps}, Fehler {float((err * w).sum().detach() / w.sum()):.2f} px")
    with torch.no_grad():
        err, w = residuals()
        mean_err = float((err * w).sum() / w.sum())
        rot = _rodrigues(aa).numpy()
    return FaceSolution((base + delta).detach().numpy(), rot, tt.detach().numpy(),
                        Intrinsics(intr.width, intr.height, intr.focal * float(torch.exp(log_f.detach())), intr.cx, intr.cy),
                        [t.index for t in tracks], mean_err)


def similarity(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None):
    """s, R, t mit s R src + t ≈ dst (Umeyama)."""
    w = np.ones(len(src)) if weights is None else weights
    w = w / w.sum()
    mu_s, mu_d = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    a, b = src - mu_s, dst - mu_d
    cov = (w[:, None] * b).T @ a
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    sign[2, 2] = np.sign(np.linalg.det(u @ vt))
    rot = u @ sign @ vt
    var = (w * (a * a).sum(1)).sum()
    s = float(np.trace(np.diag(d) @ sign) / var)
    return s, rot, mu_d - s * rot @ mu_s


# ------------------------------------------------------------------------- Kopf verformen

FACE_POINTS = 468  # ohne die zehn Irispunkte (die Augäpfel folgen den Lidern)
# Innenkante der Lippen: bei geschlossenem Mund fallen oben und unten zusammen, das würde Annys
# Lippen ineinanderschieben
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415]
SOCKET_RIGID = 0.5  # wie stark Lider und Augenhöhle zusammenbleiben
CONTACT_RADIUS = 0.004  # Meter: so nah liegende, nicht verbundene Flächen bewegen sich gemeinsam
CONTACT_RIGID = 10.0
SMOOTH = 1.0  # Glätte der Verformung (zweite Ableitung über das Netz)
SHRINK = 1e-4  # ohne Messung bleibt Annys Form
SILHOUETTE_WEIGHT = 1.0  # Kopfumriss gegenüber den Gesichtspunkten
OUTLINE_SMOOTH = 30.0  # Änderung durch den Umriss: viel glatter als die Gesichtsanpassung
EDGE_MARGIN = 4  # Pixel: Umrisspunkte am Bildrand zählen nicht
LEAK_LIMIT = 0.002  # Anteil der Maske am Rand des erlaubten Bereichs: Umriss unbrauchbar


@dataclasses.dataclass
class HeadPlacement:
    """Kopf-Koordinaten der Gesichtslösung -> Annys Ruhepose (Meter, nach Maßstab)."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def to_head(self, verts: np.ndarray) -> np.ndarray:
        return (verts - self.translation) @ self.rotation / self.scale

    def to_anny(self, pts: np.ndarray) -> np.ndarray:
        return self.scale * pts @ self.rotation.T + self.translation


def place_head(model: BodyModel, verts: np.ndarray, sol: FaceSolution) -> HeadPlacement:
    """Gemessene Gesichtsform so auf Annys Kopf legen, dass sie am besten passt (Drehung, Lage, Größe)."""
    tri, bary = load_correspondence()
    anny = surface_points(verts, model.faces, tri, bary)
    s, r, t = similarity(sol.points[:FACE_POINTS], anny[:FACE_POINTS])
    return HeadPlacement(s, r, t)


def head_cameras(sol: FaceSolution, placement: HeadPlacement, verts: np.ndarray) -> np.ndarray:
    """(K, V, 3) Körperpunkte in den Kamerakoordinaten der Kopfbilder (nur der Kopf passt)."""
    head = placement.to_head(verts)
    return np.einsum("kij,vj->kvi", sol.rotations, head) + sol.translations[:, None]


def _mesh_laplacian(n: int, faces: np.ndarray):
    import scipy.sparse as sp

    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    e = np.unique(e, axis=0)
    adj = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                        shape=(n, n)).tocsr()
    deg = np.maximum(np.asarray(adj.sum(1)).ravel(), 1)
    return sp.diags(1 / deg) @ (sp.diags(deg) - adj)


def head_masks(video: Path, sol: FaceSolution, placement: HeadPlacement, model: BodyModel,
               verts: np.ndarray, log: Log | None = None) -> dict[int, np.ndarray]:
    """Kopfumriss in jedem Bild der Gesichtslösung (Analyseauflösung).

    GrabCut, angestoßen vom Modell: innen sicher Kopf, weit außen sicher Hintergrund, dazwischen
    entscheiden die Farben. Bilder, in denen die Maske am Schädel bis an den sicheren Hintergrund
    ausläuft (heller Kopf vor heller Wand), werden übergangen; Hals und Kleidung spielen keine Rolle."""
    from .video import read_full_frames

    head = head_vertex_mask(model)
    upper = _upper_vertices(model, verts)
    faces = model.faces[head[model.faces].all(1)]
    top = faces[upper[faces].all(1)]
    rest = faces[~upper[faces].all(1)]
    cams = head_cameras(sol, placement, verts)
    where = {i: k for k, i in enumerate(sol.frames)}
    size = (sol.intr.width, sol.intr.height)
    out = {}
    for index, rgb in read_full_frames(video, set(sol.frames)):
        img = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        cam = cams[where[index]]
        model_mask = (face_buffer(cam, faces, sol.intr) > 0).astype(np.uint8)
        ys, xs = np.nonzero(model_mask)
        if len(ys) < 100:
            continue
        band = max(8, int(0.06 * np.ptp(ys)))
        gc = np.full(model_mask.shape, cv2.GC_BGD, np.uint8)
        wide = cv2.dilate(model_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * band + 1,) * 2))
        gc[wide > 0] = cv2.GC_PR_BGD
        gc[model_mask > 0] = cv2.GC_PR_FGD
        core = cv2.erode(model_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1,) * 2))
        gc[core > 0] = cv2.GC_FGD
        ys, xs = np.nonzero(gc != cv2.GC_BGD)
        y0, y1 = max(ys.min() - 10, 0), ys.max() + 11
        x0, x1 = max(xs.min() - 10, 0), xs.max() + 11
        sub = gc[y0:y1, x0:x1].copy()
        cv2.grabCut(img[y0:y1, x0:x1], sub, None, np.zeros((1, 65)), np.zeros((1, 65)), 5,
                    cv2.GC_INIT_WITH_MASK)
        mask = np.zeros(model_mask.shape, np.uint8)
        mask[y0:y1, x0:x1] = (sub == cv2.GC_FGD) | (sub == cv2.GC_PR_FGD)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            mask = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        # Ausgelaufen? Nur am Schädel prüfen (am Hals läuft die Haut natürlich weiter)
        outer = cv2.erode(wide, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band + 1,) * 2))
        d_top = cv2.distanceTransform((face_buffer(cam, top, sol.intr) == 0).astype(np.uint8), cv2.DIST_L2, 5)
        d_rest = cv2.distanceTransform((face_buffer(cam, rest, sol.intr) == 0).astype(np.uint8), cv2.DIST_L2, 5)
        leak = (mask > 0) & (outer == 0) & (d_top < d_rest)
        if leak.sum() > LEAK_LIMIT * mask.sum():
            continue
        out[index] = mask
    if log:
        log(f"Kopfumriss in {len(out)} von {len(sol.frames)} Bildern sicher bestimmt.")
    return out


def _upper_vertices(model: BodyModel, verts: np.ndarray) -> np.ndarray:
    """Schädel und Ohren: Kopfpunkte oberhalb der Nasenspitze (darunter gibt der Umriss Hals und
    Kragen wieder, nicht den Kopf; Kinn und Kiefer kommen aus den Gesichtspunkten)."""
    tri, bary = load_correspondence()
    nose_z = float(surface_points(verts, model.faces, tri[[NOSE_TIP]], bary[[NOSE_TIP]])[0, 2])
    return head_vertex_mask(model) & (verts[:, 2] > nose_z)


def fit_head_shape(model: BodyModel, verts: np.ndarray, sol: FaceSolution, placement: HeadPlacement,
                   masks: dict[int, np.ndarray] | None = None, rounds: int = 3, log: Log | None = None
                   ) -> np.ndarray:
    """Verschiebung (V, 3) der Kopfpunkte (Meter, Annys Ruhepose nach Maßstab).

    Gesichtspunkte: Annys Stellen sollen auf der gemessenen Gesichtsform liegen. Kopfumriss (falls
    masks): Schädel und Ohren oberhalb der Nase sollen in jedem Bild den Umriss ausfüllen und nicht
    überragen. Die Verformung bleibt glatt und läuft zum Hals hin aus; die Augäpfel folgen ihren
    Augenhöhlen."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    faces = model.faces
    n = len(verts)
    tri, bary = load_correspondence()
    use = np.setdiff1d(np.arange(FACE_POINTS), INNER_LIPS)
    tri, bary = tri[use], bary[use]
    target = placement.to_anny(sol.points[use])
    resid = target - surface_points(verts, faces, tri, bary)
    head = head_vertex_mask(model)
    eye_masks = list(eye_vertex_masks(model).values())
    eyes = np.logical_or.reduce(eye_masks)
    free = head & ~eyes
    nf = int(free.sum())
    col = -np.ones(n, np.int64)
    col[free] = np.arange(nf)
    corners = faces[tri]
    keep = free[corners]
    rows = np.repeat(np.arange(len(use)), 3).reshape(-1, 3)
    b_mat = sp.csr_matrix((bary[keep], (rows[keep], col[corners[keep]])), shape=(len(use), nf))
    lap = (_mesh_laplacian(n, faces) @ sp.csr_matrix((np.ones(nf), (np.flatnonzero(free), np.arange(nf))),
                                                      shape=(n, nf))).tocsr()
    # Augenhöhlen bewegen sich fast als Ganzes (sonst schieben sich die Lider in die Augäpfel)
    sockets = [col[_socket(verts, em, free)] for em in eye_masks]
    rigid = sp.csr_matrix((nf, nf))
    for idx in sockets:
        sel = sp.csr_matrix((np.ones(len(idx)), (np.arange(len(idx)), idx)), shape=(len(idx), nf))
        center = sp.eye(len(idx)) - sp.csr_matrix(np.full((len(idx), len(idx)), 1.0 / len(idx)))
        rigid = rigid + sel.T @ center @ sel
    # Flächen, die sich berühren, ohne verbunden zu sein (Ober- und Unterlippe, Lider, Ohrfalten),
    # bewegen sich gleich, damit sie sich nicht durchdringen
    a, b = _contact_pairs(verts, faces, free)
    pairs = sp.csr_matrix((np.r_[np.ones(len(a)), -np.ones(len(a))],
                           (np.r_[np.arange(len(a)), np.arange(len(a))], np.r_[col[a], col[b]])),
                          shape=(len(a), nf))
    rigid = rigid + CONTACT_RIGID * (pairs.T @ pairs)
    system = (b_mat.T @ b_mat + SMOOTH * (lap.T @ lap) + SOCKET_RIGID * rigid + SHRINK * sp.eye(nf)).tocsc()
    solve = spla.factorized(system)
    disp = np.stack([solve(b_mat.T @ resid[:, c]) for c in range(3)], -1)
    if log:
        before = np.linalg.norm(resid, axis=1)
        after = np.linalg.norm(b_mat @ disp - resid, axis=1)
        log(f"Gesicht: Abstand zu Annys Gesichtspunkten {np.median(before) * 1000:.1f} mm, "
            f"nach Verformung {np.median(after) * 1000:.1f} mm.")
    if masks:
        disp = _fit_outline(model, verts, sol, placement, masks, free, b_mat, lap, resid, disp, rounds,
                            log, rigid)
    out = np.zeros_like(verts)
    out[free] = disp
    for em in eye_masks:
        out[em] = out[_socket(verts, em, free)].mean(0)  # Augapfel wandert mit seiner Augenhöhle
    return out


def _contact_pairs(verts: np.ndarray, faces: np.ndarray, free: np.ndarray, radius: float = CONTACT_RADIUS,
                   hops: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Punktpaare, die nah beieinander liegen, aber auf dem Netz weit auseinander (mehr als hops Kanten)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    idx = np.flatnonzero(free)
    close = cKDTree(verts[idx]).query_pairs(radius, output_type="ndarray")
    if len(close) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    a, b = idx[close[:, 0]], idx[close[:, 1]]
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    adj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(verts),) * 2).tocsr()
    src = np.unique(a)
    dist = dijkstra(adj, directed=False, unweighted=True, indices=src, limit=hops)
    row = np.searchsorted(src, a)
    far = ~np.isfinite(dist[row, b])
    return a[far], b[far]


def _socket(verts: np.ndarray, eye: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Lider und Augenhöhle rund um einen Augapfel."""
    c = verts[eye].mean(0)
    radius = np.linalg.norm(verts[eye] - c, axis=1).max()
    return free & (np.linalg.norm(verts - c, axis=1) < 1.5 * radius)


def _fit_outline(model, verts, sol, placement, masks, free, b_mat, lap, resid, disp, rounds, log, rigid):
    """Feinanpassung von Schädel und Ohren an den Kopfumriss in allen Bildern.

    Je Bild liegen die Randpunkte des Modells (Punkte auf seinem Umriss, Normale quer zum Blick) auf
    dem gemessenen Umriss: vorzeichenbehafteter Abstand zur Maske, in Metern. Die Änderung gegenüber
    der reinen Gesichtsanpassung bleibt sehr glatt, damit keine Beulen entstehen."""
    faces = model.faces
    upper = _upper_vertices(model, verts)
    head = head_vertex_mask(model)
    head_faces = faces[head[faces].all(1)]
    frames = [k for k, i in enumerate(sol.frames) if i in masks]
    intr = sol.intr
    sdfs = []
    for k in frames:
        m = masks[sol.frames[k]].astype(np.uint8)
        sdfs.append(cv2.distanceTransform(1 - m, cv2.DIST_L2, 5) - cv2.distanceTransform(m, cv2.DIST_L2, 5))
    sdf = torch.tensor(np.stack(sdfs), dtype=torch.float64)[:, None]
    rot = torch.tensor(sol.rotations[frames], dtype=torch.float64)
    tr = torch.tensor(sol.translations[frames], dtype=torch.float64)
    s_inv = 1.0 / placement.scale
    r_place = torch.tensor(placement.rotation, dtype=torch.float64)
    t_place = torch.tensor(placement.translation, dtype=torch.float64)
    base = torch.tensor(verts, dtype=torch.float64)
    free_idx = torch.tensor(np.flatnonzero(free))
    b_t, lap_t, rigid_t = _sparse(b_mat), _sparse(lap), _sparse(rigid)
    resid_t = torch.tensor(resid, dtype=torch.float64)
    d0 = torch.tensor(disp, dtype=torch.float64)
    center = torch.tensor([intr.cx, intr.cy], dtype=torch.float64)

    def to_cam(points):  # (P, 3) Anny -> (F, P, 3)
        h = (points - t_place) @ r_place * s_inv
        return torch.einsum("fij,pj->fpi", rot, h) + tr[:, None]

    def rim(d):
        """Je Bild die Modellpunkte auf dem Umriss (nur Schädel und Ohren, nicht am Bildrand)."""
        v = verts.copy()
        v[free] += d
        cams = to_cam(torch.tensor(v)).numpy()
        out = []
        for f in range(len(frames)):
            cam = cams[f]
            ids = face_buffer(cam, head_faces, intr)
            inside = (ids > 0).astype(np.uint8)
            edge = cv2.distanceTransform(inside, cv2.DIST_L2, 3)  # Abstand zum Rand von innen
            uv = intr.focal * cam[:, :2] / cam[:, 2:] + [intr.cx, intr.cy]
            x, y = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
            ok = (upper & (x >= EDGE_MARGIN) & (x < intr.width - EDGE_MARGIN)
                  & (y >= EDGE_MARGIN) & (y < intr.height - EDGE_MARGIN))
            sel = np.flatnonzero(ok)
            sel = sel[edge[y[sel], x[sel]] <= 1.5]
            out.append(torch.tensor(sel))
        return out

    d = torch.nn.Parameter(d0.clone())
    n_lm = b_mat.shape[0]

    def energy(sel):
        v = base.index_add(0, free_idx, d)
        e_lm = (torch.sparse.mm(b_t, d) - resid_t).pow(2).sum()
        e_smooth = SMOOTH * torch.sparse.mm(lap_t, d).pow(2).sum() + SHRINK * d.pow(2).sum()
        e_smooth = e_smooth + OUTLINE_SMOOTH * torch.sparse.mm(lap_t, d - d0).pow(2).sum()
        e_smooth = e_smooth + SOCKET_RIGID * (d * torch.sparse.mm(rigid_t, d)).sum()
        cams = to_cam(v)
        terms = []
        for f, idx in enumerate(sel):
            if len(idx) == 0:
                continue
            c = cams[f, idx]
            uv = intr.focal * c[:, :2] / c[:, 2:] + center
            grid = torch.stack([uv[:, 0] / (intr.width - 1) * 2 - 1, uv[:, 1] / (intr.height - 1) * 2 - 1], -1)
            dist = torch.nn.functional.grid_sample(sdf[f:f + 1], grid[None, None], align_corners=True,
                                                   padding_mode="border")[0, 0, 0]
            terms.append(((dist * c[:, 2] / intr.focal) ** 2).mean())
        e_sil = SILHOUETTE_WEIGHT * n_lm * torch.stack(terms).mean()
        return e_lm + e_smooth + e_sil, e_sil

    for r in range(rounds):
        sel = rim(d.detach().numpy())
        opt = torch.optim.LBFGS([d], lr=1.0, max_iter=50, history_size=20, line_search_fn="strong_wolfe")

        def closure(opt=opt, sel=sel):
            opt.zero_grad()
            e, _ = energy(sel)
            e.backward()
            return e

        opt.step(closure)
        if log:
            with torch.no_grad():
                _, e_sil = energy(sel)
            log(f"Kopfumriss: Runde {r + 1}/{rounds}, mittlere Abweichung "
                f"{math.sqrt(float(e_sil) / n_lm / SILHOUETTE_WEIGHT) * 1000:.1f} mm")
    return d.detach().numpy()


def _sparse(m) -> torch.Tensor:
    m = m.tocoo()
    return torch.sparse_coo_tensor(np.vstack([m.row, m.col]), m.data, m.shape, dtype=torch.float64,
                                   check_invariants=False).coalesce()


# ------------------------------------------------------------------------- Hautbild des Kopfes

BLINK_LIMIT = 0.3  # Lider (halb) zu
HEAD_FEATHER = 8.0  # Texel: weicher Übergang zwischen Kopf- und Körperhautbild


def head_region(model: BodyModel) -> np.ndarray:
    """(V,) Anteil des Kopfes je Punkt (Skinning-Gewicht von Kopf und Augen): 1 am Kopf, 0 am Hals."""
    sel = np.array([n == "head" or n.startswith("eye") for n in model.bone_labels])
    w = np.where(sel[model.vertex_bone_indices], model.vertex_bone_weights, 0.0).sum(1)
    return np.clip((np.asarray(w, np.float64) - 0.3) / 0.6, 0.0, 1.0)


def eyes_open(shapes: dict[str, float] | None) -> bool:
    """Lider offen (MediaPipe-Mimikwerte eyeBlink…); ohne Werte: ja."""
    if not shapes:
        return True
    return max(shapes.get("eyeBlinkLeft", 0.0), shapes.get("eyeBlinkRight", 0.0)) <= BLINK_LIMIT


@dataclasses.dataclass
class HeadSkin:
    skin: np.ndarray  # (size, size, 3) uint8 RGB, Hautbild aus der Kopfaufnahme
    alpha: np.ndarray  # (size, size) 0..1: wie sicher jede Stelle gesehen wurde
    white: np.ndarray | None  # Farbe von Weiß in der Kopfaufnahme (linear, Grün = 1)
    iris: np.ndarray | None  # Irisfarbe (linear RGB)
    sclera: np.ndarray | None  # Farbe des Augenweiß (linear RGB)


def bake_head(model: BodyModel, verts: np.ndarray, sol: FaceSolution, placement: HeadPlacement, video: Path,
              masks: dict[int, np.ndarray] | None = None, tracks: list[FaceTrack] | None = None,
              size: int | None = None, fine_align: bool = True, log: Log | None = None) -> HeadSkin:
    """Hautbild des Kopfes aus der Kopfaufnahme (volle Auflösung). Wie beim Körper: erst grob, dann
    per optischem Fluss nachgeschoben. Die Augäpfel werden nicht projiziert (sie drehen sich und
    stecken größtenteils hinter den Lidern), sondern mit Iris- und Augenweißfarbe aus den Bildern
    neu gezeichnet."""
    from .texture import (
        SEEN_FULL,
        TEXTURE_SIZE,
        Shot,
        TextureBaker,
        _pad,
        alignment_flow,
        render_textured,
        to_linear,
        white_point,
        white_samples,
    )
    from .video import read_full_frames

    size = size or TEXTURE_SIZE
    head = placement.to_head(verts)
    region = head_region(model)
    eyes = np.logical_or.reduce(list(eye_vertex_masks(model).values()))
    region[eyes] = 0.0
    by_frame = {t.index: t for t in tracks or []}
    where = {i: k for k, i in enumerate(sol.frames)}
    intr = sol.intr
    faces = model.faces
    used = region[faces].max(1) > 0
    part = faces[used]
    uv = model.anny.texture_coordinates.detach().cpu().numpy()
    face_uv = model.anny.face_texture_coordinate_indices.detach().cpu().numpy()[used]
    forward = sol.rotations @ (placement.rotation.T @ np.array([0.0, -1.0, 0.0]))  # Gesicht in Kamera

    def shots():
        for index, rgb in read_full_frames(video, set(sol.frames)):
            k = where[index]
            f = rgb.shape[1] / intr.width
            mask = masks.get(index) if masks else None
            yield k, Shot(rgb=rgb, mask=np.ones((intr.height, intr.width), np.float32) if mask is None else mask,
                          intr=intr, to_image=np.diag([f, f, 1.0]), vertices=verts, yaw=0.0, trans=np.zeros(3),
                          camera=head @ sol.rotations[k].T + sol.translations[k], region=region)

    baker = TextureBaker(model, size)
    whites, iris, sclera = [], [], []
    for n, (k, shot) in enumerate(shots()):
        baker.add(shot)
        person = None if shot.mask is None else (shot.mask > 0.5)
        if person is not None and person.shape != shot.rgb.shape[:2]:
            person = cv2.resize(person.astype(np.uint8), shot.rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        whites.append(white_samples(shot.rgb, person))
        track = by_frame.get(sol.frames[k])
        if track is not None and -forward[k, 2] > EYE_FRONTAL and eyes_open(track.shapes):
            i_px, s_px = _eye_samples(shot.rgb, track.points * shot.to_image[0, 0])
            iris.append(to_linear(i_px))
            sclera.append(to_linear(s_px))
        if log and n % 16 == 0:
            log(f"Kopf-Hautbild: Bild {n + 1}/{len(sol.frames)} übertragen")
    skin = baker.finish()
    if fine_align:
        k_c = 1920 / max(intr.width, intr.height)
        cintr = intr.scaled(k_c)
        refined = TextureBaker(model, size, reference=skin)
        shifts = []
        for n, (_, shot) in enumerate(shots()):
            to_cmp = np.diag([k_c, k_c, 1.0]) @ np.linalg.inv(shot.to_image)
            photo = cv2.warpPerspective(shot.rgb, to_cmp, (cintr.width, cintr.height), flags=cv2.INTER_AREA)
            render, body = render_textured(verts, part, uv, face_uv, skin, 0.0, np.zeros(3), cintr,
                                           background=photo, with_mask=True, camera=shot.camera)
            flow = alignment_flow(render, photo, body)
            shifts.append(float(np.median(np.linalg.norm(flow[body], axis=-1))) / k_c if body.any() else 0.0)
            shot.flow, shot.flow_scale = flow, k_c
            refined.add(shot)
            if log and n % 16 == 0:
                log(f"Kopf-Hautbild fein ausgerichtet: Bild {n + 1}/{len(sol.frames)}")
        if log:
            log(f"Kopf: Bilder im Mittel um {np.median(shifts):.1f} Pixel nachgeschoben.")
        baker, skin = refined, refined.finish()
    m = baker.map
    alpha = np.zeros(size * size, np.float32)
    # Der ganze Kopf kommt aus der Kopfaufnahme: was sie nicht zeigt (Scheitel, Hinterkopf), wird aus
    # der gesehenen Kopfhaut ergänzt; die Körperaufnahme zeigt den Kopf nur klein und von unten
    on_head = (head_region(model)[faces[m.face]] * m.bary).sum(1)
    alpha[m.texel] = np.maximum(np.clip(baker.base_w / SEEN_FULL, 0, 1), on_head)
    filled = np.zeros(size * size, bool)
    filled[m.texel] = True
    alpha = _pad(np.repeat(alpha.reshape(size, size, 1), 3, -1), filled.reshape(size, size), 12)[..., 0]
    iris_c = np.median(np.concatenate(iris), 0) if sum(map(len, iris)) > 50 else None
    sclera_c = np.median(np.concatenate(sclera), 0) if sum(map(len, sclera)) > 50 else None
    if log:
        log(f"Augen aus {len(iris)} Bildern mit offenen Lidern: Irisfarbe "
            + ("unbekannt" if iris_c is None else "RGB " + "/".join(str(int(v)) for v in _srgb_u8(iris_c))) + ".")
    skin, alpha = paint_eyes(model, verts, skin, alpha, iris_c, sclera_c)
    return HeadSkin(skin, alpha, white_point(whites), iris_c, sclera_c)


def _srgb_u8(lin: np.ndarray) -> np.ndarray:
    from .texture import to_srgb

    return to_srgb(np.asarray(lin, np.float32)[None])[0]


# Gesichtspunkte von MediaPipe: Irismitte, Irisrand, äußerer und innerer Augenwinkel
_EYES = ((468, (469, 470, 471, 472), 33, 133), (473, (474, 475, 476, 477), 263, 362))


def _eye_samples(rgb: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel der Iris (ohne Pupille und ohne den vom Oberlid beschatteten Teil) und des Augenweiß
    (zwischen Iris und Augenwinkeln) eines Bildes; pts in Pixeln dieses Bildes."""
    iris, sclera = [], []
    h, w = rgb.shape[:2]
    for center, ring, outer, inner in _EYES:
        c = pts[center]
        r = float(np.mean(np.linalg.norm(pts[list(ring)] - c, axis=1)))
        if r < 4:
            continue
        x0, x1 = int(max(c[0] - r, 0)), int(min(c[0] + r + 1, w))
        y0, y1 = int(max(c[1] - r, 0)), int(min(c[1] + r + 1, h))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - c[0], yy - c[1]) / r
        keep = (d > 0.45) & (d < 0.85) & (yy > c[1] - 0.3 * r)
        iris.append(rgb[yy[keep], xx[keep]])
        for corner in (outer, inner):
            v = pts[corner] - c
            length = float(np.linalg.norm(v))
            for t in np.linspace(min((r + 2) / max(length, 1), 0.9), 0.7, 5):
                x, y = (c + t * v).astype(int)
                if 1 <= x < w - 1 and 1 <= y < h - 1:
                    sclera.append(rgb[y - 1:y + 2, x - 1:x + 2].reshape(-1, 3))
    empty = np.zeros((0, 3), np.uint8)
    return (np.concatenate(iris) if iris else empty), (np.concatenate(sclera) if sclera else empty)


EYE_FRONTAL = 0.8  # cos des Winkels zwischen Gesicht und Kamera für die Augenfarben (etwa 37°)
IRIS_RADIUS = 0.0059  # Meter (Iris etwa 11,8 mm)
PUPIL_RADIUS = 0.0019
DEFAULT_IRIS = np.array([0.12, 0.16, 0.22], np.float32)  # linear, blaugrau
DEFAULT_SCLERA = np.array([0.62, 0.58, 0.55], np.float32)


def paint_eyes(model: BodyModel, verts: np.ndarray, skin: np.ndarray, alpha: np.ndarray,
               iris: np.ndarray | None, sclera: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Augäpfel im Hautbild zeichnen: Augenweiß, Iris mit Speichen und dunklem Rand, Pupille."""
    from .texture import to_srgb, uv_map

    size = skin.shape[0]
    m = uv_map(model, size)
    faces = model.faces
    iris = DEFAULT_IRIS if iris is None else np.asarray(iris, np.float32)
    sclera = DEFAULT_SCLERA if sclera is None else np.asarray(sclera, np.float32)
    # Augenweiß wirkt im Video durch Schatten oft grau; etwas aufhellen, Farbton behalten
    sclera = sclera * min(1.6, 0.6 / max(float(sclera.mean()), 1e-3))
    out = skin.reshape(-1, 3).copy()
    a = alpha.reshape(-1).copy()
    for em in eye_vertex_masks(model).values():
        on = em[faces[m.face]].all(1)
        if not on.any():
            continue
        ball = verts[em]
        c = ball.mean(0)
        front = ball[np.argmin(ball[:, 1])] - c  # Anny blickt nach -Y
        radius = float(np.linalg.norm(front))
        front /= radius
        p = (verts[faces[m.face[on]]] * m.bary[on][..., None]).sum(1) - c
        cos = (p @ front) / np.maximum(np.linalg.norm(p, axis=1), 1e-9)
        theta = np.arccos(np.clip(cos, -1, 1))
        side = np.cross(front, [0.0, 0.0, 1.0])
        phi = np.arctan2(p @ np.cross(side, front), p @ side)
        t_iris = math.asin(min(IRIS_RADIUS / radius, 0.95))
        t_pupil = math.asin(min(PUPIL_RADIUS / radius, 0.5))
        rel = theta / t_iris
        streak = 1 + 0.18 * np.sin(37 * phi + 3 * np.sin(11 * phi)) * np.clip(rel * 1.5, 0, 1)
        col = iris[None] * streak[:, None]
        col = col * (1 - 0.55 * _smooth(0.78, 1.0, rel))[:, None]  # dunkler Rand der Iris
        col = col * (1 - 0.25 * (1 - _smooth(t_pupil, 1.8 * t_pupil, theta)))[:, None]
        col = np.where((theta < t_pupil)[:, None], np.array([0.008, 0.008, 0.01]), col)
        white = sclera[None] * (1 - 0.25 * _smooth(math.radians(55), math.radians(95), theta))[:, None]
        mix = _smooth(t_iris * 0.98, t_iris * 1.04, theta)[:, None]
        lin = (1 - mix) * col + mix * white
        sel = m.texel[on]
        out[sel] = to_srgb(lin.astype(np.float32))
        a[sel] = 1.0
    # Ränder der Augen-Inseln auffüllen, sonst schimmert beim Verkleinern Haut durch
    img = out.reshape(size, size, 3)
    island = np.zeros(size * size, np.float32)
    for em in eye_vertex_masks(model).values():
        island[m.texel[em[faces[m.face]].all(1)]] = 1
    island = island.reshape(size, size)
    grow = (cv2.dilate(island, np.ones((5, 5), np.uint8)) > 0) & (island == 0)
    near = cv2.dilate(img * island[..., None].astype(np.uint8), np.ones((5, 5), np.uint8))
    img[grow] = near[grow]
    return img, a.reshape(size, size)


def _smooth(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def merge_head_skin(body: np.ndarray, head: np.ndarray, alpha: np.ndarray, body_white: np.ndarray | None = None,
                    head_white: np.ndarray | None = None, log: Log | None = None) -> np.ndarray:
    """Kopf-Hautbild in das Körper-Hautbild einsetzen. Erst nimmt der Weißabgleich beider Aufnahmen
    den Farbstich des Lichts heraus (falls bekannt), dann wird die Helligkeit des Kopfes an den
    Körper angeglichen, damit am Hals keine Kante entsteht."""
    from .texture import to_linear, to_srgb

    if head.shape != body.shape:
        head = cv2.resize(head, body.shape[1::-1], interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, body.shape[1::-1], interpolation=cv2.INTER_AREA)
    lb, lh = to_linear(body), to_linear(head)
    if body_white is not None:
        lb = lb / np.asarray(body_white, np.float32)
    if head_white is not None:
        lh = lh / np.asarray(head_white, np.float32)
    sigma = body.shape[0] / 256
    bb, hb = cv2.GaussianBlur(lb, (0, 0), sigma), cv2.GaussianBlur(lh, (0, 0), sigma)
    both = (alpha > 0.8) & (hb.min(-1) > 0.02)
    gain = np.ones(3, np.float32)
    if both.sum() > 1000:
        gain = np.median(bb[both] / hb[both], axis=0).astype(np.float32)
    if log:
        log("Kopf an den Körper angeglichen (Rot/Grün/Blau " + "/".join(f"{g:.2f}" for g in gain) + ").")
    a = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), HEAD_FEATHER)[..., None]
    return to_srgb(a * lh * gain + (1 - a) * lb)


def compare_head(model: BodyModel, verts: np.ndarray, sol: FaceSolution, placement: HeadPlacement, video: Path,
                 skin: np.ndarray, height: int = 640) -> np.ndarray | None:
    """Vergleichsbild: Kopfvideo und Modell mit Hautbild aus derselben Richtung (vorn, beide Seiten,
    von unten), je Zeile Foto links, Modell rechts."""
    from .texture import render_textured
    from .video import read_full_frames

    region = head_region(model)
    faces = model.faces
    used = region[faces].max(1) > 0
    uv = model.anny.texture_coordinates.detach().cpu().numpy()
    face_uv = model.anny.face_texture_coordinate_indices.detach().cpu().numpy()[used]
    head_faces = faces[head_vertex_mask(model)[faces].all(1)]
    head = placement.to_head(verts)
    fwd = sol.rotations @ (placement.rotation.T @ np.array([0.0, -1.0, 0.0]))
    yaw = np.degrees(np.arctan2(fwd[:, 0], -fwd[:, 2]))
    pitch = np.degrees(np.arcsin(np.clip(-fwd[:, 1], -1, 1)))
    pick: dict[int, int] = {}
    for y, p in ((0, 10), (-40, 8), (40, 8), (10, 45)):
        k = int(np.argmin((yaw - y) ** 2 + (pitch - p) ** 2))
        pick.setdefault(sol.frames[k], k)
    scale = 1920 / max(sol.intr.width, sol.intr.height)
    cintr = sol.intr.scaled(scale)
    rows = {}
    for index, rgb in read_full_frames(video, set(pick)):
        cam = head @ sol.rotations[pick[index]].T + sol.translations[pick[index]]
        photo = cv2.resize(rgb, (cintr.width, cintr.height), interpolation=cv2.INTER_AREA)
        render = render_textured(verts, faces[used], uv, face_uv, skin, 0.0, np.zeros(3), cintr,
                                 background=np.full_like(photo, 235), camera=cam)
        ys, xs = np.nonzero(face_buffer(cam, head_faces, cintr))
        if len(ys) == 0:
            continue
        y0, y1 = max(ys.min() - 60, 0), min(ys.max() + 40, cintr.height)
        half = int((y1 - y0) * 0.42)
        x0, x1 = max(int(xs.mean()) - half, 0), min(int(xs.mean()) + half, cintr.width)
        tile = np.hstack([photo[y0:y1, x0:x1], np.full((y1 - y0, 8, 3), 255, np.uint8), render[y0:y1, x0:x1]])
        rows[index] = cv2.resize(tile, (int(tile.shape[1] * height / tile.shape[0]), height))
    if not rows:
        return None
    width = max(r.shape[1] for r in rows.values())
    return np.vstack([np.hstack([r, np.full((height, width - r.shape[1], 3), 255, np.uint8)])
                      for i in pick if (r := rows.get(i)) is not None])
