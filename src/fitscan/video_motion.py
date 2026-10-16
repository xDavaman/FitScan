"""Bewegung aus einem fremden Video auf das eigene Modell übertragen.

Eingabe ist ein beliebiges Video (Handy, Trainingsvideo, gerenderte Animation), auch mit Schnitten
und fahrender Kamera. Ablauf:

1. Je Bild Körperpunkte mit MediaPipe Pose (Apache 2.0, offline) und ein Maß für Bildwechsel.
2. Schnitte erkennen und das Video in Szenen teilen. Szenen, in denen die Hauptperson lange genug
   ganz zu sehen ist, werden ausgelesen (`mocap.fit_motion`: Annys Gelenke auf die Punkte, Knie und
   Ellbogen als Scharniere, glatter Verlauf).
3. Logikprüfung, damit die Bewegung menschlich bleibt:
   - Ausreißer (einzelne Bilder, in denen MediaPipe ein Glied verwechselt) werden ersetzt,
   - Gelenkgrenzen: Knie und Ellbogen beugen nur in eine Richtung und nicht weiter als ein Mensch,
     Wirbelsäule, Hals, Hand- und Fußgelenke bleiben in ihrem Bewegungsumfang,
   - Blickrichtung: eine Kamerafahrt um die Person dreht sie nicht mit,
   - Füße: Bodenkontakt erkennen, stehende Füße rutschen nicht, nichts sinkt in den Boden,
   - Sitzen/Liegen: sitzt die Person (Bank), kommt eine Bank unter Gesäß und Rücken,
   - Durchdringung: Körper mit Boden, Bank und sich selbst wird geprüft und berichtet.
4. Übertragung auf ein fertiges FitScan-Modell (koerper.glb) als glTF-Animationen. Haut, Augen und
   Formregler des Modells bleiben, das Format ist das der Bewegungsbibliothek (`motion.Motion`).

Ehrlich: Aus einem einzelnen Video ist die Tiefe (wie weit vor oder hinter dem Körper eine Hand ist)
nur geschätzt; Gegenstände (Hanteln, Bank) erkennt das Werkzeug nicht, die Bank wird aus der Haltung
erschlossen.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import roma
import torch

from . import contact, mocap
from . import motion as M
from .body import BodyModel, BodyParams
from .camera import Intrinsics

Log = Callable[[str], None]

LEGS = (23, 24, 25, 26, 27, 28)  # Hüften, Knie, Knöchel
CUT_THRESHOLD = 0.35  # Anteil der Farbverteilung, der sich bei einem Schnitt ändert
FOCAL_FACTOR = 1.15  # Brennweite in Bildpunkten / lange Bildseite (etwa 50 mm Kleinbild)

# Fremde Videos: Gesichter fehlen oft (Puppen, Masken, von hinten), MediaPipe rät dann die
# Gesichtspunkte und kippt den Kopf. Gesicht deshalb kaum gewichten, Hals und Kopf ruhiger halten.
CAPTURE_WEIGHTS = {"points": {0: 0.0, 9: 0.0, 10: 0.0}, "quiet": {"neck01": 3.0, "head": 3.0}}

# Bewegungsumfang je Gelenk (Grad, größte Drehung gegenüber Annys Bezugshaltung)
LIMITS = {"spine04": 40.0, "spine02": 40.0, "neck01": 35.0, "head": 40.0, "clavicle.L": 30.0,
          "clavicle.R": 30.0, "wrist.L": 80.0, "wrist.R": 80.0, "foot.L": 55.0, "foot.R": 55.0,
          "upperleg01.L": 135.0, "upperleg01.R": 135.0}
HINGE_MAX = {"lowerleg01": 150.0, "lowerarm01": 150.0}  # Knie und Ellbogen
HINGE_SLACK = 5.0  # so weit darf ein Scharnier überstrecken bzw. seitlich nachgeben (Grad)


# ------------------------------------------------------------------------ Video lesen


@dataclasses.dataclass
class Track:
    """Was MediaPipe im ganzen Video sieht."""

    fps: float
    width: int
    height: int
    cut: np.ndarray  # (F,) Änderung der Farbverteilung zum Bild davor, 0..1
    persons: np.ndarray  # (F,) erkannte Personen
    keypoints: list  # je Bild (33, 3) oder None
    world: list  # je Bild (33, 3) oder None

    @property
    def frames(self) -> int:
        return len(self.cut)

    def save(self, path: Path) -> Path:
        def stack(items):
            return np.stack([np.full((33, 3), np.nan, np.float32) if k is None else k for k in items])

        np.savez_compressed(path, fps=self.fps, size=[self.width, self.height], cut=self.cut,
                            persons=self.persons, keypoints=stack(self.keypoints), world=stack(self.world))
        return path

    @classmethod
    def load(cls, path: Path) -> Track:
        d = np.load(path)

        def unstack(a):
            return [None if np.isnan(k).any() else k for k in a]

        return cls(float(d["fps"]), int(d["size"][0]), int(d["size"][1]), d["cut"], d["persons"],
                   unstack(d["keypoints"]), unstack(d["world"]))


def _histogram(rgb: np.ndarray) -> np.ndarray:
    import cv2

    small = cv2.resize(rgb, (90, 160), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return h / max(h.sum(), 1.0)


def analyze(video: Path, fps: float = 30.0, log: Log | None = None) -> Track:
    from .video import read_frames
    from .vision import PoseAnalyzer

    analyzer = PoseAnalyzer(video_mode=False)  # je Bild für sich: nach einem Schnitt nichts mitschleppen
    cut, persons, kps, world = [], [], [], []
    prev = None
    size = (0, 0)
    try:
        for idx, ts, rgb in read_frames(Path(video), fps=fps):
            h = _histogram(rgb)
            cut.append(0.0 if prev is None else 0.5 * float(np.abs(h - prev).sum()))
            prev = h
            o = analyzer.analyze(rgb, idx, ts)
            persons.append(o.persons)
            kps.append(o.keypoints)
            world.append(o.world)
            size = (rgb.shape[1], rgb.shape[0])
            if log and idx % 150 == 0:
                log(f"Körperpunkte: {ts:.0f} s")
    finally:
        analyzer.close()
    return Track(fps, size[0], size[1], np.array(cut, np.float32), np.array(persons), kps, world)


# ------------------------------------------------------------------------ Szenen


@dataclasses.dataclass
class Shot:
    start: int  # erstes Bild
    end: int  # nach dem letzten Bild
    full: float  # Anteil der Bilder, in denen Beine und Hüfte sicher erkannt sind
    persons: int
    body_px: float  # mittlere Körpergröße im Bild

    @property
    def length(self) -> int:
        return self.end - self.start

    def seconds(self, fps: float) -> tuple[float, float]:
        return self.start / fps, self.end / fps


def _full_body(kp: np.ndarray | None, width: int, height: int) -> bool:
    if kp is None:
        return False
    pts = kp[list(LEGS)]
    inside = (pts[:, 0] > 0) & (pts[:, 0] < width) & (pts[:, 1] > 0) & (pts[:, 1] < height)
    return bool((pts[:, 2] > 0.5).all() and inside.all())


def find_shots(track: Track, min_seconds: float = 1.2, min_full: float = 0.6,
               threshold: float = CUT_THRESHOLD) -> list[Shot]:
    """Szenen zwischen Schnitten, in denen die Hauptperson die meiste Zeit ganz zu sehen ist.

    Als Schnitt zählt auch ein Sprung der Hüfte um mehr als ein Drittel der Körpergröße von einem
    Bild zum nächsten (weiche Überblendungen ändern die Farben kaum)."""
    n = track.frames
    full = np.array([_full_body(k, track.width, track.height) for k in track.keypoints])
    cuts = set(np.flatnonzero(track.cut > threshold).tolist())
    for i in range(1, n):
        a, b = track.keypoints[i - 1], track.keypoints[i]
        if a is None or b is None:
            continue
        size = max(np.ptp(a[:, 1]), np.ptp(b[:, 1]), 1.0)
        if np.linalg.norm(a[[23, 24], :2].mean(0) - b[[23, 24], :2].mean(0)) > size / 3:
            cuts.add(i)
    bounds = [0, *sorted(c for c in cuts if 0 < c < n), n]
    shots = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        ok = np.flatnonzero(full[a:b])
        if len(ok) == 0:
            continue
        s, e = a + ok[0], a + ok[-1] + 1  # Anfang und Ende ohne ganze Person abschneiden
        if e - s < min_seconds * track.fps or full[s:e].mean() < min_full:
            continue
        body = [np.ptp(k[:, 1]) for k in track.keypoints[s:e] if k is not None]
        shots.append(Shot(int(s), int(e), float(full[s:e].mean()), int(track.persons[s:e].max()),
                          float(np.median(body))))
    return shots


# MediaPipe-Punkte linke ↔ rechte Seite
_SIDE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
               (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
_SWAP = np.arange(33)
for _a, _b in _SIDE_PAIRS:
    _SWAP[_a], _SWAP[_b] = _b, _a
_CORE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def fix_side_swaps(track: Track, shot: Shot) -> tuple[Track, int]:
    """Links/rechts vertauscht: Bei glatten, gesichtslosen Figuren (und von hinten) springt MediaPipe
    oft zwischen Vorder- und Rückansicht. Jedes Bild wird so beschriftet, dass es am besten zum
    vorigen passt; die Beschriftung, die MediaPipe in den meisten Bildern hatte, gilt.
    Gibt eine Kopie der Spur und die Zahl umbeschrifteter Bilder zurück."""
    kps, world = list(track.keypoints), list(track.world)
    flips, prev = {}, None
    for i in range(shot.start, shot.end):
        kp = kps[i]
        if kp is None:
            continue
        flip = False
        if prev is not None:
            vis = np.minimum(kp[_CORE, 2], prev[_CORE, 2])[:, None]
            keep = float((np.abs(kp[_CORE, :2] - prev[_CORE, :2]) * vis).sum())
            swap = float((np.abs(kp[_SWAP][_CORE, :2] - prev[_CORE, :2]) * vis).sum())
            flip = swap < 0.7 * keep
        flips[i] = flip
        prev = kp[_SWAP] if flip else kp
    # bisher passend zum ersten Bild: so drehen, dass die Mehrheit unverändert bleibt
    state = flips
    if sum(state.values()) > len(state) / 2:
        state = {i: not s for i, s in state.items()}
    for i, s in state.items():
        if s:
            kps[i] = kps[i][_SWAP]
            if world[i] is not None:
                world[i] = world[i][_SWAP]
    return dataclasses.replace(track, keypoints=kps, world=world), int(sum(state.values()))


def capture(model: BodyModel, params: BodyParams, track: Track, shot: Shot, name: str,
            steps: tuple[int, int] = (150, 500), log: Log | None = None) -> mocap.Capture:
    obs = [mocap.Observation(track.keypoints[i], track.world[i]) for i in range(shot.start, shot.end)]
    intr = Intrinsics(track.width, track.height, FOCAL_FACTOR * max(track.width, track.height))
    return mocap.fit_motion(model, params, obs, intr, track.fps, name=name, steps=steps, log=log,
                            weights=CAPTURE_WEIGHTS)


# ------------------------------------------------------------------------ Logikprüfung


@dataclasses.dataclass
class Report:
    """Was die Logikprüfung gefunden und korrigiert hat."""

    name: str
    seconds: tuple[float, float] = (0.0, 0.0)
    image_error_px: float = 0.0
    outliers: int = 0  # ersetzte Einzelbilder (Summe über Gelenke)
    limited: dict[str, int] = dataclasses.field(default_factory=dict)  # Gelenk -> Bilder begrenzt
    turn_removed_deg: float = 0.0  # weggerechnete Drehung durch die Kamerafahrt
    support: str = ""  # "Bank" oder ""
    foot_slip_cm: float = 0.0
    contact_frames: float = 0.0  # Anteil der Bilder mit Bodenkontakt
    arm_depth_before_cm: float = 0.0  # Arm im Rumpf, vor dem Lösen
    arm_depth_after_cm: float = 0.0
    findings: list[str] = dataclasses.field(default_factory=list)
    side_swaps: int = 0  # Bilder mit vertauschter linker/rechter Seite, umbeschriftet

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    def lines(self) -> list[str]:
        a, b = self.seconds
        out = [f"{self.name} (Video {a:.1f} bis {b:.1f} s): mittlerer Bildfehler {self.image_error_px:.1f} px"]
        if self.side_swaps:
            out.append(f"  {self.side_swaps} Bilder mit vertauschter linker/rechter Seite umbeschriftet")
        if self.outliers:
            out.append(f"  {self.outliers} Ausreißer-Bilder (verwechselte Glieder) ersetzt")
        if self.limited:
            parts = ", ".join(f"{_label(k)} {v}×" for k, v in sorted(self.limited.items(), key=lambda kv: -kv[1]))
            out.append(f"  Gelenke auf menschlichen Bewegungsumfang begrenzt: {parts}")
        if self.turn_removed_deg > 5:
            out.append(f"  Kamerafahrt herausgerechnet (bis {self.turn_removed_deg:.0f}° Drehung)")
        if self.support:
            out.append(f"  Sitzende/liegende Haltung erkannt: {self.support} unter Gesäß und Rücken ergänzt")
        if self.arm_depth_before_cm > 0.5:
            out.append(f"  Arme steckten bis {self.arm_depth_before_cm:.1f} cm im Rumpf (anderer Körperbau als im "
                       f"Video), nach außen gedreht: jetzt {self.arm_depth_after_cm:.1f} cm")
        out.append(f"  Füße: {100 * self.contact_frames:.0f} % der Bilder mit Bodenkontakt, Rutschen höchstens "
                   f"{self.foot_slip_cm:.1f} cm")
        out += [f"  {f}" for f in self.findings] or ["  Keine Durchdringung von Boden, Bank oder Körper gefunden"]
        return out


def _label(bone: str) -> str:
    return contact.body_part(bone).replace("Unterschenkel", "Knie").replace("Unterarm", "Ellbogen")


def _rotvecs(motion: M.Motion) -> np.ndarray:
    return roma.unitquat_to_rotvec(torch.as_tensor(motion.rotations)).numpy()


def _with_rotvecs(motion: M.Motion, rv: np.ndarray) -> M.Motion:
    q = roma.rotvec_to_unitquat(torch.as_tensor(rv, dtype=torch.float32)).numpy()
    return dataclasses.replace(motion, rotations=M._continuous(q.astype(np.float64)).astype(np.float32))


def remove_outliers(motion: M.Motion, threshold_deg: float = 25.0) -> tuple[M.Motion, int]:
    """Einzelne Bilder, in denen ein Gelenk weit wegspringt und gleich zurückkommt, ersetzen."""
    q = motion.rotations.astype(np.float64).copy()
    count = 0
    for _ in range(2):  # zwei Durchgänge: auch Ausreißer über zwei Bilder
        for b in range(q.shape[1]):
            qb = q[:, b]
            for t in range(1, len(qb) - 1):
                prev, nxt = qb[t - 1], qb[t + 1]
                ang = lambda x, y: 2 * math.degrees(math.acos(min(1.0, abs(float(x @ y)))))  # noqa: E731
                if ang(prev, nxt) < threshold_deg and min(ang(qb[t], prev), ang(qb[t], nxt)) > threshold_deg:
                    mid = prev + (nxt if prev @ nxt >= 0 else -nxt)
                    qb[t] = mid / np.linalg.norm(mid)
                    count += 1
    return dataclasses.replace(motion, rotations=M._continuous(q).astype(np.float32)), count


def limit_joints(model: BodyModel, params: BodyParams, motion: M.Motion) -> tuple[M.Motion, dict[str, int]]:
    """Gelenke in den menschlichen Bewegungsumfang zurückholen."""
    points = mocap._Points(model, params)
    axes, signs = mocap._hinge_axes(model, points)
    hinge = {b: (axes[k].numpy(), float(signs[k])) for k, b in enumerate(mocap.BONES) if signs[k] != 0}
    rv = _rotvecs(motion)
    limited: dict[str, int] = {}
    for k, bone in enumerate(motion.bones):
        r = rv[:, k]
        before = r.copy()
        if bone in hinge:
            axis, sign = hinge[bone]
            along = r @ axis
            off = r - along[:, None] * axis
            slack = math.radians(HINGE_SLACK)
            top = math.radians(HINGE_MAX[bone.split(".")[0]])
            bend = np.clip(along * sign, -slack, top)
            n = np.linalg.norm(off, axis=1, keepdims=True)
            off = off * np.minimum(1.0, slack * 2 / np.maximum(n, 1e-9))  # kaum seitlich beugen
            r = bend[:, None] * sign * axis + off
        elif bone in LIMITS:
            n = np.linalg.norm(r, axis=1, keepdims=True)
            r = r * np.minimum(1.0, math.radians(LIMITS[bone]) / np.maximum(n, 1e-9))
        changed = np.linalg.norm(r - before, axis=1) > math.radians(2.0)
        if changed.any():
            limited[bone] = int(changed.sum())
        rv[:, k] = r
    return _with_rotvecs(motion, rv), limited


def _heading(rot: np.ndarray) -> np.ndarray:
    """Blickrichtung der Hüfte um die Hochachse (Radiant) aus (F, 3, 3)."""
    fwd = rot @ np.array([0.0, -1.0, 0.0])  # Anny blickt nach -Y
    return np.unwrap(np.arctan2(fwd[:, 0], -fwd[:, 1]))


def steady_heading(motion: M.Motion, keep_seconds: float = 1.5) -> tuple[M.Motion, float]:
    """Langsame Drehung der ganzen Person (meist eine Kamerafahrt um sie herum) herausrechnen.

    Schnelle Drehungen (kürzer als keep_seconds) bleiben, die Person blickt danach im Mittel nach -Y."""
    k = motion.bones.index("root")
    rot = roma.unitquat_to_rotmat(torch.as_tensor(motion.rotations[:, k])).double().numpy()
    yaw = _heading(rot)
    slow = M._smooth(yaw[:, None], motion.fps * keep_seconds / 2)[:, 0]
    fix = -slow  # drehen auf Blick -Y
    c, s = np.cos(fix), np.sin(fix)
    rz = np.zeros((len(fix), 3, 3))
    rz[:, 0, 0], rz[:, 0, 1], rz[:, 1, 0], rz[:, 1, 1], rz[:, 2, 2] = c, -s, s, c, 1.0
    rotations = motion.rotations.copy()
    q = roma.rotmat_to_unitquat(torch.as_tensor(rz @ rot)).numpy()
    rotations[:, k] = M._continuous(q[:, None].astype(np.float64))[:, 0]
    # Hüftweg mitdrehen, um die Hüfte des ersten Bilds
    root = motion.root.astype(np.float64)
    centre = root[0, :2].copy()
    xy = root[:, :2] - centre
    root[:, 0] = c * xy[:, 0] - s * xy[:, 1] + centre[0]
    root[:, 1] = s * xy[:, 0] + c * xy[:, 1] + centre[1]
    turned = float(np.degrees(np.ptp(slow)))
    return dataclasses.replace(motion, rotations=rotations.astype(np.float32), root=root.astype(np.float32)), turned


def stabilize_root(motion: M.Motion, in_place: bool = True) -> M.Motion:
    """Hüftweg beruhigen und die Szene um den Ursprung legen.

    Die Entfernung zur Kamera ist aus einem Video nur geschätzt und zittert; einzelne Bilder springen.
    Erst Medianfilter gegen Sprünge, dann waagerecht stark (in_place: Übung auf der Stelle) und
    senkrecht leicht glätten. Danach liegt die Hüfte im Mittel über dem Ursprung."""
    root = motion.root.astype(np.float64)
    r = 4
    padded = np.pad(root, ((r, r), (0, 0)), mode="edge")
    med = np.stack([np.median(padded[t:t + 2 * r + 1], axis=0) for t in range(len(root))])
    xy = M._smooth(med[:, :2], motion.fps * (0.5 if in_place else 0.08))
    z = M._smooth(med[:, 2:], motion.fps * 0.06)
    root = np.concatenate([xy - np.median(xy, axis=0), z], 1)
    return dataclasses.replace(motion, root=root.astype(np.float32))


def smooth(motion: M.Motion, sigma_s: float = 0.04) -> M.Motion:
    """Zittern glätten (Gauß über die Zeit, Drehungen als Quaternionen)."""
    sigma = sigma_s * motion.fps
    f, b = motion.rotations.shape[:2]
    q = M._smooth(motion.rotations.reshape(f, -1).astype(np.float64), sigma).reshape(f, b, 4)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    root = M._smooth(motion.root.astype(np.float64), sigma)
    return dataclasses.replace(motion, rotations=q.astype(np.float32), root=root.astype(np.float32))


def _parts_mask(model: BodyModel, parts: tuple[str, ...]) -> np.ndarray:
    return model.dominant_bone_mask(lambda n: n.split(".")[0] in parts)


def find_support(model: BodyModel, vertices: np.ndarray, bones: np.ndarray) -> list[contact.Box]:
    """Bank aus der Haltung erschließen: Oberschenkel waagerecht oder höher, Gesäß über dem Boden.

    vertices: (F, V, 3) nach dem Bodenausgleich, bones: (F, J, 4, 4). Gibt eine Sitzfläche und,
    wenn der Oberkörper sich zurücklehnt, eine Rückenlehne entlang des Rückens zurück (contact.Box)."""
    pelvis = _parts_mask(model, ("root", "pelvis", "upperleg01", "spine05"))
    back = _parts_mask(model, ("spine03", "spine02", "spine01"))
    v = np.median(vertices, axis=0)  # mittlere Haltung der Szene
    heads = np.median(bones[:, :, :3, 3], axis=0)
    seat_z = float(np.percentile(v[pelvis, 2], 2))
    thigh = []
    for side in ("L", "R"):
        d = heads[model.bone(f"lowerleg01.{side}")] - heads[model.bone(f"upperleg01.{side}")]
        thigh.append(math.degrees(math.acos(np.clip(-d[2] / max(np.linalg.norm(d), 1e-9), -1, 1))))
    if seat_z < 0.15 or min(thigh) < 45:
        return []  # steht, kniet oder liegt am Boden
    seat_pts = v[pelvis][v[pelvis, 2] < seat_z + 0.05]
    centre = seat_pts.mean(0)
    # Sitzfläche so hoch, wie es Gesäß und Oberschenkel in den meisten Bildern zulassen (tiefster Punkt über
    # der Fläche, ganze Szene); Unterschenkel und Füße dürfen davor und darunter stehen
    half = np.array([0.15, 0.2])
    mid = centre[:2] + np.array([0.0, 0.08])
    legs = _parts_mask(model, ("lowerleg01", "lowerleg02", "foot")) | model.dominant_bone_mask(
        lambda n: n.startswith("toe"))
    inside = np.all(np.abs(vertices[..., :2] - mid) < half, axis=-1) & ~legs
    lows = [float(np.percentile(f[m, 2], 1)) for f, m in zip(vertices, inside, strict=True) if m.sum() > 20]
    seat_z = min(seat_z, float(np.percentile(lows, 25))) if lows else seat_z  # selten kurz einsinken statt schweben
    if seat_z < 0.15:
        return []
    boxes = [contact.Box("Bank", np.array([mid[0], mid[1], (seat_z - 0.005) / 2]), np.eye(3),
                         np.array([half[0], half[1], (seat_z - 0.005) / 2]))]
    # Rückenlehne: Linie vom Gesäß zu den Schulterblättern (hinterste Rückenpunkte)
    bpts = v[back]
    shoulder = bpts[np.argmax(bpts[:, 1])]  # Blick -Y: hinten = +Y
    d = shoulder - np.array([centre[0], centre[1] + 0.1, seat_z])
    d[0] = 0.0
    length = float(np.linalg.norm(d))
    lean = math.degrees(math.atan2(d[1], d[2]))
    if length > 0.2 and lean > 15:  # deutlich zurückgelehnt: Lehne
        along = d / length
        normal = np.cross(np.array([1.0, 0, 0]), along)  # zeigt vom Rücken weg nach hinten/unten
        if normal[2] > 0:
            normal = -normal
        thick = 0.04
        mid = np.array([centre[0], centre[1] + 0.1, seat_z]) + d / 2 + normal * (thick + 0.005)
        rot = np.stack([np.array([1.0, 0, 0]), along, -normal], 1)
        boxes.append(contact.Box("Lehne", mid, rot, np.array([0.14, length / 2 + 0.05, thick])))
    return boxes


ARM_PARTS = ("Oberarm", "Unterarm", "Hand")
TORSO_PARTS = ("Brust", "Bauch", "Becken")


def _part_masks(model: BodyModel) -> np.ndarray:
    idx, w = model.vertex_bone_indices, model.vertex_bone_weights
    owner = idx[np.arange(len(idx)), w.argmax(1)]
    return np.array([contact.body_part(model.bone_labels[b]) for b in owner])


def _arm_hits(checker: contact.Checker, vertices: np.ndarray) -> dict[str, float]:
    """Je Seite: wie tief Oberarm, Unterarm oder Hand im Rumpf stecken (echte Dreiecksschnitte)."""
    depth = {"L": 0.0, "R": 0.0}
    for f in checker.check_frame(vertices, self_check=True):
        if f.kind != "koerper":
            continue
        a, b = f.where.split(" / ")
        for arm, other in ((a, b), (b, a)):
            if arm.split(" ")[0] in ARM_PARTS and other.split(" ")[0] in TORSO_PARTS:
                side = "L" if arm.endswith("links") else "R"
                depth[side] = max(depth[side], f.depth)
    return depth


def separate_arms(fig, motion: M.Motion, margin: float = 0.01, rounds: int = 5, every: int = 3,
                  log: Log | None = None) -> tuple[M.Motion, float, float]:
    """Arme, die im Rumpf stecken (dickerer Körper als im Video), im Schultergelenk vom Körper weg
    drehen (abspreizen), bis sie die Haut nur noch berühren. Geprüft wird jedes every-te Bild,
    dazwischen wird die Korrektur weich übergeblendet. Gibt Bewegung, größte Tiefe vorher, nachher."""
    model = fig.model
    checker = contact.Checker(model, fig.reference_vertices)
    bones = list(motion.bones)
    rotations = motion.rotations.copy()
    for side in ("L", "R"):
        if f"upperarm01.{side}" not in bones:
            bones.append(f"upperarm01.{side}")
            ident = np.zeros((motion.frames, 1, 4), np.float32)
            ident[..., 3] = 1
            rotations = np.concatenate([rotations, ident], 1)
    motion = dataclasses.replace(motion, bones=bones, rotations=rotations)
    ref = fig.reference_poses
    sample = np.unique(np.r_[np.arange(0, motion.frames, every), motion.frames - 1])
    chest = model.bone("spine01")
    worst_before = worst_after = 0.0
    for rnd in range(rounds + 1):
        sub = dataclasses.replace(motion, rotations=motion.rotations[sample], root=motion.root[sample])
        out = fig.pose(M.pose_matrices(model, sub).cpu().numpy())
        hits = [_arm_hits(checker, out["vertices"][i]) for i in range(len(sample))]
        worst = max(max(h.values()) for h in hits)
        if rnd == 0:
            worst_before = worst
        worst_after = worst
        if rnd == rounds or worst < 0.003:
            break
        if log:
            log(f"Arme vom Rumpf lösen: Runde {rnd + 1}, tiefste Stelle {100 * worst:.1f} cm")
        rotations = motion.rotations.copy()
        full_pose = fig.pose(M.pose_matrices(model, motion).cpu().numpy())["poses"]
        for side, sign in (("L", 1.0), ("R", -1.0)):
            depth = np.array([h[side] for h in hits])
            if depth.max() < 0.003:
                continue
            b = model.bone(f"upperarm01.{side}")
            p = model.bone_parents[b]
            angle = np.where(depth > 0.003, np.arctan((depth + margin) / 0.3), 0.0)
            # auch die Nachbarbilder mitnehmen, dann über alle Bilder verteilen
            angle = np.array([angle[max(0, i - 1):i + 2].max() for i in range(len(angle))])
            full = np.interp(np.arange(motion.frames), sample, angle)
            full = M._smooth(full[:, None], 1.5)[:, 0]
            delta = out["poses"][:, chest, :3, :3] @ ref[chest, :3, :3].T
            fwd = delta @ np.array([0.0, -1.0, 0.0])  # Blickrichtung der Brust (Anny blickt nach -Y)
            fwd_full = np.stack([np.interp(np.arange(motion.frames), sample, fwd[:, c]) for c in range(3)], 1)
            fwd_full /= np.linalg.norm(fwd_full, axis=1, keepdims=True)
            # Drehung um die Blickrichtung: linker Arm nach +X (außen), rechter nach -X
            world = roma.rotvec_to_rotmat(torch.as_tensor(sign * fwd_full * full[:, None])).numpy()
            pr = full_pose[:, p, :3, :3] @ ref[p, :3, :3].T
            local = np.transpose(pr, (0, 2, 1)) @ world @ pr
            k = motion.bones.index(f"upperarm01.{side}")
            cur = roma.unitquat_to_rotmat(torch.as_tensor(rotations[:, k])).double().numpy()
            q = roma.rotmat_to_unitquat(torch.as_tensor(local @ cur)).numpy()
            rotations[:, k] = M._continuous(q[:, None])[:, 0]
        motion = dataclasses.replace(motion, rotations=rotations.astype(np.float32))
    return motion, worst_before, worst_after


def _foot_slip(model: BodyModel, vertices: np.ndarray, contacts: dict[str, np.ndarray]) -> float:
    worst = 0.0
    for foot in M.FEET:
        c = contacts.get(foot)
        if c is None or not c.any():
            continue
        sole = M._sole_vertices(model, foot)
        xy = vertices[:, sole, :2].mean(1)
        i = 0
        while i < len(c):
            if not c[i]:
                i += 1
                continue
            j = i
            while j < len(c) and c[j]:
                j += 1
            worst = max(worst, float(np.linalg.norm(xy[i:j] - xy[i], axis=1).max()))
            i = j
    return 100 * worst


def _relevant(f: contact.Finding) -> bool:
    """Falten in einem Gelenk und weiches Gewebe, das sich beim Beugen leicht eindrückt
    (Bauch auf Oberschenkel, unter 2 cm), sind bei Menschen normal und stehen nicht im Bericht."""
    if f.kind == "falte":
        return False
    return not (f.kind == "koerper" and f.depth < 0.02)


def make_plausible(model: BodyModel, params: BodyParams, cap: mocap.Capture, name: str,
                   seconds: tuple[float, float] = (0.0, 0.0), steady: bool = True, fig=None,
                   check_every: int = 3, log: Log | None = None) -> tuple[M.Motion, list[contact.Box], Report]:
    """Die Logikprüfung aus dem Modulkopf. Gibt die Bewegung, die erschlossenen Möbel und den Bericht.

    fig: Zielkörper (posing.Figure, z. B. aus dem GLB); Arme und Durchdringung werden an ihm geprüft.
    Ohne gilt der Körper, mit dem ausgelesen wurde."""
    from .posing import Figure

    log = log or (lambda _: None)
    fig = fig or Figure.from_params(model, params)
    rep = Report(name, seconds, cap.reprojection_px)
    motion, rep.outliers = remove_outliers(cap.motion)
    motion, rep.limited = limit_joints(model, params, motion)
    if steady:
        motion, rep.turn_removed_deg = steady_heading(motion)
    motion = stabilize_root(smooth(motion), in_place=steady)
    motion, _ = limit_joints(model, params, motion)  # Glätten kann knapp über die Grenze schieben
    log(f"{name}: Füße auf den Boden")
    pb = M.play(model, params, motion)
    # Fremde Videos zittern in der Tiefe: Fußgeschwindigkeit großzügiger als bei eigenen Aufnahmen
    contacts = M.detect_contacts(model, pb, motion.fps, height=0.08, speed=0.8)
    motion = M.fix_ground(model, params, motion, contacts)
    pb = M.play(model, params, motion)
    support = find_support(model, pb.vertices, pb.bones)
    if support:
        rep.support = " und ".join(b.name for b in support)
    rep.contact_frames = float(np.mean(np.any(np.stack([contacts[f] for f in M.FEET], 1), 1)))
    rep.foot_slip_cm = _foot_slip(model, pb.vertices, contacts)
    log(f"{name}: Arme und Durchdringung prüfen")
    motion, rep.arm_depth_before_cm, rep.arm_depth_after_cm = separate_arms(fig, motion, log=log)
    rep.arm_depth_before_cm *= 100
    rep.arm_depth_after_cm *= 100
    _, verts = figure_playback(fig, motion)
    checker = contact.Checker(model, fig.reference_vertices)
    findings = [[f for f in checker.check_frame(verts[i], support) if _relevant(f)]
                if i % check_every == 0 else [] for i in range(motion.frames)]
    rep.findings = contact.summarize(findings, motion.fps)
    motion = dataclasses.replace(motion, name=name, source="Video (übertragen)")
    return motion, support, rep


# ------------------------------------------------------------------------ Übertragen auf ein GLB


def _ground_on_figure(vertices: np.ndarray, model: BodyModel, contacts: dict[str, np.ndarray]) -> np.ndarray:
    """Höhenausgleich für den Zielkörper (andere Beinlänge): Füße mit Kontakt auf z = 0."""
    soles = np.concatenate([M._sole_vertices(model, f) for f in M.FEET])
    lowest_sole = vertices[:, soles, 2].min(1)
    lowest = vertices[:, :, 2].min(1)
    has = np.any(np.stack([contacts[f] for f in M.FEET if f in contacts], 1), 1) if contacts else \
        np.zeros(len(vertices), bool)
    frames = np.arange(len(vertices))
    shift = np.interp(frames, frames[has], -lowest_sole[has]) if has.any() else -lowest
    shift = np.maximum(shift, -lowest)  # nie unter den Boden
    return shift


def figure_playback(fig, motion: M.Motion) -> tuple[np.ndarray, np.ndarray]:
    """Bewegung auf einem Körper aus einem GLB: (F, J, 4, 4) Knochen, (F, V, 3) Punkte, Boden z = 0."""
    model = fig.model
    out = fig.pose(M.pose_matrices(model, motion).cpu().numpy())
    ri = model.bone("root")
    ref = fig.reference_poses[ri, :3, 3]
    hip = float(ref[2] - fig.reference_vertices[:, 2].min())
    offset = motion.root.astype(np.float64) * hip - out["poses"][:, ri, :3, 3]
    poses = out["poses"].copy()
    poses[:, :, :3, 3] += offset[:, None]
    verts = out["vertices"] + offset[:, None]
    dz = _ground_on_figure(verts, model, motion.contacts)
    poses[:, :, 2, 3] += dz[:, None]
    verts[..., 2] += dz[:, None]
    return poses, verts


def animate_glb(model: BodyModel, glb_in: Path, motions: list[M.Motion], glb_out: Path) -> Path:
    """Bewegungen als glTF-Animationen in ein vorhandenes FitScan-GLB schreiben (Haut bleibt)."""
    from .animation import read_glb, write_glb_raw
    from .export import _TO_Y_UP, bone_name
    from .posing import Figure

    fig = Figure.from_glb(model, glb_in)
    gltf, blob = read_glb(glb_in)
    skin = gltf["skins"][0]
    joints = skin["joints"]
    names = [gltf["nodes"][j]["name"] for j in joints]
    preset = "unreal" if "pelvis" in names and "root" in names else "anny"
    anny_of = {bone_name(n, preset): n for n in model.bone_labels}
    parent = {}
    for i, node in enumerate(gltf["nodes"]):
        for c in node.get("children", []):
            parent[c] = i
    jindex = {j: k for k, j in enumerate(joints)}
    order = [model.bone(anny_of[n]) if n in anny_of else -1 for n in names]  # -1: Bodengelenk
    root_joint = joints[names.index(bone_name("root", preset))]

    def add(array: np.ndarray, kind: str, minmax: bool = False) -> int:
        while len(blob) % 4:
            blob.append(0)
        data = np.ascontiguousarray(array, np.float32)
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": data.nbytes}
        blob.extend(data.tobytes())
        gltf["bufferViews"].append(view)
        acc = {"bufferView": len(gltf["bufferViews"]) - 1, "componentType": 5126,
               "count": len(data), "type": kind}
        if minmax:
            acc["min"] = data.reshape(len(data), -1).min(0).tolist()
            acc["max"] = data.reshape(len(data), -1).max(0).tolist()
        gltf["accessors"].append(acc)
        return len(gltf["accessors"]) - 1

    animations = gltf.setdefault("animations", [])
    for motion in motions:
        poses, _ = figure_playback(fig, motion)
        world = _TO_Y_UP[None, None] @ poses  # (F, J, 4, 4) glTF-Achsen
        f = len(world)
        wj = np.empty((f, len(joints), 4, 4))
        for k, bi in enumerate(order):
            wj[:, k] = np.eye(4) if bi < 0 else world[:, bi]
        times = add(np.arange(f) / motion.fps, "SCALAR", minmax=True)
        samplers, channels = [], []
        for k, j in enumerate(joints):
            p = parent.get(j)
            local = wj[:, k] if p not in jindex else np.linalg.inv(wj[:, jindex[p]]) @ wj[:, k]
            q = roma.rotmat_to_unitquat(torch.as_tensor(local[:, :3, :3])).numpy()
            q = M._continuous(q[:, None].astype(np.float64))[:, 0]
            rest = np.array(gltf["nodes"][j].get("rotation", [0, 0, 0, 1.0]))
            if j != root_joint and np.abs(np.abs(q @ rest) - 1.0).max() < 1e-7:
                continue
            samplers.append({"input": times, "output": add(q, "VEC4"), "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": j, "path": "rotation"}})
            if j == root_joint:
                samplers.append({"input": times, "output": add(local[:, :3, 3], "VEC3"), "interpolation": "LINEAR"})
                channels.append({"sampler": len(samplers) - 1, "target": {"node": j, "path": "translation"}})
        animations.append({"name": motion.name, "samplers": samplers, "channels": channels})
    return write_glb_raw(gltf, blob, glb_out)


# ------------------------------------------------------------------------ Puppe wie im Trainingsvideo


def mannequin_params(model: BodyModel, params: BodyParams, muscle: float = 1.0, weight: float = 0.35,
                     smooth_rounds: int = 80) -> BodyParams:
    """Muskulöse Puppe wie in Trainingsanimationen: gleiche Größe, mehr Muskeln, wenig Fett, glatter
    Kopf ohne Gesicht (Gesicht, Ohren und Lippen weich ausgestrichen, Augen und Mundraum nach innen)."""
    phen = {**params.phenotype, "muscle": muscle, "weight": weight, "proportions": 0.7}
    puppet = BodyParams(phenotype=phen, local={}, scale=params.scale)
    rest = M._rest(model, puppet)
    v = rest["_unscaled_vertices"][0].detach().cpu().numpy().astype(np.float64)
    # Kopf geometrisch: alles über dem Kopfgelenk; Augäpfel, Zähne, Zunge sind eigene Netzteile
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    head_z = float(rest["bone_heads"][0, model.bone("head"), 2]) / puppet.scale
    e = contact._edges(model.faces)
    _, comp = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(v), len(v))))
    main = comp == np.bincount(comp).argmax()
    head = v[:, 2] > head_z - 0.03
    inner = ~main & (v[:, 2] > head_z - 0.15)  # Augen, Zähne, Zunge (Zähne liegen tiefer als das Kopfgelenk)
    # Kopf als glattes Ei wie bei einer Schaufensterpuppe: jeden Kopfpunkt vom Mittelpunkt aus auf
    # ein Ellipsoid schieben, unterhalb einer schrägen Linie Kinn → Schädelansatz weich in den Hals
    # überblenden, danach glätten
    hv = v[head & main]
    lo, hi = np.percentile(hv, 3, axis=0), np.percentile(hv, 97, axis=0)
    centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2 + 0.01, lo[2] + 0.55 * (hi[2] - lo[2])])
    radii = np.array([(hi[0] - lo[0]) * 0.47, (hi[1] - lo[1]) * 0.42, hv[:, 2].max() - centre[2]])
    face = main & (np.abs(v[:, 0]) < 0.02) & (v[:, 2] > head_z - 0.12) & (v[:, 1] < centre[1] - 0.75 * radii[1])
    chin = v[face][np.argmin(v[face, 2])]  # Kinnspitze
    back_y = centre[1] + radii[1]
    slope = (head_z - 0.01 - chin[2]) / (back_y - chin[1])
    above = v[:, 2] - (chin[2] + (v[:, 1] - chin[1]) * slope)  # Abstand über der Linie Kinn → Nacken
    above = np.where(v[:, 2] > chin[2] - 0.04, above, -1.0)  # Brust vor dem Kinn bleibt
    blend = np.clip((above + 0.04) / 0.035, 0, 1) * main  # Kinn ganz drin, darunter Hals
    blend = blend * blend * (3 - 2 * blend)

    def shell(x):
        d = (x - centre) / radii
        return centre + (x - centre) / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)

    # Mundraum (Tasche hinter den Lippen, hängt am Körpernetz): liegt tief im Kopf. Auf die Eiform
    # geschoben läge er doppelt über den Lippen und das Netz knüllt sich; er wandert deshalb nach innen.
    depth = np.linalg.norm((v - centre) / radii, axis=1)
    cavity = main & (above > 0) & (depth < 0.8) & (v[:, 2] < centre[2] - 0.2 * radii[2])
    inner = inner | cavity
    smooth_sel = (above > -0.07) & main & ~inner
    sel = smooth_sel & (blend > 0)
    edges = contact._edges(model.faces)
    edges = edges[~inner[edges].any(1)]  # beim Glätten nur Nachbarn auf der Oberfläche
    deg = np.bincount(edges.ravel(), minlength=len(v)).astype(np.float64)

    target = v.copy()
    target[sel] = v[sel] + blend[sel, None] * (shell(v)[sel] - v[sel])  # auf die Eiform schieben
    for _ in range(smooth_rounds):  # Knicke am Übergang glätten, Eiform halten
        acc = np.zeros_like(target)
        np.add.at(acc, edges[:, 0], target[edges[:, 1]])
        np.add.at(acc, edges[:, 1], target[edges[:, 0]])
        avg = acc / np.maximum(deg, 1)[:, None]
        move = smooth_sel & (deg > 0)
        target[move] += 0.5 * (avg[move] - target[move])
        target[sel] += blend[sel, None] * (shell(target)[sel] - target[sel])
    target[inner] = centre + 0.6 * (shell(v)[inner] - centre)  # verschwindet im Kopf
    if cavity.any():  # Mundtasche als glatte Haut über den Lippenspalt spannen (Lippenrand fest)
        all_e = contact._edges(model.faces)
        all_e = all_e[~(~main)[all_e].any(1)]
        deg_all = np.bincount(all_e.ravel(), minlength=len(v)).astype(np.float64)
        target[cavity] = shell(target[cavity].mean(0, keepdims=True))
        for _ in range(300):
            acc = np.zeros_like(target)
            np.add.at(acc, all_e[:, 0], target[all_e[:, 1]])
            np.add.at(acc, all_e[:, 1], target[all_e[:, 0]])
            target[cavity] = shell(acc[cavity] / np.maximum(deg_all[cavity], 1)[:, None])
    puppet.offsets = (target - v).astype(np.float32)
    return puppet


def export_puppet(model: BodyModel, puppet: BodyParams, motions: list[M.Motion], path: Path) -> Path:
    """Wie motion.export_glb, aber ohne repair_offsets (auch nicht in build_rig): Beim Glätten des Gesichts schieben sich
    Mundraum und Lidinnenseiten ineinander. Das ist innen und unsichtbar, die Reparatur würde dafür
    aber das ganze Gesicht wiederherstellen."""
    from .export import _TO_Y_UP, GlbAnimation, build_rig, vertex_normals, write_glb

    plain = puppet.copy()
    plain.offsets = None
    rig = build_rig(model, plain, preset="anny", morphs=False)
    verts = model.rest_numpy(puppet)["vertices"].astype(np.float64)
    v_y = verts @ _TO_Y_UP[:3, :3].T
    v_y[:, 1] -= model.rest_numpy(plain)["vertices"][:, 2].min()  # Füße wie im Rig auf den Boden
    rig = dataclasses.replace(rig, vertices=v_y.astype(np.float32),
                              normals=vertex_normals(v_y, model.faces).astype(np.float32))
    animations = []
    for motion in motions:
        pb = M.play(model, puppet, motion, vertices=False)
        rotations, root = M.gltf_channels(pb, rig.joint_parents)
        animations.append(GlbAnimation(motion.name, np.arange(motion.frames) / motion.fps, rotations, root))
    return write_glb(rig, path, name="Puppe", animations=animations)


# ------------------------------------------------------------------------ Gesamtablauf


def body_params(glb: Path, height: float | None, weight: float | None, model: BodyModel,
                log: Log | None = None) -> BodyParams:
    """Körper zum Auslesen: params.json neben dem GLB (FitScan-Speicher) oder aus Größe/Gewicht."""
    folder = Path(glb).parent
    if (folder / "params.json").exists():
        return BodyParams.load(folder)
    if not height or not weight:
        raise ValueError("Neben dem Modell liegt keine params.json: bitte Größe und Gewicht angeben.")
    from .fit import Person, fit_to_measurements

    params, _ = fit_to_measurements(model, Person(measures={"height": height, "weight": weight}), log=log)
    return params


def transfer(video: Path, glb: Path, folder: Path, height: float | None = None, weight: float | None = None,
             max_shots: int = 6, names: list[str] | None = None, steady: bool = True, render: bool = True,
             mannequin: bool = True, log: Log | None = None) -> dict:
    """Video -> Szenen -> Bewegungen (logisch geprüft) -> Modell mit Animationen, .blend, FBX, Vorschau."""
    log = log or print
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model = BodyModel.get()
    params = body_params(glb, height, weight, model, log)
    from .posing import Figure

    fig = Figure.from_glb(model, glb)
    cache = folder / "koerperpunkte.npz"
    if cache.exists():
        track = Track.load(cache)
    else:
        log("Video wird gelesen …")
        track = analyze(video, log=log)
        track.save(cache)
    shots = find_shots(track)
    shots = sorted(sorted(shots, key=lambda s: -s.full * min(s.length, 6 * track.fps))[:max_shots],
                   key=lambda s: s.start)
    if not shots:
        raise ValueError("Keine Szene gefunden, in der eine Person lange genug ganz zu sehen ist.")
    motions, reports, furniture = [], [], {}
    for k, shot in enumerate(shots):
        name = names[k] if names and k < len(names) else f"Szene_{k + 1:02d}"
        a, b = shot.seconds(track.fps)
        log(f"{name}: Bewegung auslesen ({a:.1f} bis {b:.1f} s)")
        fixed, swaps = fix_side_swaps(track, shot)
        cap = capture(model, params, fixed, shot, name)
        mo, support, rep = make_plausible(model, params, cap, name, (a, b), steady=steady, fig=fig, log=log)
        rep.side_swaps = swaps
        mo.save(folder / "bewegungen" / f"{name}.npz")
        motions.append(mo)
        reports.append(rep)
        furniture[name] = [{"name": s.name, "centre": s.centre.tolist(), "rotation": s.rotation.tolist(),
                            "half": np.asarray(s.half).tolist()} for s in support]
    log("Auf das Modell übertragen …")
    animated = animate_glb(model, glb, motions, folder / "modell_bewegt.glb")
    puppet_glb = None
    if mannequin:
        log("Puppe wie im Video …")
        puppet_glb = export_puppet(model, mannequin_params(model, params), motions, folder / "puppe_bewegt.glb")
    summary = {"video": str(video), "modell": str(glb), "szenen": [r.to_json() for r in reports],
               "moebel": furniture}
    (folder / "bericht.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), "utf-8")
    lines = ["# Bewegung aus Video", "", f"Video: {Path(video).name}", ""]
    for r in reports:
        lines += r.lines() + [""]
    (folder / "bericht.txt").write_text("\n".join(lines), "utf-8")
    if render:
        from .blender import find_blender

        blender = find_blender()
        if blender is None:
            log("Blender nicht gefunden: kein .blend, keine FBX, kein Vorschauvideo.")
        else:
            studio(blender, animated, folder, furniture, prefix="")
            if puppet_glb is not None:
                studio(blender, puppet_glb, folder, furniture, prefix="puppe_")
    summary["dateien"] = sorted(p.name for p in folder.iterdir())
    return summary


def studio(blender: Path, glb: Path, folder: Path, furniture: dict, render: bool = True, video: bool = True,
           prefix: str = "") -> dict:
    """Blender (externes Programm): Studio-Look wie in Trainingsvideos, .blend, FBX mit Bewegung, Vorschau."""
    import subprocess

    script = Path(__file__).with_name("blender_studio.py")
    spec = folder / "studio.json"
    spec.write_text(json.dumps({"glb": str(glb), "folder": str(folder), "furniture": furniture,
                                "motions": list(furniture), "render": render, "video": video, "prefix": prefix}), "utf-8")
    res = subprocess.run([str(blender), "-b", "--python-exit-code", "1", "-P", str(script), "--", str(spec)], capture_output=True,
                         text=True, timeout=7200)
    if res.returncode != 0:
        raise RuntimeError(f"Blender-Fehler:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    spec.unlink()
    for line in res.stdout.splitlines():
        if line.startswith("FITSCAN_STUDIO "):
            return json.loads(line.split(" ", 1)[1])
    return {}
