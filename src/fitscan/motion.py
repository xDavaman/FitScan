"""Bewegungsbibliothek: Bewegungen als Daten, die auf jedes gescannte Modell passen.

Eine Bewegung speichert je Bild die Gelenkdrehungen in Annys Form „local-ref“ (Drehung in
Weltachsen gegenüber Annys Bezugshaltung) und den Hüftpunkt (Kopf des Bones „root“) über dem
Boden. Der Hüftpunkt ist durch die Hüfthöhe des aufgenommenen Körpers geteilt und wird beim
Abspielen mit der Hüfthöhe des Zielkörpers malgenommen. So passt eine Bewegung auf jeden
Anny-Körper, und die Namen der Knochen spielen erst beim Export eine Rolle.

Koordinaten: Anny-Achsen (Z oben, Blick -Y, linke Körperseite +X), Boden bei z = 0, Meter.

Finger gehören einfach mit in `bones` (z. B. „finger2-1.R“). Gegenstände, die die Figur greift
oder trägt, hängen als `Attachment` an einem Knochen; Kontakte (Füße, Hände, Gesäß ...) stehen je
Knochen in `contacts`. Mimik steht als Kurve je ARKit-Gesichtsregler (`face_rig`, z. B. „jawOpen“)
in `face`, Werte 0 bis 1 je Bild.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import roma
import torch

from .body import BodyModel, BodyParams

FORMAT = 1
FEET = ("foot.L", "foot.R")


@dataclasses.dataclass
class Attachment:
    """Gegenstand an einem Knochen (gegriffen, getragen), von Bild start bis end einschließlich."""

    name: str  # z. B. „Hantel“
    bone: str  # Anny-Name, z. B. „wrist.R“
    start: int
    end: int
    offset: np.ndarray  # (4, 4) Lage des Gegenstands im Knochen (Meter)
    asset: str = ""  # Modell des Gegenstands in der Bibliothek, z. B. „gegenstaende/hantel.glb“

    def to_json(self) -> dict:
        return {"name": self.name, "bone": self.bone, "start": self.start, "end": self.end,
                "offset": np.asarray(self.offset, float).round(6).tolist(), "asset": self.asset}

    @classmethod
    def from_json(cls, d: dict) -> Attachment:
        return cls(d["name"], d["bone"], int(d["start"]), int(d["end"]), np.array(d["offset"], np.float64),
                   d.get("asset", ""))


@dataclasses.dataclass
class Motion:
    name: str
    fps: float
    bones: list[str]  # Anny-Namen
    rotations: np.ndarray  # (F, B, 4) Quaternion x, y, z, w
    root: np.ndarray  # (F, 3) Hüftpunkt über dem Boden, geteilt durch die Hüfthöhe
    source: str = ""
    loop: bool = False
    contacts: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)  # Knochen -> (F,) bool
    attachments: list[Attachment] = dataclasses.field(default_factory=list)
    face: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)  # ARKit-Regler -> (F,) 0..1

    @property
    def frames(self) -> int:
        return len(self.root)

    @property
    def duration(self) -> float:
        return (self.frames - 1) / self.fps

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"format": FORMAT, "name": self.name, "fps": self.fps, "bones": self.bones,
                "source": self.source, "loop": self.loop,
                "attachments": [a.to_json() for a in self.attachments]}
        extra = {f"contact:{k}": v.astype(bool) for k, v in self.contacts.items()}
        extra |= {f"face:{k}": v.astype(np.float32) for k, v in self.face.items()}
        with path.open("wb") as f:
            np.savez_compressed(f, meta=np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8),
                                rotations=self.rotations.astype(np.float32),
                                root=self.root.astype(np.float32), **extra)
        return path

    @classmethod
    def load(cls, path: Path) -> Motion:
        with np.load(Path(path)) as data:
            meta = json.loads(bytes(data["meta"]).decode("utf-8"))
            if meta.get("format", 0) > FORMAT:
                raise ValueError(f"{path}: neueres Bewegungsformat, bitte FitScan aktualisieren.")
            contacts = {k.split(":", 1)[1]: data[k].copy() for k in data.files if k.startswith("contact:")}
            face = {k.split(":", 1)[1]: data[k].copy() for k in data.files if k.startswith("face:")}
            return cls(meta["name"], float(meta["fps"]), list(meta["bones"]), data["rotations"].copy(),
                       data["root"].copy(), meta.get("source", ""), bool(meta.get("loop", False)), contacts,
                       [Attachment.from_json(a) for a in meta.get("attachments", [])], face)

    def resampled(self, fps: float) -> Motion:
        """Andere Bildrate (Drehungen sphärisch, Hüfte linear interpoliert)."""
        n = max(2, int(round(self.duration * fps)) + 1)
        t = np.linspace(0.0, self.frames - 1, n)
        i0 = np.clip(np.floor(t).astype(int), 0, self.frames - 2)
        w = (t - i0).astype(np.float32)
        q0 = torch.as_tensor(self.rotations[i0])
        q1 = torch.as_tensor(self.rotations[i0 + 1])
        q1 = torch.where(((q0 * q1).sum(-1, keepdim=True) < 0), -q1, q1)
        q = q0 * (1 - torch.as_tensor(w)[:, None, None]) + q1 * torch.as_tensor(w)[:, None, None]
        q = q / q.norm(dim=-1, keepdim=True)  # nlerp, bei 30+ Bildern pro Sekunde wie slerp
        root = self.root[i0] * (1 - w[:, None]) + self.root[i0 + 1] * w[:, None]
        near = np.round(t).astype(int)
        contacts = {k: v[near] for k, v in self.contacts.items()}
        face = {k: (v[i0] * (1 - w) + v[i0 + 1] * w).astype(np.float32) for k, v in self.face.items()}
        factor = (n - 1) / max(self.frames - 1, 1)
        attachments = [dataclasses.replace(a, start=int(round(a.start * factor)), end=int(round(a.end * factor)))
                       for a in self.attachments]
        return dataclasses.replace(self, fps=fps, rotations=q.numpy().astype(np.float32),
                                   root=root.astype(np.float32), contacts=contacts, attachments=attachments,
                                   face=face)


def from_rotvecs(name: str, fps: float, bones: list[str], rotvecs: np.ndarray, root: np.ndarray,
                 **kw) -> Motion:
    q = roma.rotvec_to_unitquat(torch.as_tensor(np.asarray(rotvecs, np.float32)))
    return Motion(name, fps, list(bones), _continuous(q.numpy()), np.asarray(root, np.float32), **kw)


def _continuous(q: np.ndarray) -> np.ndarray:
    """Vorzeichen der Quaternionen so wählen, dass benachbarte Bilder nah beieinander liegen."""
    q = q.copy()
    for i in range(1, len(q)):
        flip = (q[i] * q[i - 1]).sum(-1) < 0
        q[i][flip] *= -1
    return q


# --------------------------------------------------------------------------- Abspielen


@dataclasses.dataclass
class Playback:
    """Eine Bewegung auf einem bestimmten Körper."""

    bones: np.ndarray  # (F, J, 4, 4) Knochen in Weltkoordinaten (Anny-Achsen, Boden z = 0)
    vertices: np.ndarray | None  # (F, V, 3)
    hip_height: float


def _rest(model: BodyModel, params: BodyParams) -> dict[str, torch.Tensor]:
    offsets = None
    if params.offsets is not None:
        offsets = torch.as_tensor(params.offsets, dtype=torch.float32, device=model.device)
    return model.rest(model.phenotype_tensor(params), model.local_tensor(params), offsets, params.scale)


@torch.no_grad()
def hip_height(model: BodyModel, params: BodyParams) -> float:
    """Höhe des Hüftpunkts über dem Boden in Annys Bezugshaltung (aufrecht stehend)."""
    rest = _rest(model, params)
    verts = model.posed(rest, None, params.scale)["vertices"][0]
    return float(-verts[:, 2].min())


def pose_matrices(model: BodyModel, motion: Motion, frames: slice | None = None) -> torch.Tensor:
    """(F, J, 4, 4) Gelenkdrehungen für Anny; Knochen ohne Daten bleiben in der Bezugshaltung."""
    q = torch.as_tensor(motion.rotations[frames if frames is not None else slice(None)], device=model.device)
    idx = torch.tensor([model.bone(b) for b in motion.bones], device=model.device)
    pose = torch.eye(4, device=model.device)[None, None].repeat(len(q), len(model.bone_labels), 1, 1)
    pose[:, idx, :3, :3] = roma.unitquat_to_rotmat(q)
    return pose


@torch.no_grad()
def play(model: BodyModel, params: BodyParams, motion: Motion, vertices: bool = True,
         chunk: int = 32) -> Playback:
    rest = _rest(model, params)
    height = hip_height(model, params)
    rest_poses = rest["_unscaled_bone_poses"]
    scale = params.scale
    bones, verts = [], []
    for lo in range(0, motion.frames, chunk):
        sl = slice(lo, min(motion.frames, lo + chunk))
        pose = pose_matrices(model, motion, sl)
        transforms, poses = model.anny.get_bone_transforms(pose, rest_poses)
        world = torch.as_tensor(motion.root[sl] * height, dtype=torch.float32, device=model.device)
        poses = poses.clone()
        poses[..., :3, 3] = poses[..., :3, 3] * scale + world[:, None, :]
        bones.append(poses.cpu().numpy())
        if vertices:
            v = model.anny.get_skinned_vertices(rest["_unscaled_vertices"].expand(len(pose), -1, -1), transforms)
            verts.append((v * scale + world[:, None, :]).cpu().numpy())
    return Playback(np.concatenate(bones), np.concatenate(verts) if vertices else None, height)


# ------------------------------------------------------------------ Bodenkontakt


def detect_contacts(model: BodyModel, playback: Playback, fps: float, height: float = 0.06,
                    speed: float = 0.35) -> dict[str, np.ndarray]:
    """Je Fuß (F,) bool: Fußpunkt nah am Boden und kaum in Bewegung."""
    out = []
    for name in FEET:
        sole = _sole_vertices(model, name)
        pts = playback.vertices[:, sole] if playback.vertices is not None else None
        low = pts[..., 2].min(1) if pts is not None else playback.bones[:, model.bone(name), 2, 3]
        center = pts.mean(1) if pts is not None else playback.bones[:, model.bone(name), :3, 3]
        vel = np.zeros(len(center))
        vel[1:-1] = np.linalg.norm(center[2:, :2] - center[:-2, :2], axis=1) * fps / 2
        vel[0], vel[-1] = vel[1] if len(vel) > 2 else 0.0, vel[-2] if len(vel) > 2 else 0.0
        floor = np.percentile(low, 5)
        out.append((low - floor < height) & (vel < speed))
    contacts = _close_gaps(np.stack(out, 1), 3)
    return {name: contacts[:, k] for k, name in enumerate(FEET)}


def _close_gaps(flags: np.ndarray, size: int) -> np.ndarray:
    """Kurze Lücken und kurze Stücke entfernen (Rauschen der Erkennung)."""
    flags = flags.copy()
    for k in range(flags.shape[1]):
        f = flags[:, k]
        for value in (True, False):
            i = 0
            while i < len(f):
                if f[i] != value:
                    j = i
                    while j < len(f) and f[j] != value:
                        j += 1
                    if 0 < i and j < len(f) and j - i < size:
                        f[i:j] = value
                    i = j
                else:
                    i += 1
    return flags


_SOLE_CACHE: dict[tuple[int, str], np.ndarray] = {}


def _sole_vertices(model: BodyModel, foot: str) -> np.ndarray:
    """Punkte von Fuß und Zehen (Gewicht überwiegend auf diesen Knochen)."""
    key = (id(model), foot)
    if key not in _SOLE_CACHE:
        side = foot.split(".")[1]
        mask = model.dominant_bone_mask(lambda n: n == foot or (n.startswith("toe") and n.endswith("." + side)))
        _SOLE_CACHE[key] = np.where(mask)[0]
    return _SOLE_CACHE[key]


def fix_ground(model: BodyModel, params: BodyParams, motion: Motion,
               contacts: dict[str, np.ndarray] | None = None, smooth: float = 0.25) -> Motion:
    """Füße auf den Boden: nichts sinkt ein, stehende Füße rutschen nicht.

    Höhe: In Bildern mit Bodenkontakt kommt der tiefste Fußpunkt auf z = 0, dazwischen wird der
    Ausgleich weich übergeblendet (Sprünge behalten ihre Flugbahn). Danach wird nie unter den Boden
    gegangen. Rutschen: Solange ein Fuß steht, wird die Hüfte so nachgeführt, dass er an seinem
    Platz bleibt."""
    pb = play(model, params, motion)
    if contacts is None:
        contacts = motion.contacts if all(f in motion.contacts for f in FEET) else detect_contacts(
            model, pb, motion.fps)
    contacts = {**motion.contacts, **contacts}
    feet_contact = np.stack([contacts[f] for f in FEET], 1)
    lowest = pb.vertices[..., 2].min(1)
    any_contact = feet_contact.any(1)
    shift = np.zeros((motion.frames, 3))
    frames = np.arange(motion.frames)
    if any_contact.any():
        shift[:, 2] = np.interp(frames, frames[any_contact], -lowest[any_contact])
    else:
        shift[:, 2] = -lowest.min()
    # Waagerecht: Fuß festhalten, solange er steht
    feet_xy = []
    for name in FEET:
        sole = _sole_vertices(model, name)
        feet_xy.append(pb.vertices[:, sole, :2].mean(1))
    drift = np.zeros((motion.frames, 2))
    count = np.zeros(motion.frames)
    for k in range(2):
        f = feet_contact[:, k]
        i = 0
        while i < len(f):
            if not f[i]:
                i += 1
                continue
            j = i
            while j < len(f) and f[j]:
                j += 1
            anchor = feet_xy[k][i]
            drift[i:j] += feet_xy[k][i:j] - anchor
            count[i:j] += 1
            i = j
    has = count > 0
    drift[has] /= count[has, None]
    # Versatz aufsummieren: die Korrektur am Ende eines Schritts bleibt danach bestehen
    correction = np.zeros((motion.frames, 2))
    for t in range(1, motion.frames):
        step = -(drift[t] - drift[t - 1]) if has[t] and has[t - 1] else 0.0
        correction[t] = correction[t - 1] + step
    shift[:, :2] = correction
    exact_z = shift[:, 2].copy()
    shift = _smooth(shift, motion.fps * smooth / 4)
    shift[any_contact, 2] = exact_z[any_contact]  # bei Bodenkontakt genau auf den Boden
    root = motion.root + (shift / pb.hip_height).astype(np.float32)
    fixed = dataclasses.replace(motion, root=root.astype(np.float32), contacts=contacts)
    # Nichts unter den Boden
    lowest = play(model, params, fixed).vertices[..., 2].min(1)
    sink = np.maximum(-lowest, 0.0)
    if sink.max() > 1e-4:
        # erst verbreitern, dann glätten, damit das Glätten keine Spitze wieder im Boden lässt
        wide = np.array([sink[max(0, t - 3):t + 4].max() for t in range(len(sink))])
        lift = np.maximum(_smooth(wide[:, None], 1.0)[:, 0], sink)
        fixed.root[:, 2] += (lift / pb.hip_height).astype(np.float32)
    return fixed


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.5:
        return x
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, c], k, mode="valid") for c in range(x.shape[1])], 1)


# ----------------------------------------------------------------------- Export


def gltf_channels(playback: Playback, parents: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Drehungen (F, J+1, 4) aller Gelenke inklusive Bodengelenk und Lage des Hüftgelenks (F, 3), in
    glTF-Achsen (Y oben, Blick +Z) wie export.build_rig. parents: wie RiggedMesh.joint_parents."""
    from .export import _TO_Y_UP

    to_y = _TO_Y_UP[None, None]
    world = to_y @ playback.bones.astype(np.float64)  # (F, J, 4, 4) ohne Bodengelenk
    f, j = world.shape[:2]
    ground = np.broadcast_to(np.eye(4), (f, 1, 4, 4))
    world = np.concatenate([ground, world], 1)
    local = np.empty_like(world)
    for k, p in enumerate(parents):
        local[:, k] = world[:, k] if p < 0 else np.linalg.inv(world[:, p]) @ world[:, k]
    q = roma.rotmat_to_unitquat(torch.as_tensor(local[..., :3, :3])).numpy()
    q = _continuous(q.astype(np.float64)).astype(np.float32)
    root_index = 1  # Annys „root“ direkt unter dem Bodengelenk
    return q, local[:, root_index, :3, 3].astype(np.float32)


def export_glb(model: BodyModel, params: BodyParams, motions: list[Motion], path: Path,
               preset: str = "anny", skin: np.ndarray | None = None, eyes: np.ndarray | None = None,
               name: str = "FitScan") -> Path:
    """Modell mit Skelett, Haut und den Bewegungen als glTF-Animationen (eine je Bewegung). Hat eine
    Bewegung Mimik, bekommt das Netz die 52 Gesichtsregler als Formregler, und die Animation steuert
    sie mit (glTF „weights“)."""
    from .export import GlbAnimation, build_rig, write_glb
    from .fit import repair_offsets

    if params.offsets is not None:
        params, _ = repair_offsets(model, params)
    rig = build_rig(model, params, preset=preset, skin=skin, eyes=eyes)
    if any(m.face for m in motions):
        from .export import _TO_Y_UP
        from .face_rig import face_morphs

        rest = model.rest_numpy(params)["vertices"]
        for label, delta in face_morphs(model, rest).items():
            rig.morphs[label] = (delta @ _TO_Y_UP[:3, :3].T).astype(np.float32)
    animations = []
    for motion in motions:
        pb = play(model, params, motion, vertices=False)
        rotations, root = gltf_channels(pb, rig.joint_parents)
        animations.append(GlbAnimation(motion.name, np.arange(motion.frames) / motion.fps, rotations, root,
                                       weights=dict(motion.face)))
    return write_glb(rig, path, name=name, animations=animations)


def demo_walk(seconds: float = 3.0, fps: float = 30.0) -> Motion:
    """Einfacher Gang auf der Stelle mit Vorwärtsbewegung, für Tests."""
    n = int(seconds * fps) + 1
    t = np.arange(n) / fps
    phase = 2 * np.pi * 0.9 * t
    s, c = np.sin(phase), np.cos(phase)
    bones = ["upperleg01.L", "upperleg01.R", "lowerleg01.L", "lowerleg01.R",
             "upperarm01.L", "upperarm01.R", "lowerarm01.L", "lowerarm01.R", "spine03"]
    rv = np.zeros((n, len(bones), 3), np.float32)
    rv[:, 0, 0] = -0.45 * s  # Hüfte (negativ um X = Bein nach vorn)
    rv[:, 1, 0] = 0.45 * s
    rv[:, 2, 0] = 0.6 * np.clip(c, 0, None)  # Knie beugt beim Durchschwingen
    rv[:, 3, 0] = 0.6 * np.clip(-c, 0, None)
    rv[:, 4, 0] = 0.35 * s  # Arme gegengleich
    rv[:, 5, 0] = -0.35 * s
    rv[:, 6, 0] = -0.3
    rv[:, 7, 0] = -0.3
    rv[:, 8, 2] = 0.1 * s
    # Hüfte in Hüfthöhen: etwa 1,1 m/s nach vorn (-Y), leichtes Auf und Ab
    root = np.stack([np.zeros(n), -1.2 * t, 0.98 - 0.015 * np.cos(2 * phase)], 1)
    return from_rotvecs("Testgang", fps, bones, rv, root, source="künstlich")


def turned(motion: Motion, degrees: float) -> Motion:
    """Ganze Bewegung um die Hochachse drehen (Blick- und Laufrichtung)."""
    rz = roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, np.radians(degrees)], dtype=torch.float32))
    bones = list(motion.bones)
    rotations = motion.rotations
    if "root" not in bones:
        bones = ["root", *bones]
        ident = np.zeros((motion.frames, 1, 4), np.float32)
        ident[..., 3] = 1.0
        rotations = np.concatenate([ident, rotations], 1)
    k = bones.index("root")
    rotations = rotations.copy()
    root_rot = rz[None] @ roma.unitquat_to_rotmat(torch.as_tensor(rotations[:, k]))
    rotations[:, k] = _continuous(roma.rotmat_to_unitquat(root_rot).numpy()[:, None])[:, 0]
    root = (torch.as_tensor(motion.root) @ rz.T).numpy().astype(np.float32)
    return dataclasses.replace(motion, bones=bones, rotations=rotations, root=root)
