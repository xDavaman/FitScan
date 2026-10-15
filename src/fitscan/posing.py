"""Haltungen in verständlichen Gelenkwinkeln beschreiben und in Annys Drehungen umrechnen.

Eine Haltung ist ein dict (Knochen, Richtung) -> Grad. Die Richtungen beziehen sich auf den Körper
und werden vom Elternknochen mitgedreht:
    "runter": Knochen zum Boden schwenken (Arme an den Körper, Fuß strecken)
    "aussen": von der Körpermitte weg
    "links":  zur linken Körperseite (Wirbelsäule, Kopf seitlich)
    "vor":    nach vorn schwenken (Hüfte beugen, Arm heben, Oberkörper beugen; negativ = nach hinten)
    "spreizen": wie "aussen", aber nach "vor" (Knie bei gebeugter Hüfte nach außen)
    "dreh":   um die eigene Längsachse drehen
    "beugen": Ellbogen und Knie um ihre Scharnierachse, als Gesamtwinkel (0 = gestreckt)
Der ganze Körper kippt über ("root", "pitch"/"yaw"/"roll").

`Poser.rotations` liefert je Knochen die Drehung in Annys „local-ref“ (Weltachsen relativ zur
Bezugshaltung), wie sie `motion.Motion.rotations` speichert. Die Knochennamen sind Annys
(z. B. "upperarm01.L"); das Namensschema für Cascadeur/Unreal kommt erst beim Export.

Anny: Z oben, die Person blickt nach -Y, ihre linke Seite ist +X.
"""

from __future__ import annotations

import numpy as np

from .grip import rotation, to_quaternion

ORDER = ("beugen", "runter", "aussen", "links", "vor", "spreizen", "dreh")
HINGES = {"lowerarm01": "upperarm01", "lowerleg01": "upperleg01"}  # Gelenk -> Knochen davor
SIDES = ("L", "R")

# Richtung eines Knochens = von seinem Gelenk zum Gelenk des nächsten Glieds
_NEXT = {"upperleg01": "lowerleg01", "upperleg02": "lowerleg01", "lowerleg01": "foot",
         "lowerleg02": "foot", "foot": "toe3-1", "upperarm01": "lowerarm01", "upperarm02": "lowerarm01",
         "lowerarm01": "wrist", "lowerarm02": "wrist", "wrist": "finger3-1", "clavicle": "upperarm01",
         "shoulder01": "upperarm01", "spine05": "spine04", "spine04": "spine03", "spine03": "spine02",
         "spine02": "spine01", "spine01": "neck01", "neck01": "neck02", "neck02": "neck03",
         "neck03": "head"}


class Poser:
    def __init__(self, model, heads: np.ndarray):
        """heads: Gelenkpunkte in der Bezugshaltung (grip.reference_pose(...)["heads"])."""
        self.names = list(model.bone_labels)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.dir: dict[str, np.ndarray] = {}
        for n in self.names:
            base, _, side = n.partition(".")
            target = _NEXT.get(base)
            other = f"{target}.{side}" if side else target
            if other in self.index:
                d = heads[self.index[other]] - heads[self.index[n]]
            else:
                d = np.array([0.0, 0.0, 1.0])
            self.dir[n] = d / np.linalg.norm(d)
        # Scharniere: Achse und Beugung in der Bezugshaltung (Annys A-Pose hat gebeugte Ellbogen)
        self.hinge: dict[str, tuple[np.ndarray, float]] = {}
        for n in self.names:
            base, _, side = n.partition(".")
            if base in HINGES:
                a, b = self.dir[f"{HINGES[base]}.{side}"], self.dir[n]
                axis = np.cross(a, b)
                angle = float(np.degrees(np.arctan2(np.linalg.norm(axis), a @ b)))
                self.hinge[n] = (axis / max(np.linalg.norm(axis), 1e-9), angle)

    @staticmethod
    def toward(name: str, direction: str) -> np.ndarray:
        side = 1.0 if name.endswith(".L") else -1.0 if name.endswith(".R") else 0.0
        return {"runter": np.array([0, 0, -1.0]), "vor": np.array([0, -1.0, 0]),
                "aussen": np.array([side, 0, 0]), "spreizen": np.array([side, 0, 0]), "links": np.array([1.0, 0, 0])}[direction]

    def bone_rotation(self, name: str, values: dict[str, float]) -> np.ndarray:
        if name == "root":
            return (rotation([0, 0, 1], values.get("yaw", 0.0)) @ rotation([1, 0, 0], values.get("pitch", 0.0))
                    @ rotation([0, 1, 0], values.get("roll", 0.0)))
        delta = np.eye(3)
        d0 = self.dir[name]
        for direction in ORDER:
            if direction not in values:
                continue
            angle = values[direction]
            if abs(angle) < 1e-9 and direction != "beugen":
                continue
            cur = delta @ d0
            if direction == "beugen":
                axis, ref = self.hinge[name]
                delta = rotation(delta @ axis, angle - ref) @ delta
                continue
            if direction == "dreh":
                axis = cur
            else:
                axis = np.cross(cur, self.toward(name, direction))
                if np.linalg.norm(axis) < 1e-9:
                    continue
            delta = rotation(axis, angle) @ delta
        return delta

    def rotations(self, pose: dict) -> dict[str, np.ndarray]:
        per_bone: dict[str, dict[str, float]] = {}
        for (bone, direction), value in pose.items():
            per_bone.setdefault(bone, {})[direction] = value
        return {b: self.bone_rotation(b, v) for b, v in per_bone.items()}

    def quaternions(self, rotations: dict[str, np.ndarray]) -> np.ndarray:
        """(B, 4) xyzw für alle Knochen, nicht genannte bleiben ungedreht."""
        out = np.zeros((len(self.names), 4))
        out[:, 3] = 1.0
        for name, r in rotations.items():
            out[self.index[name]] = to_quaternion(r)
        return out

    def matrices(self, rotations: dict[str, np.ndarray]) -> np.ndarray:
        out = np.tile(np.eye(4), (len(self.names), 1, 1))
        for name, r in rotations.items():
            out[self.index[name], :3, :3] = r
        return out


# --------------------------------------------------------------------------- Bausteine


def arm(pose: dict, side: str, down: float = 40, fwd: float = 0, elbow: float = 10, out: float = 0,
        twist: float = 0, forearm: float = 0, wrist: float = 0, wrist_side: float = 0) -> dict:
    """down: Arm aus der Bezugshaltung zum Körper; fwd: nach vorn heben; elbow: Ellbogenwinkel
    (0 = gestreckt);
    twist: Oberarm drehen; forearm: Unterarm drehen (Handfläche nach hinten/vorn); wrist: Hand zur
    Handfläche hin beugen; wrist_side: Hand zur Daumenseite kippen."""
    pose[(f"upperarm01.{side}", "runter")] = down
    pose[(f"upperarm01.{side}", "aussen")] = out
    pose[(f"upperarm01.{side}", "vor")] = fwd
    pose[(f"upperarm01.{side}", "dreh")] = twist
    pose[(f"lowerarm01.{side}", "beugen")] = elbow
    pose[(f"lowerarm02.{side}", "dreh")] = forearm
    pose[(f"wrist.{side}", "vor")] = wrist
    pose[(f"wrist.{side}", "aussen")] = wrist_side
    return pose


def shoulder(pose: dict, side: str, up: float = 0, fwd: float = 0) -> dict:
    """Schulterblatt: hochziehen (Klimmzug, Kreuzheben) und nach vorn schieben."""
    pose[(f"clavicle.{side}", "runter")] = -up
    pose[(f"clavicle.{side}", "vor")] = fwd
    return pose


def leg(pose: dict, side: str, hip: float = 0, knee: float = 0, foot: float = 0, out: float = 0,
        turn: float = 0) -> dict:
    pose[(f"upperleg01.{side}", "vor")] = hip
    pose[(f"upperleg01.{side}", "spreizen")] = out
    pose[(f"upperleg01.{side}", "dreh")] = turn
    pose[(f"lowerleg01.{side}", "beugen")] = knee
    pose[(f"foot.{side}", "runter")] = foot
    return pose


SPINE = ("spine05", "spine04", "spine03", "spine02", "spine01")  # von unten nach oben


def spine(pose: dict, bend: float = 0, neck: float = 0, twist: float = 0, side: float = 0,
          turn: float = 0) -> dict:
    """bend verteilt sich auf die fünf Wirbel, neck auf Hals und Kopf (positiv = Kinn zur Brust),
    turn dreht den Kopf zur Seite."""
    for bone, share in zip(SPINE, (0.15, 0.2, 0.25, 0.2, 0.2), strict=True):
        pose[(bone, "vor")] = bend * share
        pose[(bone, "dreh")] = twist * share
        pose[(bone, "links")] = side * share
    pose[("neck01", "vor")] = neck * 0.4
    pose[("head", "vor")] = neck * 0.6
    pose[("neck01", "dreh")] = turn * 0.4
    pose[("head", "dreh")] = turn * 0.6
    return pose


def body(pose: dict, pitch: float = 0, yaw: float = 0, roll: float = 0) -> dict:
    """Ganzer Körper um die Hüfte: pitch nach vorn kippen, yaw drehen, roll seitlich."""
    pose[("root", "pitch")] = pitch
    pose[("root", "yaw")] = yaw
    pose[("root", "roll")] = roll
    return pose


def stand() -> dict:
    p: dict = {}
    for s in SIDES:
        arm(p, s, down=32, fwd=5, elbow=12)
        leg(p, s, hip=2, knee=3)
    spine(p)
    return body(p)


def pose(base: dict | None = None, **parts) -> dict:
    """pose(arms=dict(...), legs=dict(...), arm_L=..., leg_R=..., spine=..., body=..., shoulders=...)."""
    p = dict(stand() if base is None else base)
    for key, value in parts.items():
        if key == "arms":
            for s in SIDES:
                arm(p, s, **value)
        elif key == "legs":
            for s in SIDES:
                leg(p, s, **value)
        elif key == "shoulders":
            for s in SIDES:
                shoulder(p, s, **value)
        elif key in ("arm_L", "arm_R"):
            arm(p, key[-1], **value)
        elif key in ("leg_L", "leg_R"):
            leg(p, key[-1], **value)
        elif key == "spine":
            spine(p, **value)
        elif key == "body":
            body(p, **value)
        else:
            raise KeyError(key)
    return p


def pchip(times: np.ndarray, values: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Monoton kubische Interpolation (kein Überschwingen zwischen Schlüsselhaltungen)."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(times) == 1:
        return np.full_like(t, values[0], dtype=np.float64)
    h = np.diff(times)
    delta = np.diff(values) / h
    m = np.zeros_like(values)
    for k in range(1, len(values) - 1):
        if delta[k - 1] * delta[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    t = np.clip(t, times[0], times[-1])
    k = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
    s = (t - times[k]) / h[k]
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * values[k] + h10 * h[k] * m[k] + h01 * values[k + 1] + h11 * h[k] * m[k + 1]


def sample(keys: list[tuple[float, dict]], t: np.ndarray) -> list[dict]:
    """Schlüsselhaltungen (Zeit, Haltung) -> Haltung je Zeitpunkt."""
    channels = sorted({c for _, p in keys for c in p})
    times = np.array([k for k, _ in keys])
    curves = {c: pchip(times, np.array([p.get(c, 0.0) for _, p in keys]), t) for c in channels}
    return [{c: float(curves[c][i]) for c in channels} for i in range(len(t))]


# --------------------------------------------------------------------------- Figur


class Figure:
    """Ein Körper zum Abspielen von Haltungen: Ruhepunkte und Ruhe-Knochenlagen (Anny-Koordinaten).
    Gelenke drehen mit Annys Kinematik und Skinning, also genau wie beim Export."""

    def __init__(self, model, vertices: np.ndarray, bone_poses: np.ndarray):
        import torch

        self.model = model
        self._rest_v = torch.as_tensor(vertices, dtype=torch.float32)[None]
        self._rest_p = torch.as_tensor(bone_poses, dtype=torch.float32)[None]
        ref = self.pose(np.tile(np.eye(4), (1, len(model.bone_labels), 1, 1)))
        self.reference_vertices = ref["vertices"][0]
        self.reference_poses = ref["poses"][0]
        self.poser = Poser(model, self.reference_poses[:, :3, 3])

    @classmethod
    def from_params(cls, model, params) -> Figure:
        rest = model.rest_numpy(params)
        return cls(model, rest["vertices"], rest["bone_poses"])

    @classmethod
    def from_glb(cls, model, path) -> Figure:
        """Körper aus einem FitScan-GLB (Anny-Netz, Skelett mit Bodengelenk)."""
        from .animation import accessor, glb_to_anny, read_glb

        gltf, blob = read_glb(path)
        _, verts = glb_to_anny(gltf, blob, model)
        skin = gltf["skins"][0]
        ibm = accessor(gltf, blob, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
        names = [gltf["nodes"][j]["name"] for j in skin["joints"]]
        from .export import bone_name

        preset = "unreal" if "pelvis" in names and "root" in names else "anny"
        lookup = {bone_name(n, preset): k for k, n in enumerate(names)}
        to_z = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
        poses = np.stack([to_z @ np.linalg.inv(ibm[lookup[n]]) for n in model.bone_labels])
        return cls(model, verts, poses)

    def pose(self, frames: list[dict] | np.ndarray, batch: int = 64) -> dict[str, np.ndarray]:
        """frames: Liste von Drehungen je Knochen (dict Name -> 3x3, local-ref) oder (F, J, 4, 4).
        Rückgabe: vertices (F, V, 3), poses (F, J, 4, 4) Weltlage der Knochen; die Wurzel liegt im
        Ursprung, die Verschiebung über dem Boden kommt getrennt dazu."""
        import torch

        if not isinstance(frames, np.ndarray):
            frames = np.stack([self.poser.matrices(f) for f in frames]) if frames else np.zeros((0, 0, 4, 4))
        verts, poses = [], []
        anny = self.model.anny
        with torch.no_grad():
            for i in range(0, len(frames), batch):
                p = torch.as_tensor(frames[i:i + batch], dtype=torch.float32)
                transforms, bone_poses = anny.get_bone_transforms(p, self._rest_p)
                v = anny.get_skinned_vertices(self._rest_v.expand(len(p), -1, -1), transforms)
                verts.append(v.double().numpy())
                poses.append(bone_poses.double().numpy())
        return {"vertices": np.concatenate(verts), "poses": np.concatenate(poses)}
