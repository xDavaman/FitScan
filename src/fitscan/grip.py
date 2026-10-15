"""Greifen: Finger schließen sich um ein Griffstück, bis die Haut es berührt.

Ein Griff gehört zu einer Hand und einem runden Griffstück (Langhantelstange, Kurzhantelgriff,
Klimmzugstange), beschrieben durch seinen Radius. Ablauf:
1. Handfläche bestimmen: Richtung quer über die Fingerwurzeln (Zeige- zu Kleinfinger), Richtung
   der Finger und die Handflächenseite (dorthin zeigen Daumen und Fingerbeugung).
2. Das Griffstück liegt quer in der Hand an der Fingerwurzel und berührt die Haut der Handfläche.
3. Alle Fingergelenke schließen sich gleichzeitig wie bei einer echten Faust. Berührt ein Glied
   die Stange, bleiben dieses Gelenk und die Gelenke davor stehen, die äußeren schließen sich
   weiter, bis auch die Fingerkuppe anliegt.
4. Der Daumen legt sich von der anderen Seite über die Stange auf Zeige- und Mittelfinger.

Geprüft wird an der Haut selbst: Die Handpunkte werden wie beim Abspielen über Annys
Knochengewichte verformt (lineares Skinning). So passt der Griff zu jeder gescannten Hand.

Ergebnis sind Drehungen der Fingerknochen in Annys „local-ref“ (Weltachsen relativ zur
Bezugshaltung, wie `motion.Motion.rotations`) und die Lage des Griffstücks im Handgelenk-Knochen
(wie `motion.Attachment.offset`). Beides gilt für jede Arm- und Handhaltung, weil die Finger relativ
zur Hand gedreht werden.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

FINGERS = ("finger2", "finger3", "finger4", "finger5")  # Zeige-, Mittel-, Ring-, Kleinfinger
THUMB = "finger1"
MAX_BEND = (90.0, 110.0, 80.0)  # Gelenkgrenzen Grund-, Mittel-, Endgelenk (Grad)
RATIO = (1.0, 1.1, 0.75)  # wie schnell sich die Gelenke beim Schließen relativ zueinander beugen
THUMB_TURN = 60.0  # so weit dreht das Grundgelenk den Daumen zum Gegengreifen ein
THUMB_MAX = (40.0, 60.0, 80.0)
THUMB_RATIO = (0.5, 1.0, 1.0)
STEP = 1.0  # Grad je Schritt
CONTACT = 0.0008  # so nah darf die Haut an die Oberfläche (Meter), weiche Haut gibt etwas nach


def rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    a = math.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3)
    return rotation(axis, math.degrees(math.atan2(s, float(a @ b))))


def angle_of(r: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2))))


def clamp_rotation(r: np.ndarray, limit: float) -> np.ndarray:
    angle = angle_of(r)
    if angle <= limit:
        return r
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return rotation(axis, limit) if np.linalg.norm(axis) > 1e-9 else r


def to_quaternion(r: np.ndarray) -> np.ndarray:
    """3x3 -> Quaternion xyzw."""
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2
    y = math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2
    z = math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2
    x = math.copysign(x, r[2, 1] - r[1, 2])
    y = math.copysign(y, r[0, 2] - r[2, 0])
    z = math.copysign(z, r[1, 0] - r[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def axis_distance(points: np.ndarray, centre: np.ndarray, axis: np.ndarray) -> np.ndarray:
    q = points - centre
    return np.linalg.norm(q - np.outer(q @ axis, axis), axis=1)


@torch.no_grad()
def reference_pose(model, params) -> dict[str, np.ndarray]:
    """Punkte, Gelenkpunkte und Knochenlagen in Annys Bezugshaltung (alle Drehungen 0)."""
    rest = model.rest(model.phenotype_tensor(params), model.local_tensor(params),
                      None if params.offsets is None else torch.as_tensor(params.offsets, dtype=torch.float32),
                      params.scale)
    posed = model.posed(rest, None, params.scale)
    _, bone_poses = model.anny.get_bone_transforms(model._identity_pose, rest["_unscaled_bone_poses"])
    poses = bone_poses[0].double().cpu().numpy()
    poses[:, :3, 3] *= params.scale
    return {"vertices": posed["vertices"][0].double().cpu().numpy(),
            "heads": posed["bone_heads"][0].double().cpu().numpy(), "poses": poses}


class Hand:
    """Eine Hand in der Bezugshaltung: Gelenkpunkte und Hautpunkte mit Gewichten."""

    def __init__(self, model, params, side: str, ref: dict[str, np.ndarray] | None = None):
        ref = ref or reference_pose(model, params)
        self.side = side
        self.wrist_name = f"wrist.{side}"
        self.names = [self.wrist_name] + [f"metacarpal{i}.{side}" for i in range(1, 5)]
        for f in (THUMB, *FINGERS):
            self.names += [f"{f}-{k}.{side}" for k in (1, 2, 3)]
        bone = {n: model.bone(n) for n in self.names}
        self.head = {n: ref["heads"][bone[n]] for n in self.names}
        self.wrist_pose = ref["poses"][bone[self.wrist_name]]

        cols = {bone[n]: j for j, n in enumerate(self.names)}
        idx, w = model.vertex_bone_indices, model.vertex_bone_weights
        inside = np.isin(idx, list(cols)) & (w > 0)
        share = np.where(inside, w, 0).sum(1)
        keep = share > 0.3 * w.sum(1)
        self.ids = np.flatnonzero(keep)
        self.rest = ref["vertices"][self.ids]
        self.weights = np.zeros((len(self.ids), len(self.names)))
        for k, v in enumerate(self.ids):
            total = w[v].sum()
            for b, wb in zip(idx[v], w[v], strict=True):
                if wb > 0 and int(b) in cols:
                    self.weights[k, cols[int(b)]] += wb / total
        self.owner = self.weights.argmax(1)
        local = np.full(model.num_vertices, -1)
        local[self.ids] = np.arange(len(self.ids))
        f = local[model.faces]
        self.faces = f[(f >= 0).all(1)]

        # Endglieder haben keinen Kindknochen: Spitze = am weitesten außen liegender Hautpunkt
        self.tip = {}
        for f in (THUMB, *FINGERS):
            a, b = self.head[f"{f}-2.{side}"], self.head[f"{f}-3.{side}"]
            d = (b - a) / np.linalg.norm(b - a)
            pts = self.rest[self.mask([f"{f}-3.{side}"])]
            self.tip[f] = b + d * float(((pts - b) @ d).max())

        knuckles = np.array([self.head[f"{f}-1.{side}"] for f in FINGERS])
        self.knuckles = knuckles
        self.across = (knuckles[3] - knuckles[0]) / np.linalg.norm(knuckles[3] - knuckles[0])
        wrist = self.head[self.wrist_name]
        along = self.tip["finger3"] - wrist
        along = along - self.across * (along @ self.across)
        self.along = along / np.linalg.norm(along)
        normal = np.cross(self.along, self.across)
        probe = (self.tip["finger2"] + self.tip["finger3"] + self.tip[THUMB]) / 3 - knuckles.mean(0)
        self.palm = normal if probe @ normal > 0 else -normal

    def normals(self, points: np.ndarray) -> np.ndarray:
        tri = points[self.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        out = np.zeros_like(points)
        for k in range(3):
            np.add.at(out, self.faces[:, k], n)
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)

    def mask(self, bones: list[str]) -> np.ndarray:
        return np.isin(self.owner, [self.names.index(b) for b in bones])

    def chain(self, finger: str) -> tuple[list[str], list[np.ndarray]]:
        names = [f"{finger}-{k}.{self.side}" for k in (1, 2, 3)]
        return names, [self.head[n] for n in names] + [self.tip[finger]]

    def skin(self, world: dict[str, np.ndarray]) -> np.ndarray:
        """Lineares Skinning der Handpunkte. world: Knochen -> Weltdrehung relativ zur Bezugshaltung
        (das Produkt der local-ref-Drehungen entlang der Kette); fehlende Knochen bleiben stehen."""
        out = self.rest * (1.0 - self.weights.sum(1, keepdims=True))
        moved = self._joint_positions(world)
        for j, name in enumerate(self.names):
            w = self.weights[:, j]
            if not w.any():
                continue
            r = world.get(name)
            pts = self.rest if r is None else (self.rest - self.head[name]) @ r.T + moved[name]
            out += w[:, None] * pts
        return out

    def _joint_positions(self, world: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for f in (THUMB, *FINGERS):
            names, pts = self.chain(f)
            p = pts[0]
            for k, n in enumerate(names):
                if k > 0:
                    r = world.get(names[k - 1], np.eye(3))
                    p = p + r @ (pts[k] - pts[k - 1])
                out[n] = p
        return out


@dataclasses.dataclass
class Grip:
    side: str
    radius: float
    rotations: dict[str, np.ndarray]  # local-ref-Drehung (3x3) je Fingerknochen
    offset: np.ndarray  # Lage des Griffstücks (x = Stangenachse, z = zur Handfläche hinaus) im Handgelenk-Knochen
    angles: dict[str, tuple[float, ...]]
    gap: dict[str, float]  # kleinster Abstand Haut–Oberfläche je Finger in Metern (negativ = Durchdringung)

    def quaternions(self) -> dict[str, np.ndarray]:
        return {n: to_quaternion(r) for n, r in self.rotations.items()}

    def blend(self, amount: float) -> dict[str, np.ndarray]:
        """Teilweise geschlossene Hand (0 = offen, 1 = Griff), z. B. beim Zugreifen."""
        out = {}
        for n, r in self.rotations.items():
            angle = angle_of(r)
            if angle < 1e-6:
                out[n] = np.eye(3)
                continue
            axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
            out[n] = rotation(axis, angle * amount)
        return out


def _world(rots: list[np.ndarray]) -> list[np.ndarray]:
    out, acc = [], np.eye(3)
    for r in rots:
        acc = acc @ r
        out.append(acc)
    return out


def _close(hand: Hand, names, hinges, centre, axis, radius, others, limits=MAX_BEND, ratio=RATIO,
           pre: np.ndarray | None = None, obstacle: tuple | None = None):
    """Gelenke einer Kette gleichzeitig schließen, jedes Glied hält bei Berührung an.
    pre: feste Vordrehung des ersten Gelenks (Daumen zum Gegengreifen eingedreht).
    obstacle: (Punkte, Normalen) einer Haut, in die die Kette nicht eindringen darf (Finger für den Daumen)."""
    masks = [hand.mask([n]) for n in names]
    if obstacle is not None:
        from scipy.spatial import cKDTree

        tree = cKDTree(obstacle[0])
    pre = np.eye(3) if pre is None else pre
    bend = [0.0] * 3
    frozen = [False] * 3

    def rots(angles):
        out = [rotation(h, a) for a, h in zip(angles, hinges, strict=True)]
        out[0] = out[0] @ pre
        return out

    def gaps(angles):
        world = dict(others)
        world.update(zip(names, _world(rots(angles)), strict=True))
        pts = hand.skin(world)
        d = axis_distance(pts, centre, axis) - radius
        if obstacle is not None:
            dist, k = tree.query(pts)
            signed = np.einsum("ij,ij->i", pts - obstacle[0][k], obstacle[1][k]) + 0.002
            d = np.minimum(d, np.where(dist < 0.015, signed, 1.0))
        return [float(d[m].min()) if m.any() else 1.0 for m in masks]

    current = gaps(bend)
    while not all(frozen):
        trial = [b if f else min(lim, b + STEP * r)
                 for b, f, lim, r in zip(bend, frozen, limits, ratio, strict=True)]
        g = gaps(trial)
        hit = [g[k] < CONTACT and g[k] < current[k] - 1e-6 for k in range(3)]
        if any(hit):
            for k in range(max(k for k in range(3) if hit[k]) + 1):
                frozen[k] = True
            continue
        bend, current = trial, g
        frozen = [f or b >= lim for f, b, lim in zip(frozen, bend, limits, strict=True)]
    return bend, rots(bend), current


def solve_grip(hand: Hand, radius: float, depth: float = 0.2) -> Grip:
    """Finger um ein Griffstück mit radius (Meter) schließen. depth: Lage des Griffstücks von den
    Fingerwurzeln (0) zur Handwurzel (1); 0,2 ist ein normaler Kraftgriff an der Fingerwurzel."""
    side = hand.side
    axis = hand.across
    span = float(np.linalg.norm(hand.knuckles[3] - hand.knuckles[0]))
    start = hand.knuckles.mean(0)
    start = start + (hand.head[hand.wrist_name] - start) * depth

    # Stange so weit vor die Handfläche setzen, dass sie die Haut gerade berührt
    flat = hand.mask([hand.wrist_name] + [f"metacarpal{i}.{side}" for i in range(1, 5)])
    near = np.abs((hand.rest - start) @ axis) < 0.6 * span + 0.01
    pts = hand.rest[flat & near]
    h = 0.0
    while h < 0.08 and (axis_distance(pts, start + hand.palm * h, axis) - radius).min() < 0:
        h += 0.0005
    centre = start + hand.palm * h

    rotations: dict[str, np.ndarray] = {}
    angles: dict[str, tuple[float, ...]] = {}
    gap: dict[str, float] = {}
    world: dict[str, np.ndarray] = {}
    for finger in FINGERS:
        names, pts = hand.chain(finger)
        hinges = []
        for k in range(3):
            v = pts[k + 1] - pts[k]
            hinge = np.cross(v / np.linalg.norm(v), hand.palm)
            hinges.append(hinge / np.linalg.norm(hinge))
        bend, rots, g = _close(hand, names, hinges, centre, axis, radius, world)
        rotations.update(zip(names, rots, strict=True))
        world.update(zip(names, _world(rots), strict=True))
        angles[finger] = tuple(bend)
        gap[finger] = min(g)

    # Daumen: das Grundgelenk dreht ihn in die Ebene quer zur Stange, dann schließen sich alle drei
    # Gelenke um die Stange (Drehachse = Stange), bis er anliegt
    closed = hand.skin(world)
    fingers = hand.mask([f"{f}-{k}.{side}" for f in FINGERS for k in (1, 2, 3)])
    obstacle = (closed[fingers], hand.normals(closed)[fingers])
    names, pts = hand.chain(THUMB)
    d = pts[3] - pts[0]
    full = rotation_between(d, d - axis * (d @ axis))
    sign = 1.0
    r = rotation(axis, 10)
    if axis_distance((pts[0] + r @ full @ d)[None], centre, axis)[0] > axis_distance(
            (pts[0] + full @ d)[None], centre, axis)[0]:
        sign = -1.0  # Drehsinn, in dem sich der Daumen um die Stange schließt
    hinges = [axis * sign] * 3
    # Zielstelle: Daumenkuppe auf Zeige- und Mittelfinger, außen auf der Stange
    target = centre + hand.palm * (radius + 0.012) - axis * (0.3 * span)
    best = None
    for turn in np.linspace(0, 1, 7):
        for opening in range(0, 91, 10):
            pre = rotation(-hinges[0], opening) @ clamp_rotation(
                rotation_between(d, (1 - turn) * d + turn * (full @ d)), THUMB_TURN)
            bend, rots, g = _close(hand, names, hinges, centre, axis, radius, world, THUMB_MAX, THUMB_RATIO, pre,
                                   obstacle)
            if min(g) < -CONTACT:
                continue
            w = _world(rots)
            p1 = pts[0] + w[0] @ (pts[1] - pts[0])
            p2 = p1 + w[1] @ (pts[2] - pts[1])
            tip = p2 + w[2] @ (pts[3] - pts[2])
            score = float(np.linalg.norm(tip - target))
            if best is None or score < best[0]:
                best = (score, bend, rots, g, pre)
    if best is None:  # kein Griff ohne Durchdringung: Daumen bleibt offen neben der Stange
        bend, rots, g, pre = [0.0] * 3, [np.eye(3)] * 3, [0.0], np.eye(3)
    else:
        _, bend, rots, g, pre = best
    rotations.update(zip(names, rots, strict=True))
    angles[THUMB] = (round(angle_of(pre), 1), *bend)
    gap[THUMB] = min(g)

    bar = np.eye(4)
    bar[:3, 0] = axis
    bar[:3, 2] = hand.palm
    bar[:3, 1] = np.cross(hand.palm, axis)
    bar[:3, 3] = centre
    return Grip(side=side, radius=radius, rotations=rotations, offset=np.linalg.inv(hand.wrist_pose) @ bar,
                angles=angles, gap=gap)
