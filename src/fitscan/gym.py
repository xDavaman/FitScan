"""Übungen im Studio: Figur greift Geräte und bewegt sie, ohne dass etwas ineinander steckt.

Bausteine:
- Geräte als einfache Formen (Stange, Scheiben, Griff) für Anzeige und Durchdringungsprüfung.
- `orient_grip`: dreht Unterarm und Hand so, dass das Griffstück waagerecht in der gewünschten
  Richtung liegt (Langhantel quer vor dem Körper, Klimmzugstange über dem Kopf). Der größte Teil
  der Drehung geht in den Unterarm (Pronation/Supination), nur der Rest ins Handgelenk.
- Bodenkontakt: Füße bleiben stehen und stehen auf dem Boden.
- Übungen: Kreuzheben (Langhantel), Bizeps-Curl (Kurzhanteln), Klimmzug.

Ergebnis je Übung: Drehungen je Bild (local-ref, Anny-Namen), Wurzelposition je Bild,
Kontakte und die gehaltenen Geräte mit ihrer Lage im Handgelenk-Knochen, also alles, was die
Bewegungsbibliothek (`motion.Motion`, `motion.Attachment`) speichert.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from . import contact
from . import grip as G
from . import posing as P

FPS = 30
BAR_RADIUS = 0.014  # Langhantel, 28 mm
PLATE_RADIUS = 0.225  # 45-cm-Scheibe
DUMBBELL_RADIUS = 0.0165
PULLUP_RADIUS = 0.016
WRIST_LIMIT = 35.0  # so viel Restdrehung darf das Handgelenk übernehmen


@dataclasses.dataclass
class Prop:
    """Ein Gerät: Formen in eigenen Koordinaten (x = Stangenachse, Ursprung = Mitte)."""

    name: str
    shapes: list  # contact.Cylinder / contact.Box in Gerätekoordinaten

    def placed(self, matrix: np.ndarray) -> list:
        return [s.moved(matrix) for s in self.shapes]


def barbell(width: float = 2.2) -> Prop:
    shapes = [contact.Cylinder("Stange", np.zeros(3), np.array([1.0, 0, 0]), BAR_RADIUS, width / 2)]
    for sign in (-1, 1):
        shapes.append(contact.Cylinder("Scheibe", np.array([sign * 0.62, 0, 0]), np.array([1.0, 0, 0]),
                                       PLATE_RADIUS, 0.03))
    return Prop("Langhantel", shapes)


def dumbbell() -> Prop:
    shapes = [contact.Cylinder("Griff", np.zeros(3), np.array([1.0, 0, 0]), DUMBBELL_RADIUS, 0.09)]
    for sign in (-1, 1):
        shapes.append(contact.Cylinder("Gewicht", np.array([sign * 0.085, 0, 0]), np.array([1.0, 0, 0]),
                                       0.06, 0.025))
    return Prop("Kurzhantel", shapes)


def pullup_bar() -> Prop:
    return Prop("Klimmzugstange", [contact.Cylinder("Stange", np.zeros(3), np.array([1.0, 0, 0]),
                                                    PULLUP_RADIUS, 0.6)])


# --------------------------------------------------------------------------- Hände ausrichten


def _world_delta(poses: np.ndarray, ref: np.ndarray, bone: int) -> np.ndarray:
    return poses[bone, :3, :3] @ ref[bone, :3, :3].T


def _apply_world(rots: dict, name: str, parent_world: np.ndarray, w: np.ndarray) -> None:
    """Zusätzliche Weltdrehung w auf einen Knochen geben (local-ref: G' = w G)."""
    rots[name] = parent_world.T @ w @ parent_world @ rots.get(name, np.eye(3))


def orient_grip(fig: P.Figure, rots: dict, poses: np.ndarray, side: str, grip: G.Grip,
                palm: np.ndarray, axis: np.ndarray | None = None) -> float:
    """Unterarm und Hand so drehen, dass die Griffstückachse parallel zu axis liegt und die
    Handfläche in Richtung palm zeigt (palm = vom Handteller zum Griffstück). Ändert rots, gibt die
    Restdrehung im Handgelenk in Grad zurück."""
    axis = np.array([1.0, 0, 0]) if axis is None else axis
    model = fig.model
    ref = fig.reference_poses
    w_bone, f_bone, e_bone = (model.bone(f"{n}.{side}") for n in ("wrist", "lowerarm02", "lowerarm01"))
    current = (poses[w_bone] @ grip.offset)[:3, :3]
    z = palm - axis * (palm @ axis)
    z = z / np.linalg.norm(z)
    # beide Richtungen der Achse sind ein gültiger Griff; die mit der kleineren Drehung gewinnt
    needs = [np.stack([x, np.cross(z, x), z], 1) @ current.T for x in (axis, -axis)]
    need = min(needs, key=G.angle_of)  # Weltdrehung, die das Griffstück richtig legt

    # Unterarm dreht um seine Längsachse, durch das Handgelenk
    forearm_axis = poses[w_bone, :3, 3] - poses[f_bone, :3, 3]
    forearm_axis /= np.linalg.norm(forearm_axis)
    best = min(range(-180, 181), key=lambda a: G.angle_of(need @ G.rotation(forearm_axis, a).T))
    twist = G.rotation(forearm_axis, best)
    g_elbow = _world_delta(poses, ref, e_bone)
    _apply_world(rots, f"lowerarm02.{side}", g_elbow, twist)
    g_forearm = twist @ _world_delta(poses, ref, f_bone)
    rest = need @ twist.T  # erst Unterarm, dann Handgelenk: need = rest @ twist
    _apply_world(rots, f"wrist.{side}", g_forearm, rest)
    return G.angle_of(rest)


def hold(fig: P.Figure, frames: list[dict], grips: dict, palms: dict, axis: np.ndarray | None = None,
         amounts: np.ndarray | None = None) -> tuple[list[dict], np.ndarray]:
    """Für jedes Bild Finger schließen (amount 0..1) und Hände ausrichten. Gibt neue Drehungen und
    die größte Restdrehung im Handgelenk je Bild zurück."""
    out = [dict(f) for f in frames]
    for i, f in enumerate(out):
        a = 1.0 if amounts is None else float(amounts[i])
        for g in grips.values():
            f.update(g.blend(a))
    poses = fig.pose(out)["poses"]
    wrist = np.zeros(len(out))
    for i, f in enumerate(out):
        for s, g in grips.items():
            wrist[i] = max(wrist[i], orient_grip(fig, f, poses[i], s, g, palms[s], axis))
    return out, wrist


# --------------------------------------------------------------------------- Boden


def plant_feet(fig: P.Figure, vertices: np.ndarray) -> np.ndarray:
    """Wurzelverschiebung je Bild: Fußsohlen bleiben an ihrer Stelle aus Bild 0 und stehen auf z = 0."""
    sole = fig.model.dominant_bone_mask(lambda n: n.startswith(("foot", "toe")))
    offsets = np.zeros((len(vertices), 3))
    anchor = None
    for i, v in enumerate(vertices):
        z = v[sole, 2]
        low = sole.copy()
        low[sole] = z < z.min() + 0.02
        centre = v[low].mean(0)
        if anchor is None:
            anchor = centre.copy()
        offsets[i, :2] = anchor[:2] - centre[:2]
        offsets[i, 2] = -v[:, 2].min()
    offsets[:, :2] -= offsets[0, :2]
    return offsets


def bar_frames(fig: P.Figure, poses: np.ndarray, grips: dict) -> np.ndarray:
    """Lage eines beidhändig gehaltenen Griffstücks je Bild: Mitte zwischen den Händen, x = Achse."""
    out = np.tile(np.eye(4), (len(poses), 1, 1))
    for i, po in enumerate(poses):
        m = {s: po[fig.model.bone(f"wrist.{s}")] @ g.offset for s, g in grips.items()}
        a, b = m["L"][:3, 3], m["R"][:3, 3]
        x = a - b
        x /= np.linalg.norm(x)
        z = m["R"][:3, 2] - x * (m["R"][:3, 2] @ x)
        z /= np.linalg.norm(z)
        out[i, :3, :3] = np.stack([x, np.cross(z, x), z], 1)
        out[i, :3, 3] = (a + b) / 2
    return out


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Drehung a -> b, Anteil t (kürzester Weg)."""
    d = b @ a.T
    angle = G.angle_of(d)
    if angle < 1e-6:
        return b
    axis = np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]])
    if np.linalg.norm(axis) < 1e-9:
        return b if t > 0.5 else a
    return G.rotation(axis, angle * t) @ a


def frames_between(duration: float) -> np.ndarray:
    return np.arange(int(round(duration * FPS)) + 1) / FPS


def ease(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def deg(x: float) -> float:
    return math.degrees(x)


# --------------------------------------------------------------------------- Übungen


@dataclasses.dataclass
class Exercise:
    """Eine fertige Übung im Format der Bewegungsbibliothek."""

    name: str
    fps: int
    rotations: list[dict]  # je Bild: Knochen -> 3x3 (local-ref)
    root: np.ndarray  # (F, 3) Verschiebung der Wurzel in Metern (Anny-Koordinaten, Boden z = 0)
    props: list[tuple[Prop, np.ndarray]]  # Gerät und seine Weltlage je Bild (F, 4, 4)
    attachments: list[dict]  # name, bone, start, end (Bilder), offset (4x4 im Knochen)
    contacts: dict[str, np.ndarray]  # Knochen -> (F,) bool
    captions: list[tuple[float, float, str]]
    vertices: np.ndarray | None = None  # (F, V, 3) zum Prüfen und Rendern
    poses: np.ndarray | None = None  # (F, J, 4, 4) Weltlage der Knochen, inklusive root

    def quaternions(self, fig: P.Figure) -> np.ndarray:
        return np.stack([fig.poser.quaternions(r) for r in self.rotations])


def _grips(fig: P.Figure, radius: float) -> dict[str, G.Grip]:
    ref = {"vertices": fig.reference_vertices, "heads": fig.reference_poses[:, :3, 3],
           "poses": fig.reference_poses}
    return {s: G.solve_grip(G.Hand(fig.model, None, s, ref), radius) for s in P.SIDES}


def _finish(fig: P.Figure, ex: Exercise, frames: list[dict], root: np.ndarray) -> Exercise:
    out = fig.pose(frames)
    ex.vertices = out["vertices"] + root[:, None]
    poses = out["poses"].copy()
    poses[:, :, :3, 3] += root[:, None]
    ex.poses = poses
    ex.rotations = frames
    ex.root = root
    return ex


def deadlift_setup(fig: P.Figure, grips: dict, stance: float = 6.0) -> tuple[np.ndarray, dict]:
    """Startposition beim Kreuzheben suchen: Hantel auf dem Boden (Stangenmitte auf Scheibenhöhe),
    Stange über dem Mittelfuß, Schienbein an der Stange, gerader Rücken, Arme senkrecht.
    Gibt (Rumpfneigung, Oberschenkelwinkel, Schienbeinwinkel) und die Haltung zurück."""
    from scipy.optimize import minimize

    model = fig.model
    sole = model.dominant_bone_mask(lambda n: n.startswith(("foot", "toe")))
    legs = model.dominant_bone_mask(lambda n: n.startswith(("lowerleg", "upperleg")))
    palms = {s: np.array([0, 1.0, 0]) for s in P.SIDES}

    def evaluate(x):
        frames, wrist = hold(fig, [fig.poser.rotations(deadlift_pose(*x, stance=stance))], grips, palms)
        out = fig.pose(frames)
        v = out["vertices"][0]
        bar = bar_frames(fig, out["poses"], grips)[0]
        zmin = v[:, 2].min()
        foot = v[sole]
        mid = foot[foot[:, 2] < zmin + 0.02].mean(0)
        cyl = contact.Cylinder("Stange", bar[:3, 3], bar[:3, 0], BAR_RADIUS, 0.5)
        pen = max(0.0, cyl.depth(v[legs]).max() + 0.003)
        gap = np.linalg.norm((v[legs] - bar[:3, 3])[:, 1:], axis=1).min() - BAR_RADIUS
        return bar[2, 3] - zmin, bar[1, 3] - mid[1], pen, gap, wrist[0]

    def cost(x):
        z, dy, pen, gap, wrist = evaluate(x)
        return (100 * (z - PLATE_RADIUS) ** 2 + 100 * dy**2 + 1000 * pen**2 + 10 * max(gap - 0.03, 0) ** 2
                + 1e-4 * ((x[0] - 50) ** 2 + (x[2] - 15) ** 2) + 1e-3 * max(wrist - WRIST_LIMIT, 0) ** 2
                + 1e-2 * (max(40 - x[0], 0) ** 2 + max(x[0] - 62, 0) ** 2 + max(x[1] - 80, 0) ** 2
                          + max(5 - x[2], 0) ** 2))

    best = minimize(cost, [60, 70, 12], method="Nelder-Mead", options=dict(maxiter=150, xatol=0.5, fatol=1e-7))
    return best.x, deadlift_pose(*best.x, stance=stance)


def deadlift_pose(torso: float, thigh: float, shin: float, stance: float = 6.0, arms_fwd: float = 0.0) -> dict:
    """Haltung beim Kreuzheben aus drei Winkeln gegen die Senkrechte: Rumpf nach vorn, Oberschenkel
    nach vorn, Schienbein nach vorn. Die Arme hängen senkrecht, der Blick geht schräg nach vorn."""
    return P.pose(None, body=dict(pitch=torso),
                  legs=dict(hip=torso + thigh, knee=thigh + shin, foot=-shin, out=stance, turn=-8),
                  arms=dict(down=34, fwd=torso + arms_fwd, elbow=0), spine=dict(neck=-torso * 0.5),
                  shoulders=dict(up=-5))


def deadlift(fig: P.Figure, reps: int = 2) -> Exercise:
    """Kreuzheben mit der Langhantel: hinstellen, runter, greifen, heben, halten, ablegen."""
    grips = _grips(fig, BAR_RADIUS)
    palms = {s: np.array([0, 1.0, 0]) for s in P.SIDES}
    setup_x, setup = deadlift_setup(fig, grips)
    top = deadlift_pose(-4, 0, 2, arms_fwd=8)  # aufrecht, Schultern zurück, Arme vor den Oberschenkeln
    stand = P.pose(None, legs=dict(hip=2, knee=3, out=6, turn=-8))
    reach = deadlift_pose(*setup_x)
    keys = [(0.0, stand), (0.4, stand)]
    t = 1.6
    keys.append((t, reach))
    phases = []  # (Greifbeginn, Griff geschlossen, Loslassen beginnt, offen)
    for _ in range(reps):
        grip_on = t + 0.35
        lift_end = grip_on + 1.6
        keys += [(grip_on, setup), (lift_end, top), (lift_end + 0.6, top)]
        down_end = lift_end + 0.6 + 1.6
        keys += [(down_end, setup)]
        phases.append((t, grip_on, down_end, down_end + 0.35))
        t = down_end + 0.35
        keys.append((t, setup))
    keys += [(t + 1.3, stand), (t + 1.8, stand)]
    times = frames_between(keys[-1][0])
    poses = P.sample(keys, times)
    amounts = np.zeros(len(times))
    for a, b, c, d in phases:
        amounts = np.maximum(amounts, np.clip((times - a) / (b - a), 0, 1) * np.clip((d - times) / (d - c), 0, 1))
    rot = [fig.poser.rotations(p) for p in poses]
    holding = amounts > 0.999
    turn = np.zeros(len(times))  # Hände drehen sich schon beim Hinunterreichen zur Stange
    for a, _b, _c, d in phases:
        turn = np.maximum(turn, ease((times - (a - 0.9)) / 0.9) * ease((d + 0.9 - times) / 0.9))
    out = []
    for r, a in zip(rot, amounts, strict=True):
        r = dict(r)
        for g in grips.values():
            r.update(g.blend(float(ease(np.array([a]))[0])))
        out.append(r)
    posed = fig.pose(out)["poses"]
    for i, r in enumerate(out):
        if turn[i] <= 0:
            continue
        before = {k: r.get(k, np.eye(3)) for s in P.SIDES for k in (f"lowerarm02.{s}", f"wrist.{s}")}
        for s, g in grips.items():
            orient_grip(fig, r, posed[i], s, g, palms[s])
        for k, r0 in before.items():
            r[k] = _slerp(r0, r[k], float(turn[i]))
    frames = out
    posed = fig.pose(frames)
    root = plant_feet(fig, posed["vertices"])
    bars = bar_frames(fig, posed["poses"], grips)
    bars[:, :3, 3] += root
    # Stange liegt, bis sie gegriffen ist, und nach dem Ablegen wieder am Boden
    first = int(np.argmax(holding))
    world = bars.copy()
    rest_pose = bars[first].copy()
    rest_pose[2, 3] = PLATE_RADIUS
    lying = rest_pose
    for i in range(len(times)):
        if holding[i]:
            lying = bars[i].copy()
            lying[2, 3] = PLATE_RADIUS
        else:
            world[i] = lying if i > first else rest_pose
    wrist_r = fig.model.bone("wrist.R")
    offsets = []
    starts = np.flatnonzero(np.diff(holding.astype(int)) == 1) + 1
    ends = np.flatnonzero(np.diff(holding.astype(int)) == -1)
    attachments = []
    for s, e in zip(starts, ends, strict=False):
        hand = posed["poses"][s, wrist_r].copy()
        hand[:3, 3] += root[s]
        off = np.linalg.inv(hand) @ world[s]
        offsets.append(off)
        attachments.append(dict(name="Langhantel", bone="wrist.R", start=int(s), end=int(e), offset=off))
    contacts = {f"foot.{s}": np.ones(len(times), bool) for s in P.SIDES}
    contacts.update({f"wrist.{s}": holding for s in P.SIDES})
    ex = Exercise("Kreuzheben", FPS, frames, root, [(barbell(), world)], attachments, contacts,
                  [(0, 1.6, "Kreuzheben"), (1.6, 2.0, "Hände schließen sich um die Stange")])
    return _finish(fig, ex, frames, root)
