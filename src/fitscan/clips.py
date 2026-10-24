"""Kurze Bewegungen für die App: Winken, Rufen, Sprechen und Gehen.

Die Clips werden aus Schlüsselhaltungen (`posing`) auf Annys Grundkörper gerechnet und als
`motion.Motion` unter data/clips/<name>.npz gespeichert. Wie jede Bewegung der Bibliothek passen sie
auf jeden Anny-Körper (Drehungen local-ref, Hüfte in Hüfthöhen). Winken, Rufen und Sprechen tragen
die Mimik als Kurven der ARKit-Gesichtsregler (`Motion.face`, Namen wie in `face_rig`).

Alle vier sind Schleifen: Das letzte Bild gleicht dem ersten. Beim Gehen liegt das letzte Bild genau
einen Doppelschritt weiter vorn; wer die Figur auf der Stelle halten will, zieht die gleichmäßige
Vorwärtsbewegung der Hüfte ab (`Motion.root`, Richtung -Y).

Neu rechnen: `python scripts/make_clips.py`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from . import grip as G
from . import posing as P
from .motion import Motion, _continuous

FPS = 30
DATA = Path(__file__).with_name("data") / "clips"
NAMES = ("winken", "rufen", "sprechen", "gehen")


@dataclasses.dataclass
class Clip:
    motion: Motion
    rotations: list[dict]  # je Bild: Knochen -> 3x3 (local-ref), zum Prüfen und Rendern
    root: np.ndarray  # (F, 3) Verschiebung der Wurzel in Metern
    foot_error: float = 0.0  # größte Abweichung der stehenden Füße von ihrer Stelle (m)


def load(name: str) -> Motion:
    """Fertigen Clip aus der Bibliothek laden."""
    return Motion.load(DATA / f"{name}.npz")


# --------------------------------------------------------------------------- Bausteine


def _times(duration: float) -> np.ndarray:
    return np.arange(int(round(duration * FPS)) + 1) / FPS


def _ease(t: np.ndarray, start: float, end: float) -> np.ndarray:
    """0 vor start, 1 nach end, dazwischen weich."""
    x = np.clip((t - start) / (end - start), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _window(t: np.ndarray, on: float, off: float, fade: float = 0.25) -> np.ndarray:
    return _ease(t, on, on + fade) * (1 - _ease(t, off - fade, off))


def _pulse(t: np.ndarray, at: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - at) / width) ** 2)


def _blinks(t: np.ndarray, times: tuple[float, ...]) -> np.ndarray:
    return np.clip(sum(1.15 * _pulse(t, a, 0.045) for a in times), 0, 1)


class Fingers:
    """Finger einer Hand zur Handfläche krümmen: amount 0 = flach, 1 = locker zur Faust. Jedes Glied
    dreht um die Achse quer zu seiner Richtung (local-ref, Glieder bauen aufeinander auf)."""

    BEND = (50.0, 70.0, 45.0)  # Grundgelenk, Mittelgelenk, Endgelenk bei amount = 1
    THUMB = (10.0, 15.0, 15.0)

    def __init__(self, fig: P.Figure, side: str):
        ref = {"vertices": fig.reference_vertices, "heads": fig.reference_poses[:, :3, 3],
               "poses": fig.reference_poses}
        hand = G.Hand(fig.model, None, side, ref)
        self.axes: dict[str, tuple[np.ndarray, float]] = {}
        for finger in (G.THUMB, *G.FINGERS):
            names = [f"{finger}-{k}.{side}" for k in (1, 2, 3)]
            ends = [hand.head[n] for n in names] + [hand.tip[finger]]
            bend = self.THUMB if finger == G.THUMB else self.BEND
            for k, name in enumerate(names):
                d = ends[k + 1] - ends[k]
                axis = np.cross(d, hand.palm)
                self.axes[name] = (axis / np.linalg.norm(axis), bend[k])

    def blend(self, amount: float) -> dict[str, np.ndarray]:
        return {n: G.rotation(axis, angle * amount) for n, (axis, angle) in self.axes.items()}


def _hands(fig: P.Figure) -> dict[str, Fingers]:
    """Hände mit leicht gekrümmten Fingern (curl 0,3 sieht entspannt aus)."""
    return {s: Fingers(fig, s) for s in P.SIDES}


def _frames(fig: P.Figure, poses: list[dict], hands: dict[str, Fingers], curl: dict[str, np.ndarray]) -> list[dict]:
    out = []
    for i, p in enumerate(poses):
        r = fig.poser.rotations(p)
        for s, g in hands.items():
            r.update(g.blend(float(curl[s][i])))
        out.append(r)
    return out


def _stand() -> dict:
    """Locker stehen; die Hände hängen knapp neben den Oberschenkeln, damit sie auch bei kräftigeren
    Körpern ohne Korrekturformen nicht in die Beine drücken."""
    return P.pose(P.stand(), arms=dict(down=34, fwd=3, elbow=12))


def _standing_root(fig: P.Figure, pose: dict, n: int) -> np.ndarray:
    v = fig.pose([fig.poser.rotations(pose)])["vertices"][0]
    return np.tile([0.0, 0.0, -v[:, 2].min()], (n, 1))


def _motion(fig: P.Figure, name: str, frames: list[dict], root: np.ndarray, face: dict[str, np.ndarray],
            contacts: dict[str, np.ndarray]) -> Motion:
    """Nur Knochen, die sich bewegen, kommen in die Bewegung."""
    q = np.stack([fig.poser.quaternions(r) for r in frames])
    moved = np.abs(np.abs(q[..., 3]) - 1).max(0) > 1e-7
    bones = [n for n, k in zip(fig.poser.names, moved, strict=True) if k]
    q = _continuous(q[:, moved]).astype(np.float32)
    hip = -float(fig.reference_vertices[:, 2].min())
    face = {k: np.clip(v, 0, 1).astype(np.float32) for k, v in face.items()}
    return Motion(name, FPS, bones, q, (root / hip).astype(np.float32), "FitScan (Schlüsselhaltungen)", True,
                  {k: v.astype(bool) for k, v in contacts.items()}, [], face)


def _still_feet(n: int) -> dict[str, np.ndarray]:
    return {f"foot.{s}": np.ones(n, bool) for s in P.SIDES}


def _add(poses: list[dict], channel: tuple[str, str], values: np.ndarray) -> None:
    for p, v in zip(poses, values, strict=True):
        p[channel] = p.get(channel, 0.0) + float(v)


def _breathe(poses: list[dict], t: np.ndarray, period: float, degrees: float = 0.8) -> None:
    """Brustkorb hebt sich beim Einatmen ein wenig (Oberkörper streckt sich leicht)."""
    wave = degrees * 0.5 * (1 - np.cos(2 * np.pi * t / period))
    for bone in ("spine02", "spine01"):
        _add(poses, (bone, "vor"), -wave / 2)
    _add(poses, ("neck01", "vor"), wave * 0.5)


# --------------------------------------------------------------------------- Winken


# Rechter Arm zur Seite gehoben, Unterarm senkrecht, Handfläche nach vorn
WAVE_ARM = dict(down=-40, fwd=12, elbow=100, twist=-90, forearm=-15)


def winken(fig: P.Figure, hands: dict | None = None) -> Clip:
    """Freundlich winken: Arm hoch, viermal hin und her, Arm wieder ab. Lächeln dazu."""
    hands = hands or _hands(fig)
    t = _times(4.0)
    stand = _stand()
    up = P.pose(stand, arm_R=WAVE_ARM)
    up[("head", "links")] = -4  # Kopf leicht zur winkenden Seite geneigt
    up[("upperarm01.L", "runter")] = 36
    keys = [(0.0, stand), (0.25, stand), (0.85, up), (3.0, up), (3.6, stand), (4.0, stand)]
    poses = P.sample(keys, t)
    wave = np.sin(2 * np.pi * 2.0 * (t - 0.85)) * _window(t, 0.75, 3.05, 0.3)
    _add(poses, ("upperarm01.R", "runter"), 9 * wave)  # Unterarm schwingt seitlich
    _add(poses, ("wrist.R", "aussen"), 18 * np.sin(2 * np.pi * 2.0 * (t - 0.85) - 0.6) * _window(t, 0.75, 3.05, 0.3))
    _breathe(poses, t, 4.0)
    raised = _ease(t, 0.25, 0.85) * (1 - _ease(t, 3.0, 3.6))
    curl = {"L": np.full(len(t), 0.3), "R": 0.3 - 0.25 * raised}  # beim Winken Finger offen
    smile = _window(t, 0.45, 3.6, 0.4)
    face = {"mouthSmileLeft": 0.6 * smile, "mouthSmileRight": 0.6 * smile, "cheekSquintLeft": 0.3 * smile,
            "cheekSquintRight": 0.3 * smile, "eyeSquintLeft": 0.15 * smile, "eyeSquintRight": 0.15 * smile,
            "browInnerUp": 0.25 * _window(t, 0.5, 1.6, 0.35)}
    blink = _blinks(t, (0.35, 2.2, 3.75))
    face["eyeBlinkLeft"] = face["eyeBlinkRight"] = blink
    frames = _frames(fig, poses, hands, curl)
    return Clip(_motion(fig, "Winken", frames, _standing_root(fig, stand, len(t)), face, _still_feet(len(t))),
                frames, _standing_root(fig, stand, len(t)))


# --------------------------------------------------------------------------- Rufen


# Rechte Hand gewölbt neben dem Mund (Handteller 3,5 cm neben und 4,5 cm vor dem Mundwinkel,
# Ellbogen vorn); auf Annys Grundkörper mit Kopf leicht im Nacken ausgerechnet
CUP_ARM = dict(down=76.8, fwd=63.0, elbow=123.0, twist=53.4, forearm=48.5)


def rufen(fig: P.Figure, hands: dict | None = None) -> Clip:
    """Hand an den Mund und zweimal laut „Hallo!“ rufen: Kopf hoch, Augenbrauen hoch, Mund weit auf."""
    hands = hands or _hands(fig)
    t = _times(3.8)
    stand = _stand()
    call = P.pose(stand, arm_R=CUP_ARM, spine=dict(bend=6, neck=-10))
    call[("upperarm01.L", "runter")] = 35
    keys = [(0.0, stand), (0.3, stand), (0.85, call), (2.85, call), (3.45, stand), (3.8, stand)]
    poses = P.sample(keys, t)
    face: dict[str, np.ndarray] = {}
    lean = np.zeros(len(t))
    for start in (1.0, 1.95):  # „Hal-lo!“
        ha, lo = start + 0.12, start + 0.42
        face["jawOpen"] = face.get("jawOpen", 0) + 0.5 * _pulse(t, ha, 0.07) + 0.65 * _pulse(t, lo, 0.13)
        face["mouthStretchLeft"] = face.get("mouthStretchLeft", 0) + 0.25 * _pulse(t, ha, 0.07)
        face["mouthStretchRight"] = face.get("mouthStretchRight", 0) + 0.25 * _pulse(t, ha, 0.07)
        face["mouthFunnel"] = face.get("mouthFunnel", 0) + 0.4 * _pulse(t, lo + 0.04, 0.12)
        lean += 3.0 * _pulse(t, lo, 0.18)
    _add(poses, ("spine03", "vor"), lean)  # bei jedem Ruf ein kleiner Ruck nach vorn
    _add(poses, ("head", "vor"), -0.6 * lean)
    brows = _window(t, 0.8, 2.9, 0.3)
    face |= {"browInnerUp": 0.45 * brows, "browOuterUpLeft": 0.4 * brows, "browOuterUpRight": 0.4 * brows,
             "eyeWideLeft": 0.3 * brows, "eyeWideRight": 0.3 * brows}
    face["eyeBlinkLeft"] = face["eyeBlinkRight"] = _blinks(t, (0.25, 3.5))
    _breathe(poses, t, 3.8, degrees=1.5)
    raised = _ease(t, 0.3, 0.85) * (1 - _ease(t, 2.85, 3.45))
    curl = {"L": np.full(len(t), 0.3), "R": 0.3 - 0.1 * raised}  # gewölbte Hand
    frames = _frames(fig, poses, hands, curl)
    root = _standing_root(fig, stand, len(t))
    return Clip(_motion(fig, "Rufen", frames, root, face, _still_feet(len(t))), frames, root)


# --------------------------------------------------------------------------- Sprechen


# Vokale und Konsonanten als Mischung von Gesichtsreglern (Werte bei voller Stärke)
_SOUNDS = {
    "a": {"jawOpen": 0.32, "mouthStretchLeft": 0.08, "mouthStretchRight": 0.08},
    "e": {"jawOpen": 0.16, "mouthStretchLeft": 0.25, "mouthStretchRight": 0.25, "mouthSmileLeft": 0.12,
          "mouthSmileRight": 0.12},
    "i": {"jawOpen": 0.08, "mouthStretchLeft": 0.3, "mouthStretchRight": 0.3, "mouthSmileLeft": 0.2,
          "mouthSmileRight": 0.2},
    "o": {"jawOpen": 0.2, "mouthFunnel": 0.45, "mouthPucker": 0.15},
    "u": {"jawOpen": 0.08, "mouthPucker": 0.5, "mouthFunnel": 0.25},
    "m": {"mouthClose": 0.25, "mouthPressLeft": 0.3, "mouthPressRight": 0.3},
    "f": {"jawOpen": 0.05, "mouthRollLower": 0.35, "mouthUpperUpLeft": 0.15, "mouthUpperUpRight": 0.15},
}
# zwei kurze Sätze; Laute wie in „Hallo, ich bin dein Modell. Schön, dass du da bist.“
_PHRASES = ((0.3, "aoimaemoe"), (2.2, "uoaeueaii"))


def sprechen(fig: P.Figure, hands: dict | None = None) -> Clip:
    """Sprechen als Schleife (4 s): Mund und Kiefer bilden Silben, der Kopf nickt und dreht sich
    leicht mit, die Augen blinzeln; Körper steht entspannt und atmet."""
    hands = hands or _hands(fig)
    period = 4.0
    t = _times(period)
    stand = _stand()
    poses = P.sample([(0.0, stand), (period, stand)], t)
    rng = np.random.default_rng(7)
    face: dict[str, np.ndarray] = {}
    accents = []
    for start, sounds in _PHRASES:
        at = start
        for k, sound in enumerate(sounds):
            strength = rng.uniform(0.75, 1.15) * (1.15 if k in (0, 4) else 1.0)
            width = 0.055 if sound in "mf" else 0.07
            for name, value in _SOUNDS[sound].items():
                face[name] = face.get(name, 0) + value * strength * _pulse(t, at, width)
            if k in (0, 4):
                accents.append(at)
            at += rng.uniform(0.15, 0.2)
    face = {k: np.clip(v, 0, 1) for k, v in face.items()}
    for s in ("Left", "Right"):
        face[f"mouthSmile{s}"] = np.clip(face.get(f"mouthSmile{s}", 0) + 0.12, 0, 1)  # freundlich
    face["browInnerUp"] = sum(0.25 * _pulse(t, a, 0.15) for a in accents)
    face["eyeBlinkLeft"] = face["eyeBlinkRight"] = _blinks(t, (1.05, 2.0, 3.55))
    # Kopf: kleine Nicker auf den betonten Silben, langsames Drehen und Neigen (Schleife über 4 s)
    nod = sum(3.0 * _pulse(t, a + 0.05, 0.12) for a in accents)
    _add(poses, ("neck01", "vor"), 0.4 * nod + 1.2 * np.sin(2 * np.pi * t / 2.0))
    _add(poses, ("head", "vor"), 0.6 * nod)
    _add(poses, ("head", "dreh"), 4.0 * np.sin(2 * np.pi * t / period))
    _add(poses, ("head", "links"), 2.0 * np.sin(2 * np.pi * t / period + 1.0))
    _breathe(poses, t, period / 2, degrees=1.0)
    curl = {s: np.full(len(t), 0.3) for s in P.SIDES}
    frames = _frames(fig, poses, hands, curl)
    for v in face.values():  # Schleife: letztes Bild = erstes
        v[-1] = v[0]
    root = _standing_root(fig, stand, len(t))
    return Clip(_motion(fig, "Sprechen", frames, root, face, _still_feet(len(t))), frames, root)


# --------------------------------------------------------------------------- Gehen


def _pitch(angle: float, across: np.ndarray) -> np.ndarray:
    """Fuß um seine waagerechte Querachse kippen, positiv = Zehen hoch."""
    return G.rotation(across, float(angle))


def _about(frame: np.ndarray, pivot: np.ndarray, angle: float, across: np.ndarray) -> np.ndarray:
    out = frame.copy()
    r = _pitch(angle, across)
    out[:3, :3] = r @ frame[:3, :3]
    out[:3, 3] = pivot + r @ (frame[:3, 3] - pivot)
    return out


@dataclasses.dataclass
class Gait:
    """Gangbild: Zeiten als Anteil eines Doppelschritts, Winkel in Grad, Längen in Hüfthöhen."""

    period: float = 1.1  # Sekunden je Doppelschritt
    step: float = 0.62  # Schrittlänge
    stance: float = 0.62  # Anteil, in dem ein Fuß am Boden steht
    flat: float = 0.1  # bis hier rollt der Fuß nach dem Aufsetzen über die Ferse ab
    heel_rise: float = 0.36  # ab hier hebt sich die Ferse (Abrollen über den Ballen)
    strike: float = 14.0  # Zehen beim Aufsetzen angehoben
    push: float = 40.0  # Ferse beim Abstoßen angehoben (Zehen bleiben am Boden)
    reach: float = 0.42  # Sprunggelenk beim Aufsetzen vor der Hüfte, in Schrittlängen
    lift: float = 0.06  # Fuß hebt sich in der Schwungphase zusätzlich
    bow: float = 0.02  # Schwungbein schwingt leicht nach außen
    width: float = 0.8  # Abstand der Sprunggelenke von der Mitte, Anteil vom lockeren Stand
    sway: float = 0.022  # Hüfte pendelt über den Standfuß
    bob: float = 0.013  # Hüfte hebt und senkt sich zweimal je Doppelschritt
    drop: float = 0.022  # Hüfte tiefer als im Stehen (Knie nie ganz durchgestreckt)
    yaw: float = 4.0  # Becken dreht mit dem Schwungbein
    roll: float = 2.5  # Becken senkt sich auf der Schwungbeinseite
    twist: float = 7.0  # Brustkorb dreht gegen das Becken
    arms: float = 16.0  # Armschwung


# Richtwerte je Anteil des Doppelschritts (0 = Ferse setzt auf): Hüftbeugung und Knie in Grad
_HIP = (np.array([0.0, 0.15, 0.5, 0.62, 0.85, 1.0]), np.array([22.0, 18.0, -10.0, -5.0, 25.0, 22.0]))
_KNEE = (np.array([0.0, 0.15, 0.4, 0.6, 0.73, 0.9, 1.0]), np.array([5.0, 15.0, 5.0, 35.0, 60.0, 20.0, 5.0]))


def _across(fig: P.Figure, side: str, turn: np.ndarray | None = None) -> np.ndarray:
    """Waagerechte Querachse des Fußes (Fußspitze zeigt leicht nach außen); turn: Drehung des Fußes
    gegenüber der Bezugshaltung."""
    index = fig.poser.index
    foot = fig.reference_poses[index[f"foot.{side}"], :3, 3]
    ball = np.mean([fig.reference_poses[index[f"toe{k}-1.{side}"], :3, 3] for k in range(1, 6)], 0)
    ahead = (ball - foot) if turn is None else turn @ (ball - foot)
    across = np.cross(ahead * [1, 1, 0], [0.0, 0.0, 1.0])
    return across / np.linalg.norm(across)


def _foot_track(fig: P.Figure, side: str, gait: Gait, t: np.ndarray, start: float, flat0: np.ndarray,
                stride: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lage des Fußknochens je Bild (F, 4, 4), Beugung der Zehen nach oben (Grad, damit sie beim
    Abrollen flach am Boden bleiben) und Bodenkontakt (F,)."""
    sole = fig.model.dominant_bone_mask(lambda n: n == f"foot.{side}" or (n.startswith("toe") and n.endswith(side)))
    ref = fig.reference_poses[fig.poser.index[f"foot.{side}"]]
    local = (fig.reference_vertices[sole] - ref[:3, 3]) @ ref[:3, :3]  # Sohlenpunkte im Fußknochen
    across = _across(fig, side, flat0[:3, :3] @ ref[:3, :3].T)
    # Ballen: vorderster tiefer Punkt des Fußes ohne Zehen (Drehpunkt beim Abrollen; die Zehen biegen
    # sich davor nach oben). Annys Ballen liegt im Stand etwas höher als die Ferse.
    flat = local @ flat0[:3, :3].T + flat0[:3, 3]
    own = fig.model.dominant_bone_mask(lambda n: n == f"foot.{side}")[sole]
    low = own & (flat[:, 2] < flat[:, 2].min() + 0.025)
    ball = local[np.flatnonzero(low)[np.argmax(flat[low] @ np.cross([0.0, 0.0, 1.0], across))]]

    def planted(k: int) -> np.ndarray:
        f = flat0.copy()
        f[1, 3] -= k * stride
        return f

    def stance(k: int, phase: float) -> np.ndarray:
        f = planted(k)
        if phase < gait.flat:  # Ferse setzt auf, Fuß rollt über die Ferse ab
            world = local @ f[:3, :3].T + f[:3, 3]
            heel = world[np.argmax(world[:, 1])] * [1, 1, 0]
            return _about(f, heel, gait.strike * (1 - _ease(phase, 0, gait.flat)), across)
        if phase > gait.heel_rise:  # Ferse hebt sich, Drehpunkt am Ballen
            pivot = (f[:3, :3] @ ball + f[:3, 3]) * [1, 1, 0]
            out = _about(f, pivot, -gait.push * _ease(phase, gait.heel_rise, gait.stance), across)
            # der Ballen sinkt dabei auf den Boden (im flachen Stand trägt vor allem die Ferse)
            out[2, 3] -= ((local[own] @ out[:3, :3].T)[:, 2] + out[2, 3]).min()
            return out
        return f

    frames, toe, contact = [], [], []
    for time in t:
        cycle = (time - start) / gait.period
        k = int(np.floor(cycle + 1e-9))
        phase = cycle - k
        if phase <= gait.stance:
            frames.append(stance(k, phase))
            toe.append(gait.push * float(_ease(phase, gait.heel_rise, gait.stance)))
            contact.append(True)
            continue
        u = (phase - gait.stance) / (1 - gait.stance)
        a, b = stance(k, gait.stance), stance(k + 1, 0.0)
        s = 10 * u**3 - 15 * u**4 + 6 * u**5  # ruckfrei
        out = b.copy()
        out[:3, 3] = a[:3, 3] + (b[:3, 3] - a[:3, 3]) * s
        hip = stride / (2 * gait.step)
        out[2, 3] += gait.lift * hip * np.sin(np.pi * u) ** 1.5
        out[0, 3] += np.sign(flat0[0, 3]) * gait.bow * hip * np.sin(np.pi * u)  # Knie am Standbein vorbei
        angle = P.pchip(np.array([0, 0.3, 0.65, 1.0]), np.array([-gait.push, -4.0, 6.0, gait.strike]),
                        np.array([u]))[0]
        out[:3, :3] = _pitch(angle, across) @ planted(k)[:3, :3]
        frames.append(out)
        toe.append(gait.push * float(1 - _ease(u, 0, 0.35)))
        contact.append(False)
    return np.array(frames), np.array(toe), np.array(contact)


def gehen(fig: P.Figure, gait: Gait | None = None, hands: dict | None = None) -> Clip:
    """Ein Doppelschritt Gehen als nahtlose Schleife: Ferse setzt auf, Fuß rollt ab, Ferse hebt sich,
    die Zehen bleiben am Boden. Die Beine werden je Bild so gebeugt, dass der Standfuß genau an seiner
    Stelle steht (kein Rutschen). Becken dreht und senkt sich, Arme schwingen gegengleich, der Kopf
    blickt ruhig nach vorn."""
    from .poses import WALK_DOF, plant_legs

    gait = gait or Gait()
    hands = hands or _hands(fig)
    hip = -float(fig.reference_vertices[:, 2].min())
    step = gait.step * hip
    n = int(round(gait.period * FPS))
    t = np.arange(2 * n + 1) / FPS  # zwei Doppelschritte rechnen, den zweiten behalten (eingeschwungen)
    w = 2 * np.pi * t / gait.period
    stand = _stand()
    out = fig.pose([fig.poser.rotations(stand)])
    z0 = -out["vertices"][0][:, 2].min()
    poses = []
    for i in range(len(t)):
        c = np.cos(w[i])
        # Becken dreht mit dem vorderen Bein, Brustkorb dagegen, der Kopf bleibt nach vorn gerichtet
        p = P.pose(stand, spine=dict(bend=3, twist=gait.twist * c, turn=-(gait.twist - gait.yaw) * c),
                   body=dict(yaw=-gait.yaw * c, roll=-gait.roll * np.sin(w[i])))
        for s, sign in (("L", -1), ("R", 1)):  # Arm schwingt gegen das Bein seiner Seite
            fwd = 3 + sign * gait.arms * c
            P.arm(p, s, down=34, fwd=fwd, elbow=16 + 12 * max(fwd - 3, 0) / gait.arms)
            # Beinwinkel wie beim Gehen als Richtwert: Die Füße legen die Beine fest, die Richtwerte
            # entscheiden nur, wie hoch die Hüfte geht und welche Lösung gewählt wird
            phase = (t[i] / gait.period - (0.0 if s == "L" else 0.5)) % 1.0
            P.leg(p, s, hip=float(P.pchip(*_HIP, np.array([phase]))[0]),
                  knee=float(P.pchip(*_KNEE, np.array([phase]))[0]))
        poses.append(p)
    _breathe(poses, t, gait.period)
    feet, contact, toes = [], {}, {}
    for s, start, ahead in (("L", 0.0, 0.0), ("R", gait.period / 2, step)):  # rechts einen Schritt später
        f0 = out["poses"][0][fig.poser.index[f"foot.{s}"]].copy()
        f0[:3, 3] = [f0[0, 3] * gait.width, -(gait.reach * step + ahead), f0[2, 3] + z0]
        track, toe, flags = _foot_track(fig, s, gait, t, start, f0, 2 * step)
        feet.append(track)
        contact[f"foot.{s}"] = flags
        toes[s] = toe
    root = np.stack([gait.sway * hip * np.sin(w), -2 * step / gait.period * t,
                     z0 - (gait.drop - gait.bob * np.cos(2 * w - 2 * np.pi * 0.56)) * hip], 1)
    feet = np.stack(feet, 1)
    _, solved, _ = plant_legs(fig, poses, root, feet, dofs=WALK_DOF)
    # Wo das hintere Bein die Hüfte nach unten zieht, ist sie tiefer als gewünscht; dort sanft
    # hineingleiten statt zu springen: glatte Kurve, die nie über der erreichbaren Höhe liegt
    z = solved[:, 2].copy()
    kernel = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
    kernel /= kernel.sum()
    for _ in range(2):
        wrapped = np.concatenate([z[-5:-1], z, z[1:5]])  # Schleife: Anfang und Ende hängen zusammen
        z = np.minimum(z, np.convolve(wrapped, kernel, mode="same")[4:-4])
    root[:, 2] = z - 0.002
    legs, root, worst = plant_legs(fig, poses, root, feet, height=1.0, dofs=WALK_DOF)
    keep = slice(n, 2 * n + 1)
    legs, root = legs[keep], root[keep] - [0.0, root[n, 1], 0.0]
    frames = _frames(fig, legs, hands, {s: np.full(n + 1, 0.3) for s in P.SIDES})
    for s in P.SIDES:  # Zehen beim Abrollen nach oben gebogen, damit sie flach am Boden bleiben
        across = _across(fig, s)
        for r, bend in zip(frames, toes[s][keep], strict=True):
            for k in range(1, 6):
                r[f"toe{k}-1.{s}"] = _pitch(bend, across)
    return Clip(_motion(fig, "Gehen", frames, root, {}, {k: v[keep] for k, v in contact.items()}), frames, root,
                worst)


def build(fig: P.Figure) -> dict[str, Clip]:
    hands = _hands(fig)
    return {"winken": winken(fig, hands), "rufen": rufen(fig, hands), "sprechen": sprechen(fig, hands),
            "gehen": gehen(fig, hands=hands)}
