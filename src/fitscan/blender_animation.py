"""Bewegungsaufnahme für das Körpermodell: Sprung, Rolle vorwärts, Sprint mit Sturz. Läuft in Blender.

Aufruf:
    blender -b -P src/fitscan/blender_animation.py -- koerper_mimik.glb ausgabeordner [--ohne-video]

Ablauf:
1. Das GLB (Skelett, Haut, Formregler, Gesichtsregler aus face_rig.py) wird importiert.
2. Die Bewegung ist als Folge von Schlüsselhaltungen beschrieben (Gelenkwinkel in Grad, bezogen
   auf die Körperachsen). Zwischen den Haltungen wird monoton kubisch interpoliert, die Laufbewegung
   wird aus einem Schrittzyklus erzeugt.
3. Bodenkontakt: Für jedes Bild wird das verformte Netz ausgewertet. Was den Boden berührt, rutscht
   nicht (Füße beim Laufen, Rücken bei der Rolle), der tiefste Punkt liegt auf dem Boden. In der
   Luft fliegt der Körper auf einer Wurfparabel, beim Aufprall rutscht er mit Reibung aus. Die
   Verschiebung liegt auf dem Wurzelknochen (Root Motion), wie Cascadeur und Unreal es erwarten.
4. Die Mimik ist eine eigene Aktion auf den Gesichtsreglern (Shape Keys) und damit unabhängig von
   der Körperbewegung: Regler einzeln stellen, Aktion tauschen oder ganz abschalten.
5. Ausgabe: .blend (mit Boden, Kamera, Licht), FBX (Cascadeur/Unreal), GLB, Vorschauvideos.

Blender ist GPL; dieses Skript ruft Blender nur auf und wird nicht mit der App ausgeliefert.
"""

from __future__ import annotations

import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blender_glb_to_fbx import show_textures  # noqa: E402

FPS = 30
G = 9.81
BODY_ACTION = "Bewegung_Sprung_Rolle_Sturz"
FACE_ACTION = "Mimik_zur_Bewegung"
FACE_DEMO_ACTION = "Mimik_Uebersicht"

# --------------------------------------------------------------------------- Haltungen
# Kanal = (Knochen, Richtung) -> Grad. Richtungen (im Körper, vom Elternknochen mitgedreht):
#   "runter": Knochen zum Boden schwenken (Arme an den Körper, Fuß strecken)
#   "aussen": von der Körpermitte weg, "links": zur linken Körperseite (Wirbelsäule seitlich)
#   "vor":    nach vorn schwenken (Hüfte beugen, Arm heben, Oberkörper beugen; negativ = nach hinten)
#   "dreh":   um die eigene Achse drehen
# Becken: ("pelvis", "pitch") kippt den ganzen Körper nach vorn (360 = Rolle), "yaw" dreht ihn.
ORDER = ("runter", "aussen", "links", "vor", "dreh")
SIDES = ("l", "r")


def arm(pose: dict, side: str, down: float = 40, fwd: float = 0, elbow: float = 10, out: float = 0,
        twist: float = 0) -> dict:
    pose[(f"upperarm_{side}", "runter")] = down
    pose[(f"upperarm_{side}", "aussen")] = out
    pose[(f"upperarm_{side}", "vor")] = fwd
    pose[(f"upperarm_{side}", "dreh")] = twist
    pose[(f"lowerarm_{side}", "vor")] = elbow
    return pose


def leg(pose: dict, side: str, hip: float = 0, knee: float = 0, foot: float = 0, out: float = 0) -> dict:
    pose[(f"thigh_{side}", "vor")] = hip
    pose[(f"thigh_{side}", "aussen")] = out
    pose[(f"calf_{side}", "vor")] = -knee
    pose[(f"foot_{side}", "runter")] = foot
    return pose


def spine(pose: dict, bend: float = 0, neck: float = 0, twist: float = 0, side: float = 0,
          turn: float = 0) -> dict:
    """bend verteilt sich auf die fünf Wirbel, neck auf Hals und Kopf (positiv = Kinn zur Brust),
    turn dreht den Kopf zur Seite."""
    for i, share in enumerate((0.15, 0.2, 0.25, 0.2, 0.2), start=1):
        pose[(f"spine_0{i}", "vor")] = bend * share
        pose[(f"spine_0{i}", "dreh")] = twist * share
        pose[(f"spine_0{i}", "links")] = side * share
    pose[("neck_01", "vor")] = neck * 0.4
    pose[("head", "vor")] = neck * 0.6
    pose[("neck_01", "dreh")] = turn * 0.4
    pose[("head", "dreh")] = turn * 0.6
    return pose


def body(pose: dict, pitch: float = 0, yaw: float = 0, roll: float = 0) -> dict:
    pose[("pelvis", "pitch")] = pitch
    pose[("pelvis", "yaw")] = yaw
    pose[("pelvis", "roll")] = roll
    return pose


def both_arms(pose: dict, **kw) -> dict:
    for s in SIDES:
        arm(pose, s, **kw)
    return pose


def both_legs(pose: dict, **kw) -> dict:
    for s in SIDES:
        leg(pose, s, **kw)
    return pose


def stand() -> dict:
    p: dict = {}
    both_arms(p, down=40, fwd=5, elbow=12)
    both_legs(p, hip=2, knee=3, foot=0)
    spine(p, bend=0, neck=0)
    return body(p)


def pose(base: dict | None = None, **parts) -> dict:
    p = dict(stand() if base is None else base)
    for key, value in parts.items():
        if key == "arms":
            both_arms(p, **value)
        elif key == "legs":
            both_legs(p, **value)
        elif key in ("arm_l", "arm_r"):
            arm(p, key[-1], **value)
        elif key in ("leg_l", "leg_r"):
            leg(p, key[-1], **value)
        elif key == "spine":
            spine(p, **value)
        elif key == "body":
            body(p, **value)
        else:
            raise KeyError(key)
    return p


def a_pose() -> dict:
    return {}


# --------------------------------------------------------------------------- Ablauf
# Zeiten in Sekunden. Modus je Abschnitt: "boden" (Kontakt rutscht nicht), "luft" (Wurfparabel),
# "rutschen" (Aufprall, gleitet mit Reibung aus).


def run_cycle(t0: float, t1: float, cadence: float = 1.7, ramp: float = 0.6) -> list[tuple[float, dict]]:
    """Sprint: Schrittzyklus mit cadence Doppelschritten pro Sekunde, Ausholen wächst über ramp s."""
    keys = []
    t = t0
    while t <= t1 + 1e-6:
        a = min(1.0, (t - t0) / ramp)
        a = a * a * (3 - 2 * a)
        phi = 2 * math.pi * cadence * (t - t0)
        p = stand()
        for side, ph in (("l", phi), ("r", phi + math.pi)):
            hip = 48 * math.sin(ph)
            knee = 18 + 100 * ((1 + math.cos(ph + 0.5)) / 2) ** 2
            foot = 8 + 22 * max(0.0, -math.sin(ph + 0.3))
            leg(p, side, hip=a * hip + 2 * (1 - a), knee=a * knee + 3 * (1 - a), foot=a * foot)
            # Arme gegengleich zum Bein derselben Seite, Ellbogen etwa rechtwinklig
            arm(p, side, down=40 + a * 5, fwd=a * (-50 * math.sin(ph)) + 5, elbow=12 + a * 78,
                out=a * 6)
        spine(p, bend=a * 8, neck=-a * 6, twist=a * 10 * math.sin(phi))
        body(p, pitch=a * 10, yaw=-a * 6 * math.sin(phi))
        keys.append((t, p))
        t += 1.0 / FPS * 2
    return keys


def choreography() -> tuple[list[tuple[float, dict]], list[tuple[float, float, str, dict]], list, float]:
    """Gibt Schlüsselhaltungen, Bodenabschnitte, Beschriftungen und die Gesamtdauer zurück."""
    keys: list[tuple[float, dict]] = []
    s = stand()

    # Ruhepose (A-Pose) -> lockerer Stand
    keys += [(0.0, a_pose()), (0.5, s), (0.8, s)]

    # ---------------- Sprung nach vorn
    crouch = pose(s, arms=dict(down=40, fwd=-55, elbow=15), legs=dict(hip=65, knee=105, foot=-35),
                  spine=dict(bend=30, neck=-10))
    takeoff = pose(s, arms=dict(down=30, fwd=150, elbow=10), legs=dict(hip=-2, knee=4, foot=35),
                   spine=dict(bend=-8, neck=-12))
    tuck = pose(s, arms=dict(down=30, fwd=120, elbow=25), legs=dict(hip=60, knee=85, foot=20),
                spine=dict(bend=10, neck=-5))
    reach = pose(s, arms=dict(down=30, fwd=80, elbow=20), legs=dict(hip=30, knee=25, foot=15),
                 spine=dict(bend=10, neck=0))
    absorb = pose(s, arms=dict(down=30, fwd=45, elbow=25), legs=dict(hip=75, knee=105, foot=-35),
                  spine=dict(bend=30, neck=-15))
    keys += [(1.15, crouch), (1.40, takeoff), (1.62, tuck), (1.85, reach), (1.95, reach),
             (2.15, absorb), (2.65, s)]
    t_air0, t_air1 = 1.42, 1.92

    # ---------------- Rolle vorwärts
    step = pose(s, leg_l=dict(hip=35, knee=25, foot=0), leg_r=dict(hip=-10, knee=10, foot=10),
                arms=dict(down=35, fwd=40, elbow=20), spine=dict(bend=15))
    squat = pose(s, legs=dict(hip=120, knee=135, foot=-35), arms=dict(down=30, fwd=75, elbow=15),
                 spine=dict(bend=40, neck=25), body=dict(pitch=10))
    dive = pose(squat, legs=dict(hip=95, knee=110, foot=-20), arms=dict(down=30, fwd=85, elbow=25),
                spine=dict(bend=55, neck=45), body=dict(pitch=55))
    ball = pose(s, legs=dict(hip=130, knee=140, foot=10), arms=dict(down=20, fwd=70, elbow=95, out=-10),
                spine=dict(bend=70, neck=45))
    up = pose(s, legs=dict(hip=115, knee=135, foot=-30), arms=dict(down=30, fwd=70, elbow=20),
              spine=dict(bend=45, neck=15), body=dict(pitch=345))
    keys += [(3.00, step), (3.35, squat), (3.60, dive),
             (3.75, body(dict(ball), pitch=100)), (3.95, body(dict(ball), pitch=180)),
             (4.15, body(dict(ball), pitch=270)), (4.35, body(dict(up), pitch=330)),
             (4.55, body(dict(up), pitch=360)), (5.10, body(dict(s), pitch=360))]

    # ---------------- Sprint, Stolpern, Sturz
    t_run0, t_run1 = 5.25, 7.35
    for t, p in run_cycle(t_run0, t_run1):
        keys.append((t, body(p, pitch=p[("pelvis", "pitch")] + 360)))
    trip = pose(s, leg_l=dict(hip=55, knee=70, foot=0), leg_r=dict(hip=-35, knee=50, foot=30),
                arms=dict(down=30, fwd=60, elbow=30, out=10), spine=dict(bend=15, neck=-20),
                body=dict(pitch=360 + 35))
    fly = pose(s, legs=dict(hip=5, knee=35, foot=25), arms=dict(down=20, fwd=110, elbow=15, out=10),
               spine=dict(bend=0, neck=-35), body=dict(pitch=360 + 70))
    impact = pose(s, legs=dict(hip=10, knee=40, foot=30), arms=dict(down=10, fwd=120, elbow=70, out=15),
                  spine=dict(bend=-5, neck=-45), body=dict(pitch=360 + 88))
    # Liegend zeigt "vor" zum Boden: Arme und Beine deshalb seitlich (runter/aussen) statt nach vorn
    lie = pose(s, leg_l=dict(hip=0, knee=20, foot=35, out=18), leg_r=dict(hip=0, knee=5, foot=40, out=4),
               arm_l=dict(down=-115, fwd=-12, elbow=8, out=0), arm_r=dict(down=38, fwd=-18, elbow=5, out=0),
               spine=dict(bend=-5, neck=-20, turn=20), body=dict(pitch=360 + 90, roll=0))
    lie_still = pose(lie, spine=dict(bend=0, neck=-5, turn=55))
    t_trip = t_run1 + 0.15
    keys += [(t_trip, trip), (t_trip + 0.30, fly), (t_trip + 0.50, impact), (t_trip + 0.95, lie),
             (t_trip + 2.0, lie_still), (t_trip + 3.0, lie_still)]
    duration = t_trip + 3.0

    segments = [
        (0.0, t_air0, "boden", {}),
        (t_air0, t_air1, "luft", {"speed": 1.4}),
        (t_air1, t_trip + 0.22, "boden", {}),
        (t_trip + 0.22, t_trip + 0.52, "luft", {"height": 0.18}),
        (t_trip + 0.52, t_trip + 1.3, "rutschen", {"friction": 0.35}),
        (t_trip + 1.3, duration + 1, "boden", {}),
    ]
    captions = [(0.8, 2.7, "Sprung"), (2.9, 5.2, "Rolle vorwärts"), (t_run0, t_trip, "Sprint"),
                (t_trip, duration, "Stolpern und Sturz")]
    return keys, segments, captions, duration


# Mimik zur Bewegung: eigene Aktion auf den Gesichtsreglern
FACE_KEYS = [
    (0.0, {}),
    (0.9, {"browDownLeft": 0.5, "browDownRight": 0.5, "mouthPressLeft": 0.3, "mouthPressRight": 0.3}),
    (1.2, {"browDownLeft": 0.7, "browDownRight": 0.7, "jawOpen": 0.1, "mouthStretchLeft": 0.3,
           "mouthStretchRight": 0.3}),
    (1.5, {"jawOpen": 0.35, "eyeWideLeft": 0.6, "eyeWideRight": 0.6, "browInnerUp": 0.5}),
    (2.0, {"jawOpen": 0.15, "eyeSquintLeft": 0.4, "eyeSquintRight": 0.4}),
    (2.6, {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8, "cheekSquintLeft": 0.4, "cheekSquintRight": 0.4}),
    (3.3, {"mouthPucker": 0.4, "browDownLeft": 0.3, "browDownRight": 0.3}),
    (3.6, {"eyeBlinkLeft": 0.85, "eyeBlinkRight": 0.85, "mouthPressLeft": 0.6, "mouthPressRight": 0.6}),
    (4.4, {"eyeBlinkLeft": 0.85, "eyeBlinkRight": 0.85, "mouthPressLeft": 0.6, "mouthPressRight": 0.6}),
    (4.7, {"eyeWideLeft": 0.3, "eyeWideRight": 0.3, "jawOpen": 0.1}),
    (5.2, {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0, "cheekSquintLeft": 0.5, "cheekSquintRight": 0.5,
           "eyeSquintLeft": 0.3, "eyeSquintRight": 0.3}),
    (5.7, {"jawOpen": 0.3, "browDownLeft": 0.6, "browDownRight": 0.6, "mouthStretchLeft": 0.5,
           "mouthStretchRight": 0.5}),
    (7.3, {"jawOpen": 0.35, "browDownLeft": 0.7, "browDownRight": 0.7, "mouthStretchLeft": 0.6,
           "mouthStretchRight": 0.6}),
    (7.55, {"jawOpen": 0.8, "eyeWideLeft": 1.0, "eyeWideRight": 1.0, "browInnerUp": 1.0,
            "browOuterUpLeft": 0.8, "browOuterUpRight": 0.8}),
    (8.0, {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0, "mouthFrownLeft": 0.8, "mouthFrownRight": 0.8,
           "noseSneerLeft": 0.7, "noseSneerRight": 0.7, "browDownLeft": 0.9, "browDownRight": 0.9,
           "jawOpen": 0.2, "mouthStretchLeft": 0.6, "mouthStretchRight": 0.6}),
    (8.9, {"eyeSquintLeft": 0.7, "eyeSquintRight": 0.7, "mouthFrownLeft": 0.6, "mouthFrownRight": 0.6,
           "browDownLeft": 0.6, "browDownRight": 0.6, "browInnerUp": 0.5, "eyeBlinkLeft": 0.3,
           "eyeBlinkRight": 0.3}),
    (10.5, {"eyeSquintLeft": 0.5, "eyeSquintRight": 0.5, "mouthFrownLeft": 0.4, "mouthFrownRight": 0.4,
            "browInnerUp": 0.6, "eyeBlinkLeft": 0.2, "eyeBlinkRight": 0.2, "jawOpen": 0.1}),
]


# --------------------------------------------------------------------------- Interpolation


def pchip(times: np.ndarray, values: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Monotone kubische Interpolation (Fritsch-Carlson): weich, ohne Überschwingen."""
    h = np.diff(times)
    delta = np.diff(values) / h
    m = np.zeros_like(values)
    if len(values) > 2:
        same = delta[:-1] * delta[1:] > 0
        w1 = 2 * h[1:] + h[:-1]
        w2 = h[1:] + 2 * h[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            harm = (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:])
        m[1:-1] = np.where(same, harm, 0.0)
    i = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
    s = np.clip((t - times[i]) / h[i], 0.0, 1.0)
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * values[i] + h10 * h[i] * m[i] + h01 * values[i + 1] + h11 * h[i] * m[i + 1]


def sample_channels(keys: list[tuple[float, dict]], frames: np.ndarray) -> dict:
    keys = sorted(keys, key=lambda k: k[0])
    channels = sorted({c for _, p in keys for c in p})
    times = np.array([k[0] for k in keys])
    out = {}
    for c in channels:
        vals = np.array([p.get(c, 0.0) for _, p in keys], dtype=np.float64)
        out[c] = pchip(times, vals, frames / FPS)
    return out


# --------------------------------------------------------------------------- Skelett


class Rig:
    CHAIN = {"thigh": "calf", "calf": "foot", "foot": "middletoe_01", "upperarm": "lowerarm",
             "lowerarm": "hand", "hand": "middle_01", "clavicle": "upperarm", "shoulder": "upperarm"}

    def __init__(self, arm_obj):
        self.obj = arm_obj
        self.bones = arm_obj.data.bones
        self.rest = {b.name: b.matrix_local.to_3x3().normalized() for b in self.bones}
        self.dir = {b.name: self._direction(b) for b in self.bones}

    def _direction(self, bone) -> Vector:
        name = bone.name
        base, _, side = name.rpartition("_") if name.endswith(("_l", "_r")) else (name, "", "")
        target = self.CHAIN.get(base)
        if target is not None and f"{target}_{side}" in self.bones:
            d = self.bones[f"{target}_{side}"].head_local - bone.head_local
        elif name.startswith("spine_0") and name != "spine_05":
            d = self.bones[f"spine_0{int(name[-1]) + 1}"].head_local - bone.head_local
        elif name in ("spine_05", "neck_01", "neck_02", "neck_03", "head", "pelvis"):
            d = Vector((0.0, 0.0, 1.0))
        else:
            d = bone.matrix_local.to_3x3() @ Vector((0.0, 1.0, 0.0))
        return d.normalized()

    @staticmethod
    def toward(name: str, direction: str) -> Vector:
        side = 1.0 if name.endswith("_l") else -1.0 if name.endswith("_r") else 0.0
        return {"runter": Vector((0, 0, -1)), "vor": Vector((0, -1, 0)), "aussen": Vector((side, 0, 0)),
                "links": Vector((1, 0, 0))}[direction]

    def local_rotation(self, name: str, values: dict[str, float]) -> Quaternion:
        """Gelenkwinkel (Körperachsen, vom Elternknochen mitgedreht) -> lokale Knochendrehung."""
        if name == "pelvis":
            delta = (Matrix.Rotation(math.radians(values.get("yaw", 0.0)), 3, "Z")
                     @ Matrix.Rotation(math.radians(values.get("pitch", 0.0)), 3, "X")
                     @ Matrix.Rotation(math.radians(values.get("roll", 0.0)), 3, "Y"))
        else:
            delta = Matrix.Identity(3)
            d0 = self.dir[name]
            for direction in ORDER:
                angle = values.get(direction, 0.0)
                if abs(angle) < 1e-6:
                    continue
                cur = delta @ d0
                if direction == "dreh":
                    axis = cur
                else:
                    axis = cur.cross(self.toward(name, direction))
                    if axis.length < 1e-6:
                        continue
                delta = Matrix.Rotation(math.radians(angle), 3, axis.normalized()) @ delta
        b = self.rest[name]
        return (b.inverted() @ delta @ b).to_quaternion()


# --------------------------------------------------------------------------- Aufbau


def import_model(glb: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb, bone_heuristic="TEMPERANCE", merge_vertices=True)
    arm_obj = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
    mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH")
    for o in (arm_obj, mesh):
        if o.animation_data:
            o.animation_data_clear()
    return arm_obj, mesh


def key_body(rig: Rig, channels: dict, frames: np.ndarray) -> bpy.types.Action:
    arm_obj = rig.obj
    arm_obj.animation_data_create()
    action = bpy.data.actions.new(BODY_ACTION)
    action.use_fake_user = True
    arm_obj.animation_data.action = action
    per_bone: dict[str, dict[str, np.ndarray]] = {}
    for (bone, direction), vals in channels.items():
        per_bone.setdefault(bone, {})[direction] = vals
    for bone, chans in per_bone.items():
        pb = arm_obj.pose.bones[bone]
        pb.rotation_mode = "QUATERNION"
        quats = []
        prev = None
        for i in range(len(frames)):
            q = rig.local_rotation(bone, {k: float(v[i]) for k, v in chans.items()})
            if prev is not None and prev.dot(q) < 0:
                q.negate()
            quats.append(q)
            prev = q
        path = f'pose.bones["{bone}"].rotation_quaternion'
        for c in range(4):
            fc = action.fcurves.new(path, index=c, action_group=bone)
            fc.keyframe_points.add(len(frames))
            data = np.empty(2 * len(frames))
            data[0::2] = frames
            data[1::2] = [q[c] for q in quats]
            fc.keyframe_points.foreach_set("co", data)
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
            fc.update()
    return action


def evaluated_vertices(mesh) -> np.ndarray:
    dg = bpy.context.evaluated_depsgraph_get()
    ev = mesh.evaluated_get(dg)
    me = ev.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mw = np.array(ev.matrix_world)
    return co @ mw[:3, :3].T + mw[:3, 3]


def mode_at(segments, t: float):
    for t0, t1, mode, params in segments:
        if t0 <= t < t1:
            return mode, params, t0, t1
    return "boden", {}, t, t


def solve_ground(mesh, frames: np.ndarray, segments) -> np.ndarray:
    """Verschiebung des ganzen Körpers je Bild: Bodenkontakt ohne Rutschen, Flug, Ausrutschen."""
    scene = bpy.context.scene
    offsets = np.zeros((len(frames), 3))
    prev_pts = None
    velocity = np.zeros(2)
    history: list[np.ndarray] = []
    air_start = None
    for i, f in enumerate(frames):
        scene.frame_set(int(f))
        pts = evaluated_vertices(mesh)
        t = f / FPS
        mode, params, t0, t1 = mode_at(segments, t)
        low = pts[:, 2].min()
        off = offsets[i - 1].copy() if i else np.zeros(3)
        if mode == "luft":
            if air_start is None:
                air_start = (offsets[i - 1].copy() if i else np.zeros(3))
                if "speed" in params:
                    velocity = np.array([0.0, -params["speed"]])
                elif len(history) >= 2:
                    velocity = (history[-1] - history[max(0, len(history) - 6)]) / (min(5, len(history) - 1) / FPS)
            dur = t1 - t0
            s = (t - t0) / dur
            height = params.get("height", G * dur * dur / 8)
            off[:2] = off[:2] + velocity / FPS
            off[2] = 4 * height * s * (1 - s) - low
        else:
            air_start = None
            if mode == "rutschen":
                velocity = velocity * math.exp(-1.0 / FPS / params.get("friction", 0.3))
                off[:2] = off[:2] + velocity / FPS
            elif prev_pts is not None:
                # Was den Boden berührt, bleibt stehen (weich gewichtet, damit der Kontakt sauber wechselt)
                w = np.exp(-(pts[:, 2] - low) / 0.006)
                w[pts[:, 2] - low > 0.03] = 0.0
                w /= w.sum()
                move = ((pts[:, :2] - prev_pts[:, :2]) * w[:, None]).sum(0)
                off[:2] = off[:2] - move
            off[2] = -low
        offsets[i] = off
        history.append(off[:2].copy())
        prev_pts = pts
    return offsets


def key_root(arm_obj, frames: np.ndarray, offsets: np.ndarray) -> None:
    bone = arm_obj.data.bones[0]
    while bone.parent is not None:
        bone = bone.parent
    inv = bone.matrix_local.to_3x3().inverted()
    action = arm_obj.animation_data.action
    path = f'pose.bones["{bone.name}"].location'
    local = np.array([inv @ Vector(o) for o in offsets])
    for c in range(3):
        fc = action.fcurves.new(path, index=c, action_group=bone.name)
        fc.keyframe_points.add(len(frames))
        data = np.empty(2 * len(frames))
        data[0::2] = frames
        data[1::2] = local[:, c]
        fc.keyframe_points.foreach_set("co", data)
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
        fc.update()


def key_face(mesh, keys, name: str, hold: float = 0.0) -> bpy.types.Action | None:
    """Mimik als eigene Aktion auf den Shape Keys (unabhängig von der Körperaktion)."""
    shape_keys = mesh.data.shape_keys
    if shape_keys is None:
        return None
    names = [k.name for k in shape_keys.key_blocks[1:]]
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    shape_keys.animation_data_create()
    shape_keys.animation_data.action = action
    channels = sorted({c for _, p in keys for c in p if c in names})
    times = np.array([k[0] for k in keys])
    frames = np.arange(int(round(times[0] * FPS)), int(round(times[-1] * FPS)) + 1)
    for c in channels:
        vals = np.array([p.get(c, 0.0) for _, p in keys])
        seq = np.clip(pchip(times, vals, frames / FPS), 0.0, 1.0)
        fc = action.fcurves.new(f'key_blocks["{c}"].value', action_group="Mimik")
        # Schlüssel nur an den Haltungen, Blender interpoliert weich dazwischen
        for t, v in zip(times, vals, strict=True):
            kp = fc.keyframe_points.insert(t * FPS + 1, float(v))
            kp.interpolation = "BEZIER"
        del seq
        fc.update()
    return action


# --------------------------------------------------------------------------- Szene, Video


def make_scene(arm_obj, mesh, frames, offsets, captions):
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = int(frames[0])
    scene.frame_end = int(frames[-1])
    show_textures([mesh])

    # Boden mit Karomuster, damit man die Bewegung sieht
    lo = offsets[:, 1].min() - 4
    hi = offsets[:, 1].max() + 4
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, (lo + hi) / 2, 0))
    floor = bpy.context.active_object
    floor.name = "Boden"
    floor.scale = (40, hi - lo, 1)
    mat = bpy.data.materials.new("Boden")
    mat.use_nodes = True
    nt = mat.node_tree
    checker = nt.nodes.new("ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 1.0
    checker.inputs["Color1"].default_value = (0.42, 0.42, 0.40, 1)
    checker.inputs["Color2"].default_value = (0.30, 0.30, 0.29, 1)
    coord = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(coord.outputs["Object"], checker.inputs["Vector"])
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.8
    nt.links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
    floor.data.materials.append(mat)
    # Karos 1 m groß, unabhängig von der Skalierung: Objektkoordinaten mal Größe
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (floor.scale[0], floor.scale[1], 1)
    nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], checker.inputs["Vector"])

    world = bpy.data.worlds.new("Himmel")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.6, 0.68, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    scene.world = world
    sun = bpy.data.objects.new("Sonne", bpy.data.lights.new("Sonne", "SUN"))
    sun.data.energy = 3.0
    sun.data.angle = math.radians(8)
    sun.rotation_euler = (math.radians(50), 0, math.radians(35))
    scene.collection.objects.link(sun)

    # Kamera fährt seitlich mit (geglättete Bahn des Körpers)
    cam = bpy.data.objects.new("Kamera", bpy.data.cameras.new("Kamera"))
    cam.data.lens = 35
    scene.collection.objects.link(cam)
    scene.camera = cam
    k = np.exp(-0.5 * (np.arange(-30, 31) / 12.0) ** 2)
    k /= k.sum()
    path = np.stack([np.convolve(np.pad(offsets[:, c], 30, mode="edge"), k, mode="valid") for c in (0, 1)], 1)
    for i, f in enumerate(frames):
        target = Vector((path[i, 0], path[i, 1], 0.75))
        cam.location = target + Vector((5.2, 1.2, 0.75))
        cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert("location", frame=int(f))
        cam.keyframe_insert("rotation_euler", frame=int(f))
    add_captions(cam, captions)

    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 16
    scene.eevee.use_soft_shadows = True
    scene.eevee.use_gtao = True
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    return cam


def add_captions(cam, captions, size: float = 0.05, pos=(-0.30, 0.15, -1.0), center: bool = False):
    mat = bpy.data.materials.get("Schrift") or bpy.data.materials.new("Schrift")
    mat.use_nodes = True
    em = mat.node_tree.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1, 1, 1, 1)
    em.inputs["Strength"].default_value = 1.0
    out = mat.node_tree.nodes["Material Output"]
    mat.node_tree.links.new(em.outputs["Emission"], out.inputs["Surface"])
    objs = []
    for i, (t0, t1, text) in enumerate(captions):
        curve = bpy.data.curves.new(f"Text{i}", "FONT")
        curve.body = text
        curve.size = size
        if center:
            curve.align_x = "CENTER"
        obj = bpy.data.objects.new(f"Beschriftung {i}", curve)
        obj.data.materials.append(mat)
        bpy.context.scene.collection.objects.link(obj)
        obj.parent = cam
        obj.location = pos
        for f, hidden in ((0, True), (int(t0 * FPS), False), (int(t1 * FPS), True)):
            obj.hide_render = hidden
            obj.hide_viewport = hidden
            obj.keyframe_insert("hide_render", frame=f)
            obj.keyframe_insert("hide_viewport", frame=f)
        objs.append(obj)
    return objs


def render_video(path: str, start: int, end: int, size=(1280, 720)) -> None:
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = start, end
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.filepath = path
    bpy.ops.render.render(animation=True)


def face_demo(arm_obj, mesh, out_dir: str, render: bool) -> dict:
    """Jeder Gesichtsregler einzeln, Körper in Ruhe: eigene Aktion und Nahaufnahme-Video."""
    shape_keys = mesh.data.shape_keys
    names = [k.name for k in shape_keys.key_blocks[1:] if not k.name.startswith(("weight_", "muscle_"))]
    step = 0.6
    keys = [(0.0, {})]
    captions = []
    for i, n in enumerate(names):
        t = 0.4 + i * step
        keys += [(t, {}), (t + step * 0.45, {n: 1.0}), (t + step * 0.75, {n: 1.0})]
        captions.append((t, t + step, f"{FACE_LABELS.get(n, n)}  ({n})"))
    end = 0.4 + len(names) * step + 0.4
    keys.append((end, {}))
    action = key_face(mesh, keys, FACE_DEMO_ACTION)
    info = {"regler": names, "dauer_s": end}
    if not render:
        return info
    scene = bpy.context.scene
    arm_obj.animation_data.action = None
    for pb in arm_obj.pose.bones:
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
    bpy.context.view_layer.update()
    head = arm_obj.matrix_world @ arm_obj.pose.bones["head"].head
    cam = bpy.data.objects.new("Kamera Gesicht", bpy.data.cameras.new("Kamera Gesicht"))
    cam.data.lens = 85
    scene.collection.objects.link(cam)
    target = head + Vector((0, -0.06, 0.07))
    cam.location = target + Vector((0.35, -0.75, 0.05))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam
    for o in scene.objects:
        if o.name.startswith("Beschriftung"):
            o.animation_data_clear()
            o.hide_render = True
    # 85 mm in 0,5 m Abstand: das Bild ist nur gut 20 cm breit, die Schrift also klein und mittig
    add_captions(cam, captions, size=0.0075, pos=(0.0, -0.088, -0.5), center=True)
    light = bpy.data.objects.new("Gesichtslicht", bpy.data.lights.new("Gesichtslicht", "AREA"))
    light.data.energy = 18
    light.data.size = 0.8
    light.location = cam.location + Vector((-0.5, -0.2, 0.4))
    light.rotation_euler = (head - light.location).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(light)
    shape_keys.animation_data.action = action
    render_video(os.path.join(out_dir, "mimik_regler.mp4"), 0, int(end * FPS), size=(600, 600))
    return info


FACE_LABELS = {
    "jawOpen": "Mund auf (Kiefer)", "jawForward": "Kiefer vor", "jawLeft": "Kiefer nach links",
    "jawRight": "Kiefer nach rechts", "mouthClose": "Lippen zu bei offenem Kiefer",
    "mouthDimpleLeft": "Grübchen links", "mouthDimpleRight": "Grübchen rechts",
    "mouthLowerDownLeft": "Unterlippe runter links", "mouthLowerDownRight": "Unterlippe runter rechts",
    "mouthUpperUpLeft": "Oberlippe hoch links", "mouthUpperUpRight": "Oberlippe hoch rechts",
    "mouthRollLower": "Unterlippe einrollen", "mouthRollUpper": "Oberlippe einrollen",
    "mouthShrugLower": "Unterlippe vorschieben", "mouthShrugUpper": "Oberlippe vorschieben",
    "tongueOut": "Zunge raus", "mouthSmileLeft": "Lächeln links", "mouthSmileRight": "Lächeln rechts",
    "mouthFrownLeft": "Mundwinkel runter links", "mouthFrownRight": "Mundwinkel runter rechts",
    "mouthPucker": "Kussmund", "mouthFunnel": "Trichtermund", "mouthLeft": "Mund nach links",
    "mouthRight": "Mund nach rechts", "mouthPressLeft": "Lippen pressen links",
    "mouthPressRight": "Lippen pressen rechts", "mouthStretchLeft": "Mund breit links",
    "mouthStretchRight": "Mund breit rechts", "browInnerUp": "Brauen innen hoch",
    "browDownLeft": "Braue runter links", "browDownRight": "Braue runter rechts",
    "browOuterUpLeft": "Braue außen hoch links", "browOuterUpRight": "Braue außen hoch rechts",
    "eyeBlinkLeft": "Lid zu links", "eyeBlinkRight": "Lid zu rechts", "eyeWideLeft": "Auge weit links",
    "eyeWideRight": "Auge weit rechts", "eyeSquintLeft": "Auge zusammenkneifen links",
    "eyeSquintRight": "Auge zusammenkneifen rechts", "cheekPuff": "Backen aufblasen",
    "cheekSquintLeft": "Wange hoch links", "cheekSquintRight": "Wange hoch rechts",
    "noseSneerLeft": "Nase rümpfen links", "noseSneerRight": "Nase rümpfen rechts",
    "eyeLookUpLeft": "Blick hoch links", "eyeLookUpRight": "Blick hoch rechts",
    "eyeLookDownLeft": "Blick runter links", "eyeLookDownRight": "Blick runter rechts",
    "eyeLookInLeft": "Blick zur Nase links", "eyeLookInRight": "Blick zur Nase rechts",
    "eyeLookOutLeft": "Blick nach außen links", "eyeLookOutRight": "Blick nach außen rechts",
}


# --------------------------------------------------------------------------- Export


def save_images(out_dir: str, stem: str) -> dict:
    """Haut- und Augenbild als eigene Dateien: Cascadeur liest Bilder nicht aus der FBX, und Blenders
    FBX-Export bettet eingepackte Bilder ohne Dateipfad nur einmal ein (das Augenbild fehlte)."""
    files = {}
    for img in bpy.data.images:
        if img.size[0] == 0:
            continue
        if img.packed_file is None:
            continue
        ext = ".png" if img.packed_file.data[:4] == b"\x89PNG" else ".jpg"
        path = os.path.join(out_dir, f"{stem}_{img.name.lower()}{ext}")
        with open(path, "wb") as f:
            f.write(img.packed_file.data)  # Originalbild, ohne erneute Kompression
        img.filepath = path
        files[img.name.lower()] = path
    return files


def export_all(arm_obj, mesh, out_dir: str, stem: str) -> dict:
    files = save_images(out_dir, stem)
    bpy.ops.object.select_all(action="DESELECT")
    arm_obj.select_set(True)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    fbx = os.path.join(out_dir, f"{stem}_unreal.fbx")
    bpy.ops.export_scene.fbx(
        filepath=fbx, use_selection=True, object_types={"ARMATURE", "MESH"}, add_leaf_bones=False,
        apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL", bake_anim=True,
        bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False, bake_anim_force_startend_keying=True,
        bake_anim_simplify_factor=0.0, use_armature_deform_only=False, mesh_smooth_type="FACE",
        axis_forward="-Z", axis_up="Y", primary_bone_axis="Y", secondary_bone_axis="X",
        path_mode="COPY", embed_textures=True)
    files["fbx"] = fbx
    glb = os.path.join(out_dir, f"{stem}_unreal.glb")
    bpy.ops.export_scene.gltf(
        filepath=glb, export_format="GLB", use_selection=True, export_animations=True,
        export_animation_mode="ACTIVE_ACTIONS", export_morph=True, export_morph_animation=True,
        export_skins=True, export_force_sampling=True, export_frame_range=True, export_yup=True)
    files["glb"] = glb
    return files


def main(argv: list[str]) -> None:
    args = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    if len(args) < 2:
        raise SystemExit("Aufruf: blender -b -P blender_animation.py -- eingabe.glb ausgabeordner [--ohne-video]")
    glb, out_dir = args[0], args[1]
    render = "--ohne-video" not in args
    stem = "bewegung"
    os.makedirs(out_dir, exist_ok=True)

    arm_obj, mesh = import_model(glb)
    rig = Rig(arm_obj)
    keys, segments, captions, duration = choreography()
    frames = np.arange(0, int(round(duration * FPS)) + 1)
    channels = sample_channels(keys, frames)
    action = key_body(rig, channels, frames)
    offsets = solve_ground(mesh, frames, segments)
    key_root(arm_obj, frames, offsets)
    face_action = key_face(mesh, [(min(t, duration), p) for t, p in FACE_KEYS if t <= duration + 1e-6]
                           + [(duration, FACE_KEYS[-1][1])], FACE_ACTION)
    cam = make_scene(arm_obj, mesh, frames, offsets, captions)

    files = export_all(arm_obj, mesh, out_dir, stem)
    # Mimik-Übersicht als zweite Aktion speichern, im .blend aktiv bleibt die Mimik zur Bewegung
    demo = face_demo(arm_obj, mesh, out_dir, render=False)
    mesh.data.shape_keys.animation_data.action = face_action
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = int(frames[0]), int(frames[-1])
    scene.frame_set(int(frames[0]))
    bpy.ops.file.pack_all()
    blend = os.path.join(out_dir, f"{stem}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    files["blend"] = blend

    travel = offsets[-1] - offsets[0]
    info = {
        "dauer_s": round(duration, 2), "bilder": len(frames), "fps": FPS,
        "weg_m": round(float(np.hypot(travel[0], travel[1])), 2),
        "sprunghoehe_m": round(float(offsets[:, 2].max()), 3),
        "aktionen": {"koerper": action.name, "mimik": face_action.name if face_action else None,
                     "mimik_uebersicht": FACE_DEMO_ACTION},
        "gesichtsregler": demo["regler"], "dateien": files,
    }
    if render:
        scene.camera = cam
        render_video(os.path.join(out_dir, "bewegung.mp4"), int(frames[0]), int(frames[-1]))
        info["dateien"]["video"] = os.path.join(out_dir, "bewegung.mp4")
        face_demo(arm_obj, mesh, out_dir, render=True)
        info["dateien"]["video_mimik"] = os.path.join(out_dir, "mimik_regler.mp4")
    print("FITSCAN_INFO " + json.dumps(info))


if __name__ == "__main__":
    main(sys.argv)
