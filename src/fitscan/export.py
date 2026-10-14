"""Export des riggbaren Körpermodells: GLB (mit Skelett, Gewichten, Formreglern) und OBJ.

Eigener Exporter ohne fremde Laufzeit-Abhängigkeiten. Ziel sind Blender und Cascadeur:
- echte Größe in Metern, Y nach oben, Füße auf dem Boden (y = 0), Blick nach +Z
- Ruhepose aus Anny (A-Pose), symmetrisches Skelett mit einheitlichen Seiten-Endungen
- vier Gelenkeinflüsse pro Punkt (das lesen alle Programme)
- keine UV-Nähte, damit das Netz geschlossen bleibt
- Formregler (Morph Targets) für mehr/weniger Gewicht und Muskeln
"""

from __future__ import annotations

import dataclasses
import json
import struct
from pathlib import Path

import numpy as np

from .body import BodyModel, BodyParams
from .mesh_check import MeshReport, check_mesh, vertex_normals

GROUND_JOINT = {"anny": "ground", "unreal": "root"}
PRESETS = ("anny", "unreal")

# Anny -> Namen wie beim Unreal-Mannequin (von Cascadeur, Unreal und vielen Werkzeugen erkannt)
_UNREAL_BASE = {
    "root": "pelvis", "pelvis": "hip", "upperleg01": "thigh", "upperleg02": "thigh_twist_01",
    "lowerleg01": "calf", "lowerleg02": "calf_twist_01", "foot": "foot",
    "toe1-1": "bigtoe_01", "toe1-2": "bigtoe_02",
    "toe2-1": "indextoe_01", "toe2-2": "indextoe_02", "toe2-3": "indextoe_03",
    "toe3-1": "middletoe_01", "toe3-2": "middletoe_02", "toe3-3": "middletoe_03",
    "toe4-1": "ringtoe_01", "toe4-2": "ringtoe_02", "toe4-3": "ringtoe_03",
    "toe5-1": "littletoe_01", "toe5-2": "littletoe_02", "toe5-3": "littletoe_03",
    "spine05": "spine_01", "spine04": "spine_02", "spine03": "spine_03", "spine02": "spine_04",
    "spine01": "spine_05", "clavicle": "clavicle", "shoulder01": "shoulder",
    "upperarm01": "upperarm", "upperarm02": "upperarm_twist_01",
    "lowerarm01": "lowerarm", "lowerarm02": "lowerarm_twist_01", "wrist": "hand",
    "finger1-1": "thumb_01", "finger1-2": "thumb_02", "finger1-3": "thumb_03",
    "metacarpal1": "index_metacarpal", "finger2-1": "index_01", "finger2-2": "index_02",
    "finger2-3": "index_03",
    "metacarpal2": "middle_metacarpal", "finger3-1": "middle_01", "finger3-2": "middle_02",
    "finger3-3": "middle_03",
    "metacarpal3": "ring_metacarpal", "finger4-1": "ring_01", "finger4-2": "ring_02",
    "finger4-3": "ring_03",
    "metacarpal4": "pinky_metacarpal", "finger5-1": "pinky_01", "finger5-2": "pinky_02",
    "finger5-3": "pinky_03",
    "neck01": "neck_01", "neck02": "neck_02", "neck03": "neck_03", "head": "head", "eye": "eye",
}

MORPH_STEP = 0.25  # wie weit ein Formregler auf 1.0 den Phänotyp verschiebt
MORPHS = (("weight_up", "weight", +1), ("weight_down", "weight", -1),
          ("muscle_up", "muscle", +1), ("muscle_down", "muscle", -1))

SKIN_COLOR = [0.80, 0.62, 0.52, 1.0]


def bone_name(name: str, preset: str = "anny") -> str:
    if preset == "anny":
        return name
    if preset != "unreal":
        raise ValueError(f"Unbekanntes Namensschema: {preset}")
    base, _, side = name.partition(".")
    mapped = _UNREAL_BASE.get(base, base.replace("-", "_"))
    return f"{mapped}_{side.lower()}" if side else mapped


# ------------------------------------------------------------------------- Rig vorbereiten


@dataclasses.dataclass
class RiggedMesh:
    """Alles, was ein Exporter braucht, schon in Y-oben-Koordinaten mit Füßen auf y = 0."""

    vertices: np.ndarray  # (V, 3) float32
    normals: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3) uint32
    joint_names: list[str]
    joint_parents: list[int]  # -1 = keine Eltern
    joint_global: np.ndarray  # (J, 4, 4) Ruhepose
    joints4: np.ndarray  # (V, 4) uint16
    weights4: np.ndarray  # (V, 4) float32, Summe 1
    morphs: dict[str, np.ndarray]  # Name -> (V, 3) Verschiebungen
    report: MeshReport


# Anny (Z oben, Blick -Y) -> glTF (Y oben, Blick +Z): (x, y, z) -> (x, z, -y)
_TO_Y_UP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def top4_weights(indices: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-weights, axis=1)[:, :4]
    j = np.take_along_axis(indices, order, 1)
    w = np.take_along_axis(weights, order, 1).astype(np.float64)
    w[w < 1e-4] = 0.0
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    j = np.where(w > 0, j, 0)
    return j.astype(np.uint16), w.astype(np.float32)


def build_rig(model: BodyModel, params: BodyParams, preset: str = "anny",
              morphs: bool = True) -> RiggedMesh:
    if params.offsets is not None:
        from .fit import repair_offsets

        params, _ = repair_offsets(model, params)
    rest = model.rest_numpy(params)
    verts = rest["vertices"].astype(np.float64)
    template = None
    if params.offsets is not None:
        plain = params.copy()
        plain.offsets = None
        template = model.rest_numpy(plain)["vertices"]
    digits = model.dominant_bone_mask(lambda n: n.startswith(("toe", "finger", "metacarpal")))
    report = check_mesh(verts, model.faces, template=template, digit_faces=digits[model.faces].all(1))

    to_y = _TO_Y_UP.copy()
    to_y[1, 3] = -verts[:, 2].min()  # Füße auf den Boden

    def convert(v: np.ndarray) -> np.ndarray:
        return v @ to_y[:3, :3].T + to_y[:3, 3]

    v_y = convert(verts)
    bone_global = to_y[None] @ rest["bone_poses"].astype(np.float64)

    # Zusätzliches Bodengelenk als oberste Wurzel (Cascadeur und Spiele-Engines erwarten das)
    ground = np.eye(4)
    names = [GROUND_JOINT[preset]] + [bone_name(n, preset) for n in model.bone_labels]
    parents = [-1] + [p + 1 if p is not None and p >= 0 else 0 for p in model.bone_parents]
    globals_ = np.concatenate([ground[None], bone_global], 0)

    j4, w4 = top4_weights(model.vertex_bone_indices, model.vertex_bone_weights)
    j4 = np.where(w4 > 0, j4 + 1, 0).astype(np.uint16)  # wegen des Bodengelenks verschoben

    morph_deltas: dict[str, np.ndarray] = {}
    if morphs:
        for name, key, sign in MORPHS:
            p = params.copy()
            value = float(np.clip(p.phenotype.get(key, 0.5) + sign * MORPH_STEP, 0.0, 1.0))
            if abs(value - p.phenotype.get(key, 0.5)) < 1e-3:
                continue
            p.phenotype[key] = value
            other = convert(model.rest_numpy(p)["vertices"].astype(np.float64))
            morph_deltas[name] = (other - v_y).astype(np.float32)

    return RiggedMesh(
        vertices=v_y.astype(np.float32),
        normals=vertex_normals(v_y, model.faces).astype(np.float32),
        faces=model.faces.astype(np.uint32),
        joint_names=names,
        joint_parents=parents,
        joint_global=globals_,
        joints4=j4,
        weights4=w4,
        morphs=morph_deltas,
        report=report,
    )


# ------------------------------------------------------------------------- GLB schreiben


def _matrix_to_trs(m: np.ndarray) -> tuple[list[float], list[float]]:
    t = m[:3, 3]
    r = m[:3, :3]
    # Rotationsmatrix -> Quaternion (x, y, z, w)
    tr = np.trace(r)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return [float(x) for x in t], [float(x) for x in q]


class _Buffer:
    def __init__(self):
        self.data = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def add(self, array: np.ndarray, component: int, kind: str, target: int | None = None,
            minmax: bool = False) -> int:
        while len(self.data) % 4:
            self.data.append(0)
        raw = np.ascontiguousarray(array).tobytes()
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(raw)}
        if target is not None:
            view["target"] = target
        self.data.extend(raw)
        self.views.append(view)
        acc = {"bufferView": len(self.views) - 1, "componentType": component,
               "count": int(array.shape[0]), "type": kind}
        if minmax:
            flat = array.reshape(array.shape[0], -1)
            acc["min"] = [float(x) for x in flat.min(0)]
            acc["max"] = [float(x) for x in flat.max(0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1


FLOAT, UINT16, UINT32 = 5126, 5123, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


def write_glb(rig: RiggedMesh, path: Path, name: str = "FitScan") -> Path:
    buf = _Buffer()
    pos = buf.add(rig.vertices, FLOAT, "VEC3", ARRAY_BUFFER, minmax=True)
    nor = buf.add(rig.normals, FLOAT, "VEC3", ARRAY_BUFFER)
    jnt = buf.add(rig.joints4, UINT16, "VEC4", ARRAY_BUFFER)
    wgt = buf.add(rig.weights4, FLOAT, "VEC4", ARRAY_BUFFER)
    idx = buf.add(rig.faces.reshape(-1), UINT32, "SCALAR", ELEMENT_ARRAY_BUFFER)
    targets, target_names = [], []
    for morph_name, delta in rig.morphs.items():
        targets.append({"POSITION": buf.add(delta, FLOAT, "VEC3", ARRAY_BUFFER, minmax=True)})
        target_names.append(morph_name)
    ibm = np.linalg.inv(rig.joint_global).astype(np.float32)
    ibm_acc = buf.add(ibm.transpose(0, 2, 1).reshape(-1, 16), FLOAT, "MAT4")  # spaltenweise

    # 0: Hülle für das Skelett, 1: Netz (laut glTF-Regel nicht unterhalb von Knoten mit Transform)
    nodes: list[dict] = [{"name": name, "children": []}, {"name": f"{name}_Body", "mesh": 0, "skin": 0}]
    first_joint = len(nodes)
    for j, (jname, parent) in enumerate(zip(rig.joint_names, rig.joint_parents, strict=True)):
        local = rig.joint_global[j] if parent < 0 else np.linalg.inv(rig.joint_global[parent]) @ rig.joint_global[j]
        t, q = _matrix_to_trs(local)
        node = {"name": jname, "translation": t, "rotation": q}
        nodes.append(node)
    for j, parent in enumerate(rig.joint_parents):
        if parent < 0:
            nodes[0]["children"].append(first_joint + j)
        else:
            nodes[first_joint + parent].setdefault("children", []).append(first_joint + j)

    primitive = {
        "attributes": {"POSITION": pos, "NORMAL": nor, "JOINTS_0": jnt, "WEIGHTS_0": wgt},
        "indices": idx, "material": 0, "mode": 4,
    }
    mesh: dict = {"name": f"{name}_Body", "primitives": [primitive]}
    if targets:
        primitive["targets"] = targets
        mesh["weights"] = [0.0] * len(targets)
        mesh["extras"] = {"targetNames": target_names}

    gltf = {
        "asset": {"version": "2.0", "generator": "FitScan"},
        "scene": 0,
        "scenes": [{"name": name, "nodes": [0, 1]}],
        "nodes": nodes,
        "meshes": [mesh],
        "skins": [{"name": f"{name}_Rig", "joints": list(range(first_joint, first_joint + len(rig.joint_names))),
                   "inverseBindMatrices": ibm_acc, "skeleton": first_joint}],
        "materials": [{"name": "Haut", "pbrMetallicRoughness": {
            "baseColorFactor": SKIN_COLOR, "metallicFactor": 0.0, "roughnessFactor": 0.6}}],
        "accessors": buf.accessors,
        "bufferViews": buf.views,
        "buffers": [{"byteLength": len(buf.data)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    while len(buf.data) % 4:
        buf.data.append(0)
    total = 12 + 8 + len(js) + 8 + len(buf.data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON"))
        f.write(js)
        f.write(struct.pack("<I4s", len(buf.data), b"BIN\x00"))
        f.write(bytes(buf.data))
    return path


def write_obj(rig: RiggedMesh, path: Path) -> Path:
    """Reines Netz ohne Skelett, z. B. für 3D-Druck oder Bildhauer-Programme."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# FitScan Körpermodell, Meter, Y oben", "o FitScan_Body"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in rig.vertices]
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in rig.normals]
    lines += [f"f {a}//{a} {b}//{b} {c}//{c}" for a, b, c in (rig.faces.astype(np.int64) + 1)]
    path.write_text("\n".join(lines) + "\n", "utf-8")
    return path


# ------------------------------------------------------------------------- Cascadeur-Prüfung


def cascadeur_check(rig: RiggedMesh) -> list[str]:
    """Hinweise, falls das Modell gegen Cascadeurs Importregeln verstößt (leer = alles gut)."""
    problems = []
    height = float(rig.vertices[:, 1].max() - rig.vertices[:, 1].min())
    if not 0.5 < height < 2.6:
        problems.append(f"Größe {height:.2f} m wirkt nicht wie echte Größe in Metern.")
    if abs(float(rig.vertices[:, 1].min())) > 1e-3:
        problems.append("Füße stehen nicht auf dem Boden (y = 0).")
    scales = [np.linalg.norm(m[:3, :3], axis=0) for m in rig.joint_global]
    if not np.allclose(scales, 1.0, atol=1e-3):
        problems.append("Gelenke haben einen Maßstab ungleich 1.")
    pos = {n: rig.joint_global[i][:3, 3] for i, n in enumerate(rig.joint_names)}
    left = [n for n in pos if n.endswith((".L", "_l"))]
    for n in left:
        other = n[:-2] + (".R" if n.endswith(".L") else "_r")
        if other not in pos:
            problems.append(f"Gegenstück zu {n} fehlt.")
            continue
        mirrored = pos[n] * np.array([-1.0, 1.0, 1.0])
        if np.linalg.norm(mirrored - pos[other]) > 0.01:
            problems.append(f"{n} und {other} liegen nicht spiegelgleich.")
    if len(set(rig.joint_names)) != len(rig.joint_names):
        problems.append("Gelenknamen sind nicht eindeutig.")
    return problems


def export_body(model: BodyModel, params: BodyParams, folder: Path, name: str = "koerper",
                presets: tuple[str, ...] = PRESETS, obj: bool = True) -> dict[str, Path | list[str]]:
    """Schreibt GLB je Namensschema und OBJ in einen Ordner. Gibt Pfade und Prüfhinweise zurück."""
    folder = Path(folder)
    out: dict[str, Path | list[str]] = {}
    for preset in presets:
        rig = build_rig(model, params, preset=preset)
        if not rig.report.ok:
            raise ValueError(rig.report.summary())
        suffix = "" if preset == "anny" else f"_{preset}"
        out[f"glb{suffix}"] = write_glb(rig, folder / f"{name}{suffix}.glb", name="FitScan")
        if preset == presets[0]:
            out["hinweise"] = cascadeur_check(rig) + rig.report.warnings
            if obj:
                out["obj"] = write_obj(rig, folder / f"{name}.obj")
    return out
