"""Bewegung und Mimik für ein fertiges Körpermodell.

1. Gesichtsregler (face_rig.py) für die Kopfform der Person berechnen und in das GLB schreiben.
2. Blender (blender_animation.py) bewegt das Modell: Sprung, Rolle vorwärts, Sprint mit Sturz,
   dazu eine eigene Mimik-Aktion. Ausgabe: .blend, FBX (Cascadeur, Unreal), GLB, Vorschauvideos.

Das GLB von FitScan führt die Punkte nach Annys Nummerierung (an UV-Nähten doppelt), deshalb lässt
sich jedes exportierte koerper*.glb wieder Annys Netz zuordnen, ohne die Scan-Daten zu brauchen.
"""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import numpy as np

from .blender import find_blender

SCRIPT = Path(__file__).with_name("blender_animation.py")
FLOAT = 5126


# ------------------------------------------------------------------------- GLB lesen/schreiben


def read_glb(path: Path) -> tuple[dict, bytearray]:
    data = Path(path).read_bytes()
    magic, _, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"Keine GLB-Datei: {path}")
    js_len, _ = struct.unpack_from("<I4s", data, 12)
    gltf = json.loads(data[20:20 + js_len])
    bin_len, _ = struct.unpack_from("<I4s", data, 20 + js_len)
    start = 28 + js_len
    return gltf, bytearray(data[start:start + bin_len])


def write_glb_raw(gltf: dict, blob: bytearray, path: Path) -> Path:
    while len(blob) % 4:
        blob.append(0)
    gltf["buffers"] = [{"byteLength": len(blob)}]
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack("<I4s", len(js), b"JSON"))
        f.write(js)
        f.write(struct.pack("<I4s", len(blob), b"BIN\x00"))
        f.write(bytes(blob))
    return path


_TYPES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_DTYPES = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}


def accessor(gltf: dict, blob: bytearray, index: int) -> np.ndarray:
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(_DTYPES[acc["componentType"]])
    width = _TYPES[acc["type"]]
    offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride", dtype.itemsize * width)
    raw = np.frombuffer(bytes(blob), np.uint8, count=stride * (acc["count"] - 1) + dtype.itemsize * width,
                        offset=offset)
    rows = np.lib.stride_tricks.as_strided(raw, (acc["count"], dtype.itemsize * width), (stride, 1))
    return np.ascontiguousarray(rows).view(dtype).reshape(acc["count"], width).squeeze()


def add_morph_targets(gltf: dict, blob: bytearray, deltas: dict[str, np.ndarray]) -> None:
    """Formregler (je GLB-Punkt, glTF-Koordinaten) an alle Teile des ersten Netzes anhängen."""
    mesh = gltf["meshes"][0]
    names = mesh.setdefault("extras", {}).setdefault("targetNames", [])
    for name, delta in deltas.items():
        delta = np.ascontiguousarray(delta, np.float32)
        while len(blob) % 4:
            blob.append(0)
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": len(blob), "byteLength": delta.nbytes,
                                    "target": 34962})
        blob.extend(delta.tobytes())
        gltf["accessors"].append({"bufferView": len(gltf["bufferViews"]) - 1, "componentType": FLOAT,
                                  "count": len(delta), "type": "VEC3",
                                  "min": delta.min(0).tolist(), "max": delta.max(0).tolist()})
        acc = len(gltf["accessors"]) - 1
        for prim in mesh["primitives"]:
            prim.setdefault("targets", []).append({"POSITION": acc})
        names.append(name)
    mesh["weights"] = [0.0] * len(names)


# ------------------------------------------------------------------------- Zuordnung zu Anny


def glb_to_anny(gltf: dict, blob: bytearray, model) -> tuple[np.ndarray, np.ndarray]:
    """Gibt je GLB-Punkt die Anny-Nummer und die Anny-Punkte (Z oben, Meter) zurück."""
    prim = gltf["meshes"][0]["primitives"][0]
    pos = accessor(gltf, blob, prim["attributes"]["POSITION"]).astype(np.float64)
    # Punkte sind nach Anny-Nummer sortiert, Doppelte an UV-Nähten liegen direkt hintereinander
    new = np.ones(len(pos), bool)
    new[1:] = np.any(pos[1:] != pos[:-1], axis=1)
    index = np.cumsum(new) - 1
    if index[-1] + 1 != model.num_vertices:
        raise ValueError(f"GLB passt nicht zu Anny: {index[-1] + 1} statt {model.num_vertices} Punkte.")
    faces = np.concatenate([accessor(gltf, blob, p["indices"]).reshape(-1, 3)
                            for p in gltf["meshes"][0]["primitives"]])
    mine = {tuple(sorted(f)) for f in index[faces].tolist()}
    theirs = {tuple(sorted(f)) for f in model.faces.tolist()}
    if mine != theirs:
        raise ValueError("GLB passt nicht zu Annys Dreiecken (anderes Netz?).")
    first = np.flatnonzero(new)
    y_up = pos[first]
    anny = np.stack([y_up[:, 0], -y_up[:, 2], y_up[:, 1]], 1)  # glTF (Y oben) -> Anny (Z oben)
    return index, anny


def add_face_controls(glb: Path, target: Path, model=None) -> list[str]:
    """Schreibt eine Kopie des GLB mit den Gesichtsreglern. Gibt deren Namen zurück."""
    from .body import BodyModel
    from .face_rig import face_morphs

    model = model or BodyModel.get()
    gltf, blob = read_glb(glb)
    index, anny = glb_to_anny(gltf, blob, model)
    morphs = face_morphs(model, anny)
    known = set(gltf["meshes"][0].get("extras", {}).get("targetNames", []))
    deltas = {}
    for name, d in morphs.items():
        if name in known:
            continue
        y_up = np.stack([d[:, 0], d[:, 2], -d[:, 1]], 1)
        deltas[name] = y_up[index]
    add_morph_targets(gltf, blob, deltas)
    write_glb_raw(gltf, blob, target)
    return list(deltas)


# ------------------------------------------------------------------------- Blender


def animate(glb: Path, out_dir: Path, video: bool = True, blender: Path | None = None,
            timeout: int = 7200, model=None) -> dict:
    """Gesichtsregler hinzufügen, Bewegung aufnehmen, exportieren und (optional) Videos rendern."""
    blender = blender or find_blender()
    if blender is None:
        raise FileNotFoundError(
            "Blender nicht gefunden. Blender kostenlos von blender.org installieren oder den Pfad "
            "zu blender.exe in der Umgebungsvariable FITSCAN_BLENDER angeben.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with_face = out_dir / "koerper_mimik.glb"
    controls = add_face_controls(Path(glb), with_face, model=model)
    cmd = [str(blender), "-b", "--factory-startup", "-P", str(SCRIPT), "--", str(with_face), str(out_dir)]
    if not video:
        cmd.append("--ohne-video")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    for line in proc.stdout.splitlines():
        if line.startswith("FITSCAN_INFO "):
            info = json.loads(line[len("FITSCAN_INFO "):])
            info["neue_gesichtsregler"] = controls
            info["dateien"]["glb_mimik"] = str(with_face)
            return info
    raise RuntimeError("Bewegungsaufnahme in Blender fehlgeschlagen:\n" + (proc.stderr or proc.stdout)[-3000:])
