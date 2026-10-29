"""Anny-Daten und Prüfwerte für die Swift-App (`apple/FitScanCore`).

Die App rechnet Annys Körper selbst, offline und ohne PyTorch. Dieses Modul schreibt dafür
- `anny_v1.fsan`: Vorlage, Formen (dünn besetzt, je Form auf int16 skaliert), Knochen und
  Gewichte. Der Datenteil ist mit Deflate gepackt, wie ihn Apples `NSData.decompressed(using:
  .zlib)` liest.
- `referenz.json`: Prüfwerte vom PC (Punkte, Knochen, Maße, Anpassung an Maße). Die Swift-Tests
  rechnen dagegen.

`AnnyData` liest die Datei mit numpy genau so, wie FitScanCore sie liest und rechnet. So lässt
sich der Rechenweg der App hier gegen Anny prüfen, auch ohne Swift-Compiler.

Aufbau der Datei (alles Little Endian):
    "FSAN" | u32 Version | u32 Länge JSON | u32 Länge Daten | u32 gepackte Länge (0 = ungepackt)
    | JSON | Daten
Im JSON stehen Namen, Anker und für jedes Feld Lage, Typ und Form im (entpackten) Datenteil.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

from .body import PHENOTYPES, BodyModel, BodyParams

MAGIC = b"FSAN"
VERSION = 1
ALIGN = 16
_DTYPES = {"f32": np.float32, "i16": np.int16, "u16": np.uint16, "u32": np.uint32, "u8": np.uint8}


# ----------------------------------------------------------------------------- schreiben


def _variations() -> list[tuple[str, list[str]]]:
    from anny.models.model_data import PHENOTYPE_VARIATIONS

    return [(k, list(v)) for k, v in PHENOTYPE_VARIATIONS.items()]


def _anchors(model: BodyModel) -> dict[str, list[float]]:
    return {k: [float(x) for x in v.cpu().numpy()] for k, v in
            ((k, model.anny.anchors[k]) for k in ("age", "gender", "muscle", "weight", "height",
                                                   "proportions", "cupsize", "firmness"))}


def build_arrays(model: BodyModel) -> tuple[dict, dict[str, np.ndarray]]:
    a = model.anny
    np32 = lambda t: t.detach().cpu().numpy().astype(np.float32)  # noqa: E731
    shapes = a.blendshapes.detach().cpu().numpy().astype(np.float64)  # (S, V, 3)
    num_shapes, num_vertices = shapes.shape[:2]
    mag = np.abs(shapes).max(-1)
    start = np.zeros(num_shapes + 1, np.uint32)
    scale = np.zeros(num_shapes, np.float32)
    vert, delta = [], []
    for s in range(num_shapes):
        idx = np.nonzero(mag[s] > 0)[0]
        m = float(mag[s].max()) if len(idx) else 0.0
        q = m / 32767.0 if m > 0 else 1.0
        scale[s] = q
        vert.append(idx.astype(np.uint16))
        delta.append(np.round(shapes[s, idx] / q).astype(np.int16))
        start[s + 1] = start[s] + len(idx)
    if num_vertices > 65535:
        raise ValueError("Zu viele Punkte für u16-Indizes.")
    arrays = {
        "template_vertices": np32(a.template_vertices),
        "faces": model.faces.astype(np.uint32),
        "vertex_bone_indices": model.vertex_bone_indices.astype(np.uint8),
        "vertex_bone_weights": model.vertex_bone_weights.astype(np.float32),
        "phenotype_mask": a.stacked_phenotype_blend_shapes_mask.detach().cpu().numpy().astype(np.uint8),
        "template_bone_heads": np32(a.template_bone_heads),
        "bone_heads_blendshapes": np32(a.bone_heads_blendshapes),
        "bone_template_orientation": np32(a.bone_template_orientation_matrices),
        "bone_orientation_blendshapes": np32(a.bone_orientation_blendshapes),
        "reference_bone_orientations": np32(a.reference_bone_orientations),
        "shape_start": start,
        "shape_scale": scale,
        "shape_vertex": np.concatenate(vert),
        "shape_delta": np.concatenate(delta),
        "texture_coordinates": np32(a.texture_coordinates),
        "face_texture_indices": a.face_texture_coordinate_indices.detach().cpu().numpy().astype(np.uint32),
    }
    meta = {
        "format": "fitscan-anny",
        "version": VERSION,
        "num_vertices": int(num_vertices),
        "num_bones": len(model.bone_labels),
        "num_shapes": int(num_shapes),
        "num_macro_shapes": int(arrays["phenotype_mask"].shape[0]),
        "phenotype_labels": list(model.phenotype_labels),
        "local_labels": list(model.local_labels),
        "bone_labels": list(model.bone_labels),
        "bone_parents": [int(p) if p is not None else -1 for p in model.bone_parents],
        "phenotype_variations": _variations(),
        "anchors": _anchors(model),
        "fixed_phenotypes": {"african": 0.5, "asian": 0.5, "caucasian": 0.5, "cupsize": 0.5,
                             "firmness": 0.5},
    }
    return meta, arrays


def write_data(model: BodyModel, path: Path, compress: bool = True) -> Path:
    meta, arrays = build_arrays(model)
    payload = bytearray()
    layout = {}
    for name, arr in arrays.items():
        while len(payload) % ALIGN:
            payload.append(0)
        kind = next(k for k, t in _DTYPES.items() if arr.dtype == t)
        raw = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<")).tobytes()
        layout[name] = {"offset": len(payload), "dtype": kind, "shape": list(arr.shape)}
        payload.extend(raw)
    meta["arrays"] = layout
    js = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    packed = b""
    if compress:
        c = zlib.compressobj(9, zlib.DEFLATED, -15)  # rohes Deflate, wie COMPRESSION_ZLIB bei Apple
        packed = c.compress(bytes(payload)) + c.flush()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", VERSION, len(js), len(payload), len(packed)))
        f.write(js)
        f.write(packed if compress else bytes(payload))
    return path


# ------------------------------------------------------------------------------- lesen


def interpolation_weights(value: float, anchors: list[float]) -> list[float]:
    """Wie anny.utils.interpolation.linear_interpolation_coefficients (ohne Extrapolation)."""
    n = len(anchors)
    v = min(max(value, anchors[0]), anchors[-1])
    idx = next((i for i, a in enumerate(anchors) if a >= v), n)  # searchsorted(side="left")
    idx = min(max(idx, 1), n - 1)
    lo, hi = anchors[idx - 1], anchors[idx]
    alpha = (v - lo) / (hi - lo)
    out = [0.0] * n
    out[idx - 1] = 1.0 - alpha
    out[idx] = alpha
    return out


def special_procrustes(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    d = np.sign(np.linalg.det(u) * np.linalg.det(vt))
    u[..., :, -1] *= d[..., None]
    return u @ vt


class AnnyData:
    """Liest `anny_v1.fsan` und rechnet Annys Ruhepose so, wie FitScanCore es tut."""

    def __init__(self, path: Path):
        raw = Path(path).read_bytes()
        if raw[:4] != MAGIC:
            raise ValueError("Keine FitScan-Anny-Datei.")
        version, js_len, payload_len, packed_len = struct.unpack("<IIII", raw[4:20])
        if version != VERSION:
            raise ValueError(f"Unbekannte Version {version}.")
        self.meta = json.loads(raw[20:20 + js_len].decode("utf-8"))
        body = raw[20 + js_len:]
        payload = zlib.decompress(body, -15) if packed_len else body
        if len(payload) != payload_len:
            raise ValueError("Datenteil hat die falsche Länge.")
        self.arrays = {}
        for name, spec in self.meta["arrays"].items():
            dt = np.dtype(_DTYPES[spec["dtype"]]).newbyteorder("<")
            count = int(np.prod(spec["shape"]))
            self.arrays[name] = np.frombuffer(payload, dt, count, spec["offset"]).reshape(spec["shape"])
        self.phenotype_labels = self.meta["phenotype_labels"]
        self.local_labels = self.meta["local_labels"]
        self.bone_labels = self.meta["bone_labels"]

    def coefficients(self, phenotype: dict[str, float], local: dict[str, float]) -> np.ndarray:
        fixed = self.meta["fixed_phenotypes"]
        anchors = self.meta["anchors"]
        value = lambda k: float(phenotype.get(k, fixed.get(k, 0.5)))  # noqa: E731
        phens: list[float] = []
        for feature, keys in self.meta["phenotype_variations"]:
            if feature == "race":
                vals = [value(k) for k in keys]
                total = sum(vals)
                phens.extend([v / total if total > 0 else 1 / 3 for v in vals])
            else:
                phens.extend(interpolation_weights(value(feature), anchors[feature]))
        mask = self.arrays["phenotype_mask"].astype(bool)
        p = np.asarray(phens)
        macro = np.where(mask, p[None, :], 1.0).prod(1)
        loc = np.zeros(2 * len(self.local_labels))
        for i, name in enumerate(self.local_labels):
            v = float(local.get(name, 0.0))
            loc[2 * i] = max(v, 0.0)
            loc[2 * i + 1] = max(-v, 0.0)
        return np.concatenate([macro, loc])

    def rest(self, params: BodyParams) -> dict[str, np.ndarray]:
        c = self.coefficients(params.phenotype, params.local)
        ar = self.arrays
        verts = ar["template_vertices"].astype(np.float64).copy()
        start, scale = ar["shape_start"], ar["shape_scale"]
        for s in np.nonzero(c)[0]:
            a, b = int(start[s]), int(start[s + 1])
            np.add.at(verts, ar["shape_vertex"][a:b], (c[s] * scale[s]) * ar["shape_delta"][a:b].astype(np.float64))
        if params.offsets is not None:
            verts += params.offsets
        heads = ar["template_bone_heads"] + np.einsum("s,sjk->jk", c, ar["bone_heads_blendshapes"])
        orient = ar["bone_template_orientation"] + np.einsum("s,sjkl->jkl", c, ar["bone_orientation_blendshapes"])
        rot = special_procrustes(orient.astype(np.float64))
        poses = np.zeros((len(heads), 4, 4))
        poses[:, :3, :3] = rot
        poses[:, :3, 3] = heads * params.scale
        poses[:, 3, 3] = 1.0
        return {"vertices": verts * params.scale, "bone_heads": heads * params.scale, "bone_poses": poses}


# ---------------------------------------------------------------------------- Prüfwerte

# Künstliche Personen (keine echten Daten): Regler für den Körpervergleich
REFERENCE_BODIES = {
    "standard": BodyParams(),
    "kraeftig": BodyParams(
        phenotype={"gender": 0.0, "age": 0.55, "muscle": 0.8, "weight": 0.65, "height": 0.7, "proportions": 0.4},
        local={"measure-waist-circ-incr": 0.3, "measure-hips-circ-incr": -0.4, "measure-upperarm-circ-incr": 0.5},
        scale=1.02),
    "frau_60": BodyParams(
        phenotype={"gender": 1.0, "age": 0.72, "muscle": 0.35, "weight": 0.6, "height": 0.3, "proportions": 0.6},
        local={"measure-bust-circ-incr": 0.2, "measure-calf-circ-incr": -0.3}, scale=0.98),
    "divers_jung": BodyParams(
        phenotype={"gender": 0.5, "age": 0.5, "muscle": 0.5, "weight": 0.3, "height": 0.5, "proportions": 0.5},
        local={"measure-neck-circ-incr": 0.6, "measure-thigh-circ-incr": -0.2}),
}

# Künstliche Eingaben für die Anpassung an Maße
REFERENCE_PEOPLE = {
    "nur_groesse_gewicht": {"gender": "maennlich", "age_years": 30,
                            "measures": {"height": 178.0, "weight": 76.0}},
    "mit_massband": {"gender": "maennlich", "age_years": 35,
                     "measures": {"height": 181.0, "weight": 88.0, "chest": 106.0, "waist": 92.0,
                                  "hips": 97.0, "thigh": 58.0, "upperarm": 37.0, "neck": 40.0}},
    "frau": {"gender": "weiblich", "age_years": 45,
             "measures": {"height": 166.0, "weight": 64.0, "waist": 78.0, "hips": 100.0}},
}

SAMPLE_STEP = 23  # jeder 23. Punkt kommt in die Prüfwerte


def reference(model: BodyModel, fit_steps: int = 240) -> dict:
    from .fit import Person, fit_to_measurements
    from .measure import measure_params

    sample = list(range(0, model.num_vertices, SAMPLE_STEP))
    bodies = {}
    for name, params in REFERENCE_BODIES.items():
        rest = model.rest_numpy(params)
        bodies[name] = {
            "params": params.to_json(),
            "sample_vertices": sample,
            "vertices": np.round(rest["vertices"][sample].astype(np.float64), 7).tolist(),
            "bone_heads": np.round(rest["bone_heads"].astype(np.float64), 7).tolist(),
            "bone_rotations": np.round(rest["bone_poses"][:, :3, :3].astype(np.float64), 6).tolist(),
            "measures": measure_params(model, params),
        }
    fits = {}
    for name, spec in REFERENCE_PEOPLE.items():
        person = Person(gender=spec["gender"], age_years=spec["age_years"], measures=dict(spec["measures"]))
        params, measures = fit_to_measurements(model, person, steps=fit_steps)
        fits[name] = {"person": spec, "pc_params": params.to_json(), "pc_measures": measures}
    masks = {
        "arm_vertices": int(model.arm_vertex_mask().sum()),
    }
    return {"format": "fitscan-referenz", "version": VERSION, "sample_step": SAMPLE_STEP,
            "phenotypes": PHENOTYPES, "bodies": bodies, "fits": fits, "masks": masks,
            "glb": {"joints": len(model.bone_labels) + 1, "vertices": model.num_vertices,
                    "triangles": int(len(model.faces)),
                    "morphs": ["weight_up", "weight_down", "muscle_up", "muscle_down"]}}


# ---------------------------------------------------------------------------- Fotoscan

def photo_settings() -> dict:
    """Einstellungen des Foto- und Videoscans für FitScanCore (`fotoscan.json`)."""
    from . import fit, photofit
    from .camera import DEFAULT_FOV_LONG_SIDE_DEG, KEYPOINT_OFFSETS, MEDIAPIPE_TO_BONE

    return {
        "format": "fitscan-fotoscan", "version": 1,
        "work_side": photofit.WORK_SIDE, "contour_points": photofit.CONTOUR_POINTS,
        "pose_bones": list(fit.POSE_BONES), "video_locals": list(fit.VIDEO_LOCALS),
        "keypoints": {str(k): {"bone": b, "offset": [round(float(x), 6) for x in KEYPOINT_OFFSETS[k]]}
                      for k, b in MEDIAPIPE_TO_BONE.items()},
        "silhouette_weight": fit.SILHOUETTE_WEIGHT, "local_reg": fit.LOCAL_REG, "pose_reg": photofit.POSE_REG,
        "keypoint_weight_shape": fit.KEYPOINT_WEIGHT_SHAPE, "muscle_starts": list(fit.MUSCLE_STARTS),
        "learning_rate": photofit.LR, "probe_every": photofit.PROBE_EVERY, "steps": list(photofit.STEPS),
        "fov_long_side_deg": DEFAULT_FOV_LONG_SIDE_DEG,
    }


def mask_runs(mask: np.ndarray) -> list[int]:
    """Maske zeilenweise als abwechselnde Lauflängen, beginnend mit Hintergrund."""
    flat = mask.reshape(-1).astype(np.int8)
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(flat)) + 1, [len(flat)]])
    runs = np.diff(bounds).astype(int).tolist()
    return [0, *runs] if flat[0] else runs


def mask_from_runs(runs: list[int], width: int, height: int) -> np.ndarray:
    flat = np.zeros(width * height, bool)
    pos = 0
    for i, n in enumerate(runs):
        if i % 2:
            flat[pos:pos + n] = True
        pos += n
    return flat.reshape(height, width)


PHOTO_BODY = "kraeftig"  # künstlicher Testkörper für die Fotoscan-Prüfwerte
PHOTO_POSE_SEED = 0.1


def photo_reference(model: BodyModel, data: AnnyData) -> dict:
    """Prüfwerte des Fotoscans: vier künstliche Bilder, Zwischenwerte des Rechenwegs und das Ergebnis."""
    import math

    import torch

    from . import photofit, synthetic
    from .body import param_to_age
    from .camera import MEDIAPIPE_TO_BONE, Intrinsics
    from .fit import FitFrame, Person
    from .measure import measure_params

    params = REFERENCE_BODIES[PHOTO_BODY]
    filmed = synthetic.film(model, params, arm_deg=4.0, views=4, width=360, height=640, seed=5)
    frames, frames_json = [], []
    for f in filmed:
        mask = f.mask > 0.5
        kp = np.zeros((33, 3))
        for i in MEDIAPIPE_TO_BONE:
            kp[i] = np.round(f.keypoints[i].astype(np.float64), 4)
        intr = Intrinsics(f.intr.width, f.intr.height, float(f.intr.focal), float(f.intr.cx), float(f.intr.cy))
        frames.append(FitFrame(mask=mask.astype(np.float32), intr=intr, keypoints=kp, yaw_init=f.yaw_init,
                               calibrated=False))
        frames_json.append({"width": intr.width, "height": intr.height, "focal": intr.focal, "cx": intr.cx,
                            "cy": intr.cy, "runs": mask_runs(mask), "yaw_hint": f.yaw_init,
                            "keypoints": {str(i): kp[i].tolist() for i in MEDIAPIPE_TO_BONE}})

    # Zwischenwerte an einem festen Zustand, mit dem Körper so, wie die App ihn rechnet
    view = photofit.prepare_view(model, frames[0])
    rig = photofit.Rig(model)
    pose = np.array([[PHOTO_POSE_SEED * math.sin(3 * j + c) for c in range(3)] for j in range(len(rig.pose_bones))])
    rest = data.rest(BodyParams())
    verts, heads = rest["vertices"].astype(np.float64), rest["bone_heads"].astype(np.float64)
    rs, ts = photofit.transforms_np(rig, heads, pose)
    posed = rig.skin(torch.as_tensor(verts), torch.as_tensor(rs), torch.as_tensor(ts)).numpy()
    b = view.kp_bones
    kp3 = np.einsum("pij,pj->pi", rs[b], heads[b]) + ts[b]
    yaw = 0.3
    trans = photofit._align_translation(torch.as_tensor(posed), torch.tensor(yaw, dtype=torch.float64), view).numpy()
    loss, g_v, g_k, g_y, g_t = photofit.view_loss_np(posed, kp3, yaw, trans, view, 0.5)
    g_rest, g_pose = photofit.skin_gradient_np(rig, verts, heads, pose, g_v, b, g_k)
    sample = list(range(0, model.num_vertices, SAMPLE_STEP))
    h, w = view.dt.shape
    pixels = [(x, y) for y in range(0, h, 53) for x in range(0, w, 37)]

    gender = "maennlich" if params.phenotype["gender"] < 0.5 else "weiblich"
    truth = measure_params(model, params)
    person = Person(gender=gender, age_years=round(param_to_age(params.phenotype["age"]), 1),
                    measures={"height": truth["height"], "weight": truth["weight"]})
    fitted, measures, results = photofit.fit_photos(model, person, frames)
    return {
        "body": PHOTO_BODY, "truth": truth, "calibrated": False, "frames": frames_json,
        "person": {"gender": person.gender, "age_years": person.age_years, "measures": person.measures},
        "prepared": {
            "width": w, "height": h, "focal": view.intr.focal, "px_height": view.px_height,
            "bbox_center": list(view.bbox_center), "pixels": int(view.mask.sum()),
            "contour_count": len(view.contour), "contour": view.contour[::20].tolist(),
            "dt": [[x, y, round(float(view.dt[y, x]), 5)] for x, y in pixels],
            "kp_bones": [model.bone_labels[i] for i in b],
        },
        "terms": {
            "pose": pose.tolist(), "yaw": yaw, "align_trans": trans.tolist(), "keypoint_weight": 0.5,
            "sample": sample, "posed": posed[sample].tolist(), "loss": loss, "g_yaw": g_y,
            "g_trans": g_t.tolist(), "g_points": g_v[sample].tolist(), "g_keypoints": g_k.tolist(),
            "g_rest": g_rest[sample].tolist(), "g_pose": g_pose.tolist(),
        },
        "fit": {"measures": measures, "yaw_deg": [r.yaw_deg for r in results], "iou": [r.iou for r in results],
                "params": fitted.to_json()},
    }
