"""Anpassung des Körpermodells an eingegebene Maße (und später an Videobilder).

Die eingegebenen Maße sind harte Vorgaben: Körpergröße wird exakt über den Maßstab getroffen,
Gewicht und Umfänge über Phänotyp- und lokale Formregler.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import cv2
import numpy as np
import torch

from .body import GENDER_VALUES, BodyModel, BodyParams, age_to_param, axis_angle_to_matrix
from .camera import KEYPOINT_OFFSETS, MEDIAPIPE_TO_BONE, Intrinsics, project, render_mask, to_camera
from .measure import CIRCUMFERENCES, MeasureProbe, measure_params

# Welcher lokale Anny-Regler zu welchem Maß gehört
LOCAL_FOR_MEASURE = {
    "chest": ["measure-bust-circ-incr"],
    "waist": ["measure-waist-circ-incr"],
    "hips": ["measure-hips-circ-incr"],
    "thigh": ["measure-thigh-circ-incr"],
    "calf": ["measure-calf-circ-incr"],
    "upperarm": ["measure-upperarm-circ-incr"],
    "neck": ["measure-neck-circ-incr"],
    "inseam": ["measure-upperleg-height-incr", "measure-lowerleg-height-incr"],
}

# Gewichtung der Maße im Fehler (relativer Fehler)
MEASURE_WEIGHTS = {"height": 40.0, "weight": 6.0, "inseam": 6.0, **{k: 6.0 for k in CIRCUMFERENCES}}

FREE_PHENOTYPES = ["muscle", "weight", "height", "proportions"]

Log = Callable[[str], None]


@dataclasses.dataclass
class Person:
    """Was der Mensch eingibt."""

    gender: str = "maennlich"  # maennlich | weiblich | divers
    age_years: float = 30.0
    measures: dict[str, float] = dataclasses.field(default_factory=dict)  # cm / kg

    def validate(self) -> list[str]:
        problems = []
        h = self.measures.get("height")
        w = self.measures.get("weight")
        if not h or not (100 <= h <= 230):
            problems.append("Bitte die Körpergröße in cm angeben (100 bis 230).")
        if not w or not (30 <= w <= 250):
            problems.append("Bitte das Gewicht in kg angeben (30 bis 250).")
        if self.gender not in GENDER_VALUES:
            problems.append("Geschlecht muss maennlich, weiblich oder divers sein.")
        for key in CIRCUMFERENCES:
            v = self.measures.get(key)
            if v is not None and not (15 <= v <= 220):
                problems.append(f"Der Wert für {key} ({v} cm) wirkt unplausibel.")
        if h and self.measures.get("inseam") and not (0.3 * h <= self.measures["inseam"] <= 0.6 * h):
            problems.append("Die Schrittlänge passt nicht zur Körpergröße.")
        return problems


class ShapeVariables(torch.nn.Module):
    """Die frei veränderlichen Größen einer Anpassung, in unbeschränkter Form."""

    def __init__(self, model: BodyModel, person: Person, init: BodyParams | None,
                 extra_locals: list[str] | None = None, free_gender: bool | None = None):
        super().__init__()
        self.model = model
        init = init or BodyParams()
        self.fixed = {"gender": GENDER_VALUES[person.gender], "age": age_to_param(person.age_years)}
        self.free_gender = person.gender == "divers" if free_gender is None else free_gender
        free = list(FREE_PHENOTYPES) + (["gender"] if self.free_gender else [])
        self.free = free
        start = torch.tensor([init.phenotype.get(k, 0.5) for k in free], dtype=torch.float32)
        self.pheno_raw = torch.nn.Parameter(torch.logit(start.clamp(0.02, 0.98)).to(model.device))

        wanted = set(extra_locals or [])
        for key in person.measures:
            wanted.update(LOCAL_FOR_MEASURE.get(key, []))
        wanted.update(k for k, v in init.local.items() if abs(v) > 1e-6)
        self.local_names = [n for n in model.local_labels if n in wanted]
        lstart = torch.tensor([init.local.get(n, 0.0) for n in self.local_names], dtype=torch.float32)
        self.local_raw = torch.nn.Parameter(torch.atanh(lstart.clamp(-0.95, 0.95)).to(model.device))
        self.log_scale = torch.nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init.scale))),
                                                         device=model.device))
        self.base_local = {k: v for k, v in init.local.items() if k not in self.local_names}
        self.offsets: torch.Tensor | None = None
        if init.offsets is not None:
            self.offsets = torch.as_tensor(init.offsets, dtype=torch.float32, device=model.device)

    def phenotype_tensor(self) -> torch.Tensor:
        vals = []
        free_vals = torch.sigmoid(self.pheno_raw)
        for k in self.model.phenotype_labels:
            if k in self.free:
                vals.append(free_vals[self.free.index(k)])
            else:
                vals.append(torch.tensor(self.fixed.get(k, 0.5), device=self.model.device))
        return torch.stack(vals)[None]

    def local_values(self) -> torch.Tensor:
        return torch.tanh(self.local_raw)

    def local_tensor(self) -> torch.Tensor:
        full = torch.zeros(len(self.model.local_labels), device=self.model.device)
        for k, v in self.base_local.items():
            full[self.model.local_labels.index(k)] = v
        if self.local_names:
            idx = torch.tensor([self.model.local_labels.index(n) for n in self.local_names],
                               device=self.model.device)
            full = full.index_put((idx,), self.local_values())
        return full[None]

    def scale(self) -> torch.Tensor:
        return torch.exp(self.log_scale)

    def rest(self) -> dict[str, torch.Tensor]:
        return self.model.rest(self.phenotype_tensor(), self.local_tensor(), self.offsets, self.scale())

    def to_params(self) -> BodyParams:
        with torch.no_grad():
            pheno = self.phenotype_tensor()[0].cpu().tolist()
            local = dict(self.base_local)
            for n, v in zip(self.local_names, self.local_values().cpu().tolist(), strict=True):
                local[n] = v
            return BodyParams(
                phenotype=dict(zip(self.model.phenotype_labels, pheno, strict=True)),
                local=local,
                scale=float(self.scale()),
                offsets=None if self.offsets is None else self.offsets.detach().cpu().numpy(),
            )


def measurement_loss(values: dict[str, torch.Tensor], targets: dict[str, float]) -> torch.Tensor:
    loss = torch.zeros((), device=next(iter(values.values())).device)
    for key, target in targets.items():
        if key in values and target:
            rel = (values[key] - target) / target
            loss = loss + MEASURE_WEIGHTS.get(key, 1.0) * rel**2
    return loss


def fit_to_measurements(
    model: BodyModel,
    person: Person,
    init: BodyParams | None = None,
    steps: int = 240,
    log: Log | None = None,
) -> tuple[BodyParams, dict[str, float]]:
    """Passt Anny an die eingegebenen Maße an. Gibt Parameter und die erreichten Maße zurück."""
    problems = person.validate()
    if problems:
        raise ValueError(" ".join(problems))
    targets = {k: float(v) for k, v in person.measures.items() if v}
    var = ShapeVariables(model, person, init)
    # Startmaßstab grob auf die Größe setzen
    with torch.no_grad():
        rest = var.rest()
        v = rest["vertices"][0]
        h = float((v[:, 2].max() - v[:, 2].min()) * 100)
        var.log_scale += torch.log(torch.tensor(targets["height"] / h, device=model.device))

    opt = torch.optim.Adam(var.parameters(), lr=0.05)
    probe = None
    for step in range(steps):
        if step % 40 == 0:
            with torch.no_grad():
                rest = var.rest()
                probe = MeasureProbe(model, rest["vertices"][0].cpu().numpy(),
                                     rest["bone_heads"][0].cpu().numpy())
        opt.zero_grad()
        rest = var.rest()
        values = probe.evaluate(rest["vertices"][0])
        loss = measurement_loss(values, targets)
        loss = loss + 0.004 * (var.local_values() ** 2).sum()
        loss = loss + 0.004 * ((torch.sigmoid(var.pheno_raw) - 0.5) ** 2).sum()
        loss.backward()
        opt.step()
        if log and step % 40 == 0:
            log(f"Maße anpassen: Schritt {step}/{steps}, Fehler {loss.item():.5f}")

    params = exact_height(model, var.to_params(), targets["height"])
    return params, measure_params(model, params)


def exact_height(model: BodyModel, params: BodyParams, height_cm: float) -> BodyParams:
    """Die eingegebene Körpergröße ist verbindlich: kleiner Maßstabsausgleich am Ende."""
    rest = model.rest_numpy(params)
    z = rest["vertices"][:, 2]
    params.scale *= height_cm / ((z.max() - z.min()) * 100)
    return params


# --------------------------------------------------------------------------------------
# Anpassung an Videobilder (Silhouetten + Körperpunkte)
# --------------------------------------------------------------------------------------

POSE_BONES = ["upperarm01.L", "upperarm01.R", "lowerarm01.L", "lowerarm01.R",
              "upperleg01.L", "upperleg01.R", "spine03", "neck01"]

VIDEO_LOCALS = [
    "torso-scale-depth-incr", "torso-scale-horiz-incr", "torso-vshape-incr",
    "stomach-pregnant-incr", "buttocks-volume-incr", "hip-scale-depth-incr",
    "hip-scale-horiz-incr", "measure-shoulder-dist-incr", "measure-upperarm-length-incr",
    "measure-lowerarm-length-incr", "measure-upperleg-height-incr", "measure-lowerleg-height-incr",
    "measure-napetowaist-dist-incr", "measure-bust-circ-incr", "measure-waist-circ-incr",
    "measure-hips-circ-incr", "measure-thigh-circ-incr", "measure-calf-circ-incr",
    "measure-upperarm-circ-incr", "measure-neck-circ-incr",
]

# Regionen, in denen keine Feinverschiebungen erlaubt sind (Masken dort zu ungenau)
NO_OFFSET_BONES = ("head", "eye", "finger", "metacarpal", "wrist", "toe", "foot")

WORK_SIDE = 640  # Arbeitsauflösung der Masken in Pixeln (lange Seite)

# Gewicht der Bildterme gegenüber Maßen und Reglern. Die Bildfehler sind auf die Körperhöhe in
# Pixeln normiert und daher sehr klein (2 px Abweichung bei 300 px Höhe ≈ 4e-5).
SILHOUETTE_WEIGHT = 100.0
OFFSET_SMOOTH = 4.0e4  # Glätte der Feinform (Unterschied benachbarter Punkte)
OFFSET_SIZE = 2.0e3  # Feinform klein halten, die Form kommt vor allem aus den Reglern
LOCAL_REG = 0.0005  # Regler sanft zur Mitte ziehen (gegen unnatürliche Formen)
# Körperpunkte richten Lage und Haltung aus; für die Form zählen vor allem die Silhouetten,
# denn MediaPipe-Punkte liegen nicht exakt auf Annys Gelenken.
KEYPOINT_WEIGHT_SHAPE = 0.5
MUSCLE_STARTS = (0.25, 0.75)  # zusätzliche Startwerte für den Muskelanteil


@dataclasses.dataclass
class FitFrame:
    mask: np.ndarray  # (H, W) 0..1
    intr: Intrinsics
    keypoints: np.ndarray | None = None  # (33, 3) Pixel + Sichtbarkeit (MediaPipe-Reihenfolge)
    yaw_init: float | None = None
    calibrated: bool = True  # False: Punkte liegen exakt auf Annys Gelenken (künstliche Testdaten)


@dataclasses.dataclass
class FrameFitResult:
    yaw_deg: float
    iou: float
    # Bild zur Kontrolle: grün = passt, rot = nur im Video (Modell zu schmal), blau = nur im Modell
    overlay: np.ndarray | None = dataclasses.field(default=None, repr=False, compare=False)

    def to_json(self) -> dict:
        return {"yaw_deg": self.yaw_deg, "iou": self.iou}


def _prepare_frame(frame: FitFrame, device) -> dict:
    factor = WORK_SIDE / max(frame.mask.shape)
    intr = frame.intr.scaled(factor)
    mask = cv2.resize(frame.mask.astype(np.float32), (intr.width, intr.height),
                      interpolation=cv2.INTER_AREA) > 0.5
    ys, xs = np.where(mask)
    if len(ys) < 50:
        raise ValueError("Maske leer")
    dt = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)[:, 0, :].astype(np.float32)
    idx = np.linspace(0, len(contour) - 1, min(400, len(contour))).astype(int)
    kp_bones, kp_xy, kp_w, kp_off = [], [], [], []
    if frame.keypoints is not None:
        for mp_idx, bone in MEDIAPIPE_TO_BONE.items():
            x, y, vis = frame.keypoints[mp_idx]
            if vis > 0.5:
                kp_bones.append(bone)
                kp_xy.append([x * factor, y * factor])
                kp_w.append(float(vis))
                kp_off.append(KEYPOINT_OFFSETS[mp_idx] if frame.calibrated else np.zeros(3, np.float32))
    return {
        "intr": intr,
        "mask": mask,
        "dt": torch.as_tensor(dt, device=device)[None, None],
        "contour": torch.as_tensor(contour[idx], device=device),
        "px_height": float(ys.max() - ys.min()),
        "bbox_center": ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2),
        "kp_bones": kp_bones,
        "kp_xy": torch.as_tensor(np.array(kp_xy, np.float32).reshape(-1, 2), device=device),
        "kp_w": torch.as_tensor(np.array(kp_w, np.float32), device=device),
        "kp_off": torch.as_tensor(np.array(kp_off, np.float32).reshape(-1, 3), device=device),
    }


def _frame_loss(verts, heads, yaw, trans, prep, model: BodyModel, kp_weight: float = 1.0) -> torch.Tensor:
    intr = prep["intr"]
    kp3d = heads[:0]
    if prep["kp_bones"]:
        idx = torch.tensor([model.bone(b) for b in prep["kp_bones"]], device=verts.device)
        height = verts[:, 2].max() - verts[:, 2].min()
        kp3d = heads[idx] + prep["kp_off"] * height  # MediaPipe-Punkte sitzen nicht genau im Gelenk
    pts = torch.cat([verts, kp3d], 0)
    uv = project(to_camera(pts, yaw[None], trans[None]), intr)[0]
    uv_v, uv_h = uv[: len(verts)], uv[len(verts):]
    h_px = prep["px_height"]
    grid = torch.stack([uv_v[:, 0] / (intr.width - 1) * 2 - 1, uv_v[:, 1] / (intr.height - 1) * 2 - 1], -1)
    d = torch.nn.functional.grid_sample(prep["dt"], grid[None, None], align_corners=True,
                                        padding_mode="border")[0, 0, 0]
    l_in = ((d / h_px) ** 2).mean()
    l_cov = (torch.cdist(prep["contour"], uv_v).min(1).values / h_px).pow(2).mean()
    loss = 4.0 * l_in + 4.0 * l_cov
    if prep["kp_bones"]:
        diff = (uv_h - prep["kp_xy"]) / h_px
        loss = loss + kp_weight * (prep["kp_w"] * diff.pow(2).sum(-1)).mean()
    return loss


def _align_translation(verts: torch.Tensor, yaw: torch.Tensor, prep: dict) -> torch.Tensor:
    """Verschiebung, bei der die Silhouette bei dieser Drehung so groß ist und so liegt wie im Bild."""
    intr = prep["intr"]
    cam = to_camera(verts, yaw[None], torch.zeros(1, 3, device=verts.device))[0]
    cx, cy = prep["bbox_center"]
    span = float(cam[:, 1].max() - cam[:, 1].min())
    t = torch.tensor([0.0, 0.0, intr.focal * span / prep["px_height"] - float(cam[:, 2].mean())],
                     device=verts.device)
    for _ in range(3):
        uv = project(cam + t, intr)
        z = float((cam[:, 2] + t[2]).mean())
        px_h = float(uv[:, 1].max() - uv[:, 1].min())
        t[2] = (t[2] + float(cam[:, 2].mean())) * px_h / prep["px_height"] - float(cam[:, 2].mean())
        u_c = float(uv[:, 0].max() + uv[:, 0].min()) / 2
        v_c = float(uv[:, 1].max() + uv[:, 1].min()) / 2
        t[0] += (cx - u_c) * z / intr.focal
        t[1] += (cy - v_c) * z / intr.focal
    return t


def _initial_view(verts, heads, prep: dict, model: BodyModel, yaw_hint: float | None,
                  refine_steps: int = 25) -> tuple[float, list[float]]:
    """Drehung und Lage eines Bildes: Kandidaten prüfen, die besten kurz nachschärfen."""
    cands = [i * math.pi / 6 for i in range(12)]
    if yaw_hint is not None:
        cands += [yaw_hint, -yaw_hint, yaw_hint + math.pi, math.pi - yaw_hint]
    scored = []
    with torch.no_grad():
        for c in cands:
            y = torch.tensor(c, dtype=torch.float32, device=verts.device)
            t = _align_translation(verts, y, prep)
            scored.append((float(_frame_loss(verts, heads, y, t, prep, model)), c, t))
    scored.sort(key=lambda s: s[0])
    best = None
    for _, c, t in scored[:4]:
        y = torch.nn.Parameter(torch.tensor(c, dtype=torch.float32, device=verts.device))
        tt = torch.nn.Parameter(t.clone())
        opt = torch.optim.Adam([y, tt], lr=0.02)
        for _ in range(refine_steps):
            opt.zero_grad()
            loss = _frame_loss(verts, heads, y, tt, prep, model)
            loss.backward()
            opt.step()
        with torch.no_grad():
            final = float(_frame_loss(verts, heads, y, tt, prep, model))
        if best is None or final < best[0]:
            best = (final, float(y.detach()), tt.detach().cpu().tolist())
    return best[1], best[2]


def fit_to_frames(
    model: BodyModel,
    person: Person,
    frames: list[FitFrame],
    init: BodyParams | None = None,
    steps: tuple[int, int, int] = (80, 160, 140),
    log: Log | None = None,
    silhouette_weight: float = SILHOUETTE_WEIGHT,
) -> tuple[BodyParams, dict[str, float], list[FrameFitResult]]:
    """Passt Form, Haltung und Feinform an Silhouetten aus mehreren Blickwinkeln an."""
    if not frames:
        raise ValueError("Keine brauchbaren Bilder für die Anpassung.")
    problems = person.validate()
    if problems:
        raise ValueError(" ".join(problems))
    device = model.device
    targets = {k: float(v) for k, v in person.measures.items() if v}
    if init is None:
        init, _ = fit_to_measurements(model, person, log=log)
    var = ShapeVariables(model, person, init, extra_locals=VIDEO_LOCALS)
    preps = [_prepare_frame(f, device) for f in frames]
    k = len(preps)

    pose_aa = torch.nn.Parameter(torch.zeros(len(POSE_BONES), 3, device=device))
    pose_idx = [model.bone(b) for b in POSE_BONES]

    def pose_tensor():
        pose = torch.eye(4, device=device)[None, None].repeat(1, len(model.bone_labels), 1, 1)
        return pose.index_put((torch.tensor([0], device=device)[:, None],
                               torch.tensor(pose_idx, device=device)[None]),
                              axis_angle_to_matrix(pose_aa)[None])

    # Feinform: Verschiebung je Punkt entlang der Flächennormale (verhindert Knicke und
    # umgeklappte Dreiecke), mit weichem Übergang zu den gesperrten Bereichen.
    edges_np = _mesh_edges(model.faces)
    offset_gate = torch.as_tensor(_soft_gate(model, edges_np), device=device)
    disp = torch.nn.Parameter(torch.zeros(model.num_vertices, device=device))
    normals = torch.zeros(model.num_vertices, 3, device=device)
    faces_t = torch.as_tensor(model.faces, device=device)
    edges = torch.as_tensor(edges_np, device=device)

    def offsets() -> torch.Tensor:
        return (offset_gate * disp)[:, None] * normals

    def body(with_offsets: bool):
        var.offsets = offsets() if with_offsets else None
        rest = var.rest()
        posed = model.posed(rest, pose_tensor(), var.scale())
        return rest, posed["vertices"][0], posed["bone_heads"][0]

    # Startwerte für Drehung und Verschiebung pro Bild
    with torch.no_grad():
        _, verts, heads = body(False)
    yaw_init, trans_init = [], []
    for i, (prep, frame) in enumerate(zip(preps, frames, strict=True)):
        y, t = _initial_view(verts, heads, prep, model, frame.yaw_init)
        yaw_init.append(y)
        trans_init.append(t)
        if log:
            log(f"Bild {i + 1}/{k}: Startdrehung {math.degrees(y) % 360:.0f}°")
    trans = torch.nn.Parameter(torch.tensor(trans_init, dtype=torch.float32, device=device))
    yaws = torch.nn.Parameter(torch.tensor(yaw_init, dtype=torch.float32, device=device))

    def run_stage(name: str, n_steps: int, params: list, with_offsets: bool, kp_w: float,
                  first: bool = True) -> float:
        nonlocal normals
        opt = torch.optim.Adam([
            {"params": [p for p in params if p is not disp], "lr": 0.02},
            {"params": [p for p in params if p is disp], "lr": 0.001},
        ])
        probe, last = None, float("inf")
        for step in range(n_steps):
            if step % 50 == 0:
                with torch.no_grad():
                    if with_offsets:
                        plain, _, _ = body(False)
                        normals = _vertex_normals(plain["vertices"][0] / var.scale(), faces_t)
                        if step == 0 and first and init.offsets is not None:
                            # Re-Scan: Feinform des letzten Scans als Start
                            start = torch.as_tensor(init.offsets, dtype=torch.float32, device=device)
                            disp.copy_((start * normals).sum(-1))
                    rest, _, _ = body(with_offsets)
                    probe = MeasureProbe(model, rest["vertices"][0].cpu().numpy(),
                                         rest["bone_heads"][0].cpu().numpy())
            opt.zero_grad()
            rest, verts, heads = body(with_offsets)
            loss = sum(_frame_loss(verts, heads, yaws[i], trans[i], preps[i], model, kp_w)
                       for i in range(k)) / k
            loss = silhouette_weight * loss
            loss = loss + 0.5 * measurement_loss(probe.evaluate(rest["vertices"][0]), targets)
            loss = loss + LOCAL_REG * (var.local_values() ** 2).sum()
            loss = loss + 0.01 * (pose_aa ** 2).sum()
            if with_offsets:
                d = offset_gate * disp
                loss = loss + OFFSET_SMOOTH * (d[edges[:, 0]] - d[edges[:, 1]]).pow(2).mean()
                loss = loss + OFFSET_SIZE * d.pow(2).mean()
            loss.backward()
            opt.step()
            last = loss.item()
            if log and step % 40 == 0:
                log(f"{name}: Schritt {step}/{n_steps}, Fehler {last:.5f}")
        return last

    view_params = [yaws, trans, pose_aa]
    shape_params = [*view_params, *var.parameters()]
    run_stage("Lage und Haltung", steps[0], view_params, False, 1.0)

    # Körperform: Muskelanteil und Fettanteil sehen bei gleichem Gewicht ähnlich aus. Mehrere
    # Startwerte für die Muskeln kurz anrechnen und mit dem besten weitermachen.
    tracked = [yaws, trans, pose_aa, *var.parameters()]
    snapshot = [t.detach().clone() for t in tracked]
    muscle_i = var.free.index("muscle")
    start_muscle = float(torch.sigmoid(var.pheno_raw[muscle_i].detach()))
    trial_steps = max(1, steps[1] // 3)
    best = None
    for m0 in dict.fromkeys([start_muscle, MUSCLE_STARTS[0], MUSCLE_STARTS[1]]):
        with torch.no_grad():
            for t, v in zip(tracked, snapshot, strict=True):
                t.copy_(v)
            var.pheno_raw[muscle_i] = torch.logit(torch.tensor(m0, device=device))
        score = run_stage(f"Körperform (Start Muskeln {m0:.2f})", trial_steps, shape_params, False,
                          KEYPOINT_WEIGHT_SHAPE)
        if best is None or score < best[0]:
            best = (score, [t.detach().clone() for t in tracked])
    with torch.no_grad():
        for t, v in zip(tracked, best[1], strict=True):
            t.copy_(v)
    run_stage("Körperform", steps[1] - trial_steps, shape_params, False, KEYPOINT_WEIGHT_SHAPE)
    run_stage("Feinform", steps[2], [*shape_params, disp], True, KEYPOINT_WEIGHT_SHAPE)

    var.offsets = offsets().detach()
    params, repaired = repair_offsets(model, var.to_params())
    if log and repaired:
        log(f"Feinform an {repaired} Punkten geglättet (umgeklappte Dreiecke verhindert).")
    var.offsets = None if params.offsets is None else torch.as_tensor(params.offsets, device=device)
    params = exact_height(model, params, targets["height"])

    # Passgenauigkeit pro Bild
    results = []
    with torch.no_grad():
        rest = var.rest()
        posed = model.posed(rest, pose_tensor(), var.scale())
        verts = posed["vertices"][0].cpu().numpy()
    for i, prep in enumerate(preps):
        rendered = render_mask(verts, model.faces, float(yaws[i].detach()), trans[i].detach().cpu().numpy(),
                               prep["intr"]) > 0
        inter = (rendered & prep["mask"]).sum()
        union = (rendered | prep["mask"]).sum()
        overlay = np.full((*rendered.shape, 3), 245, np.uint8)
        overlay[rendered & prep["mask"]] = (110, 180, 110)
        overlay[prep["mask"] & ~rendered] = (220, 60, 60)
        overlay[rendered & ~prep["mask"]] = (60, 100, 230)
        results.append(FrameFitResult(math.degrees(float(yaws[i].detach())) % 360, float(inter / max(union, 1)),
                                      overlay))
    return params, measure_params(model, params), results


def _soft_gate(model: BodyModel, edges: np.ndarray, rings: int = 8) -> np.ndarray:
    """1 = Feinform erlaubt, 0 = gesperrt (Kopf, Hände, Füße); dazwischen ein weicher Übergang."""
    gate = (~model.dominant_bone_mask(lambda n: n.startswith(NO_OFFSET_BONES))).astype(np.float32)
    count = np.bincount(edges.ravel(), minlength=len(gate)).astype(np.float32)
    for _ in range(rings):
        acc = np.zeros_like(gate)
        np.add.at(acc, edges[:, 0], gate[edges[:, 1]])
        np.add.at(acc, edges[:, 1], gate[edges[:, 0]])
        gate = np.minimum(gate, acc / np.maximum(count, 1))
    return gate


def _vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    vn = torch.zeros_like(verts).index_add_(0, faces[:, 0], fn).index_add_(0, faces[:, 1], fn)
    vn = vn.index_add_(0, faces[:, 2], fn)
    return vn / vn.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def repair_offsets(model: BodyModel, params: BodyParams, rounds: int = 12) -> tuple[BodyParams, int]:
    """Nimmt die Feinform dort zurück, wo sie Dreiecke umklappt oder Flächen ineinander schiebt.
    Gibt die Zahl der betroffenen Punkte zurück."""
    from .mesh_check import self_intersections

    if params.offsets is None:
        return params, 0
    plain = params.copy()
    plain.offsets = None
    template = model.rest_numpy(plain)["vertices"]
    faces = model.faces

    def fnormals(v):
        return np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])

    ref = fnormals(template)
    base = self_intersections(template, faces)
    offsets = params.offsets.astype(np.float32).copy()
    touched: set[int] = set()
    for i in range(rounds):
        trial = params.copy()
        trial.offsets = offsets
        verts = model.rest_numpy(trial)["vertices"]
        bad_faces = set(np.flatnonzero((fnormals(verts) * ref).sum(1) < 0).tolist())
        for a, b in self_intersections(verts, faces) - base:
            bad_faces.update((a, b))
        if not bad_faces:
            break
        bad = np.unique(faces[sorted(bad_faces)])
        ring = np.unique(faces[np.isin(faces, bad).any(1)])
        touched.update(ring.tolist())
        offsets[ring] *= 0.0 if i == rounds - 1 else 0.5
    out = params.copy()
    out.offsets = offsets
    return out, len(touched)


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)
