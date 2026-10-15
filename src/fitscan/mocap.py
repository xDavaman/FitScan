"""Bewegung aus einem Handyvideo (Motion Capture ohne Anzug).

Die Person ist schon gescannt, Körperform und Knochenlängen stehen also fest. Je Bild liefert
MediaPipe Körperpunkte im Bild und grob in 3D. Gesucht sind je Bild die Gelenkdrehungen und die
Lage der Hüfte, bei denen Annys Gelenke genau auf diesen Punkten landen. Alle Bilder werden
gemeinsam angepasst: Bewegungen müssen glatt sein, Knie und Ellbogen sind Scharniere. Danach
wird der Boden aus den stehenden Füßen bestimmt und die Bewegung waagerecht gestellt.

Das Handy steht dabei fest (Stativ, Regal). Gerechnet wird zuerst in einer waagerechten
Kamera: Anny-Achsen, die Kamera blickt entlang +Y. Neigt sich das Handy, liegt der Boden dort
schief; das gleicht der letzte Schritt aus.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import roma
import torch

from .body import BodyModel, BodyParams
from .camera import KEYPOINT_OFFSETS, Intrinsics
from .motion import Motion, _rest, hip_height

Log = Callable[[str], None]

# Gelenke, die je Bild frei sind (dazu die Drehung der Hüfte „root“)
BONES = ["spine04", "spine02", "neck01", "head",
         "clavicle.L", "clavicle.R", "upperarm01.L", "upperarm01.R", "lowerarm01.L", "lowerarm01.R",
         "wrist.L", "wrist.R", "upperleg01.L", "upperleg01.R", "lowerleg01.L", "lowerleg01.R",
         "foot.L", "foot.R"]
HINGES = {"lowerleg01.L": "lowerleg02.L", "lowerleg01.R": "lowerleg02.R",
          "lowerarm01.L": "wrist.L", "lowerarm01.R": "wrist.R"}
QUIET = {"spine04": 1.0, "spine02": 1.0, "neck01": 1.0, "head": 0.5, "clavicle.L": 4.0, "clavicle.R": 4.0,
         "wrist.L": 1.0, "wrist.R": 1.0, "foot.L": 1.0, "foot.R": 1.0}

# MediaPipe-Punkte an Knochenköpfen (Index -> Bone), wie camera.MEDIAPIPE_TO_BONE
JOINT_POINTS = {11: "upperarm01.L", 12: "upperarm01.R", 13: "lowerarm01.L", 14: "lowerarm01.R",
                15: "wrist.L", 16: "wrist.R", 23: "upperleg01.L", 24: "upperleg01.R",
                25: "lowerleg01.L", 26: "lowerleg01.R", 27: "foot.L", 28: "foot.R"}
# MediaPipe-Punkte im Gesicht -> Punkt des Face Landmarkers (data/face_landmarks_anny.json)
# Augen und Ohren fehlen bewusst: In Seitenansichten rät MediaPipe sie und dreht damit den Kopf
FACE_POINTS = {0: 1, 9: 291, 10: 61}
FOOT_POINTS = (29, 30, 31, 32)  # Ferse links/rechts, Fußspitze links/rechts
# Gliedmaßen für die Richtungen aus MediaPipes 3D-Punkten (Maßstab spielt so keine Rolle)
LIMBS = [(11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28),
         (29, 31), (30, 32), (24, 23), (12, 11), (23, 11), (24, 12)]  # Rumpf: Neigung
POINT_WEIGHT = {**{k: 1.0 for k in JOINT_POINTS}, **{k: 0.3 for k in FACE_POINTS},
                **{k: 0.7 for k in FOOT_POINTS}}

WEIGHTS = {"3d": 0.05, "hinge": 0.05, "prior": 0.002, "smooth_pose": 2e-4, "smooth_trans": 4e-4,
           "points": {}, "quiet": {}}

# Anny (x, y, z) -> waagerechte Kamera (x, -z, y), siehe camera.to_camera
_M = torch.tensor([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


@dataclasses.dataclass
class Observation:
    """Was MediaPipe in einem Bild sieht."""

    keypoints: np.ndarray | None  # (33, 3) Pixel x, y und Sichtbarkeit
    world: np.ndarray | None  # (33, 3) Meter, Hüftmitte als Ursprung, Achsen wie das Bild


@dataclasses.dataclass
class Capture:
    motion: Motion
    camera_pitch_deg: float  # Neigung des Handys (positiv = nach unten)
    camera_height: float  # Meter über dem Boden
    reprojection_px: float  # mittlerer Abstand der Gelenke zu den Punkten im Bild
    frames_used: int
    frames_total: int

    def to_json(self) -> dict:
        return {"bewegung": self.motion.name, "bilder": self.frames_total, "bilder_mit_person": self.frames_used,
                "fps": self.motion.fps, "dauer_s": round(self.motion.duration, 2),
                "handy_neigung_grad": round(self.camera_pitch_deg, 1), "handy_hoehe_m": round(self.camera_height, 2),
                "abweichung_im_bild_px": round(self.reprojection_px, 1)}


# ------------------------------------------------------------------------ Körperpunkte


class _Points:
    """Wo auf Annys Körper die MediaPipe-Punkte sitzen: Knochenköpfe (mit geeichtem Versatz, der
    sich mit dem Knochen dreht) und Hautpunkte (Gesicht, Ferse, Fußspitze)."""

    def __init__(self, model: BodyModel, params: BodyParams):
        self.model = model
        rest = _rest(model, params)
        self.rest = rest
        with torch.no_grad():
            ref_t, ref_poses = model.anny.get_bone_transforms(model._identity_pose, rest["_unscaled_bone_poses"])
            verts = model.anny.get_skinned_vertices(rest["_unscaled_vertices"], ref_t)[0] * params.scale
        self.scale = params.scale
        body_height = float(verts[:, 2].max() - verts[:, 2].min())
        self.joint_ids = list(JOINT_POINTS)
        self.joint_bones = torch.tensor([model.bone(JOINT_POINTS[k]) for k in self.joint_ids])
        # Versatz im Knochen gespeichert: dreht sich mit
        offs = []
        for k in self.joint_ids:
            world = torch.as_tensor(KEYPOINT_OFFSETS[k], dtype=torch.float32) * body_height
            # Datei: x links, y hinten, z oben, wie Anny
            rot = ref_poses[0, model.bone(JOINT_POINTS[k]), :3, :3]
            offs.append(rot.T @ world)
        self.joint_offsets = torch.stack(offs)
        # Hautpunkte: Gesicht über die Eichung des Face Landmarkers, Füße aus der Form
        face = json.loads((Path(__file__).with_name("data") / "face_landmarks_anny.json").read_text("utf-8"))
        faces = model.faces
        ids, bary = [], []
        for lm in FACE_POINTS.values():
            ids.append(faces[face["dreieck"][lm]])
            bary.append(face["bary"][lm])
        v = verts.cpu().numpy()
        for side in ("L", "R"):  # Ferse
            sole = np.where(model.dominant_bone_mask(lambda n, s=side: n == f"foot.{s}"))[0]
            low = sole[v[sole, 2] < v[sole, 2].min() + 0.03]
            ids.append([low[np.argmax(v[low, 1])]] * 3)
            bary.append([1.0, 0.0, 0.0])
        for side in ("L", "R"):  # Fußspitze (vorderstes Ende der Zehen)
            toes = np.where(model.dominant_bone_mask(lambda n, s=side: n.startswith("toe") and n.endswith("." + s)))[0]
            ids.append([toes[np.argmin(v[toes, 1])]] * 3)
            bary.append([1.0, 0.0, 0.0])
        self.skin_ids = list(FACE_POINTS) + list(FOOT_POINTS)
        self.skin_vertices = torch.as_tensor(np.array(ids, np.int64))
        self.skin_bary = torch.as_tensor(np.array(bary, np.float32))
        flat = self.skin_vertices.reshape(-1)
        self.skin_rest = rest["_unscaled_vertices"][0, flat]
        self.skin_bone_idx = model.anny.vertex_bone_indices[flat]
        self.skin_bone_w = model.anny.vertex_bone_weights[flat]
        self.ids = self.joint_ids + self.skin_ids

    def _heads(self, bone_poses: torch.Tensor) -> torch.Tensor:
        rest_poses = self.rest["_unscaled_bone_poses"]
        heads, _ = self.model.anny.get_bone_ends(self.rest["bone_heads"] / self.scale,
                                                 self.rest["bone_heads"] / self.scale, rest_poses, bone_poses)
        return heads * self.scale

    def __call__(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pose (F, J, 4, 4) -> Punkte (F, P, 3) relativ zur Hüfte und Knochenköpfe (F, J, 3)."""
        transforms, poses = self.model.anny.get_bone_transforms(pose, self.rest["_unscaled_bone_poses"])
        heads = self._heads(poses)
        rot = poses[:, self.joint_bones, :3, :3]
        joints = heads[:, self.joint_bones] + (rot @ self.joint_offsets[None, :, :, None])[..., 0]
        t = transforms[:, self.skin_bone_idx]  # (F, S*3, K, 4, 4)
        p = torch.cat([self.skin_rest, torch.ones_like(self.skin_rest[:, :1])], -1)
        skinned = ((t @ p[None, :, None, :, None])[..., :3, 0] * self.skin_bone_w[None, ..., None]).sum(2)
        skinned = skinned.reshape(len(pose), -1, 3, 3) * self.scale
        skin = (skinned * self.skin_bary[None, :, :, None]).sum(2)
        return torch.cat([joints, skin], 1), heads


# ------------------------------------------------------------------------ Anpassung


def _pose(model: BodyModel, root_rv: torch.Tensor, rv: torch.Tensor) -> torch.Tensor:
    f = len(root_rv)
    pose = torch.eye(4)[None, None].repeat(f, len(model.bone_labels), 1, 1)
    idx = torch.tensor([model.bone(b) for b in BONES])
    pose[:, 0, :3, :3] = roma.rotvec_to_rotmat(root_rv)
    pose[:, idx, :3, :3] = roma.rotvec_to_rotmat(rv)
    return pose


def _initial_root(world: np.ndarray | None) -> np.ndarray:
    """Drehung der Hüfte aus MediaPipes 3D-Punkten (Hüften quer, Schultern oben)."""
    if world is None:
        return np.zeros(3, np.float32)
    w = np.stack([world[:, 0], world[:, 2], -world[:, 1]], 1)  # MediaPipe -> Anny in der Kamera
    left = w[23] - w[24]
    up = (w[11] + w[12]) / 2 - (w[23] + w[24]) / 2
    left /= max(np.linalg.norm(left), 1e-6)
    up -= left * (up @ left)
    up /= max(np.linalg.norm(up), 1e-6)
    back = np.cross(up, left)  # Anny: x links, y hinten, z oben -> y = z × x
    rot = np.stack([left, back, up], 1)
    return roma.rotmat_to_rotvec(torch.as_tensor(rot, dtype=torch.float32)).numpy()


def _robust(sq: torch.Tensor, c: float) -> torch.Tensor:
    """Geman-McClure auf quadrierte Abstände (einzelne Fehlgriffe von MediaPipe zählen wenig)."""
    return sq / (sq + c * c)


def fit_motion(model: BodyModel, params: BodyParams, observations: list[Observation], intr: Intrinsics,
               fps: float, name: str = "Bewegung", steps: tuple[int, int] = (150, 600),
               log: Log | None = None, weights: dict | None = None, tilt: np.ndarray | None = None) -> Capture:
    """Gelenkdrehungen und Hüftlage je Bild aus den MediaPipe-Punkten.

    tilt: Drehung Kamera -> waagerechte Kamera (3, 3), wenn die Neigung des Handys bekannt ist
    (senkrechte Kanten im Hintergrund, level.estimate, oder Lagesensor). Ohne wird sie aus dem
    Boden unter den Füßen geschätzt; das klappt nur, wenn die Füße an verschiedenen Stellen
    auftreten, die nicht alle gleich weit vom Handy entfernt sind."""
    weights = {**WEIGHTS, **(weights or {})}
    total = len(observations)
    used = [i for i, o in enumerate(observations) if o.keypoints is not None]
    if len(used) < 2:
        raise ValueError("In zu wenigen Bildern ist eine Person zu sehen.")
    first, last = used[0], used[-1]
    obs = observations[first:last + 1]
    f = len(obs)
    points = _Points(model, params)
    n_pts = len(points.ids)
    kp = np.zeros((f, n_pts, 2), np.float32)
    vis = np.zeros((f, n_pts), np.float32)
    w3 = np.zeros((f, n_pts, 3), np.float32)
    w3_ok = np.zeros(f, np.float32)
    for i, o in enumerate(obs):
        if o.keypoints is None:
            continue
        kp[i] = o.keypoints[points.ids, :2]
        vis[i] = o.keypoints[points.ids, 2]
        if o.world is not None:
            w = o.world[points.ids]
            w3[i] = np.stack([w[:, 0], w[:, 2], -w[:, 1]], 1)  # MediaPipe -> Anny-Achsen
            w3_ok[i] = 1.0
    vis = np.where(vis > 0.3, vis, 0.0)
    point_w = {**POINT_WEIGHT, **{int(k): v for k, v in weights["points"].items()}}
    kp_t = torch.as_tensor(kp)
    vis_t = torch.as_tensor(vis) * torch.tensor([point_w[k] for k in points.ids])[None]
    w3_t, w3_ok_t = torch.as_tensor(w3), torch.as_tensor(w3_ok)
    limb_a = torch.tensor([points.ids.index(a) for a, _ in LIMBS])
    limb_b = torch.tensor([points.ids.index(b) for _, b in LIMBS])
    md = w3_t[:, limb_b] - w3_t[:, limb_a]
    limb_dir = md / md.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    vis_raw = torch.as_tensor(vis)
    limb_w = torch.minimum(vis_raw[:, limb_a], vis_raw[:, limb_b]) * w3_ok_t[:, None]
    px_scale = float(np.median([np.ptp(o.keypoints[[0, 27, 28], 1]) for o in obs if o.keypoints is not None]))
    px_scale = max(px_scale, 50.0)
    k_focal, cx, cy = intr.focal, intr.cx, intr.cy

    # Startwerte: Drehung aus den 3D-Punkten, Lage aus Größe und Mitte im Bild
    root0 = np.stack([_initial_root(o.world) for o in obs])
    root0 = _fill_missing(root0, vis[:, 0] > 0)
    height = hip_height(model, params)
    trans0 = []
    for o in obs:
        if o.keypoints is None:
            trans0.append(None)
            continue
        hip_px = o.keypoints[[23, 24], :2].mean(0)
        sh_px = o.keypoints[[11, 12], :2].mean(0)
        torso_px = max(float(np.linalg.norm(sh_px - hip_px)), 10.0)
        z = k_focal * 0.5 * height / 0.86 / torso_px  # Rumpf etwa 0,5 m bei 0,86 m Hüfthöhe
        trans0.append([(hip_px[0] - cx) * z / k_focal, (hip_px[1] - cy) * z / k_focal, z])
    trans0 = _fill_missing(np.array([t if t is not None else [np.nan] * 3 for t in trans0], np.float32),
                           ~np.isnan(np.array([t[0] if t is not None else np.nan for t in trans0])))

    root_rv = torch.nn.Parameter(torch.as_tensor(root0, dtype=torch.float32))
    trans = torch.nn.Parameter(torch.as_tensor(trans0, dtype=torch.float32))
    rv = torch.nn.Parameter(torch.zeros(f, len(BONES), 3))
    hinge_axes, hinge_sign = _hinge_axes(model, points)
    quiet_w = {**QUIET, **weights["quiet"]}
    quiet = torch.tensor([quiet_w.get(b, 0.0) for b in BONES])

    def evaluate(full: bool, robust3d: bool = True):
        pose = _pose(model, root_rv, rv if full else rv.detach() * 0)
        pts, heads = points(pose)
        cam = pts @ _M.T + trans[:, None, :]
        z = cam[..., 2].clamp_min(0.3)
        uv = torch.stack([k_focal * cam[..., 0] / z + cx, k_focal * cam[..., 1] / z + cy], -1)
        d2 = ((uv - kp_t) ** 2).sum(-1) / px_scale ** 2
        l2d = (vis_t * _robust(d2, 0.05)).sum() / vis_t.sum().clamp_min(1)
        d = pts[:, limb_b] - pts[:, limb_a]
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sq = ((d - limb_dir) ** 2).sum(-1)
        l3d = (limb_w * (_robust(sq, 0.5) if robust3d else sq)).sum() / limb_w.sum().clamp_min(1)
        return l2d, l3d, pose

    def smooth(x: torch.Tensor) -> torch.Tensor:
        if len(x) < 3:
            return x.sum() * 0
        return ((x[2:] - 2 * x[1:-1] + x[:-2]) ** 2).mean() * fps ** 2

    # Stufe 1: nur Lage und Drehung des Rumpfs
    opt = torch.optim.Adam([root_rv, trans], lr=0.02)
    for step in range(steps[0]):
        opt.zero_grad()
        l2d, l3d, _ = evaluate(False)
        loss = l2d + weights["3d"] * l3d + 1e-4 * (smooth(root_rv) + smooth(trans))
        loss.backward()
        opt.step()
        if log and step % 50 == 0:
            log(f"Rumpf ausrichten: Schritt {step}/{steps[0]}, Fehler {loss.item():.4f}")
    # Stufe 2: Gelenke grob nach MediaPipes 3D-Richtungen stellen (ohne Ausreißerschutz, damit auch
    # weit entfernte Lösungen gefunden werden, etwa ob das Knie zur Kamera oder von ihr weg zeigt)
    opt = torch.optim.Adam([rv], lr=0.03)
    for _ in range(steps[0]):
        opt.zero_grad()
        _, l3d, _ = evaluate(True, robust3d=False)
        loss = (l3d + weights["hinge"] * _hinge_loss(rv, hinge_axes, hinge_sign)
                + weights["prior"] * (quiet[None, :, None] * rv ** 2).sum(-1).mean()
                + weights["smooth_pose"] * smooth(rv))
        loss.backward()
        opt.step()
    # Stufe 3: alle Gelenke und Lage nach Bildpunkten und 3D-Richtungen
    opt = torch.optim.Adam([root_rv, trans, rv], lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps[1], eta_min=0.001)
    for step in range(steps[1]):
        opt.zero_grad()
        l2d, l3d, _ = evaluate(True)
        hinge = _hinge_loss(rv, hinge_axes, hinge_sign)
        prior = (quiet[None, :, None] * rv ** 2).sum(-1).mean()
        loss = (l2d + weights["3d"] * l3d + weights["hinge"] * hinge + weights["prior"] * prior
                + weights["smooth_pose"] * (smooth(rv) + smooth(root_rv)) + weights["smooth_trans"] * smooth(trans))
        loss.backward()
        opt.step()
        sched.step()
        if log and step % 100 == 0:
            log(f"Gelenke anpassen: Schritt {step}/{steps[1]}, Fehler {loss.item():.4f}")

    with torch.no_grad():
        l2d, _, pose = evaluate(True)
        pts, heads = points(pose)
        cam = pts @ _M.T + trans[:, None, :]
        uv = torch.stack([k_focal * cam[..., 0] / cam[..., 2] + cx, k_focal * cam[..., 1] / cam[..., 2] + cy], -1)
        err = ((uv - kp_t).norm(dim=-1) * (vis_t > 0)).sum() / (vis_t > 0).sum()
    capture = _to_world(model, params, points, root_rv.detach(), rv.detach(), trans.detach(), fps, name, height,
                         tilt)
    return dataclasses.replace(capture, reprojection_px=float(err), frames_used=len(used), frames_total=total)


def _fill_missing(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Fehlende Bilder (keine Person erkannt) aus den Nachbarn auffüllen."""
    x = x.copy()
    idx = np.arange(len(x))
    if ok.sum() == 0:
        return np.nan_to_num(x)
    for c in range(x.shape[1]):
        x[:, c] = np.interp(idx, idx[ok], x[ok, c])
    return x


def _hinge_axes(model: BodyModel, points: _Points) -> tuple[torch.Tensor, torch.Tensor]:
    """Beugeachse von Knie und Ellbogen in Annys Bezugshaltung und die Richtung, in die sie beugen
    (Knie nach hinten, Ellbogen nach vorn)."""
    with torch.no_grad():
        _, poses = model.anny.get_bone_transforms(model._identity_pose, points.rest["_unscaled_bone_poses"])
        heads = points._heads(poses)[0]
    axes, signs = [], []
    for b in BONES:
        if b not in HINGES:
            axes.append(torch.zeros(3))
            signs.append(0.0)
            continue
        d = heads[model.bone(HINGES[b])] - heads[model.bone(b)]
        d = d / d.norm()
        forward = torch.tensor([0.0, -1.0, 0.0])
        axis = torch.linalg.cross(d, forward)
        axis = axis / axis.norm()
        # Beugen = Drehung um axis mit positivem Winkel bewegt die Spitze nach vorn (Ellbogen)
        # bzw. nach hinten (Knie)
        bend_forward = b.startswith("lowerarm")
        axes.append(axis)
        signs.append(1.0 if bend_forward else -1.0)
    return torch.stack(axes), torch.tensor(signs)


def _hinge_loss(rv: torch.Tensor, axes: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    mask = signs != 0
    a = axes[mask]
    r = rv[:, mask]
    along = (r * a).sum(-1)
    off = r - along[..., None] * a
    wrong = torch.relu(-(along * signs[mask]) - 0.05)  # Überstrecken
    return (off ** 2).sum(-1).mean() + 4.0 * (wrong ** 2).mean()


# ------------------------------------------------------------- Boden und Welt


def _to_world(model: BodyModel, params: BodyParams, points: _Points, root_rv: torch.Tensor, rv: torch.Tensor,
              trans: torch.Tensor, fps: float, name: str, height: float, tilt: np.ndarray | None) -> Capture:
    """Alles waagerecht stellen (bekannte Neigung oder Boden aus den tiefsten Fußpunkten), Boden auf z = 0."""
    with torch.no_grad():
        pose = _pose(model, root_rv, rv)
        pts, _ = points(pose)
    # Hüftlage aus der Kamera zurück in Anny-Achsen (waagerechte Kamera)
    hip = trans @ _M  # M orthogonal: p = M^T (cam - 0)
    feet_idx = [points.ids.index(k) for k in FOOT_POINTS]
    feet = (pts[:, feet_idx] + hip[:, None]).reshape(-1, 3).numpy()
    if tilt is not None:
        # Kamera -> waagerechte Kamera, in Anny-Achsen: p_anny = M^T p_cam
        normal = (_M.T.numpy() @ np.asarray(tilt, np.float64).T @ _M.numpy() @ np.array([0.0, 0.0, 1.0]))
        z = feet @ normal
        offset = float(z[z < np.percentile(z, 30)].mean())
    else:
        normal, offset = _ground_plane(feet)
    # Drehung, die die Bodennormale auf +Z legt
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, z)
    s = np.linalg.norm(axis)
    angle = math.atan2(s, float(normal @ z))
    level = roma.rotvec_to_rotmat(torch.as_tensor(axis / max(s, 1e-9) * angle, dtype=torch.float32))
    pitch = math.degrees(math.atan2(-normal[1], normal[2]))
    root_rot = level[None] @ roma.rotvec_to_rotmat(root_rv)
    hip_w = hip @ level.T
    hip_w[:, 2] -= offset  # Boden liegt nach der Drehung bei z = offset
    rotations = torch.cat([roma.rotmat_to_unitquat(root_rot)[:, None], roma.rotvec_to_unitquat(rv)], 1)
    motion = Motion(name, fps, ["root", *BONES], _continuous_np(rotations.numpy()),
                    (hip_w.numpy() / height).astype(np.float32), source="Video")
    return Capture(motion, pitch, -offset, 0.0, 0, 0)  # die Kamera liegt im Ursprung


def _continuous_np(q: np.ndarray) -> np.ndarray:
    from .motion import _continuous

    return _continuous(q.astype(np.float64)).astype(np.float32)


def _ground_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Ebene durch die tiefsten Fußpunkte (Boden). Gibt Normale (nach oben) und Abstand zurück."""
    z = points[:, 2]
    low = points[z < np.percentile(z, 30)]
    if len(low) < 6:
        return np.array([0.0, 0.0, 1.0]), float(z.min())
    center = low.mean(0)
    _, _, vt = np.linalg.svd(low - center)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    if n[2] < math.cos(math.radians(35)):  # unplausibel schief: waagerecht annehmen
        n = np.array([0.0, 0.0, 1.0])
    return n, float(n @ center)
