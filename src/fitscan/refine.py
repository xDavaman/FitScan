"""Feinform aus allen Blickwinkeln: Schulter-, Arm-, Brust- und Rückenmuskeln, Bauch, Gesäß und
Waden als echte Form statt nur als Schatten im Hautbild.

Beim Drehen zeigt jedes Videobild den Körper von einer anderen Seite im Umriss. Rund 50 Umrisse
legen die Form ringsum fest: von vorn Schultern, Arme, Taille und Beine, von der Seite das Profil
von Brust, Bauch, Rücken und Gesäß. Gesucht ist eine glatte Verschiebung jedes Punktes entlang
seiner Normalen, bei der in jedem Bild

- die Randlinie des Modells (wo die Fläche genau quer zum Blick steht, nicht verdeckt) auf dem
  Umriss liegt,
- kein Punkt des Modells außerhalb des Umrisses landet.

Lage und Haltung jedes Bildes werden dabei mit nachgeschärft (beim Drehen tritt man etwas auf der
Stelle, die Arme bewegen sich). Die Feinform bewegt sich in glatten Schritten (jede Änderung verteilt
sich auf die Nachbarschaft), sonst entstünde aus Maskenrauschen Riefen von Punkt zu Punkt. Annys Netz, Skelett und Gewichte bleiben gleich, die Verschiebung
landet in den Feinform-Offsets.

Grenze: Ein Umriss zeigt nur, was von irgendeiner Seite am Rand liegt. Mulden, die nie am Rand
liegen (Rinne der Wirbelsäule, Linien zwischen den Bauchmuskeln), bleiben so glatt wie die Vorlage.
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np
import torch

from .body import BodyModel, BodyParams
from .camera import Intrinsics, yaw_matrix
from .fit import (
    FitFrame,
    Log,
    View,
    _mesh_edges,
    _rest_tensors,
    _soft_gate,
    measurement_loss,
    repair_offsets,
    view_pose_matrices,
)
from .measure import MeasureProbe
from .texture import face_buffer

DETAIL_SCALE = 2.0  # Umrisse in doppelter Analyseauflösung (bei 1280er Bildern etwa 1 mm je Pixel)
MARGIN = 24  # Pixel um die Person herum
RIM_EDGE = 1.5  # Pixel: so nah am Rand des gerenderten Umrisses zählt ein Punkt der Randlinie
RIM_DEPTH = 0.02  # Meter: so weit darf ein Randpunkt hinter der vordersten Fläche liegen
ROBUST = 0.008  # Meter: größere Abweichungen zählen kaum mehr (Hand vor der Hüfte, Maskenfehler)
INSIDE_TOL = 0.0015  # Meter: so weit darf ein Punkt über den Umriss hinausragen (Maskenrauschen)
INSIDE_WEIGHT = 0.5
SMOOTH = 1.0  # Krümmung der Feinform über das Netz (zusätzlich zu LARGE_STEPS)
SIZE = 1.0e-3  # Feinform klein halten, wo die Umrisse nichts sagen
POSE_REG = 1.0e-4  # Haltung je Bild bleibt nahe an der Ausrichtung fürs Hautbild
# Lage und Drehung je Bild bleiben nahe an der Ausrichtung (fit_views, mit Füßen und Körperpunkten):
# ohne Fußsohlen im Umriss ist der Abstand zur Kamera schwach bestimmt; Körper dünner und Kamera
# näher sähen sonst fast gleich aus. Eine Abweichung um VIEW_SIGMA kostet so viel wie 2 mm am Rand.
VIEW_REG = 4.0e-6
VIEW_SIGMA = (0.01, math.radians(1.0))  # Meter, Radiant
ROUNDS = 4  # neue Randpunkte je Runde
STEPS = 60  # Schritte je Runde
LARGE_STEPS = 20.0  # Glätte der Schritte (λ): größer = glattere, großflächigere Formänderungen
DISP_LR = 2e-3  # Schrittweite der Feinform
POSE_LR = 3e-3  # Schrittweite der Haltung je Bild (Radiant)
VIEW_LR = 1e-3  # Schrittweite von Drehung und Lage je Bild
MEASURE_WEIGHT = 2e-3  # eingegebene Maßbandwerte gegenüber den Umrissen (1 cm ≈ 2 mm Umrissabstand)
MEASURED = ("chest", "waist", "hips", "thigh", "calf", "upperarm", "neck", "inseam")
# Hände und Füße: Finger, Handgelenk und Füße bewegen sich frei und sind in der Maske ungenau
IGNORE_BONES = ("finger", "metacarpal", "wrist", "toe", "foot")


@dataclasses.dataclass
class Outlines:
    """Umrisse aller Bilder als vorzeichenbehafteter Abstand (Pixel, positiv außerhalb), auf eine
    gemeinsame Größe zugeschnitten."""

    sdf: torch.Tensor  # (K, 1, H, W)
    intrs: list[Intrinsics]  # je Bild: Kamera des Ausschnitts
    masks: list[np.ndarray]  # (H, W) bool

    @property
    def size(self) -> tuple[int, int]:
        return self.sdf.shape[-1], self.sdf.shape[-2]


def outlines(frames: list[FitFrame], scale: float = DETAIL_SCALE, margin: int = MARGIN) -> Outlines:
    """Masken vergrößern (die weiche MediaPipe-Maske gibt den Rand feiner als ein Pixel her),
    auf die Person zuschneiden und in Abstandsfelder umrechnen."""
    crops, intrs = [], []
    for f in frames:
        intr = f.intr.scaled(scale)
        m = cv2.resize(f.mask.astype(np.float32), (intr.width, intr.height), interpolation=cv2.INTER_LINEAR) > 0.5
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        if n > 2:
            m = lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        ys, xs = np.nonzero(m)
        y0, x0 = max(int(ys.min()) - margin, 0), max(int(xs.min()) - margin, 0)
        y1, x1 = min(int(ys.max()) + margin + 1, intr.height), min(int(xs.max()) + margin + 1, intr.width)
        crops.append(m[y0:y1, x0:x1])
        intrs.append(Intrinsics(x1 - x0, y1 - y0, intr.focal, intr.cx - x0, intr.cy - y0))
    h = max(c.shape[0] for c in crops)
    w = max(c.shape[1] for c in crops)
    sdf = np.empty((len(crops), 1, h, w), np.float32)
    masks = []
    for k, c in enumerate(crops):
        m = np.zeros((h, w), bool)
        m[: c.shape[0], : c.shape[1]] = c
        masks.append(m)
        intrs[k] = dataclasses.replace(intrs[k], width=w, height=h)
        out = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        inside = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        sdf[k, 0] = np.where(m, 0.5 - inside, out - 0.5)
    return Outlines(torch.from_numpy(sdf), intrs, masks)


def _to_camera(verts: torch.Tensor, yaw: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """(K, V, 3) Anny -> (K, V, 3) Kamera, je Bild eigene Drehung und Verschiebung."""
    r = verts @ yaw_matrix(yaw).transpose(-1, -2)
    return torch.stack([r[..., 0], -r[..., 2], r[..., 1]], -1) + trans[:, None]


@dataclasses.dataclass
class RefineReport:
    rim_before_mm: float  # mittlerer Abstand der Randpunkte zum Umriss vor der Verfeinerung
    rim_after_mm: float
    iou_before: float  # mittlere Deckung von Modell und Umriss (doppelte Analyseauflösung)
    iou_after: float
    shift_mm: dict[str, float]  # mittlere |Verschiebung| je Körperbereich
    repaired: int = 0

    def to_json(self) -> dict:
        return {"randabstand_vorher_mm": round(self.rim_before_mm, 2),
                "randabstand_nachher_mm": round(self.rim_after_mm, 2),
                "deckung_vorher": round(self.iou_before, 4), "deckung_nachher": round(self.iou_after, 4),
                "verschiebung_mm": {k: round(v, 1) for k, v in self.shift_mm.items()},
                "geglaettet": self.repaired}


@dataclasses.dataclass
class _RimPoints:
    """Randlinie des Modells in allen Bildern: Punkte auf Netzkanten, an denen die Normale genau
    quer zum Sehstrahl steht (p = (1 - t) * a + t * b), nur wo sie den Umriss bildet."""

    view: torch.Tensor  # (P,)
    a: torch.Tensor
    b: torch.Tensor
    t: torch.Tensor  # (P,)
    counts: torch.Tensor  # (K,) Punkte je Bild


class _Rim:
    def __init__(self, model: BodyModel, use: np.ndarray):
        self.faces = model.faces
        e = _mesh_edges(model.faces)
        self.edges = e[use[e].all(1)]

    def select(self, cams: np.ndarray, normals: np.ndarray, out: Outlines) -> _RimPoints:
        views, aa, bb, tt = [], [], [], []
        for k, cam in enumerate(cams):
            intr = out.intrs[k]
            ids = face_buffer(cam.astype(np.float64), self.faces, intr)
            inside = (ids > 0).astype(np.uint8)
            to_out = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
            to_in = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 3)
            s = (normals[k] * cam).sum(1) / np.linalg.norm(cam, axis=1)
            a, b = self.edges[:, 0], self.edges[:, 1]
            cross = s[a] * s[b] < 0
            a, b = a[cross], b[cross]
            t = s[a] / (s[a] - s[b])
            p = cam[a] + t[:, None] * (cam[b] - cam[a])
            z = np.maximum(p[:, 2], 1e-3)
            xi = np.round(intr.focal * p[:, 0] / z + intr.cx).astype(int)
            yi = np.round(intr.focal * p[:, 1] / z + intr.cy).astype(int)
            ok = (xi >= 2) & (xi < intr.width - 2) & (yi >= 2) & (yi < intr.height - 2)
            a, b, t, p, xi, yi = a[ok], b[ok], t[ok], p[ok], xi[ok], yi[ok]
            near = np.maximum(to_out[yi, xi], to_in[yi, xi]) <= RIM_EDGE
            a, b, t, p, xi, yi = a[near], b[near], t[near], p[near], xi[near], yi[near]
            # verdeckt? (z. B. der hintere Arm am Rand des Rumpfes)
            f = ids[yi, xi] - 1
            front = cam[self.faces[np.maximum(f, 0)]][:, :, 2].min(1)
            keep = (f < 0) | (p[:, 2] <= front + RIM_DEPTH)
            views.append(np.full(int(keep.sum()), k))
            aa.append(a[keep])
            bb.append(b[keep])
            tt.append(t[keep])
        view = np.concatenate(views)
        return _RimPoints(torch.as_tensor(view), torch.as_tensor(np.concatenate(aa)),
                          torch.as_tensor(np.concatenate(bb)), torch.as_tensor(np.concatenate(tt), dtype=torch.float32),
                          torch.as_tensor(np.bincount(view, minlength=len(cams))))


def _laplacian(model: BodyModel) -> torch.Tensor:
    import scipy.sparse as sp

    n = model.num_vertices
    e = _mesh_edges(model.faces)
    adj = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    adj = adj.tocsr()
    deg = np.maximum(np.asarray(adj.sum(1)).ravel(), 1)
    lap = (sp.eye(n) - sp.diags(1 / deg) @ adj).tocoo()
    return torch.sparse_coo_tensor(np.vstack([lap.row, lap.col]), lap.data.astype(np.float32), (n, n),
                                   check_invariants=False).coalesce()


class _SmoothSolve(torch.autograd.Function):
    """x = (I + λ L)⁻¹ u mit symmetrischer Netz-Laplace-Matrix: die Feinform wird über u bewegt, jede
    Änderung verteilt sich glatt auf die Nachbarschaft ("Large Steps in Inverse Rendering",
    Nicolet u. a. 2021). So entsteht kein Rauschen von Punkt zu Punkt."""

    @staticmethod
    def forward(ctx, u, solve):
        ctx.solve = solve
        return torch.as_tensor(solve(u.detach().cpu().numpy().astype(np.float64)), dtype=u.dtype, device=u.device)

    @staticmethod
    def backward(ctx, grad):
        g = ctx.solve(grad.detach().cpu().numpy().astype(np.float64))
        return torch.as_tensor(g, dtype=grad.dtype, device=grad.device), None


def _smooth_solver(model: BodyModel, lam: float):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    n = model.num_vertices
    e = _mesh_edges(model.faces)
    adj = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    adj = adj.tocsr()
    lap = sp.diags(np.asarray(adj.sum(1)).ravel()) - adj
    return spla.factorized((sp.eye(n) + lam * lap).tocsc())


class _UniformAdam:
    """Adam mit einer gemeinsamen Schrittweite für alle Punkte (statt je Punkt): schwach bestimmte
    Stellen wandern nicht von selbst davon (Nicolet u. a. 2021)."""

    def __init__(self, param: torch.Tensor, lr: float, betas=(0.9, 0.999), eps: float = 1e-12):
        self.p, self.lr, self.b1, self.b2, self.eps = param, lr, betas[0], betas[1], eps
        self.m = torch.zeros_like(param)
        self.v = 0.0
        self.t = 0

    @torch.no_grad()
    def step(self) -> None:
        g = self.p.grad
        self.t += 1
        self.m.mul_(self.b1).add_(g, alpha=1 - self.b1)
        self.v = self.b2 * self.v + (1 - self.b2) * float((g * g).max())
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        self.p.sub_(self.lr * m_hat / (math.sqrt(v_hat) + self.eps))


def _vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """(K, V, 3) -> (K, V, 3) Einheitsnormalen."""
    tri = verts[:, faces]
    fn = torch.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0], dim=-1)
    vn = torch.zeros_like(verts)
    for i in range(3):
        vn = vn.index_add(1, faces[:, i], fn)
    return vn / vn.norm(dim=-1, keepdim=True).clamp_min(1e-12)


REGIONS = {
    "Schultern und Arme": ("shoulder", "upperarm", "lowerarm", "clavicle"),
    "Brust und Rücken": ("spine01", "spine02", "spine03"),
    "Bauch und Hüfte": ("spine04", "spine05", "pelvis", "root"),
    "Beine": ("upperleg", "lowerleg"),
}


def refine_shape(model: BodyModel, params: BodyParams, frames: list[FitFrame], views: list[View],
                 measures: dict[str, float] | None = None, rounds: int = ROUNDS, steps: int = STEPS,
                 smooth: float = SMOOTH, scale: float = DETAIL_SCALE, log: Log | None = None
                 ) -> tuple[BodyParams, list[View], RefineReport]:
    """Feinform und Ansichten aus allen Umrissen. frames/views wie für das Hautbild (fit_views).

    measures: eingegebene Maße (cm); Umfänge und Schrittlänge mit Maßbandwert bleiben erhalten."""
    if len(frames) != len(views) or not frames:
        raise ValueError("Zu jedem Bild gehört eine Ansicht.")
    device = model.device
    out = outlines(frames, scale)
    k = len(frames)
    faces_np = model.faces
    faces = torch.as_tensor(faces_np, device=device)
    ignore = model.dominant_bone_mask(lambda n: n.startswith(IGNORE_BONES))
    use = ~ignore
    use_t = torch.as_tensor(use, device=device)
    gate = torch.as_tensor(_soft_gate(model, _mesh_edges(faces_np)), device=device)
    lap = _laplacian(model).to(device)
    rim_finder = _Rim(model, use)

    with torch.no_grad():
        rest = _rest_tensors(model, params)
        base = rest["_unscaled_vertices"]
        normals = _vertex_normals(base, faces)[0]
    focal = torch.tensor([i.focal for i in out.intrs], device=device)[:, None]
    center = torch.tensor([[i.cx, i.cy] for i in out.intrs], device=device, dtype=torch.float32)
    w, h = out.size
    sdf = out.sdf.to(device)

    yaws = torch.nn.Parameter(torch.tensor([v.yaw for v in views], dtype=torch.float32, device=device))
    trans = torch.nn.Parameter(torch.tensor(np.stack([v.trans for v in views]), dtype=torch.float32, device=device))
    pose0 = torch.tensor(np.stack([v.pose for v in views]), dtype=torch.float32, device=device)
    pose = torch.nn.Parameter(pose0.clone())
    yaw0, trans0 = yaws.detach().clone(), trans.detach().clone()
    raw = torch.nn.Parameter(torch.zeros(model.num_vertices, device=device))
    solve = _smooth_solver(model, LARGE_STEPS)

    def displacement() -> torch.Tensor:
        return gate * _SmoothSolve.apply(raw, solve)

    def rest_vertices() -> torch.Tensor:
        """Ruhepose mit Feinform und Maßstab (zum Messen), (V, 3)."""
        return (base[0] + displacement()[:, None] * normals) * params.scale

    def posed(with_disp: bool = True) -> torch.Tensor:
        r = dict(rest)
        if with_disp:
            r["_unscaled_vertices"] = base + (displacement()[:, None] * normals)[None]
        return model.posed(r, view_pose_matrices(model, pose), params.scale)["vertices"]

    def distances(cams: torch.Tensor) -> torch.Tensor:
        """Abstand jedes Punktes zum Umriss, in Metern (positiv außerhalb). (K, V)"""
        z = cams[..., 2].clamp_min(1e-3)
        uv = focal[..., None] * cams[..., :2] / z[..., None] + center[:, None]
        grid = torch.stack([uv[..., 0] / (w - 1) * 2 - 1, uv[..., 1] / (h - 1) * 2 - 1], -1)
        px = torch.nn.functional.grid_sample(sdf, grid[:, None], align_corners=True, padding_mode="border")
        return px[:, 0, 0] * z / focal

    sdf_flat = sdf.reshape(-1)

    def rim_distances(cams: torch.Tensor, rp: _RimPoints) -> torch.Tensor:
        """Abstand der Randlinienpunkte zum Umriss, in Metern. (P,)"""
        p = cams[rp.view, rp.a] * (1 - rp.t)[:, None] + cams[rp.view, rp.b] * rp.t[:, None]
        z = p[:, 2].clamp_min(1e-3)
        f = focal[rp.view, 0]
        x = (f * p[:, 0] / z + center[rp.view, 0]).clamp(0, w - 1.001)
        y = (f * p[:, 1] / z + center[rp.view, 1]).clamp(0, h - 1.001)
        x0, y0 = x.detach().floor(), y.detach().floor()
        fx, fy = x - x0, y - y0
        i = rp.view * (h * w) + y0.long() * w + x0.long()
        px = ((1 - fx) * (1 - fy) * sdf_flat[i] + fx * (1 - fy) * sdf_flat[i + 1]
              + (1 - fx) * fy * sdf_flat[i + w] + fx * fy * sdf_flat[i + w + 1])
        return px * z / f

    def rims() -> _RimPoints:
        with torch.no_grad():
            v = posed()
            cams = _to_camera(v, yaws, trans)
            n = _vertex_normals(cams, faces)
        return rim_finder.select(cams.cpu().numpy(), n.cpu().numpy(), out)

    def rim_error(rp: _RimPoints) -> float:
        with torch.no_grad():
            return float(rim_distances(_to_camera(posed(), yaws, trans), rp).abs().mean()) * 1000

    def view_ious() -> list[float]:
        with torch.no_grad():
            cams = _to_camera(posed(), yaws, trans).cpu().numpy().astype(np.float64)
        ious = []
        for i in range(k):
            r = face_buffer(cams[i], faces_np, out.intrs[i]) > 0
            ious.append(float((r & out.masks[i]).sum() / max((r | out.masks[i]).sum(), 1)))
        return ious

    sel = rims()
    rim_before, iou_before = rim_error(sel), float(np.mean(view_ious()))
    if log:
        log(f"Feinform: {k} Umrisse, Randpunkte im Mittel {rim_before:.1f} mm neben dem Umriss "
            f"(Deckung {iou_before * 100:.1f} %).")
    c2 = ROBUST ** 2
    targets = {key: float(v) for key, v in (measures or {}).items() if key in MEASURED and v}
    shape_opt = _UniformAdam(raw, DISP_LR)
    for r in range(rounds):
        if r:
            sel = rims()
        probe = None
        if targets:
            with torch.no_grad():
                now = rest_vertices()
            probe = MeasureProbe(model, now.cpu().numpy(), (rest["bone_heads"][0]).cpu().numpy())
        per_view = 1.0 / (sel.counts.clamp_min(1).float() * int((sel.counts > 0).sum()))

        def energy(sel=sel, per_view=per_view, probe=probe):
            cams = _to_camera(posed(), yaws, trans)
            r_rim = rim_distances(cams, sel)
            e_rim = (per_view[sel.view] * c2 * torch.log1p(r_rim ** 2 / c2)).sum()
            e_in = (torch.relu(distances(cams)[:, use_t] - INSIDE_TOL) ** 2).mean()
            gd = displacement()
            e_smooth = smooth * (torch.sparse.mm(lap, gd[:, None]) ** 2).mean()
            loss = e_rim + INSIDE_WEIGHT * e_in + e_smooth + SIZE * (gd ** 2).mean()
            if probe is not None:
                loss = loss + MEASURE_WEIGHT * measurement_loss(probe.evaluate(rest_vertices()), targets)
            e_view = (((trans - trans0) / VIEW_SIGMA[0]) ** 2).sum(1) + ((yaws - yaw0) / VIEW_SIGMA[1]) ** 2
            return loss + POSE_REG * ((pose - pose0) ** 2).sum() / k + VIEW_REG * e_view.mean()

        opt = torch.optim.Adam([{"params": [pose], "lr": POSE_LR}, {"params": [yaws, trans], "lr": VIEW_LR}])
        for _ in range(steps):
            opt.zero_grad()
            raw.grad = None
            energy().backward()
            opt.step()
            shape_opt.step()
        if log:
            log(f"Feinform: Runde {r + 1}/{rounds}, Randpunkte im Mittel {rim_error(sel):.1f} mm neben dem Umriss.")

    sel = rims()
    rim_after = rim_error(sel)
    with torch.no_grad():
        shift = displacement().cpu().numpy()
    result = params.copy()
    extra = (shift[:, None] * normals.cpu().numpy()).astype(np.float32)
    result.offsets = extra if params.offsets is None else (params.offsets + extra).astype(np.float32)
    result, repaired = repair_offsets(model, result)
    new_views = [View(float(yaws[i].detach()), trans[i].detach().cpu().numpy().astype(np.float32),
                      pose[i].detach().cpu().numpy().astype(np.float32)) for i in range(k)]
    with torch.no_grad():
        rest = _rest_tensors(model, result)
        base = rest["_unscaled_vertices"]
        raw.zero_()
    ious = view_ious()
    for v, iou in zip(new_views, ious, strict=True):
        v.iou = iou
    iou_after = float(np.mean(ious))
    regions = {}
    for name, prefixes in REGIONS.items():
        m = model.dominant_bone_mask(lambda n, p=prefixes: n.startswith(p))
        if m.any():
            regions[name] = float(np.abs(shift[m]).mean() * params.scale * 1000)
    report = RefineReport(rim_before, rim_after, iou_before, iou_after, regions, repaired)
    if log:
        log(f"Feinform fertig: Randpunkte {rim_before:.1f} → {rim_after:.1f} mm neben dem Umriss, "
            f"Deckung {iou_before * 100:.1f} → {iou_after * 100:.1f} %.")
    return result, new_views, report


__all__ = ["Outlines", "RefineReport", "outlines", "refine_shape"]
