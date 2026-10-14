"""Körpermaße am Modell, so wie ein Maßband sie nimmt.

Umfänge werden als konvexe Hülle eines Schnitts durch die passende Körperregion gemessen
(ein Maßband überspannt Dellen). Für die Anpassung werden die Schnittpunkte als feste
Kanten-Interpolationen gespeichert ("Probe"); damit sind die Maße in PyTorch
differenzierbar. Die Probe wird während einer Anpassung regelmäßig neu erzeugt.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from scipy.spatial import ConvexHull

from .body import BodyModel, BodyParams

# Schlüssel -> (deutscher Name, Einheit)
MEASURES: dict[str, tuple[str, str]] = {
    "height": ("Körpergröße", "cm"),
    "weight": ("Gewicht", "kg"),
    "chest": ("Brustumfang", "cm"),
    "waist": ("Taillenumfang", "cm"),
    "hips": ("Hüftumfang", "cm"),
    "thigh": ("Oberschenkelumfang", "cm"),
    "calf": ("Wadenumfang", "cm"),
    "upperarm": ("Oberarmumfang", "cm"),
    "neck": ("Halsumfang", "cm"),
    "inseam": ("Schrittlänge", "cm"),
}
CIRCUMFERENCES = ["chest", "waist", "hips", "thigh", "calf", "upperarm", "neck"]

BODY_DENSITY = 980.0  # kg/m³, wie in Annys Anthropometrie


@dataclasses.dataclass
class Ring:
    a: np.ndarray  # Vertex-Indizes der Kantenanfänge
    b: np.ndarray  # Vertex-Indizes der Kantenenden
    t: np.ndarray  # Interpolation entlang der Kante
    origin: np.ndarray
    normal: np.ndarray

    def points(self, verts: np.ndarray) -> np.ndarray:
        return verts[self.a] + self.t[:, None] * (verts[self.b] - verts[self.a])


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def section_ring(
    verts: np.ndarray, edges: np.ndarray, origin: np.ndarray, normal: np.ndarray,
    near: np.ndarray | None = None, max_dist: float | None = None,
) -> Ring | None:
    """Schnitt der Kanten mit einer Ebene, reduziert auf die konvexe Hülle (Maßband)."""
    normal = normal / np.linalg.norm(normal)
    d = (verts - origin) @ normal
    da, db = d[edges[:, 0]], d[edges[:, 1]]
    cross = (da * db) < 0
    if cross.sum() < 3:
        return None
    a, b = edges[cross, 0], edges[cross, 1]
    t = da[cross] / (da[cross] - db[cross])
    pts = verts[a] + t[:, None] * (verts[b] - verts[a])
    if near is not None and max_dist is not None:
        keep = np.linalg.norm(pts - near, axis=1) < max_dist
        if keep.sum() < 3:
            return None
        a, b, t, pts = a[keep], b[keep], t[keep], pts[keep]
    u, v = _plane_basis(normal)
    p2 = np.stack([(pts - origin) @ u, (pts - origin) @ v], axis=1)
    try:
        hull = ConvexHull(p2)
    except Exception:
        return None
    order = hull.vertices  # gegen den Uhrzeigersinn geordnet
    return Ring(a=a[order], b=b[order], t=t[order], origin=origin, normal=normal)


def ring_length(ring: Ring, verts: np.ndarray) -> float:
    p = ring.points(verts)
    return float(np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1).sum())


class MeasureProbe:
    """Legt fest, wo gemessen wird, und misst dann differenzierbar an beliebigen Vertices."""

    def __init__(self, model: BodyModel, verts: np.ndarray, bone_heads: np.ndarray):
        self.model = model
        faces = model.faces
        z = verts[:, 2]
        arm = model.arm_vertex_mask()

        def region_edges(mask: np.ndarray) -> np.ndarray:
            f = faces[mask[faces].all(axis=1)]
            return _unique_edges(f)

        def head(name: str) -> np.ndarray:
            return bone_heads[model.bone(name)]

        torso_edges = region_edges(~arm)
        thigh_edges = region_edges(model.dominant_bone_mask(lambda n: n.startswith("upperleg") and n.endswith(".L")))
        calf_edges = region_edges(model.dominant_bone_mask(lambda n: n.startswith("lowerleg") and n.endswith(".L")))
        uarm_edges = region_edges(model.dominant_bone_mask(lambda n: n.startswith("upperarm") and n.endswith(".L")))

        self.top = int(np.argmax(z))
        self.bottom = int(np.argmin(z))
        height = z[self.top] - z[self.bottom]

        knee_z = head("lowerleg01.L")[2]
        midline = (np.abs(verts[:, 0]) < 0.004 * height) & (z > knee_z) & (z < head("spine05")[2]) & ~arm
        cand = np.where(midline)[0]
        self.crotch = int(cand[np.argmin(z[cand])]) if len(cand) else int(np.argmin(np.abs(z - knee_z)))
        crotch_z = z[self.crotch]
        armpit_z = head("upperarm01.L")[2] - 0.05 * height
        span = armpit_z - crotch_z
        up = np.array([0.0, 0.0, 1.0])

        def band(edges, z0, z1, pick, near=None, max_dist=None, steps=14):
            best, best_len = None, None
            for zz in np.linspace(z0, z1, steps):
                origin = np.array([0.0, 0.0, zz]) if near is None else np.array([near[0], near[1], zz])
                ring = section_ring(verts, edges, origin, up, near=None if near is None else origin, max_dist=max_dist)
                if ring is None:
                    continue
                length = ring_length(ring, verts)
                if best is None or (length > best_len if pick == "max" else length < best_len):
                    best, best_len = ring, length
            return best

        def along_bone(edges, start, end, t0, t1, pick, steps=10, max_dist=None):
            direction = end - start
            best, best_len = None, None
            for tt in np.linspace(t0, t1, steps):
                origin = start + tt * direction
                ring = section_ring(verts, edges, origin, direction, near=origin,
                                    max_dist=max_dist or 0.15 * height)
                if ring is None:
                    continue
                length = ring_length(ring, verts)
                if best is None or (length > best_len if pick == "max" else length < best_len):
                    best, best_len = ring, length
            return best

        leg_len = crotch_z - z[self.bottom]
        thigh_xy = head("upperleg02.L")
        calf_xy = head("lowerleg02.L")
        ankle_z = head("foot.L")[2]
        self.rings: dict[str, Ring | None] = {
            "hips": band(torso_edges, crotch_z + 0.03 * span, crotch_z + 0.35 * span, "max"),
            "waist": band(torso_edges, crotch_z + 0.35 * span, crotch_z + 0.75 * span, "min"),
            "chest": band(torso_edges, crotch_z + 0.70 * span, armpit_z, "max"),
            "thigh": band(thigh_edges, crotch_z - 0.15 * leg_len, crotch_z - 0.02 * leg_len, "max",
                          near=thigh_xy, max_dist=0.2 * height),
            "calf": band(calf_edges, ankle_z + 0.3 * (knee_z - ankle_z), knee_z - 0.05 * (knee_z - ankle_z),
                         "max", near=calf_xy, max_dist=0.15 * height),
            "upperarm": along_bone(uarm_edges, head("upperarm01.L"), head("lowerarm01.L"), 0.3, 0.7, "max"),
            "neck": along_bone(torso_edges, head("neck01"), head("head"), 0.3, 0.8, "min",
                               max_dist=0.08 * height),
        }
        self.faces_t = torch.as_tensor(faces, dtype=torch.long, device=model.device)

    def evaluate(self, verts: torch.Tensor) -> dict[str, torch.Tensor]:
        """verts: (V, 3) in Metern. Rückgabe in cm bzw. kg."""
        out: dict[str, torch.Tensor] = {}
        out["height"] = (verts[self.top, 2] - verts[self.bottom, 2]) * 100
        out["inseam"] = (verts[self.crotch, 2] - verts[self.bottom, 2]) * 100
        f = self.faces_t
        v0, v1, v2 = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
        volume = (torch.cross(v0, v1, dim=-1) * v2).sum(-1).sum() / 6.0
        out["weight"] = volume.abs() * BODY_DENSITY
        for name, ring in self.rings.items():
            if ring is None:
                continue
            a = torch.as_tensor(ring.a, device=verts.device)
            b = torch.as_tensor(ring.b, device=verts.device)
            t = torch.as_tensor(ring.t, dtype=verts.dtype, device=verts.device)[:, None]
            p = verts[a] + t * (verts[b] - verts[a])
            out[name] = torch.linalg.norm(p - torch.roll(p, 1, dims=0), dim=1).sum() * 100
        return out


def measure_params(model: BodyModel, params: BodyParams) -> dict[str, float]:
    rest = model.rest_numpy(params)
    probe = MeasureProbe(model, rest["vertices"], rest["bone_heads"])
    vals = probe.evaluate(torch.as_tensor(rest["vertices"], dtype=torch.float32, device=model.device))
    return {k: round(float(v), 1) for k, v in vals.items()}


def measure_vertices(model: BodyModel, verts: np.ndarray, bone_heads: np.ndarray) -> dict[str, float]:
    probe = MeasureProbe(model, verts, bone_heads)
    vals = probe.evaluate(torch.as_tensor(verts, dtype=torch.float32, device=model.device))
    return {k: round(float(v), 1) for k, v in vals.items()}
