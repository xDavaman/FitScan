"""Prüfwerkzeug für Bewegungen: steckt der Körper im Boden, in einem Gerät oder in sich selbst?

Für jedes Bild einer Bewegung bekommt `check_frame` die verformten Punkte des Körpers und meldet:
- Boden: Punkte unter z = 0 (Anny-Koordinaten, Z oben), mit Tiefe.
- Geräte: Punkte innerhalb einer Stange, Hantelscheibe, Bank oder Matte (einfache Grundformen).
- Körper in sich: Dreiecke, die einander schneiden (z. B. Unterarm im Oberschenkel). Was schon in
  der Bezugshaltung schneidet (Augäpfel in den Lidern, Mundraum), zählt nicht.

Die Treffer werden nach Körperteilen zusammengefasst, damit man sieht, was wo hängt.
"""

from __future__ import annotations

import dataclasses

import numpy as np

FLOOR_TOLERANCE = 0.001  # 1 mm, so weit gibt eine Fußsohle oder ein Rücken nach
OBJECT_TOLERANCE = 0.001


# --------------------------------------------------------------------------- Grundformen


@dataclasses.dataclass
class Cylinder:
    """Stange, Griff oder Hantelscheibe: Mitte, Achse, Radius, halbe Länge (Meter)."""

    name: str
    centre: np.ndarray
    axis: np.ndarray
    radius: float
    half_length: float

    def depth(self, points: np.ndarray) -> np.ndarray:
        """Wie tief jeder Punkt in der Form steckt (positiv = innen)."""
        axis = self.axis / np.linalg.norm(self.axis)
        q = points - self.centre
        along = q @ axis
        radial = np.linalg.norm(q - np.outer(along, axis), axis=1)
        return np.minimum(self.radius - radial, self.half_length - np.abs(along))

    def moved(self, matrix: np.ndarray) -> Cylinder:
        return dataclasses.replace(self, centre=matrix[:3, :3] @ self.centre + matrix[:3, 3],
                                   axis=matrix[:3, :3] @ self.axis)


@dataclasses.dataclass
class Box:
    """Bank, Matte, Kasten: Mitte, Drehung (3x3, Spalten = Achsen), halbe Kantenlängen."""

    name: str
    centre: np.ndarray
    rotation: np.ndarray
    half: np.ndarray

    def depth(self, points: np.ndarray) -> np.ndarray:
        local = (points - self.centre) @ self.rotation
        return (np.asarray(self.half) - np.abs(local)).min(axis=1)

    def moved(self, matrix: np.ndarray) -> Box:
        return dataclasses.replace(self, centre=matrix[:3, :3] @ self.centre + matrix[:3, 3],
                                   rotation=matrix[:3, :3] @ self.rotation)


# --------------------------------------------------------------------------- Selbstdurchdringung


def _edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def _grid_pairs(lo_a, hi_a, lo_b, hi_b, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Grobsuche: Paare (a, b), deren Hüllquader sich in derselben Gitterzelle treffen."""
    origin = np.minimum(lo_a.min(0), lo_b.min(0))

    def cells(lo, hi):
        i0 = np.floor((lo - origin) / cell).astype(np.int64)
        i1 = np.floor((hi - origin) / cell).astype(np.int64)
        ids, keys = [], []
        span = i1 - i0
        for dx in range(int(span[:, 0].max()) + 1):
            for dy in range(int(span[:, 1].max()) + 1):
                for dz in range(int(span[:, 2].max()) + 1):
                    ok = (span[:, 0] >= dx) & (span[:, 1] >= dy) & (span[:, 2] >= dz)
                    idx = np.flatnonzero(ok)
                    c = i0[idx] + np.array([dx, dy, dz])
                    keys.append((c[:, 0] * 73856093) ^ (c[:, 1] * 19349663) ^ (c[:, 2] * 83492791))
                    ids.append(idx)
        return np.concatenate(ids), np.concatenate(keys)

    ia, ka = cells(lo_a, hi_a)
    ib, kb = cells(lo_b, hi_b)
    order = np.argsort(kb, kind="stable")
    kb, ib = kb[order], ib[order]
    left = np.searchsorted(kb, ka, "left")
    right = np.searchsorted(kb, ka, "right")
    counts = right - left
    a = np.repeat(ia, counts)
    starts = np.repeat(left - np.cumsum(counts) + counts, counts) + np.arange(counts.sum())
    b = ib[starts]
    # echte Hüllquader-Überlappung, dann doppelte Paare entfernen
    ok = np.all((lo_a[a] <= hi_b[b]) & (lo_b[b] <= hi_a[a]), axis=1)
    code = np.unique(a[ok].astype(np.int64) * len(lo_b) + b[ok])
    return code // len(lo_b), code % len(lo_b)


def _segment_hits_triangle(p, q, a, b, c) -> np.ndarray:
    """Möller-Trumbore für viele Strecken p-q gegen Dreiecke a-b-c (vektorisiert)."""
    d = q - p
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = p - a
    u = inv * np.einsum("ij,ij->i", s, h)
    qv = np.cross(s, e1)
    v = inv * np.einsum("ij,ij->i", d, qv)
    t = inv * np.einsum("ij,ij->i", e2, qv)
    return ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)


def intersecting_faces(vertices: np.ndarray, faces: np.ndarray, edges: np.ndarray | None = None,
                       cell: float = 0.012) -> np.ndarray:
    """Paare (Kante, Dreieck), bei denen eine Netzkante ein nicht benachbartes Dreieck durchstößt.
    Rückgabe: (N, 2) mit Kanten- und Dreiecksnummer."""
    edges = _edges(faces) if edges is None else edges
    p, q = vertices[edges[:, 0]], vertices[edges[:, 1]]
    tri = vertices[faces]
    ea, fb = _grid_pairs(np.minimum(p, q), np.maximum(p, q), tri.min(1), tri.max(1), cell)
    shared = (edges[ea, 0, None] == faces[fb]).any(1) | (edges[ea, 1, None] == faces[fb]).any(1)
    ea, fb = ea[~shared], fb[~shared]
    hit = _segment_hits_triangle(p[ea], q[ea], tri[fb, 0], tri[fb, 1], tri[fb, 2])
    return np.stack([ea[hit], fb[hit]], 1)


# --------------------------------------------------------------------------- Prüfung


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(out, faces[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


@dataclasses.dataclass
class Finding:
    kind: str  # "boden", "geraet", "koerper"
    where: str  # Körperteil(e)
    depth: float  # Meter: wie tief es steckt (bei "falte": Zahl der durchstoßenen Dreiecke)
    detail: str = ""


class Checker:
    """Prüft Bilder einer Bewegung. Einmal pro Körper anlegen, dann check_frame je Bild."""

    def __init__(self, model, reference_vertices: np.ndarray):
        self.model = model
        self.faces = model.faces
        self.edges = _edges(self.faces)
        idx, w = model.vertex_bone_indices, model.vertex_bone_weights
        owner_bone = idx[np.arange(len(idx)), w.argmax(1)]
        self.vertex_part = np.array([body_part(model.bone_labels[b]) for b in owner_bone])
        base = intersecting_faces(reference_vertices, self.faces, self.edges)
        self.baseline = {(int(e), int(f)) for e, f in base}

    def check_frame(self, vertices: np.ndarray, objects=(), self_check: bool = True) -> list[Finding]:
        out: list[Finding] = []
        below = -vertices[:, 2]
        if below.max() > FLOOR_TOLERANCE:
            out += self._grouped("boden", below > FLOOR_TOLERANCE, below)
        for obj in objects:
            depth = obj.depth(vertices)
            if depth.max() > OBJECT_TOLERANCE:
                out += self._grouped("geraet", depth > OBJECT_TOLERANCE, depth, obj.name)
        if self_check:
            hits = intersecting_faces(vertices, self.faces, self.edges)
            new = [(int(e), int(f)) for e, f in hits if (int(e), int(f)) not in self.baseline]
            pairs: dict[tuple[str, str], list] = {}
            for e, f in new:
                a = self.vertex_part[self.edges[e, 0]]
                b = self.vertex_part[self.faces[f, 0]]
                if a == b:
                    continue  # Falten innerhalb eines Körperteils (z. B. Kniekehle) zählen getrennt
                pairs.setdefault(tuple(sorted((a, b))), []).append(vertices[self.edges[e, 0]])
            if pairs:
                normals = _vertex_normals(vertices, self.faces)
            found = []
            for (a, b), points in pairs.items():
                depth = max(self._sink(vertices, normals, a, b, np.array(points)),
                            self._sink(vertices, normals, b, a, np.array(points)))
                found.append(Finding("koerper", f"{a} / {b}", depth, f"{len(points)} Dreiecke"))
            out += sorted(found, key=lambda f: -f.depth)
            folds: dict[str, int] = {}
            for e, f in new:
                a = self.vertex_part[self.edges[e, 0]]
                if a == self.vertex_part[self.faces[f, 0]]:
                    folds[a] = folds.get(a, 0) + 1
            for part, n in folds.items():
                if n >= 3:
                    out.append(Finding("falte", part, float(n)))
        return out

    def _sink(self, vertices, normals, part: str, other: str, near: np.ndarray, radius: float = 0.06) -> float:
        """Wie tief Punkte von part (nahe der Schnittstelle) unter der Haut von other liegen."""
        from scipy.spatial import cKDTree

        mine = np.flatnonzero(self.vertex_part == part)
        theirs = np.flatnonzero(self.vertex_part == other)
        close = cKDTree(near).query(vertices[mine], distance_upper_bound=radius)[0] < np.inf
        mine = mine[close]
        if len(mine) == 0 or len(theirs) == 0:
            return 0.0
        _, k = cKDTree(vertices[theirs]).query(vertices[mine])
        nearest = theirs[k]
        signed = np.einsum("ij,ij->i", vertices[mine] - vertices[nearest], normals[nearest])
        return float(max(0.0, -signed.min()))

    def _grouped(self, kind: str, mask: np.ndarray, depth: np.ndarray, detail: str = "") -> list[Finding]:
        out = []
        for part in np.unique(self.vertex_part[mask]):
            sel = mask & (self.vertex_part == part)
            out.append(Finding(kind, str(part), float(depth[sel].max()), detail))
        return sorted(out, key=lambda f: -f.depth)


def body_part(bone: str) -> str:
    """Anny-Knochen -> verständlicher Körperteil (deutsch, mit Seite)."""
    base, _, side = bone.partition(".")
    seite = {"L": " links", "R": " rechts"}.get(side, "")
    groups = (
        (("finger", "metacarpal", "wrist"), "Hand"),
        (("lowerarm",), "Unterarm"),
        (("upperarm", "shoulder", "clavicle"), "Oberarm"),
        (("upperleg",), "Oberschenkel"),
        (("lowerleg",), "Unterschenkel"),
        (("foot", "toe"), "Fuß"),
        (("neck", "head", "jaw", "eye", "tongue"), "Kopf"),
        (("spine01", "spine02", "breast"), "Brust"),
        (("spine03", "spine04", "spine05"), "Bauch"),
        (("pelvis", "root"), "Becken"),
    )
    for prefixes, label in groups:
        if base.startswith(prefixes):
            return label + seite
    return base + seite


def summarize(findings_per_frame: list[list[Finding]], fps: float = 30.0) -> list[str]:
    """Kurzbericht: je Art und Körperteil, in welchen Bildern und wie tief."""
    agg: dict[tuple[str, str], list] = {}
    for i, findings in enumerate(findings_per_frame):
        for f in findings:
            entry = agg.setdefault((f.kind, f.where), [i, i, 0.0, 0])
            entry[1] = i
            entry[2] = max(entry[2], f.depth)
            entry[3] += 1
    lines = []
    for (kind, where), (first, last, worst, n) in sorted(agg.items()):
        when = f"{first / fps:.2f}–{last / fps:.2f} s, {n} Bilder"
        if kind in ("boden", "geraet", "koerper"):
            lines.append(f"{kind}: {where} bis {worst * 1000:.0f} mm tief ({when})")
        else:
            lines.append(f"{kind}: {where}, bis {worst:.0f} Dreiecke ({when})")
    return lines

