"""Prüft ein Körpernetz vor dem Export: wasserdicht, sauber orientiert, keine kaputten Flächen.

Das Netz besteht aus mehreren geschlossenen Teilen (Körper, Augen, Mundraum). Jeder Teil muss
für sich wasserdicht sein. Zusätzlich wird geprüft, ob Feinverschiebungen aus der Anpassung
Flächen umgeklappt haben (Vergleich mit der Anny-Vorlage).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import trimesh


@dataclasses.dataclass
class MeshReport:
    vertices: int
    faces: int
    components: int
    watertight: bool
    winding_consistent: bool
    outward: bool
    degenerate_faces: int
    nonfinite_vertices: int
    flipped_faces: int
    intersections: int
    volume_m3: float
    problems: list[str]
    warnings: list[str] = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        if self.ok:
            return (f"Netz in Ordnung: {self.vertices} Punkte, {self.faces} Dreiecke, "
                    f"{self.components} geschlossene Teile, wasserdicht, ohne Selbstüberschneidung.")
        return "Netz fehlerhaft: " + " ".join(self.problems)


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return np.cross(v1 - v0, v2 - v0)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flächengewichtete Normalen pro Punkt (Einheitslänge)."""
    fn = face_normals(vertices, faces)
    vn = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    length = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(length, 1e-12)


def _segment_hits_triangle(p0, p1, a, b, c, eps=1e-9) -> np.ndarray:
    """Vektorisierter Möller-Trumbore-Test: schneidet die Strecke p0-p1 das Dreieck abc?"""
    d = p1 - p0
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = (e1 * h).sum(1)
    ok = np.abs(det) > 1e-15
    inv = 1.0 / np.where(ok, det, 1.0)
    s = p0 - a
    u = (s * h).sum(1) * inv
    q = np.cross(s, e1)
    v = (d * q).sum(1) * inv
    t = (e2 * q).sum(1) * inv
    return ok & (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)


def self_intersections(vertices: np.ndarray, faces: np.ndarray) -> set[tuple[int, int]]:
    """Paare sich schneidender Dreiecke (ohne Nachbarn mit gemeinsamem Punkt)."""
    tri = vertices[faces]
    lo, hi = tri.min(1), tri.max(1)
    edge = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    cell = 2.0 * float(np.median(edge))
    c0 = np.floor(lo / cell).astype(np.int64)
    c1 = np.floor(hi / cell).astype(np.int64)
    keys, owners = [], []
    span = c1 - c0
    for dx in range(int(span[:, 0].max()) + 1):
        for dy in range(int(span[:, 1].max()) + 1):
            for dz in range(int(span[:, 2].max()) + 1):
                sel = (span[:, 0] >= dx) & (span[:, 1] >= dy) & (span[:, 2] >= dz)
                c = c0[sel] + np.array([dx, dy, dz])
                keys.append((c[:, 0] * 73856093) ^ (c[:, 1] * 19349663) ^ (c[:, 2] * 83492791))
                owners.append(np.where(sel)[0])
    keys, owners = np.concatenate(keys), np.concatenate(owners)
    order = np.argsort(keys, kind="stable")
    keys, owners = keys[order], owners[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    sizes = np.diff(np.r_[starts, len(keys)])
    pairs = []
    for size in np.unique(sizes[sizes > 1]):
        groups = starts[sizes == size][:, None] + np.arange(size)[None]
        ia, ib = np.triu_indices(size, 1)
        members = owners[groups]
        pairs.append(np.stack([members[:, ia].ravel(), members[:, ib].ravel()], 1))
    if not pairs:
        return set()
    pairs = np.sort(np.concatenate(pairs), axis=1)
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    i, j = pairs[:, 0], pairs[:, 1]
    overlap = np.all((lo[i] <= hi[j]) & (lo[j] <= hi[i]), axis=1)
    shared = (faces[i][:, :, None] == faces[j][:, None, :]).any((1, 2))
    i, j = i[overlap & ~shared], j[overlap & ~shared]
    hit = np.zeros(len(i), bool)
    for x, y in ((i, j), (j, i)):
        a, b, c = tri[y, 0], tri[y, 1], tri[y, 2]
        for k in range(3):
            hit |= _segment_hits_triangle(tri[x, k], tri[x, (k + 1) % 3], a, b, c)
    return set(zip(i[hit].tolist(), j[hit].tolist(), strict=True))


def check_mesh(vertices: np.ndarray, faces: np.ndarray, template: np.ndarray | None = None,
               min_area: float = 1e-12, intersections: bool = True,
               digit_faces: np.ndarray | None = None) -> MeshReport:
    """template: dieselbe Form ohne Feinform (für umgeklappte Dreiecke und neue Überschneidungen).
    digit_faces: Dreiecke an Fingern und Zehen; Berührungen zwischen ihnen sind normal."""
    problems: list[str] = []
    nonfinite = int((~np.isfinite(vertices)).any(axis=1).sum())
    if nonfinite:
        problems.append(f"{nonfinite} Punkte haben ungültige Koordinaten.")
        vertices = np.nan_to_num(vertices)

    mesh = trimesh.Trimesh(vertices, faces, process=False, validate=False)
    parts = mesh.split(only_watertight=False)
    open_parts = sum(1 for p in parts if not p.is_watertight)
    watertight = bool(mesh.is_watertight) and open_parts == 0
    if not watertight:
        problems.append(f"Netz hat Löcher ({open_parts} von {len(parts)} Teilen offen).")
    winding = bool(mesh.is_winding_consistent)
    if not winding:
        problems.append("Flächen sind uneinheitlich orientiert.")
    volume = float(mesh.volume) if watertight else float("nan")
    outward = watertight and volume > 0
    if watertight and not outward:
        problems.append("Normalen zeigen nach innen.")

    area = 0.5 * np.linalg.norm(face_normals(vertices, faces), axis=1)
    degenerate = int((area < min_area).sum())
    if degenerate:
        problems.append(f"{degenerate} Dreiecke ohne Fläche.")

    warnings: list[str] = []
    crossing = 0
    if intersections:
        labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(faces))
        main = labels == np.bincount(labels).argmax()  # Körper; Augen und Mundraum sitzen bewusst darin
        digits = np.zeros(len(faces), bool) if digit_faces is None else digit_faces

        def relevant(pairs: set[tuple[int, int]]) -> set[tuple[int, int]]:
            return {(i, j) for i, j in pairs if main[i] and main[j] and not (digits[i] and digits[j])}

        found = relevant(self_intersections(vertices, faces))
        if template is not None:
            base = relevant(self_intersections(template, faces))
            crossing = len(found - base)
            if crossing:
                problems.append(f"{crossing} Dreieckspaare durchdringen sich durch die Feinanpassung.")
            found &= base
        if found:
            warnings.append(f"{len(found)} Dreieckspaare berühren sich durch die Körperform "
                            "(z. B. Oberschenkel innen); für Animation unkritisch.")

    flipped = 0
    if template is not None:
        a = face_normals(vertices, faces)
        b = face_normals(template, faces)
        flipped = int(((a * b).sum(1) < 0).sum())
        if flipped:
            problems.append(f"{flipped} Dreiecke durch die Feinanpassung umgeklappt.")

    return MeshReport(
        vertices=len(vertices), faces=len(faces), components=len(parts), watertight=watertight,
        winding_consistent=winding, outward=outward, degenerate_faces=degenerate,
        nonfinite_vertices=nonfinite, flipped_faces=flipped, intersections=crossing,
        volume_m3=volume, problems=problems, warnings=warnings,
    )
