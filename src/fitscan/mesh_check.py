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
    volume_m3: float
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        if self.ok:
            return (f"Netz in Ordnung: {self.vertices} Punkte, {self.faces} Dreiecke, "
                    f"{self.components} geschlossene Teile, wasserdicht.")
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


def check_mesh(vertices: np.ndarray, faces: np.ndarray, template: np.ndarray | None = None,
               min_area: float = 1e-12) -> MeshReport:
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
        nonfinite_vertices=nonfinite, flipped_faces=flipped, volume_m3=volume, problems=problems,
    )
