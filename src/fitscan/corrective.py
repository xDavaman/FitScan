"""Korrekturformen: Haut gibt dort nach, wo sich Körperteile beim starken Beugen berühren.

Lineares Skinning lässt beim tiefen Hüftbeugen den Oberschenkel in den Bauch und beim vollen
Ellbogenbeugen den Unterarm in den Bizeps laufen. Für jedes Gelenk werden einmal pro Körper bei ein
paar Beugewinkeln Verschiebungen der Ruhepunkte berechnet, die die Haut auf eine gemeinsame
Kontaktfläche legen und weich in die Umgebung auslaufen. Beim Abspielen werden sie nach dem
gemessenen Gelenkwinkel überblendet und vor dem Skinning addiert – wie Formregler (Morph Targets)
in glTF, Blender oder Cascadeur."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import posing as P
from .contact import _vertex_normals, body_part, winding

MARGIN = 0.002  # so weit bleiben die Häute nach der Korrektur auseinander (m)
REACH = 0.10  # so weit läuft eine Korrektur um die Berührung herum aus (m)


@dataclass
class Joint:
    """Ein Gelenk mit Korrekturformen: welcher Winkel sie steuert und welche Teile sich berühren."""

    name: str  # z. B. "huefte"
    side: str
    pairs: list[tuple[str, str]]  # Körperteile, die sich nicht durchdringen sollen
    angles: tuple[float, ...]  # Beugewinkel der Stützstellen (Grad)

    def angle(self, model, poses: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Beugewinkel je Bild aus den Knochenlagen (F, J, 4, 4)."""
        s = self.side
        if self.name == "huefte":
            pelvis, a, b = (model.bone(n) for n in ("root", f"upperleg01.{s}", f"lowerleg01.{s}"))
            rest = reference[b, :3, 3] - reference[a, :3, 3]
            # Oberschenkel im Becken, gemessen gegen seine Richtung in der Bezugshaltung
            down = np.einsum("fij,jk,k->fi", poses[:, pelvis, :3, :3], reference[pelvis, :3, :3].T, rest)
            thigh = poses[:, b, :3, 3] - poses[:, a, :3, 3]
        else:
            chain = ("upperleg01", "lowerleg01", "foot") if self.name == "knie" else ("upperarm01", "lowerarm01", "wrist")
            a, b, c = (model.bone(f"{n}.{s}") for n in chain)
            down = poses[:, b, :3, 3] - poses[:, a, :3, 3]
            thigh = poses[:, c, :3, 3] - poses[:, b, :3, 3]
        cos = np.einsum("fi,fi->f", down, thigh) / np.linalg.norm(down, axis=1) / np.linalg.norm(thigh, axis=1)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    def share(self, rest: np.ndarray) -> np.ndarray:
        """Anteil jedes Punkts an dieser Seite (V,): die Hüftform wird mit beiden Beinen berechnet
        und weich über die Körpermitte (6 cm) auf links (+X) und rechts aufgeteilt."""
        if self.name != "huefte":
            return np.ones(len(rest))
        x = rest[:, 0] if self.side == "L" else -rest[:, 0]
        return np.clip(0.5 + x / 0.06, 0, 1)

    def pose(self, angle: float) -> dict:
        """Haltung zum Berechnen der Form: nur dieses Gelenk auf dieser Seite gebeugt."""
        if self.name == "huefte":  # beide Beine: der Bauch liegt dann zwischen beiden Oberschenkeln
            return P.pose(P.stand(), legs=dict(hip=angle, knee=min(angle, 100), out=4, turn=-8))
        if self.name == "knie":  # Oberschenkel waagerecht, Wade legt sich an
            return P.pose(P.stand(), **{f"leg_{self.side}": dict(hip=90, knee=angle, out=4, turn=-8)})
        return P.pose(P.stand(), **{f"arm_{self.side}": dict(down=24, fwd=10, elbow=angle)})


def joints() -> list[Joint]:
    out = []
    for s, seite in (("L", "links"), ("R", "rechts")):
        out.append(Joint("huefte", s, [(t, f"Oberschenkel {x}") for x in ("links", "rechts")
                                       for t in ("Bauch", f"Becken {x}", "Brust")],
                         (30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0)))
        out.append(Joint("ellbogen", s, [(f"Oberarm {seite}", f"Unterarm {seite}")],
                         (60.0, 80.0, 100.0, 120.0, 135.0, 150.0)))
        out.append(Joint("knie", s, [(f"Oberschenkel {seite}", f"Unterschenkel {seite}")],
                         (90.0, 110.0, 125.0, 140.0, 150.0, 160.0)))
    return out


def _laplacian(faces: np.ndarray, n: int):
    from scipy import sparse

    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    a.data[:] = 1.0
    deg = np.asarray(a.sum(1)).ravel()
    return sparse.diags(deg) - a


def _pushes(vertices, normals, faces, parts, pairs, radius=0.06):
    """Für jeden Punkt, der in einem anderen Teil steckt (Windungszahl), die Verschiebung nach außen
    (halbe Tiefe + Rand); die andere Seite bekommt die andere Hälfte."""
    from scipy.spatial import cKDTree

    push = np.zeros_like(vertices)
    hit = np.zeros(len(vertices), bool)
    for a, b in pairs:
        for mine, theirs in ((a, b), (b, a)):
            i = np.flatnonzero(parts == mine)
            j = np.flatnonzero(parts == theirs)
            if len(i) == 0 or len(j) == 0:
                continue
            dist, k = cKDTree(vertices[j]).query(vertices[i], distance_upper_bound=radius)
            ok = np.isfinite(dist)
            i, k = i[ok], j[k[ok]]
            if len(i) == 0:
                continue
            inside = winding(vertices[i], vertices, faces) > 1.0
            i, k = i[inside], k[inside]
            # Hinaus entlang der Hautnormale des anderen Teils, so weit wie der nächste Hautpunkt
            # entfernt ist (beide Seiten je die Hälfte: so laufen die Häute nicht aneinander vorbei)
            depth = np.linalg.norm(vertices[k] - vertices[i], axis=1)
            step = normals[k] * (depth / 2 + MARGIN)[:, None]
            better = np.linalg.norm(step, axis=1) > np.linalg.norm(push[i], axis=1)
            push[i[better]] = step[better]
            hit[i] = True
    return push, hit


def relax(lap, delta: np.ndarray, region: np.ndarray, iterations: int = 10) -> np.ndarray:
    """Verschiebung im Bereich region glätten (je Schritt halb zum Mittel der Nachbarn): benachbarte
    Punkte, die in einer Falte verschieden weit geschoben wurden, sollen keine Zacken bilden."""
    from scipy import sparse

    deg = lap.diagonal()
    avg = sparse.diags(1.0 / np.maximum(deg, 1)) @ (sparse.diags(deg) - lap)
    out = delta.copy()
    for _ in range(iterations):
        out[region] = 0.5 * out[region] + 0.5 * (avg @ out)[region]
    return out


def solve_shape(model, skinned: np.ndarray, pairs, rest: np.ndarray, steps: int = 8,
                smooth: float = 1.0, relaxing: int = 10) -> np.ndarray:
    """Weltverschiebung je Punkt (V, 3), die die Teile in pairs auseinanderlegt und glatt ausläuft.
    rest: dieselben Punkte in der Bezugshaltung (dort berührt sich nichts)."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from scipy.spatial import cKDTree

    faces = model.faces
    idx, w = model.vertex_bone_indices, model.vertex_bone_weights
    parts = np.array([body_part(model.bone_labels[b]) for b in idx[np.arange(len(idx)), w.argmax(1)]])
    involved = np.isin(parts, [p for pair in pairs for p in pair])
    lap = _laplacian(faces, len(skinned))
    delta = np.zeros_like(skinned)
    held = np.zeros(len(skinned), bool)  # Punkte mit Zielverschiebung (bleiben gehalten)
    target = np.zeros_like(skinned)
    region = None
    for _ in range(steps):
        v = skinned + delta
        push, hit = _pushes(v, _vertex_normals(v, faces), faces, parts, pairs)
        if not hit.any():
            break
        if region is None:
            near = cKDTree(skinned[hit]).query(skinned, distance_upper_bound=REACH)[0] < np.inf
            region = np.flatnonzero(near & involved)
        target[hit] = delta[hit] + push[hit]
        held |= hit
        r = region
        pos = np.full(len(skinned), -1)
        pos[r] = np.arange(len(r))
        fixed = pos[np.flatnonzero(held)]
        keep = fixed >= 0
        fixed, goal = fixed[keep], target[np.flatnonzero(held)[keep]]
        lr = lap[r][:, r]  # Punkte außerhalb bleiben fest (Verschiebung 0)
        c = sparse.coo_matrix((np.ones(len(fixed)), (np.arange(len(fixed)), fixed)), shape=(len(fixed), len(r)))
        lhs = (smooth * lr + 10.0 * c.T @ c).tocsc()  # harmonisch: kein Überschwingen
        for d in range(3):
            delta[r, d] = spsolve(lhs, 10.0 * c.T @ goal[:, d])
        if relaxing:
            delta = relax(lap, delta, r, relaxing)
    return delta


def skin_linear(model, transforms: np.ndarray) -> np.ndarray:
    """Lineare Anteile des Skinnings je Punkt (V, 3, 3): Welt-Verschiebung = A @ Ruhe-Verschiebung."""
    idx, w = model.vertex_bone_indices, model.vertex_bone_weights
    return np.einsum("vk,vkij->vij", w, transforms[idx][..., :3, :3])


def build(fig: P.Figure, joint: Joint, step: float = 5.0) -> dict[float, np.ndarray]:
    """Korrekturformen eines Gelenks: Ruhe-Verschiebung (V, 3) je gemessenem Beugewinkel. Der Winkel wird in
    kleinen Schritten erhöht und die Form jeweils weitergeschoben, damit die Berührung immer flach
    bleibt und innen/außen sicher erkannt wird."""
    model = fig.model
    delta = np.zeros_like(fig.reference_vertices)
    share = joint.share(fig.reference_vertices)[:, None]
    shapes = {}
    for a in np.arange(step, max(joint.angles) + step / 2, step):
        frame = fig.poser.rotations(joint.pose(float(a)))
        out = fig.pose([frame], offsets=delta[None], transforms=True)
        d = solve_shape(model, out["vertices"][0], joint.pairs, fig.reference_vertices)
        if np.abs(d).max() > 0:
            delta = delta + np.linalg.solve(skin_linear(model, out["transforms"][0]), d[..., None])[..., 0]
        if any(abs(a - k) < step / 2 for k in joint.angles):
            # Schlüssel ist der gemessene Winkel, mit dem offsets() später nachschlägt
            measured = joint.angle(model, out["poses"], fig.reference_poses)[0]
            shapes[round(float(measured), 2)] = delta * share
    return shapes


class Correctives:
    """Alle Korrekturformen eines Körpers. offsets() liefert je Bild die Ruhe-Verschiebung, die
    Figure.pose vor dem Skinning addiert."""

    def __init__(self, model, reference_poses: np.ndarray, shapes: dict[str, tuple[np.ndarray, np.ndarray]]):
        self.model = model
        self.reference_poses = reference_poses
        self.shapes = shapes  # "huefte.L" -> (Winkel (K,), Verschiebungen (K, V, 3))
        self.joints = {f"{j.name}.{j.side}": j for j in joints()}

    @classmethod
    def build(cls, fig: P.Figure) -> Correctives:
        shapes = {}
        for j in joints():
            got = build(fig, j)
            angles = np.array(sorted(got))
            shapes[f"{j.name}.{j.side}"] = (angles, np.stack([got[a] for a in angles]).astype(np.float32))
        return cls(fig.model, fig.reference_poses, shapes)

    def save(self, path) -> None:
        data = {"reference_poses": self.reference_poses}
        for k, (a, d) in self.shapes.items():
            data[f"{k}|winkel"], data[f"{k}|form"] = a, d
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, model, path) -> Correctives:
        d = np.load(path)
        keys = {k.split("|")[0] for k in d.files if "|" in k}
        return cls(model, d["reference_poses"], {k: (d[f"{k}|winkel"], d[f"{k}|form"]) for k in keys})

    def weights(self, poses: np.ndarray) -> dict[str, np.ndarray]:
        """Je Form und Bild das Gewicht jeder Stützstelle (F, K): stückweise linear im Winkel, unter
        der ersten Stützstelle linear bis 0 bei gestrecktem Gelenk."""
        out = {}
        for key, (angles, _) in self.shapes.items():
            a = self.joints[key].angle(self.model, poses, self.reference_poses)
            knots = np.concatenate([[0.0], angles])
            w = np.zeros((len(a), len(angles)))
            for k in range(len(angles)):
                basis = np.zeros(len(knots))
                basis[k + 1] = 1
                w[:, k] = np.interp(a, knots, basis)  # über dem letzten Winkel: volle letzte Form
            out[key] = w
        return out

    def offsets(self, poses: np.ndarray) -> np.ndarray:
        """Ruhe-Verschiebung je Bild (F, V, 3) aus den Knochenlagen (F, J, 4, 4)."""
        n = next(iter(self.shapes.values()))[1].shape[1]
        out = np.zeros((len(poses), n, 3), np.float32)
        for key, w in self.weights(poses).items():
            out += np.einsum("fk,kvc->fvc", w.astype(np.float32), self.shapes[key][1])
        return out
