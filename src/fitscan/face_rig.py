"""Gesichtsregler (Mimik) als Formregler: Mund, Kiefer, Lider, Brauen, Wangen, Nase, Blick.

Annys Skelett hat im Gesicht nur die beiden Augenknochen. Die Mimik entsteht deshalb als
Formregler (glTF Morph Targets, in Blender Shape Keys, in FBX Blendshapes). Jeder Regler wirkt
allein, lässt sich mit anderen mischen und ist unabhängig von der Körperbewegung.

Namen wie bei Apple ARKit (jawOpen, mouthSmileLeft, eyeBlinkRight, ...): Die Gesichtserkennung
auf dem iPhone liefert genau diese Werte, damit kann die App das Gesicht später live steuern.
„Left“/„Right“ meint wie bei ARKit die Seite der Person.

Berechnung für jede Person aus ihrer eigenen Kopfform:
- Die 478 Gesichtspunkte von MediaPipe sitzen fest auf Annys Netz (data/face_landmarks_anny.json).
- Jeder Regler verschiebt einige dieser Punkte (in Millimetern, im Gesichtskoordinatensystem:
  x zur linken Seite der Person, y nach oben, z aus dem Gesicht heraus) und hält andere fest.
- Die übrige Haut folgt glatt (biharmonische Interpolation über das Netz, wie gespannte Haut).
- Kiefer und Lider drehen sich statt sich zu verschieben: Kiefer um die Kiefergelenke, Lider
  um die Augapfelmitte (sie gleiten so über das Auge), die Augäpfel drehen sich für den Blick.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# ------------------------------------------------------------------------- Gesichtspunkte
# MediaPipe-Nummern. Paare (rechte Seite der Person, linke Seite der Person).
LIP_UPPER_OUT = [185, 40, 39, 37, 0, 267, 269, 270, 409]
LIP_LOWER_OUT = [146, 91, 181, 84, 17, 314, 405, 321, 375]
LIP_UPPER_IN = [191, 80, 81, 82, 13, 312, 311, 310, 415]
LIP_LOWER_IN = [95, 88, 178, 87, 14, 317, 402, 318, 324]
CORNER = ([61, 78], [291, 308])
LIP_MID = [0, 13, 14, 17]
UPPER_NEAR_CORNER = ([185, 40, 191, 80], [409, 270, 415, 310])
LOWER_NEAR_CORNER = ([146, 91, 95, 88], [375, 321, 324, 318])
CHIN = [152, 148, 377, 176, 400, 199, 175, 200, 18, 83, 313, 201, 421, 32, 262]
JAW_LOW = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]
NOSE = [1, 2, 4, 5, 6, 19, 94, 195, 197, 168, 98, 327, 129, 358, 64, 294, 102, 331, 48, 278, 115, 344,
        97, 326, 141, 370, 125, 354]
NOSE_TIP = [1, 2, 4, 5, 19, 94]
ALAR = ([129, 64, 102, 48, 98], [358, 294, 331, 278, 327])
CHEEK_BONE = [116, 345, 117, 346, 118, 347, 123, 352, 50, 280, 101, 330, 111, 340]
CHEEK = ([205, 50, 187, 207, 36, 147, 123, 101, 118], [425, 280, 411, 427, 266, 376, 352, 330, 347])
NASOLABIAL = ([206, 203, 216, 92], [426, 423, 436, 322])
BROW_UPPER = ([70, 63, 105, 66, 107], [300, 293, 334, 296, 336])
BROW_LOWER = ([46, 53, 52, 65, 55], [276, 283, 282, 295, 285])
BROW_INNER = ([107, 66, 55, 65], [336, 296, 285, 295])
BROW_OUTER = ([70, 63, 46, 53], [300, 293, 276, 283])
BROW_MID = ([105, 52], [334, 282])
FOREHEAD = [10, 151, 9, 108, 337, 69, 299, 104, 333, 67, 297, 109, 338, 8]
GLABELLA = [9, 8, 168]
LID_UPPER = ([246, 161, 160, 159, 158, 157, 173], [466, 388, 387, 386, 385, 384, 398])
LID_LOWER = ([7, 163, 144, 145, 153, 154, 155], [249, 390, 373, 374, 380, 381, 382])
LID_UPPER_MID = (159, 386)
LID_LOWER_MID = (145, 374)
EYE_CORNERS = ([33, 133], [263, 362])
ORBIT_LOW = ([111, 117, 118, 119, 120, 121, 128, 230, 229, 228, 31, 226], [340, 346, 347, 348, 349, 350, 357,
             450, 449, 448, 261, 446])
EAR = [234, 454, 127, 356, 162, 389, 93, 323, 21, 251, 132, 361, 58, 288]
JAW_HINGE = (234, 454)

MOUTH = sorted(set(LIP_UPPER_OUT + LIP_LOWER_OUT + LIP_UPPER_IN + LIP_LOWER_IN + CORNER[0] + CORNER[1]))
REFERENCE_WIDTH = 0.092  # Meter: Abstand der äußeren Augenwinkel einer mittleren Person

JAW_ANGLE = 20.0  # Grad bei jawOpen = 1
LOOK_UP, LOOK_DOWN, LOOK_SIDE = 22.0, 28.0, 30.0  # Grad
BLINK_MEET = 0.8  # die Lider treffen sich bei 80 % des Wegs vom Oberlid zum Unterlid
REGION_REACH = 0.035  # Meter: weiter von jedem Gesichtspunkt entfernt bewegt sich nichts


@dataclasses.dataclass
class Face:
    """Gesichtskoordinaten einer Person (alles in den Koordinaten des Netzes)."""

    points: np.ndarray  # (478, 3) Gesichtspunkte auf dem Netz
    origin: np.ndarray
    axes: np.ndarray  # (3, 3) Zeilen: x (linke Seite der Person), y (oben), z (aus dem Gesicht)
    scale: float  # Gesichtsbreite / Referenz

    def vec(self, mm) -> np.ndarray:
        """Millimeter im Gesichtssystem -> Meter im Netzsystem, mit der Kopfgröße skaliert."""
        return np.asarray(mm, np.float64) @ self.axes * 0.001 * self.scale

    def local(self, p: np.ndarray) -> np.ndarray:
        return (p - self.origin) @ self.axes.T


def face_frame(points: np.ndarray) -> Face:
    right, left = points[33], points[263]
    x = left - right
    width = float(np.linalg.norm(x))
    x /= width
    up = points[9] - points[152]
    y = up - x * (up @ x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    nose_out = points[1] - (right + left) / 2
    if nose_out @ z < 0:  # MediaPipe-Seiten vertauscht: Achsen so drehen, dass z aus dem Gesicht zeigt
        x, z = -x, -z
    return Face(points=points, origin=(right + left) / 2, axes=np.stack([x, y, z]),
                scale=width / REFERENCE_WIDTH)


def sides(face: Face, pair: tuple[list[int] | int, list[int] | int]) -> dict[str, list[int]]:
    """{"Right": ..., "Left": ...} nach der wirklichen Lage auf dem Netz (x > 0 = links)."""
    a, b = ([p] if isinstance(p, int) else list(p) for p in pair)
    xa = face.local(face.points[a])[:, 0].mean()
    xb = face.local(face.points[b])[:, 0].mean()
    return {"Right": a, "Left": b} if xa < xb else {"Right": b, "Left": a}


# ------------------------------------------------------------------------- glatte Verformung


class Solver:
    """Biharmonische Interpolation auf dem Gesichtsbereich des Netzes."""

    def __init__(self, verts: np.ndarray, faces: np.ndarray, region: np.ndarray):
        import scipy.sparse as sp

        self.verts = verts
        self.faces = faces
        n = len(verts)
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        e = e[region[e].all(1)]
        e = np.unique(np.sort(e, 1), axis=0)
        adj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
        adj = adj + adj.T
        deg = np.asarray(adj.sum(1)).ravel()
        self.index = np.flatnonzero(region)
        lap = (sp.diags(deg) - adj).tocsr()[self.index][:, self.index]
        self.bilap = (lap.T @ lap).tocsr()
        # Rand des Bereichs (Nachbarn außerhalb) und unverbundene Teile bleiben fest
        all_e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        border = np.zeros(n, bool)
        cross = region[all_e[:, 0]] != region[all_e[:, 1]]
        border[all_e[cross].ravel()] = True
        self.border = border & region
        self.isolated = region & (deg == 0)
        self.pos = {v: i for i, v in enumerate(self.index)}

    def solve(self, known: dict[int, np.ndarray], dims: int = 3) -> np.ndarray:
        """known: Netzpunkt -> Wert. Gibt Werte für alle Netzpunkte zurück (außerhalb 0)."""
        import scipy.sparse.linalg as spla

        n_r = len(self.index)
        fixed = np.zeros(n_r, bool)
        value = np.zeros((n_r, dims))
        for v in np.flatnonzero(self.border | self.isolated):
            fixed[self.pos[v]] = True
        for v, d in known.items():
            if v in self.pos:
                fixed[self.pos[v]] = True
                value[self.pos[v]] = d
        free = ~fixed
        a_ff = self.bilap[free][:, free]
        a_fk = self.bilap[free][:, fixed]
        rhs = -(a_fk @ value[fixed])
        solved = spla.splu(a_ff.tocsc()).solve(rhs)
        value[free] = solved
        out = np.zeros((len(self.verts), dims))
        out[self.index] = value
        return out


def _vertices_of(face_tri: np.ndarray, faces: np.ndarray, landmarks) -> list[int]:
    return sorted({int(v) for lm in landmarks for v in faces[face_tri[lm]]})


class FaceRig:
    def __init__(self, verts: np.ndarray, faces: np.ndarray, tri: np.ndarray, bary: np.ndarray,
                 eyes: dict[str, np.ndarray], skin_region: np.ndarray):
        """verts (V, 3), faces (F, 3), Gesichtspunkte als Dreieck + baryzentrische Koordinaten,
        eyes: Augapfel-Masken {"Left", "Right"}, skin_region: Netzpunkte, die sich überhaupt
        bewegen dürfen (Kopf, ohne Augäpfel)."""
        self.verts = verts.astype(np.float64)
        self.faces = faces
        self.tri = tri
        points = (self.verts[faces[tri]] * bary[..., None]).sum(1)
        self.face = face_frame(points)
        near = np.linalg.norm(self.verts[:, None, :] - points[None, ::4, :], axis=2).min(1)
        self.region = skin_region & (near < REGION_REACH * self.face.scale)
        self.solver = Solver(self.verts, faces, self.region)
        self.eyes = eyes
        self.eye_center = {s: self.verts[m].mean(0) for s, m in eyes.items()}
        mouth = self.face.local(points[[61, 291, 13, 14]]) * 1000 / self.face.scale
        self.lips_y = float(mouth[:, 1].mean())  # Höhe des Mundspalts in mm (Gesichtssystem)

    def verts_of(self, landmarks) -> list[int]:
        return _vertices_of(self.tri, self.faces, landmarks)

    # ----------------------------------------------------------------- Bausteine
    def field(self, moves: dict[int, np.ndarray], fixed=()) -> np.ndarray:
        """Verschiebung: moves = Gesichtspunkt -> Millimeter (Gesichtssystem), fixed bleibt."""
        known: dict[int, np.ndarray] = {}
        for lm in fixed:
            for v in self.verts_of([lm]):
                known[v] = np.zeros(3)
        acc: dict[int, list[np.ndarray]] = {}
        for lm, mm in moves.items():
            for v in self.verts_of([lm]):
                acc.setdefault(v, []).append(self.face.vec(mm))
        for v, ds in acc.items():
            known[v] = np.mean(ds, 0)
        return self.solver.solve(known)

    def weights(self, one, zero, half=()) -> np.ndarray:
        known = {}
        for lms, val in ((zero, 0.0), (half, 0.5), (one, 1.0)):
            for v in self.verts_of(lms):
                known[v] = np.array([val])
        return np.clip(self.solver.solve(known, dims=1)[:, 0], 0.0, 1.0)

    def rotate(self, weights: np.ndarray, pivot: np.ndarray, axis: np.ndarray, degrees: float,
               mask: np.ndarray | None = None) -> np.ndarray:
        sel = np.flatnonzero(weights > 1e-4 if mask is None else mask)
        rel = self.verts[sel] - pivot
        angle = np.radians(degrees) * (weights[sel] if mask is None else 1.0)
        k = axis / np.linalg.norm(axis)
        cos, sin = np.cos(angle)[..., None], np.sin(angle)[..., None]
        rot = rel * cos + np.cross(k, rel) * sin + k * (rel @ k)[..., None] * (1 - cos)
        out = np.zeros_like(self.verts)
        out[sel] = rot - rel
        return out

    def lm(self, idx) -> np.ndarray:
        return self.face.points[idx]

    # ----------------------------------------------------------------- Regler
    def jaw_open(self) -> np.ndarray:
        f = self.face
        lower = LIP_LOWER_OUT + LIP_LOWER_IN + CHIN + JAW_LOW
        upper = LIP_UPPER_OUT + LIP_UPPER_IN + NOSE + CHEEK_BONE + EAR[:10] + FOREHEAD
        w = self.weights(lower, upper, half=CORNER[0] + CORNER[1])
        hinge = (self.lm(JAW_HINGE[0]) + self.lm(JAW_HINGE[1])) / 2 + f.vec([0, -8, -6])
        return self.rotate(w, hinge, f.axes[0], JAW_ANGLE)

    def _mouth_fixed(self) -> list[int]:
        return NOSE + FOREHEAD + EAR + sum(LID_UPPER, []) + sum(LID_LOWER, []) + sum(BROW_UPPER, []) \
            + sum(BROW_LOWER, []) + [152]

    def mouth_corner(self, side: str, corner, upper, lower, cheek=(0, 0, 0), fix_mid: bool = True) -> np.ndarray:
        s = 1.0 if side == "Left" else -1.0
        moves = {}
        for lm in sides(self.face, CORNER)[side]:
            moves[lm] = (s * corner[0], corner[1], corner[2])
        for lm in sides(self.face, UPPER_NEAR_CORNER)[side]:
            moves[lm] = (s * upper[0], upper[1], upper[2])
        for lm in sides(self.face, LOWER_NEAR_CORNER)[side]:
            moves[lm] = (s * lower[0], lower[1], lower[2])
        for lm in sides(self.face, CHEEK)[side][:4]:
            moves[lm] = (s * cheek[0], cheek[1], cheek[2])
        other = "Right" if side == "Left" else "Left"
        fixed = self._mouth_fixed() + sides(self.face, CORNER)[other] + (LIP_MID if fix_mid else [])
        return self.field(moves, fixed)

    def lips_all(self, fn) -> np.ndarray:
        """Alle Lippenpunkte nach fn(lokale Lage in mm) -> Verschiebung in mm."""
        moves = {lm: fn(self.face.local(self.lm(lm)) * 1000 / self.face.scale) for lm in MOUTH}
        return self.field(moves, self._mouth_fixed())

    def lid(self, side: str, which: str, fraction: float) -> np.ndarray:
        """Lid um die Augapfelmitte drehen; fraction = Anteil der Lidspalte (positiv = schließen)."""
        f = self.face
        c = self.eye_center[side]
        up_mid = self.lm(sides(f, LID_UPPER_MID)[side][0]) - c
        lo_mid = self.lm(sides(f, LID_LOWER_MID)[side][0]) - c
        a_up = np.arctan2(up_mid @ f.axes[1], up_mid @ f.axes[2])
        a_lo = np.arctan2(lo_mid @ f.axes[1], lo_mid @ f.axes[2])
        gap = np.degrees(a_up - a_lo)
        lids_up, lids_lo = sides(f, LID_UPPER)[side], sides(f, LID_LOWER)[side]
        brow = sides(f, BROW_LOWER)[side] + sides(f, BROW_UPPER)[side]
        corners = sides(f, EYE_CORNERS)[side]
        orbit = sides(f, ORBIT_LOW)[side]
        if which == "upper":
            w = self.weights(lids_up, lids_lo + brow + corners + GLABELLA)
            return self.rotate(w, c, f.axes[0], fraction * gap)
        w = self.weights(lids_lo, lids_up + corners + orbit + NOSE)
        return self.rotate(w, c, f.axes[0], -fraction * gap)

    def look(self, side: str, axis: str, degrees: float) -> np.ndarray:
        f = self.face
        ax = f.axes[0] if axis == "x" else f.axes[1]
        return self.rotate(np.ones(len(self.verts)), self.eye_center[side], ax, degrees, mask=self.eyes[side])

    def brows(self, moves_by_group: list[tuple[tuple, tuple]], side: str | None) -> np.ndarray:
        moves = {}
        for group, mm in moves_by_group:
            for sd in ("Left", "Right") if side is None else (side,):
                s = 1.0 if sd == "Left" else -1.0
                for lm in sides(self.face, group)[sd]:
                    moves[lm] = (s * mm[0], mm[1], mm[2])
        fixed = sum(LID_UPPER, []) + sum(LID_LOWER, []) + NOSE_TIP + MOUTH + EAR[:10] + [10]
        if side is not None:
            other = "Right" if side == "Left" else "Left"
            fixed += sides(self.face, BROW_UPPER)[other] + sides(self.face, BROW_LOWER)[other]
        return self.field(moves, fixed)

    # ----------------------------------------------------------------- alle Regler
    def morphs(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        out["jawOpen"] = self.jaw_open()
        for side in ("Left", "Right"):
            out[f"mouthSmile{side}"] = self.mouth_corner(side, (4, 6, -3), (2, 3, -1), (2, 3.5, -1), (1, 3, 1.5))
            out[f"mouthFrown{side}"] = self.mouth_corner(side, (1, -5, -0.5), (0.5, -2.5, 0), (0.5, -2.5, 0))
            out[f"mouthStretch{side}"] = self.mouth_corner(side, (5, -2, -2), (2.5, -0.5, -1), (2.5, -1.5, -1))
            out[f"mouthPress{side}"] = self.mouth_corner(side, (0, 0, -1), (0, -0.8, -1.5), (0, 0.8, -1.5))
            out[f"cheekSquint{side}"] = self.field(
                {lm: (0, 3, 0.5) for lm in sides(self.face, CHEEK)[side] + sides(self.face, ORBIT_LOW)[side]}
                | {lm: (0, 1.2, 0) for lm in sides(self.face, LID_LOWER)[side]},
                self._mouth_fixed()[:len(NOSE + FOREHEAD + EAR)] + MOUTH + sum(LID_UPPER, []))
            sign = 1.0 if side == "Left" else -1.0
            out[f"noseSneer{side}"] = self.field(
                {lm: (0, 3, 0.5) for lm in sides(self.face, ALAR)[side]}
                | {lm: (0, 2, 0.5) for lm in sides(self.face, NASOLABIAL)[side]}
                | {lm: (-sign * 1.0, -1.5, 0) for lm in sides(self.face, BROW_INNER)[side]},
                NOSE_TIP + MOUTH + EAR + FOREHEAD + sum(LID_UPPER, []))
        out["mouthPucker"] = self.lips_all(lambda p: (-0.3 * p[0], -0.05 * (p[1] - self.lips_y), 5.5))
        out["mouthFunnel"] = self.lips_all(
            lambda p: (-0.2 * p[0], 2.5 if p[1] > self.lips_y else -3.0, 4.0))
        for side, s in (("Left", 1.0), ("Right", -1.0)):
            out[f"mouth{side}"] = self.lips_all(
                lambda p, s=s: (s * 5.0, 0.0, -2.0 if p[0] * s > 15 else (1.0 if p[0] * s < -15 else 0.0)))
        out["browInnerUp"] = self.brows([(BROW_INNER, (0, 7, 0.5)), (BROW_MID, (0, 3, 0))], None)
        for side in ("Left", "Right"):
            out[f"browDown{side}"] = self.brows([(BROW_INNER, (-3, -5, 0.5)), (BROW_MID, (-1, -4.5, 0)),
                                                  (BROW_OUTER, (0, -3, 0))], side)
            out[f"browOuterUp{side}"] = self.brows([(BROW_OUTER, (0, 7, 0)), (BROW_MID, (0, 3.5, 0))], side)
        for side in ("Left", "Right"):
            out[f"eyeBlink{side}"] = self.lid(side, "upper", BLINK_MEET) + self.lid(side, "lower", 1 - BLINK_MEET)
            out[f"eyeWide{side}"] = self.lid(side, "upper", -0.3)
            out[f"eyeSquint{side}"] = self.lid(side, "lower", 0.3) + self.lid(side, "upper", 0.1)
        out["cheekPuff"] = self._cheek_puff()
        for side in ("Left", "Right"):
            inward = LOOK_SIDE if side == "Left" else -LOOK_SIDE  # Drehung um y: positiv = nach links
            out[f"eyeLookUp{side}"] = self.look(side, "x", -LOOK_UP)
            out[f"eyeLookDown{side}"] = self.look(side, "x", LOOK_DOWN)
            out[f"eyeLookIn{side}"] = self.look(side, "y", -inward)
            out[f"eyeLookOut{side}"] = self.look(side, "y", inward)
        return {k: v.astype(np.float32) for k, v in out.items()}

    def _cheek_puff(self) -> np.ndarray:
        moves = {}
        for side, s in (("Left", 1.0), ("Right", -1.0)):
            for lm in sides(self.face, CHEEK)[side]:
                moves[lm] = (s * 5.0, -0.5, 3.0)
            for lm in sides(self.face, NASOLABIAL)[side]:
                moves[lm] = (s * 2.5, 0, 3.0)
        moves.update({lm: (0, 0, 1.5) for lm in LIP_UPPER_OUT + LIP_LOWER_OUT})
        return self.field(moves, NOSE + EAR + sum(LID_UPPER, []) + sum(LID_LOWER, []) + FOREHEAD + [152])


def face_morphs(verts: np.ndarray, faces: np.ndarray, tri: np.ndarray, bary: np.ndarray,
                eyes: dict[str, np.ndarray], skin_region: np.ndarray) -> dict[str, np.ndarray]:
    """Alle Gesichtsregler als Verschiebung je Netzpunkt (gleiches Koordinatensystem wie verts)."""
    return FaceRig(verts, faces, tri, bary, eyes, skin_region).morphs()


def morphs_for_model(model, verts: np.ndarray) -> dict[str, np.ndarray]:
    """Gesichtsregler für Annys Netz (verts: Ruhepose der Person, Anny-Koordinaten)."""
    from .head import eye_vertex_masks, load_correspondence

    tri, bary = load_correspondence()
    masks = eye_vertex_masks(model)
    eyes = {"Left": masks["L"], "Right": masks["R"]}
    head = model.dominant_bone_mask(lambda n: n.startswith(("head", "neck03")))
    region = head & ~eyes["Left"] & ~eyes["Right"]
    return face_morphs(verts, model.faces, tri, bary, eyes, region)
