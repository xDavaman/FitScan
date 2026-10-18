"""Haltungen mit Auflage: Sitzen, Liegen, Yoga und Mobility.

Eine Haltung ist hier mehr als Gelenkwinkel: Hände, Füße, Rücken oder Gesäß müssen dort aufliegen,
wo sie beim Menschen aufliegen, und wer steht, steht im Gleichgewicht. Jede Haltung nennt deshalb
ein paar freie Winkel und ihre Auflagestellen; `solve` stellt die freien Winkel so ein, dass alle
Auflagestellen gleichzeitig den Boden (oder die Bank) berühren, die Sohlen und Handflächen flach
liegen und der Körperschwerpunkt über der Standfläche liegt.

Danach gibt die Haut nach (`soften`): Was in Boden oder Bank sinken würde, wird auf die Fläche
gelegt und läuft weich in die Umgebung aus (Gesäß auf der Bank, Rücken auf der Matte). Wo sich
Körperteile berühren (Gesäß auf den Fersen in der Kindhaltung, Wade an Oberschenkel in der Hocke),
legt dieselbe Rechnung wie bei den Korrekturformen die Häute aufeinander.

Ergebnis: Drehungen (local-ref, Anny-Namen), Wurzelverschiebung, Ruhe-Verschiebung der Haut und
die Bank als `contact.Box`, also alles, was die Bewegungsbibliothek speichert.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

from . import contact
from . import corrective as K
from . import posing as P

BENCH_HEIGHT = 0.45  # Hantelbank
BENCH_HALF = np.array([0.14, 0.6, 0.0])  # halbe Breite und Länge, Höhe kommt aus BENCH_HEIGHT
SOFT_REACH = 0.08  # so weit läuft das Nachgeben der Haut um die Auflage aus (m)
SOFT_DEPTH = 0.03  # so tief darf ein weicher Bauch in der Bauchlage eingedrückt werden (m)

# Auflagestellen: Knochen, deren Hautpunkte dazugehören (je Seite)
_GROUPS = {
    "ferse": ("foot",),
    "zehen": ("toe",),
    "handballen": ("wrist", "metacarpal"),
    "finger": ("finger2-3", "finger3-3", "finger4-3", "finger5-3"),
    "schienbein": ("lowerleg",),
    "oberschenkel": ("upperleg",),
    "gesaess": ("pelvis",),
    "becken": ("pelvis", "root"),
    "ruecken": ("spine02", "spine03", "spine04"),
    "unterbauch": ("spine05", "spine04"),
    "brust": ("spine01", "spine02", "breast"),
    "kopf": ("head", "neck"),
    "unterarm": ("lowerarm",),
}


@dataclasses.dataclass
class Spec:
    """Eine Haltung: Bauplan aus Parametern, welche davon frei sind und was aufliegen muss."""

    name: str
    title: str  # Beschriftung (deutsch)
    build: Callable[..., dict]  # Parameter -> Haltung (posing-dict)
    params: dict[str, float]  # Ausgangswerte
    free: dict[str, tuple[float, float]]  # frei einstellbare Parameter mit Grenzen
    floor: tuple[str, ...]  # Auflagestellen am Boden, z. B. "ferse.L", "handballen"
    bench: tuple[str, ...] = ()  # Auflagestellen auf der Bank
    balance: bool = False  # Schwerpunkt über der Standfläche
    sink: float = 0.004  # so weit gibt die Haut an der Auflage nach (m)
    touching: tuple[tuple[str, str], ...] = ()  # Körperteile, die sich berühren dürfen
    holds: dict[str, float] = dataclasses.field(default_factory=dict)  # "a|b": kleinster Abstand, Ziel (m)
    soft: tuple[str, ...] = ()  # Stellen, die bis SOFT_DEPTH eingedrückt werden dürfen (Bauch)


@dataclasses.dataclass
class Still:
    """Eine fertige Haltung."""

    spec: Spec
    params: dict[str, float]
    pose: dict  # Haltung (posing-dict), zum Überblenden in Bewegungen
    rotations: dict  # Knochen -> 3x3 (local-ref)
    root: np.ndarray  # (3,) Wurzelverschiebung, Boden z = 0
    offsets: np.ndarray  # (V, 3) Ruhe-Verschiebung (Korrekturformen + Nachgeben)
    vertices: np.ndarray  # (V, 3) Weltpunkte
    poses: np.ndarray  # (J, 4, 4) Weltlage der Knochen
    props: list  # contact.Box (Bank)
    residual: dict[str, float]  # Abstand jeder Auflagestelle zur Fläche vor dem Nachgeben (m)
    path: tuple[np.ndarray, np.ndarray] | None = None  # Berührungsformen auf dem Weg vom Stehen her

    def along(self, progress: np.ndarray) -> np.ndarray:
        """Berührungsformen (F, V, 3) für einen Fortschritt 0 (Stehen) .. 1 (diese Haltung)."""
        n = len(self.offsets)
        if self.path is None:
            return np.zeros((len(progress), n, 3), np.float32)
        ts, shapes = self.path
        knots = np.concatenate([[0.0], ts])
        out = np.zeros((len(progress), n, 3), np.float32)
        for k in range(len(ts)):
            basis = np.zeros(len(knots))
            basis[k + 1] = 1
            w = np.interp(np.clip(progress, 0, 1), knots, basis)
            out += w[:, None, None].astype(np.float32) * shapes[k].astype(np.float32)
        return out


def groups(model) -> dict[str, np.ndarray]:
    """Punktmasken der Auflagestellen, je Seite ("ferse.L") und beidseitig ("ferse")."""
    out = {}
    for name, prefixes in _GROUPS.items():
        for s in ("L", "R", ""):
            if s:
                mask = model.dominant_bone_mask(lambda n, s=s, p=prefixes: n.startswith(p) and n.endswith("." + s))
                out[f"{name}.{s}"] = mask
            else:
                mask = model.dominant_bone_mask(lambda n, p=prefixes: n.startswith(p))
                out[name] = mask
    return out


def centre_of_mass(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Schwerpunkt des geschlossenen Netzes bei gleicher Dichte (Tetraeder zum Ursprung)."""
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c))
    return (vol[:, None] * (a + b + c)).sum(0) / (4 * vol.sum())


def balance(vertices: np.ndarray, faces: np.ndarray, ground: float, margin: float = 0.02) -> float:
    """Wie weit der Schwerpunkt (von oben gesehen) außerhalb der Standfläche liegt, mit etwas Rand
    nach innen (m, 0 = steht sicher). Standfläche = konvexe Hülle der Punkte am Boden."""
    from scipy.spatial import ConvexHull, QhullError

    com = centre_of_mass(vertices, faces)[:2]
    support = vertices[vertices[:, 2] < ground + 0.01, :2]
    try:
        hull = ConvexHull(support)
        outside = (hull.equations[:, :2] @ com + hull.equations[:, 2]).max()  # > 0: außerhalb
    except QhullError:  # steht nur auf einer Kante oder einem Punkt
        a, b = support[np.argmin(support @ [1, 1])], support[np.argmax(support @ [1, 1])]
        ab = b - a
        t = np.clip((com - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
        outside = np.linalg.norm(com - (a + t * ab))
    return float(max(0.0, outside + margin))


def _expand(names: tuple[str, ...], masks: dict[str, np.ndarray]) -> list[str]:
    """"ferse" -> ["ferse.L", "ferse.R"]; Stellen in der Körpermitte ("kopf") bleiben eins."""
    out = []
    for n in names:
        out += [n] if "." in n or not masks[f"{n}.L"].any() else [f"{n}.L", f"{n}.R"]
    return out


class Solver:
    """Stellt die freien Parameter einer Haltung ein (Nelder-Mead, wenige Parameter)."""

    def __init__(self, fig: P.Figure):
        self.fig = fig
        self.groups = groups(fig.model)

    def evaluate(self, spec: Spec, params: dict[str, float]) -> tuple[np.ndarray, dict]:
        out = self.fig.pose([self.fig.poser.rotations(spec.build(**params))])
        return out["vertices"][0], out

    def cost(self, spec: Spec, v: np.ndarray) -> tuple[float, dict[str, float]]:
        g = self.groups
        floor = _expand(spec.floor, g)
        bench = _expand(spec.bench, g)
        ground = min(v[g[n], 2].min() for n in floor)
        res = {n: float(v[g[n], 2].min() - ground) for n in floor}
        for n in bench:
            res[n] = float(v[g[n], 2].min() - ground - BENCH_HEIGHT)
        # nichts außer den Auflagestellen darf tiefer liegen als der Boden (bzw. in der Bank)
        allowed = np.zeros(len(v), bool)
        for n in floor:
            allowed |= g[n]
        soft = np.zeros(len(v), bool)
        for n in _expand(spec.soft, g):
            soft |= g[n]
        rest = ~allowed & ~soft
        res["sonst"] = float(max(0.0, ground - v[rest, 2].min()))
        if soft.any():
            res["weich"] = float(max(0.0, ground - SOFT_DEPTH - v[soft & ~allowed, 2].min()))
        if spec.balance:
            res["gleichgewicht"] = balance(v, self.fig.model.faces, ground)
        for key, target in spec.holds.items():
            from scipy.spatial import cKDTree

            a, b = key.split("|")
            d = cKDTree(v[g[b]]).query(v[g[a]])[0].min()
            res[key] = float(d - target)
        return float(sum(x * x for x in res.values())), res

    def solve(self, spec: Spec, iterations: int = 300) -> dict[str, float]:
        from scipy.optimize import minimize

        names = list(spec.free)
        lo = np.array([spec.free[n][0] for n in names])
        hi = np.array([spec.free[n][1] for n in names])
        x0 = np.array([spec.params[n] for n in names], float)

        def params(x):
            p = dict(spec.params)
            p.update(zip(names, np.clip(x, lo, hi), strict=True))
            return p

        def f(x):
            v, _ = self.evaluate(spec, params(x))
            c, _ = self.cost(spec, v)
            outside = np.maximum(lo - x, 0) + np.maximum(x - hi, 0)
            # kleine Abweichung von der Vorlage, Grenzen hart
            return 1e4 * c + 1e-5 * ((x - x0) ** 2).sum() + 1e-2 * (outside**2).sum()

        if not names:
            return dict(spec.params)
        simplex = np.vstack([x0, x0 + np.diag(0.2 * (hi - lo))])  # Schritte in Grad, nicht in Prozent
        best = minimize(f, x0, method="Nelder-Mead",
                        options=dict(maxiter=iterations, xatol=0.2, fatol=1e-6, initial_simplex=simplex))
        return params(best.x)


# --------------------------------------------------------------------------- Nachgeben der Haut


def _spread(model, region: np.ndarray, held: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Harmonische Verschiebung (V, 3): gehaltene Punkte erreichen ihr Ziel, der Rand von region bleibt."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    lap = K._laplacian(model.faces, len(goal))
    pos = np.full(len(goal), -1)
    pos[region] = np.arange(len(region))
    fixed = pos[held]
    keep = fixed >= 0
    fixed, target = fixed[keep], goal[held[keep]]
    c = sparse.coo_matrix((np.ones(len(fixed)), (np.arange(len(fixed)), fixed)), shape=(len(fixed), len(region)))
    # kleines Gewicht auf 0: lose Teile ohne gehaltene Punkte (Augäpfel) bleiben, wo sie sind
    lhs = (lap[region][:, region] + 100.0 * c.T @ c + 1e-4 * sparse.identity(len(region))).tocsc()
    out = np.zeros_like(goal)
    for d in range(3):
        out[region, d] = spsolve(lhs, 100.0 * c.T @ target[:, d])
    return out


def flatten(model, vertices: np.ndarray, surfaces: list[tuple[float, np.ndarray | None]],
            steps: int = 5) -> np.ndarray:
    """Weltverschiebung (V, 3), die Haut unter einer waagerechten Fläche auf die Fläche legt.
    surfaces: (Höhe, Bereich (lo, hi) in x/y oder None für den ganzen Boden)."""
    from scipy.spatial import cKDTree

    delta = np.zeros_like(vertices)
    held = np.zeros(len(vertices), bool)
    goal = np.zeros_like(vertices)
    region = None
    for _ in range(steps):
        v = vertices + delta
        hit = np.zeros(len(v), bool)
        for height, area in surfaces:
            inside = v[:, 2] < height - 1e-4
            if area is not None:
                inside &= np.all((v[:, :2] >= area[0]) & (v[:, :2] <= area[1]), axis=1)
            goal[inside, 2] = delta[inside, 2] + height - v[inside, 2]
            hit |= inside
        if not hit.any():
            break
        held |= hit
        if region is None:
            region = np.flatnonzero(cKDTree(vertices[hit]).query(vertices, distance_upper_bound=SOFT_REACH)[0] < np.inf)
        delta = _spread(model, region, np.flatnonzero(held), goal)
    return _follow(model, vertices, delta)


def _follow(model, vertices: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Lose Teile (Augäpfel, Zähne) bewegen sich wie die nächste Haut des Körpers."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n, labels = connected_components(K._laplacian(model.faces, len(vertices)) != 0, directed=False)
    if n == 1:
        return delta
    main = labels == np.bincount(labels).argmax()
    k = cKDTree(vertices[main]).query(vertices[~main])[1]
    delta = delta.copy()
    delta[~main] = delta[main][k]
    return delta


def to_rest(model, transforms: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Weltverschiebung -> Ruhe-Verschiebung (vor dem Skinning)."""
    return np.linalg.solve(K.skin_linear(model, transforms), world[..., None])[..., 0]


def blend(a: dict, b: dict, t: float) -> dict:
    """Haltung zwischen a (t = 0) und b (t = 1), Winkel für Winkel."""
    return {k: (1 - t) * a.get(k, 0.0) + t * b.get(k, 0.0) for k in set(a) | set(b)}


def _base(fig: P.Figure, rotations: dict) -> np.ndarray:
    if fig.correctives is None:
        return np.zeros_like(fig.reference_vertices)
    return fig.correctives.offsets(fig.pose([rotations])["poses"])[0]


def touch(fig: P.Figure, pose: dict, pairs, steps: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Ruhe-Verschiebungen, die Körperteile in pairs aufeinanderlegen. Die Haltung wird vom
    Stehen aus in Schritten angefahren und die Verschiebung mitgeführt, damit jede Berührung flach
    bleibt und innen/außen sicher erkannt wird (wie corrective.build). Rückgabe: Fortschritt je
    Schritt (S,) und die Form bis dahin (S, V, 3); die letzte gilt für die Haltung selbst."""
    acc = np.zeros_like(fig.reference_vertices)
    start = P.stand()
    ts = np.linspace(0, 1, steps + 1)[1:]
    shapes = []
    for t in ts:
        rotations = fig.poser.rotations(blend(start, pose, float(t)))
        base = _base(fig, rotations)
        out = fig.pose([rotations], offsets=(base + acc)[None], transforms=True)
        d = K.solve_shape(fig.model, out["vertices"][0], list(pairs), fig.reference_vertices)
        if np.abs(d).max() > 0:
            acc = acc + to_rest(fig.model, out["transforms"][0], d)
        shapes.append(acc.astype(np.float32))
    return ts, np.stack(shapes)


def soften(fig: P.Figure, spec: Spec, pose: dict, root: np.ndarray, bench: contact.Box | None):
    """Ruhe-Verschiebung (V, 3) samt Korrekturformen: Berührungen zwischen Körperteilen
    aufeinanderlegen, dann die Haut an Boden und Bank nachgeben lassen. Gibt dazu den Weg der
    Berührungsformen zurück (siehe touch) oder None."""
    model = fig.model
    rotations = fig.poser.rotations(pose)
    offsets = _base(fig, rotations)
    path = None
    if spec.touching:
        path = touch(fig, pose, spec.touching)
        offsets = offsets + path[1][-1]
    out = fig.pose([rotations], offsets=offsets[None], transforms=True)
    v = out["vertices"][0] + root
    surfaces = [(0.0, None)]
    if bench is not None:
        top = bench.centre[2] + bench.half[2]
        surfaces.append((top, np.stack([bench.centre[:2] - bench.half[:2], bench.centre[:2] + bench.half[:2]])))
    d = flatten(model, v, surfaces)
    return offsets + to_rest(model, out["transforms"][0], d), path


# --------------------------------------------------------------------------- Ganze Haltung


def make(fig: P.Figure, spec: Spec, solver: Solver | None = None) -> Still:
    """Haltung einstellen, auf Boden/Bank legen und die Haut nachgeben lassen."""
    solver = solver or Solver(fig)
    params = solver.solve(spec)
    pose = spec.build(**params)
    rotations = fig.poser.rotations(pose)
    out = fig.pose([rotations], transforms=True)
    v = out["vertices"][0]
    _, residual = solver.cost(spec, v)
    g = solver.groups
    ground = min(v[g[n], 2].min() for n in _expand(spec.floor, g))
    root = np.array([0.0, 0.0, -ground - spec.sink])
    bench = None
    if spec.bench:
        seat = np.zeros(len(v), bool)
        for n in _expand(spec.bench, g):
            seat |= g[n]
        p = v[seat] + root
        low = p[p[:, 2] < p[:, 2].min() + 0.02]
        top = p[:, 2].min() + spec.sink * 3  # das Gesäß gibt mehr nach als eine Fußsohle
        half = BENCH_HALF.copy()
        half[2] = top / 2
        # Bank hinter der Person (Blick nach -Y), Vorderkante gut 10 cm vor dem tiefsten Gesäßpunkt
        centre = np.array([low[:, 0].mean(), low[:, 1].mean() - 0.12 + half[1], top / 2])
        bench = contact.Box("Bank", centre, np.eye(3), half)
    offsets, path = soften(fig, spec, pose, root, bench)
    final = fig.pose([rotations], offsets=offsets[None])
    poses = final["poses"][0].copy()
    poses[:, :3, 3] += root
    return Still(spec, params, pose, rotations, root, offsets, final["vertices"][0] + root, poses,
                 [bench] if bench is not None else [], residual, path)


# --------------------------------------------------------------------------- Haltungen


def _sit(knee=90.0, foot=0.0, lean=5.0, arm=20.0, elbow=30.0):
    return P.pose(None, body=dict(pitch=lean), legs=dict(hip=90 + lean, knee=knee, foot=foot, out=6, turn=-6),
                  arms=dict(down=22, fwd=arm + lean, elbow=elbow, forearm=40, wrist=10), spine=dict(neck=-lean))


def _supine(pitch=-90.0, neck=0.0, hip=0.0, arms=0.0):
    return P.pose(None, body=dict(pitch=pitch), spine=dict(neck=neck),
                  legs=dict(hip=hip, knee=4, foot=30, out=6, turn=-25),
                  arms=dict(down=32, fwd=arms, elbow=8, forearm=-40))


def _prone(pitch=90.0, neck=0.0, hip=0.0, arms=0.0, foot=60.0):
    return P.pose(None, body=dict(pitch=pitch), spine=dict(neck=neck, turn=75),
                  legs=dict(hip=hip, knee=4, foot=foot, out=4),
                  arms=dict(down=32, fwd=arms, elbow=8, forearm=60))


def _tree(roll=0.0, out=0.0, hip=40.0, spread=50.0, turn=-60.0, knee=145.0):
    return P.pose(None, body=dict(roll=roll), leg_L=dict(hip=2, out=out, turn=-5),
                  leg_R=dict(hip=hip, out=spread, knee=knee, turn=turn, foot=25),
                  arms=dict(down=-130, elbow=15))


def _warrior(front=95.0, wide=60.0, back=35.0, foot_f=0.0, foot_b=0.0):
    return P.pose(None, leg_L=dict(out=wide, turn=-85, knee=front, foot=foot_f),
                  leg_R=dict(out=back, turn=10, foot=foot_b),
                  arms=dict(down=-40, elbow=0), spine=dict(turn=70))


def _overhead(p: dict, spread: float) -> dict:
    """Arme über Kopf schulterbreit auseinander: nach dem Heben seitlich spreizen, damit der Kopf
    zwischen den Oberarmen Platz hat."""
    for s in P.SIDES:
        p[(f"upperarm01.{s}", "spreizen")] = spread
    return p


def _dog(pitch=135.0, hip=105.0, foot=-25.0, wrist=-45.0, forearm=-48.0):
    return _overhead(P.pose(None, body=dict(pitch=pitch), legs=dict(hip=hip, knee=0, foot=foot, out=3),
                            arms=dict(down=32, fwd=170, elbow=0, forearm=forearm, wrist=wrist),
                            shoulders=dict(up=10)), 35)


def _child(pitch=115.0, hip=145.0, knee=160.0, neck=10.0, arms=165.0, foot=60.0, wrist=-20.0):
    return _overhead(P.pose(None, body=dict(pitch=pitch), legs=dict(hip=hip, knee=knee, foot=foot, out=30, turn=15),
                            arms=dict(down=32, fwd=arms, elbow=5, forearm=-45, wrist=wrist), spine=dict(neck=neck),
                            shoulders=dict(up=10)), 35)


def _cobra(bend=-40.0, arms=10.0, elbow=70.0, wrist=-70.0, pitch=90.0, forearm=-45.0):
    return P.pose(None, body=dict(pitch=pitch), legs=dict(knee=0, foot=60, out=3), spine=dict(bend=bend, neck=-15),
                  arms=dict(down=32, fwd=arms, elbow=elbow, forearm=forearm, wrist=wrist))


def _squat(pitch=40.0, hip=140.0, foot=-40.0):
    return P.pose(None, body=dict(pitch=pitch), legs=dict(hip=hip, knee=150, foot=foot, out=20, turn=-20),
                  arms=dict(down=22, fwd=50 + pitch, elbow=60), spine=dict(neck=-pitch * 0.6))


def _lunge(front=125.0, back_hip=-30.0, back_knee=95.0, foot_f=-35.0, foot_b=-50.0, hip=95.0):
    return P.pose(None, leg_L=dict(hip=hip, knee=front, foot=foot_f, turn=-4),
                  leg_R=dict(hip=back_hip, knee=back_knee, foot=foot_b, turn=-4),
                  arms=dict(down=38, fwd=3, elbow=12))


def _fold(pitch=90.0, bend=40.0, arms=135.0, foot=0.0):
    return P.pose(None, body=dict(pitch=pitch), legs=dict(hip=pitch, knee=5, foot=foot), spine=dict(bend=bend, neck=20),
                  arms=dict(down=20, fwd=arms, elbow=10, wrist=-30))



def catalogue() -> dict[str, Spec]:
    specs = [
        Spec("sitzen_bank", "Sitzen auf der Hantelbank", _sit, dict(knee=90, foot=0, lean=5, arm=20, elbow=30),
             dict(knee=(70, 110), foot=(-25, 25)), ("ferse", "zehen"), bench=("gesaess",), sink=0.003,
             touching=(("Hand links", "Oberschenkel links"), ("Hand rechts", "Oberschenkel rechts"),
                       ("Hand links", "Unterschenkel links"), ("Hand rechts", "Unterschenkel rechts"),
                       ("Bauch", "Oberschenkel links"), ("Bauch", "Oberschenkel rechts"))),
        Spec("rueckenlage", "Rückenlage", _supine, dict(pitch=-90, neck=0, hip=0, arms=0),
             dict(pitch=(-100, -80), neck=(-15, 25), hip=(-10, 15), arms=(-20, 10)),
             ("kopf", "ruecken", "gesaess", "ferse", "handballen"), sink=0.008),
        Spec("bauchlage", "Bauchlage, Kopf zur Seite", _prone, dict(pitch=90, neck=0, hip=0, arms=0, foot=60),
             dict(pitch=(80, 100), neck=(-20, 20), hip=(-15, 10), arms=(-10, 25), foot=(30, 80)),
             ("kopf", "brust", "oberschenkel", "zehen", "handballen"), sink=0.008, soft=("unterbauch", "schienbein")),
        Spec("baum", "Yoga: Baum", _tree, dict(roll=0, out=0, hip=40, spread=50, turn=-60, knee=145),
             dict(roll=(-10, 10), out=(-10, 10), hip=(10, 45), spread=(40, 90), turn=(-110, -20), knee=(120, 165)),
             ("ferse.L", "zehen.L"), balance=True, sink=0.002, holds={"ferse.R|oberschenkel.L": 0.0},
             touching=(("Fuß rechts", "Oberschenkel links"), ("Fuß rechts", "Unterschenkel links"))),
        Spec("krieger2", "Yoga: Krieger II", _warrior, dict(front=95, wide=60, back=35, foot_f=0, foot_b=0),
             dict(front=(80, 110), wide=(45, 75), back=(25, 45), foot_f=(-30, 30), foot_b=(-30, 30)),
             ("ferse", "zehen"), balance=True, sink=0.002),
        Spec("hund", "Yoga: Herabschauender Hund", _dog, dict(pitch=137, hip=101, foot=-25, wrist=-45, forearm=-48),
             dict(pitch=(120, 150), hip=(90, 112), foot=(-45, 0), wrist=(-90, 0), forearm=(-90, 0)),
             ("handballen", "finger", "ferse", "zehen"), sink=0.003,
             touching=(("Bauch", "Oberschenkel links"), ("Bauch", "Oberschenkel rechts"))),
        Spec("kind", "Yoga: Kindhaltung", _child, dict(pitch=115, hip=145, knee=160, neck=10, arms=165, foot=60, wrist=-20),
             dict(pitch=(95, 135), hip=(125, 160), knee=(145, 170), neck=(-10, 40), arms=(140, 180),
                  foot=(40, 80), wrist=(-60, 10)),
             ("schienbein", "zehen", "kopf", "handballen", "finger"), sink=0.005,
             touching=(("Bauch", "Oberschenkel links"), ("Bauch", "Oberschenkel rechts"),
                       ("Brust", "Oberschenkel links"), ("Brust", "Oberschenkel rechts"),
                       ("Oberschenkel links", "Unterschenkel links"),
                       ("Oberschenkel rechts", "Unterschenkel rechts"),
                       ("Becken links", "Fuß links"), ("Becken rechts", "Fuß rechts"),
                       ("Bauch", "Fuß links"), ("Bauch", "Fuß rechts"), ("Fuß links", "Fuß rechts"))),
        Spec("kobra", "Yoga: Kobra", _cobra, dict(bend=-40, arms=10, elbow=70, wrist=-70, pitch=90, forearm=-45),
             dict(bend=(-50, -25), arms=(-15, 60), elbow=(40, 120), wrist=(-90, -30), pitch=(80, 100),
                  forearm=(-90, 0)),
             ("oberschenkel", "zehen", "handballen", "finger"), sink=0.005, soft=("unterbauch",)),
        Spec("hocke", "Mobility: tiefe Hocke", _squat, dict(pitch=40, hip=140, foot=-40),
             dict(pitch=(20, 60), hip=(120, 150), foot=(-55, -20)), ("ferse", "zehen"), balance=True,
             sink=0.002,
             touching=(("Oberschenkel links", "Unterschenkel links"), ("Oberschenkel rechts", "Unterschenkel rechts"),
                       ("Bauch", "Oberschenkel links"), ("Bauch", "Oberschenkel rechts"),
                       ("Becken links", "Oberschenkel links"), ("Becken rechts", "Oberschenkel rechts"))),
        Spec("ausfallschritt", "Mobility: tiefer Ausfallschritt", _lunge,
             dict(front=125, back_hip=-30, back_knee=95, foot_f=-35, foot_b=-50, hip=95),
             dict(front=(90, 140), back_hip=(-60, -5), back_knee=(60, 130), foot_f=(-50, 10), foot_b=(-80, 0),
                  hip=(70, 110)),
             ("ferse.L", "zehen.L", "schienbein.R", "zehen.R"), balance=True, sink=0.003),
        Spec("vorbeuge", "Mobility: Vorbeuge", _fold, dict(pitch=90, bend=40, arms=135, foot=0),
             dict(pitch=(70, 110), bend=(20, 60), arms=(100, 170), foot=(-10, 10)),
             ("ferse", "zehen", "finger"), balance=True, sink=0.002,
             touching=(("Bauch", "Oberschenkel links"), ("Bauch", "Oberschenkel rechts"),
                       ("Brust", "Oberschenkel links"), ("Brust", "Oberschenkel rechts"))),
    ]
    return {s.name: s for s in specs}


# --------------------------------------------------------------------------- Bewegungen


def bench_prop(box: contact.Box):
    """Die Bank als Gerät der Bewegungsbibliothek (Ursprung in der Bankmitte)."""
    from . import gym

    m = np.eye(4)
    m[:3, :3], m[:3, 3] = box.rotation, box.centre
    return gym.Prop("Hantelbank", [dataclasses.replace(box, centre=np.zeros(3), rotation=np.eye(3))]), m


PROGRESS = ("fortschritt", "t")  # Hilfskanal in Schlüsselhaltungen: 0 = Stehen, 1 = Zielhaltung


def _surfaces(ex, i: int) -> list:
    """Auflageflächen in Bild i: Boden und die Oberseite jeder Bank."""
    out = [(0.0, None)]
    for prop, mats in ex.props:
        for b in prop.placed(mats[i]):
            if isinstance(b, contact.Box):
                top = b.centre[2] + b.half[2]
                out.append((top, np.stack([b.centre[:2] - b.half[:2], b.centre[:2] + b.half[:2]])))
    return out


LEG_DOF = (("upperleg01", "vor"), ("upperleg01", "spreizen"), ("upperleg01", "dreh"), ("lowerleg01", "beugen"),
           ("foot", "runter"), ("foot", "aussen"))


def _rotvec(r: np.ndarray) -> np.ndarray:
    """Drehvektor (Achse * Winkel im Bogenmaß) einer 3x3-Drehung."""
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    s = np.linalg.norm(axis)
    angle = np.arctan2(s / 2, (np.trace(r) - 1) / 2)
    return axis * (angle / s) if s > 1e-12 else np.zeros(3)


def _bones(fig: P.Figure, poses: list[dict]) -> np.ndarray:
    """Weltlage aller Knochen (N, J, 4, 4), Wurzel im Ursprung, ohne Haut."""
    import torch

    mats = np.stack([fig.poser.matrices(fig.poser.rotations(p)) for p in poses])
    with torch.no_grad():
        _, bone_poses = fig.model.anny.get_bone_transforms(torch.as_tensor(mats, dtype=torch.float32),
                                                           fig._rest_p)
    return bone_poses.double().numpy()


def _with(pose: dict, keys: list, x: np.ndarray) -> dict:
    out = dict(pose)
    out.update(zip(keys, x[:-1], strict=True))
    return out


def _foot_error(bones: np.ndarray, foot: list[int], shift: np.ndarray, feet: np.ndarray) -> np.ndarray:
    """Abstand der Füße zu ihrer Stelle: Lage in cm, Richtung in 1/10 rad."""
    r = []
    for k, j in enumerate(foot):
        r += list((bones[j][:3, 3] + shift - feet[k][:3, 3]) * 100)
        r += list(_rotvec(feet[k][:3, :3] @ bones[j][:3, :3].T) * 10)
    return np.array(r)


def plant_legs(fig: P.Figure, poses: list[dict], root: np.ndarray, feet: np.ndarray,
               iterations: int = 12) -> tuple[list[dict], np.ndarray, float]:
    """Beine je Bild so beugen, dass beide Füße genau an ihrer Stelle stehen (Lage und Richtung wie in
    feet: (2, 4, 4) Weltlage von foot.L/foot.R). Frei sind je Bein Hüfte (3), Knie und Sprunggelenk (2)
    sowie die Höhe der Hüfte; die Haltung weicht so wenig wie möglich von der gewünschten ab.
    Rückgabe: Haltungen, Wurzelverschiebung (F, 3), größte Abweichung der Füße (m)."""
    keys = [(f"{b}.{s}", d) for s in P.SIDES for b, d in LEG_DOF]
    foot = [fig.poser.index[f"foot.{s}"] for s in P.SIDES]
    n = len(keys) + 1
    scale = np.array([1.0] * len(keys) + [0.01])  # Schritt: Grad bzw. cm
    weight = np.array([0.01] * len(keys) + [0.03]) / scale  # Abweichung von der Wunschhaltung
    out, roots, worst = [], root.astype(float).copy(), 0.0
    previous = None
    for i, want in enumerate(poses):
        goal = np.array([want.get(k, 0.0) for k in keys] + [root[i, 2]])
        x = goal.copy()
        if previous is not None:  # vom letzten Bild aus weiter (Hüfthöhe als Abstand zur Wunschhöhe)
            x[:-1], x[-1] = previous[:-1], goal[-1] + previous[-1]

        def residual(x, bones, i=i, goal=goal):
            shift = np.array([root[i, 0], root[i, 1], x[-1]])
            return np.concatenate([_foot_error(bones, foot, shift, feet), (x - goal) * weight])

        damping = 1e-2
        for _ in range(iterations):
            trials = [x] + [x + np.eye(n)[k] * scale[k] * 0.5 for k in range(n)]
            bones = _bones(fig, [_with(want, keys, t) for t in trials])
            r0 = residual(x, bones[0])
            jac = np.stack([(residual(trials[k + 1], bones[k + 1]) - r0) / (scale[k] * 0.5) for k in range(n)], 1)
            h = jac.T @ jac
            step = -np.linalg.solve(h + damping * np.diag(np.diag(h) + 1e-9), jac.T @ r0)
            new = x + step
            r1 = residual(new, _bones(fig, [_with(want, keys, new)])[0])
            if r1 @ r1 < r0 @ r0:
                x, damping = new, damping * 0.3
            else:
                damping *= 10
            if np.abs(step / scale).max() < 0.05:
                break
        r = residual(x, _bones(fig, [_with(want, keys, x)])[0])
        worst = max(worst, float(np.abs(r[[0, 1, 2, 6, 7, 8]]).max()) / 100)
        out.append(_with(want, keys, x))
        roots[i, 2] = x[-1]
        previous = np.concatenate([x[:-1], [x[-1] - goal[-1]]])
    return out, roots, worst


def stand_over(fig: P.Figure, poses: list[dict], root: np.ndarray, feet: np.ndarray, centre: np.ndarray,
               rounds: int = 3) -> tuple[list[dict], np.ndarray, float]:
    """Füße festhalten (plant_legs) und die Hüfte waagerecht so verschieben, dass der Schwerpunkt über
    centre (Mitte der Standfläche, xy) liegt: niemand kippt beim Hinsetzen oder in der Hocke."""
    root = root.astype(float).copy()
    for _ in range(rounds):
        legs, root, worst = plant_legs(fig, poses, root, feet)
        v = fig.pose([fig.poser.rotations(p) for p in legs])["vertices"] + root[:, None]
        com = np.stack([centre_of_mass(x, fig.model.faces) for x in v])
        root[:, :2] += centre[:2] - com[:, :2]
    legs, root, worst = plant_legs(fig, poses, root, feet)
    return legs, root, worst


def flow(fig: P.Figure, name: str, keys: list[tuple[float, dict]], still: Still, captions: list,
         props=(), balanced: Callable[[np.ndarray], np.ndarray] | None = None):
    """Bewegung aus Schlüsselhaltungen mit stehenden Füßen: Die Füße stehen in jedem Bild genau dort,
    wo sie in der Zielhaltung still stehen (die Beine werden passend gebeugt), und der Schwerpunkt liegt
    über den Füßen. balanced(progress) -> Anteil 0..1, wie weit das Gleichgewicht gilt (beim Sitzen
    trägt die Bank). Die Schlüssel tragen ihren Fortschritt zur Zielhaltung (PROGRESS); danach richten
    sich die Berührungsformen. Wo die Haut in Boden oder Bank steckt, gibt sie in jedem Bild nach."""
    from . import gym

    times = gym.frames_between(keys[-1][0])
    sampled = P.sample(keys, times)
    progress = np.array([p.pop(PROGRESS, 0.0) for p in sampled])
    first = fig.pose([fig.poser.rotations(p) for p in sampled])["vertices"]
    root = gym.plant_feet(fig, first)
    top = int(np.argmax(progress))
    root += still.root - root[top]
    feet = still.poses[[fig.poser.index[f"foot.{s}"] for s in P.SIDES]]
    centre = gym._sole_centre(fig, still.vertices)
    share = np.ones(len(times)) if balanced is None else balanced(progress)
    legs, planted, worst = stand_over(fig, sampled, root, feet, centre)
    root[:, 2] = planted[:, 2]
    root[:, :2] += share[:, None] * (planted[:, :2] - root[:, :2])
    if (share < 1).any():  # wo die Bank trägt, gilt die Hüftlage aus den Schlüsseln: Beine neu
        legs, root, worst = plant_legs(fig, sampled, root, feet)
    frames = [fig.poser.rotations(p) for p in legs]
    poses = fig.pose(frames)["poses"]
    offsets = still.along(progress)
    if fig.correctives is not None:
        offsets += fig.correctives.offsets(poses)
    out = fig.pose(frames, offsets=offsets, transforms=True)
    contacts = {f"foot.{s}": np.ones(len(times), bool) for s in P.SIDES}
    ex = gym.Exercise(name, gym.FPS, frames, root, list(props), [], contacts, captions)
    ex.foot_error = worst
    v = out["vertices"] + root[:, None]
    for i in range(len(frames)):
        surfaces = _surfaces(ex, i)
        if not any((v[i, :, 2] < h - 5e-4).any() for h, _ in surfaces):
            continue
        d = flatten(fig.model, v[i], surfaces)
        offsets[i] += to_rest(fig.model, out["transforms"][i], d).astype(np.float32)
    final = fig.pose(frames, offsets=offsets)
    ex.vertices = final["vertices"] + root[:, None]
    world = final["poses"].copy()
    world[:, :, :3, 3] += root[:, None]
    ex.poses = world
    return ex


def _key(pose: dict, progress: float) -> dict:
    out = dict(pose)
    out[PROGRESS] = progress
    return out


def _lean(pose: dict, degrees: float) -> dict:
    """Oberkörper nach vorn neigen, Beine bleiben, wo sie sind (wie beim Aufstehen vom Stuhl)."""
    out = dict(pose)
    out[("root", "pitch")] = out.get(("root", "pitch"), 0.0) + degrees
    for s in P.SIDES:
        out[(f"upperleg01.{s}", "vor")] = out.get((f"upperleg01.{s}", "vor"), 0.0) + degrees
    out[("neck01", "vor")] = out.get(("neck01", "vor"), 0.0) - degrees * 0.4
    out[("head", "vor")] = out.get(("head", "vor"), 0.0) - degrees * 0.4
    return out


def _reps(start: dict, mid: dict, end: dict, reps: int, down: float, hold: float, up: float,
          rest: float = 0.6, at: float = 0.55) -> tuple[list, float]:
    """Schlüssel für: stehen, über mid in end, halten, zurück, stehen (reps mal)."""
    keys = [(0.0, _key(start, 0)), (rest, _key(start, 0))]
    t = rest
    for _ in range(reps):
        keys += [(t + down * at, _key(mid, at)), (t + down, _key(end, 1)), (t + down + hold, _key(end, 1)),
                 (t + down + hold + up * (1 - at), _key(mid, at)), (t + down + hold + up, _key(start, 0)),
                 (t + down + hold + up + rest, _key(start, 0))]
        t += down + hold + up + rest
    return keys, t


def sit_down(fig: P.Figure, seated: Still, reps: int = 2):
    """Auf die Hantelbank setzen und wieder aufstehen: Füße bleiben stehen, beim Hinsetzen und
    Aufstehen neigt sich der Oberkörper nach vorn, das Gesäß gibt auf der Bank nach."""
    from . import gym

    stand = blend(P.stand(), seated.pose, 0.0)
    mid = _lean(blend(stand, seated.pose, 0.7), 22)
    keys, t = _reps(stand, mid, seated.pose, reps, down=1.6, hold=1.4, up=1.6, at=0.7)

    props = [bench_prop(box) for box in seated.props]
    props = [(prop, np.repeat(m[None], len(gym.frames_between(t)), 0)) for prop, m in props]
    return flow(fig, "Hinsetzen", keys, seated, [(0.0, t + 1, "Auf die Bank setzen und aufstehen, Füße bleiben stehen")],
                props=props, balanced=lambda progress: np.clip((1 - progress) / 0.15, 0, 1))


def squat(fig: P.Figure, deep: Still, reps: int = 2):
    """In die tiefe Hocke und wieder hoch, Fersen bleiben am Boden."""
    stand = blend(P.stand(), deep.pose, 0.0)
    mid = _lean(blend(stand, deep.pose, 0.5), 12)
    keys, t = _reps(stand, mid, deep.pose, reps, down=1.8, hold=1.5, up=1.6, at=0.5)
    return flow(fig, "Hocke", keys, deep, [(0.0, t + 1, "Tiefe Hocke: runter und hoch, Fersen bleiben am Boden")])
