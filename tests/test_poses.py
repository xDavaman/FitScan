import numpy as np
import pytest

from fitscan import contact as C
from fitscan import poses as H
from fitscan import posing as P
from fitscan.body import BodyParams


@pytest.fixture(scope="module")
def fig(model):
    return P.Figure.from_params(model, BodyParams())


def _cube():
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    f = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
                  [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]])
    return v, f


def test_centre_of_mass_and_balance():
    v, f = _cube()
    assert H.centre_of_mass(v + 3.0, f) == pytest.approx([3.5, 3.5, 3.5])
    # Standfläche ist die ganze Unterseite: Schwerpunkt mittig, steht sicher
    assert H.balance(v, f, 0.0) == 0.0
    # Würfel kippt über eine Kante: nur noch die Kante steht am Boden, Schwerpunkt daneben
    tilted = v.copy()
    tilted[:, 2] += tilted[:, 0] * 0.5
    assert H.balance(tilted, f, 0.0) > 0.4


def test_flatten_lays_skin_on_floor(model, fig):
    v = fig.reference_vertices - [0, 0, fig.reference_vertices[:, 2].min() + 0.005]
    d = H.flatten(model, v, [(0.0, None)])
    after = v + d
    assert after[:, 2].min() > -0.001
    # nur die Fußsohlen und ihre Umgebung bewegen sich
    assert np.abs(d[v[:, 2] > 0.15]).max() < 1e-6


def test_along_interpolates_contact_shapes(model, fig):
    n = len(fig.reference_vertices)
    still = H.Still(None, {}, {}, {}, np.zeros(3), np.zeros((n, 3)), None, None, [], {},
                    (np.array([0.5, 1.0]), np.stack([np.full((n, 3), 1.0), np.full((n, 3), 3.0)])))
    out = still.along(np.array([0.0, 0.25, 0.75, 1.0, 1.5]))[:, 0, 0]
    assert out == pytest.approx([0, 0.5, 2, 3, 3])


def test_supine_rests_on_floor(model, fig):
    spec = H.catalogue()["rueckenlage"]
    still = H.make(fig, spec)
    # Hinterkopf, Rücken, Gesäß und Fersen liegen auf, nichts steckt im Boden
    for name, gap in still.residual.items():
        if name != "sonst":
            assert gap < 0.015, name
    findings = C.Checker(model, fig.reference_vertices).check_frame(still.vertices, self_check=False)
    assert max([f.depth for f in findings if f.kind == "boden"], default=0.0) < 0.002
    # die Haut liegt flach auf: viele Punkte des Rückens berühren den Boden
    assert (still.vertices[:, 2] < 0.002).sum() > 50


def test_bench_sits_under_glutes(model, fig):
    still = H.make(fig, H.catalogue()["sitzen_bank"])
    (bench,) = still.props
    top = bench.centre[2] + bench.half[2]
    assert top == pytest.approx(H.BENCH_HEIGHT, abs=0.03)
    assert bench.depth(still.vertices).max() < 0.003
    feet = model.dominant_bone_mask(lambda n: n.startswith(("foot", "toe")))
    assert still.vertices[feet, 2].min() == pytest.approx(0.0, abs=0.005)
