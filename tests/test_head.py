import math

import numpy as np
import pytest

from fitscan import head as H
from fitscan.body import BodyParams
from fitscan.camera import Intrinsics
from fitscan.fit import repair_offsets


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


@pytest.fixture(scope="module")
def anny_face(model):
    verts = model.rest_numpy(BodyParams())["vertices"].astype(np.float64)
    tri, bary = H.load_correspondence()
    return verts, H.surface_points(verts, model.faces, tri, bary)


def test_similarity_recovers_transform():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(60, 3))
    rot = _rotation([0.3, -1.0, 0.4], 0.7)
    dst = 1.3 * src @ rot.T + [0.1, -0.2, 0.3]
    s, r, t = H.similarity(src, dst)
    assert s == pytest.approx(1.3, abs=1e-9)
    assert np.allclose(r, rot, atol=1e-9)
    assert np.allclose(t, [0.1, -0.2, 0.3], atol=1e-9)


def test_correspondence_sits_on_annys_face(anny_face):
    """Die Gesichtspunkte liegen an den richtigen Stellen: Nasenspitze vorn, Kinn unten,
    rechter Augenwinkel (aus Sicht der Person) auf der rechten Seite (-X)."""
    _, pts = anny_face
    face = pts[: H.FACE_POINTS]
    assert pts[H.NOSE_TIP, 1] < np.percentile(face[:, 1], 2)  # Anny blickt nach -Y
    assert pts[152, 2] < np.percentile(face[:, 2], 2)
    assert pts[33, 0] < 0 < pts[263, 0]
    assert abs(pts[33, 0] + pts[263, 0]) < 0.003  # symmetrisch


def _synthetic_tracks(shape, intr, rng, count=24, noise=0.5):
    c = shape.mean(0)
    to_cam = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # Anny -> Kamera (Gesicht zur Kamera)
    tracks = []
    for k in range(count):
        rot = (_rotation([1, 0, 0], math.radians(rng.uniform(-15, 25)))
               @ _rotation([0, 1, 0], math.radians(rng.uniform(-50, 50))) @ to_cam)
        cam = (shape - c) @ rot.T + [rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), 0.42]
        uv = intr.focal * cam[:, :2] / cam[:, 2:] + [intr.cx, intr.cy]
        tracks.append(H.FaceTrack(k, (uv + rng.normal(0, noise, uv.shape)).astype(np.float32)))
    return tracks


def test_solve_face_recovers_shape(anny_face):
    """Aus Gesichtspunkten in vielen Ansichten folgt die Form (bis auf die Größe)."""
    _, template = anny_face
    rng = np.random.default_rng(1)
    c = template.mean(0)
    true = template.copy()
    true[:, 0] = c[0] + (true[:, 0] - c[0]) * 1.06  # breiteres Gesicht
    true[:, 2] -= 0.1 * np.maximum(c[2] - true[:, 2], 0)  # längeres Kinn
    intr = Intrinsics.phone_default(720, 1280)
    sol = H.solve_face(_synthetic_tracks(true, intr, rng), intr, template, steps=400)
    assert sol.error_px < 1.5

    def gap(shape):
        s, r, t = H.similarity(shape, true)
        return np.median(np.linalg.norm(s * shape @ r.T + t - true, axis=1))

    assert gap(sol.points) < 1.5e-3
    assert gap(sol.points) < 0.4 * gap(template)


def test_fit_head_shape_follows_face(model, anny_face):
    """Annys Kopf wird glatt so verformt, dass ihre Gesichtspunkte auf der gemessenen Form liegen;
    der Hals bleibt, und das Netz durchdringt sich nicht."""
    verts, template = anny_face
    c = template.mean(0)
    true = template.copy()
    true[:, 0] = c[0] + (true[:, 0] - c[0]) * 1.05
    true[:, 2] -= 0.08 * np.maximum(c[2] - true[:, 2], 0)
    sol = H.FaceSolution(true - true.mean(0), np.eye(3)[None], np.zeros((1, 3)),
                         Intrinsics.phone_default(720, 1280), [0], 0.0)
    place = H.place_head(model, verts, sol)
    disp = H.fit_head_shape(model, verts, sol, place)
    tri, bary = H.load_correspondence()
    use = np.setdiff1d(np.arange(H.FACE_POINTS), H.INNER_LIPS)
    before = np.linalg.norm(H.surface_points(verts, model.faces, tri, bary)[use] - place.to_anny(sol.points)[use], axis=1)
    after = np.linalg.norm(H.surface_points(verts + disp, model.faces, tri, bary)[use]
                           - place.to_anny(sol.points)[use], axis=1)
    assert np.median(after) < 0.25 * np.median(before)
    assert np.abs(disp[~H.head_vertex_mask(model)]).max() == 0.0
    params = BodyParams()
    params.offsets = disp.astype(np.float32)
    _, touched = repair_offsets(model, params)
    assert touched < 50


def test_eyes_open():
    assert H.eyes_open(None)
    assert H.eyes_open({"eyeLookOutLeft": 0.6, "eyeBlinkLeft": 0.1})
    assert not H.eyes_open({"eyeBlinkRight": 0.6})


def test_merge_head_skin_matches_colours():
    body = np.full((256, 256, 3), (180, 140, 120), np.uint8)
    head = np.full((256, 256, 3), (200, 150, 110), np.uint8)
    head[100:140, 100:140] = (60, 40, 30)  # Tattoo/Augenbraue im Kopfbild
    alpha = np.zeros((256, 256), np.float32)
    alpha[60:200, 60:200] = 1.0
    out = H.merge_head_skin(body, head, alpha).astype(int)
    # Hautton des Kopfes an den Körper angeglichen, Einzelheiten bleiben, außen unverändert
    assert np.abs(out[80, 80] - [180, 140, 120]).max() <= 3
    assert out[120, 120].max() < 90
    assert (out[5, 5] == [180, 140, 120]).all()


def test_paint_eyes_draws_iris_and_pupil(model):
    from fitscan.texture import to_linear, uv_map

    size = 2048
    verts = model.rest_numpy(BodyParams())["vertices"].astype(np.float64)
    skin = np.full((size, size, 3), 200, np.uint8)
    alpha = np.zeros((size, size), np.float32)
    iris = to_linear(np.array([[70, 110, 150]], np.uint8))[0]
    out, a = H.paint_eyes(model, verts, skin, alpha, iris, None)
    m = uv_map(model, size)
    faces = model.faces
    for em in H.eye_vertex_masks(model).values():
        on = em[faces[m.face]].all(1)
        p = (verts[faces[m.face[on]]] * m.bary[on][..., None]).sum(1)
        c = verts[em].mean(0)
        front = verts[em][np.argmin(verts[em][:, 1])] - c
        cos = ((p - c) @ front) / np.linalg.norm(p - c, axis=1) / np.linalg.norm(front)
        col = out.reshape(-1, 3)[m.texel[on]].astype(int)
        assert col[cos > 0.995].max() < 40  # Pupille vorn in der Mitte
        ring = (cos > 0.95) & (cos < 0.98)
        assert (col[ring, 2] > col[ring, 0]).mean() > 0.8  # blaue Iris
        assert col[(cos > 0.4) & (cos < 0.6)].mean() > 150  # Augenweiß
        assert (a.reshape(-1)[m.texel[on]] == 1).all()
