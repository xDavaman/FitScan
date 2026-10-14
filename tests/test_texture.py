import json
import math
import struct

import numpy as np
import pytest
import torch

from fitscan.body import BodyParams
from fitscan.camera import Intrinsics, to_camera
from fitscan.export import build_rig, write_glb, write_obj
from fitscan.texture import SEEN_FULL, Shot, TextureBaker, face_buffer, render_textured, uv_map

SIZE = 512


@pytest.fixture(scope="module")
def known_skin():
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    return np.stack([xx * 255 / SIZE, yy * 255 / SIZE, 128 + 60 * np.sin(xx / 9.0)], -1).astype(np.uint8)


def test_uv_map_matches_uv_area(model):
    m = uv_map(model, SIZE)
    assert np.allclose(m.bary.sum(1), 1.0, atol=1e-5)
    # so viele Texel, wie die UV-Inseln Fläche haben
    q = m.uv[m.face_uv]
    d1, d2 = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
    area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]).sum()
    assert len(m.texel) / SIZE**2 == pytest.approx(area, abs=0.03)


def test_bake_recovers_known_skin(model, known_skin):
    """Ein Körper mit bekanntem Hautbild wird aus 12 Richtungen gerendert; das daraus gewonnene
    Hautbild muss dem Original an allen gesehenen Stellen entsprechen."""
    verts = model.rest_numpy(BodyParams())["vertices"]
    uv = model.anny.texture_coordinates.numpy()
    face_uv = model.anny.face_texture_coordinate_indices.numpy()
    intr = Intrinsics.phone_default(540, 960)
    trans = np.array([0.0, 0.85, 3.0])
    baker = TextureBaker(model, SIZE)
    for k in range(12):
        yaw = math.radians(30 * k)
        rgb = render_textured(verts, model.faces, uv, face_uv, known_skin, yaw, trans, intr)
        cam = to_camera(torch.as_tensor(verts), torch.tensor([yaw]),
                        torch.as_tensor(trans, dtype=torch.float32)[None])
        mask = (face_buffer(cam[0].numpy().astype(np.float64), model.faces, intr) > 0).astype(np.float32)
        baker.add(Shot(rgb=rgb, mask=mask, intr=intr, to_image=np.eye(3), vertices=verts, yaw=yaw, trans=trans))
    assert baker.coverage() > 0.6
    skin = baker.finish()
    m = baker.map
    seen = baker.base_w >= SEEN_FULL
    got = skin.reshape(-1, 3)[m.texel[seen]].astype(np.float32)
    want = known_skin.reshape(-1, 3)[m.texel[seen]].astype(np.float32)
    assert np.median(np.abs(got - want)) < 4.0
    assert np.mean(np.abs(got - want)) < 8.0
    # Lücken sind gefüllt, nichts bleibt schwarz
    assert skin.reshape(-1, 3)[m.texel].max(1).min() > 0


def test_textured_glb_and_obj(model, known_skin, tmp_path):
    rig = build_rig(model, BodyParams(), skin=known_skin)
    path = write_glb(rig, tmp_path / "haut.glb")
    data = path.read_bytes()
    length, _ = struct.unpack("<I4s", data[12:20])
    gltf = json.loads(data[20:20 + length])
    prim = gltf["meshes"][0]["primitives"][0]
    count = gltf["accessors"][prim["attributes"]["POSITION"]]["count"]
    assert gltf["accessors"][prim["attributes"]["TEXCOORD_0"]]["count"] == count
    assert count == len(np.unique(np.stack([rig.faces.ravel(), rig.face_uv.ravel()], -1), axis=0))
    assert gltf["images"][0]["mimeType"] == "image/jpeg"
    assert gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] == {"index": 0}
    # gedoppelte Punkte an den Nähten liegen exakt aufeinander: geometrisch weiter geschlossen
    src, uv, faces = rig.split_at_seams()
    assert faces.max() + 1 == len(src) == count
    assert np.all((uv >= 0) & (uv <= 1))

    obj = write_obj(rig, tmp_path / "haut.obj")
    text = obj.read_text("utf-8")
    assert "mtllib haut.mtl" in text and "usemtl Haut" in text
    assert text.count("\nvt ") == len(rig.uv)
    assert text.count("\nv ") == len(rig.vertices)  # Punkte nicht gedoppelt
    assert "map_Kd haut_haut.jpg" in (tmp_path / "haut.mtl").read_text("utf-8")
    assert (tmp_path / "haut_haut.jpg").stat().st_size > 1000


def test_eyes_get_own_material(model, known_skin, tmp_path):
    """Mit Augenbild: Augäpfel als zweiter Netzteil mit eigenem Material und Bild, gleiche Punkte,
    Formregler auf beiden Teilen; die Augen-UVs füllen das Augenbild."""
    from fitscan.texture import eye_faces, eye_layout

    eyes = np.full((256, 256, 3), 230, np.uint8)
    rig = build_rig(model, BodyParams(), skin=known_skin, eyes=eyes)
    on = np.logical_or.reduce(list(eye_faces(model).values()))
    assert on.sum() > 100 and (rig.eye_faces == on).all()
    uv, face_uv, eye = eye_layout(model)
    left, right = (uv[face_uv[sel]].reshape(-1, 2) for sel in eye_faces(model).values())
    assert left[:, 0].max() <= 0.5 <= right[:, 0].min()
    assert np.ptp(left, 0).max() > 0.45 and np.ptp(right, 0).max() > 0.45  # Insel füllt ihre Hälfte
    path = write_glb(rig, tmp_path / "augen.glb")
    data = path.read_bytes()
    length, _ = struct.unpack("<I4s", data[12:20])
    gltf = json.loads(data[20:20 + length])
    body, ball = gltf["meshes"][0]["primitives"]
    assert body["attributes"] == ball["attributes"] and body["targets"] == ball["targets"]
    assert gltf["accessors"][ball["indices"]]["count"] == 3 * on.sum()
    assert gltf["accessors"][body["indices"]]["count"] == 3 * (~on).sum()
    assert [m["name"] for m in gltf["materials"]] == ["Haut", "Augen"]
    assert gltf["materials"][ball["material"]]["pbrMetallicRoughness"]["baseColorTexture"] == {"index": 1}
    assert len(gltf["images"]) == len(gltf["textures"]) == 2
    obj = write_obj(rig, tmp_path / "augen.obj")
    text = obj.read_text("utf-8")
    assert "usemtl Haut" in text and "usemtl Augen" in text
    assert text.count("\nf ") == len(model.faces)
    assert "map_Kd augen_augen.jpg" in (tmp_path / "augen.mtl").read_text("utf-8")


@pytest.mark.slow
def test_scan_texture_end_to_end(tmp_path, model):
    """Künstliches Drehvideo mit bekanntem Hautbild: Prüfung, Ausrichtung je Bild, zwei Durchgänge
    mit Feinausrichtung. Das Ergebnis muss dem bekannten Hautbild am Rumpf entsprechen."""
    from fitscan.pipeline import bake_skin, best_recording, check_recordings, level_camera
    from fitscan.synthetic import example_body, film_video
    from fitscan.video import select_for_fit

    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    skin = np.full((SIZE, SIZE, 3), (200, 150, 125), np.uint8)
    skin[((xx // 32) + (yy // 32)) % 2 == 0] = (90, 60, 50)  # grobes Schachbrett wie ein großes Tattoo
    params, _ = example_body(2)
    video = film_video(model, params, tmp_path / "haut.mp4", seconds=4.0, fps=6.0, skin=skin)
    rec = best_recording(check_recordings([video], fps=6.0))
    chosen = select_for_fit(rec.frames)
    cam = level_camera(chosen, [a.obs.mask for a in chosen])
    out = bake_skin(model, params, rec, cam, tmp_path, max_frames=10, size=SIZE)
    assert out is not None and out.shape == (SIZE, SIZE, 3)
    assert (tmp_path / "haut.jpg").exists()
    views = json.loads((tmp_path / "ansichten.json").read_text("utf-8"))
    assert len(views) >= 8 and np.median([v["iou"] for v in views]) > 0.85
    m = uv_map(model, SIZE)
    torso = model.dominant_bone_mask(lambda n: n.startswith("spine"))[model.faces[m.face]].all(1)
    got = out.reshape(-1, 3)[m.texel[torso]].astype(np.float32)
    want = skin.reshape(-1, 3)[m.texel[torso]].astype(np.float32)
    assert np.median(np.abs(got - want).mean(1)) < 15


def test_white_balance_removes_cast():
    from fitscan.texture import neutralize, white_point, white_samples

    rng = np.random.default_rng(0)
    wall = np.array([238, 222, 196], np.float32)  # weiße Wand unter warmem Licht
    img = np.clip(wall * rng.uniform(0.75, 1.0, (240, 320, 1)), 0, 255).astype(np.uint8)
    person = np.zeros((240, 320), bool)
    person[60:200, 120:200] = True
    img[person] = (200, 140, 110)  # Haut zählt nicht
    white = white_point([white_samples(img, person)])
    assert white is not None and white[0] > 1.05 and white[2] < 0.9
    fixed = neutralize(img, white).astype(int)
    assert np.abs(fixed[10, 10] - fixed[10, 10].mean()).max() <= 3
    # kein heller Hintergrund: keine Korrektur
    assert white_point([white_samples(np.full((100, 100, 3), 40, np.uint8))]) is None


def test_bake_region_limits_to_part(model, known_skin):
    """Mit region trägt ein Bild nur zu diesem Teil des Körpers bei (z. B. Kopf aus der Kopfaufnahme)."""
    verts = model.rest_numpy(BodyParams())["vertices"]
    uv = model.anny.texture_coordinates.numpy()
    face_uv = model.anny.face_texture_coordinate_indices.numpy()
    intr = Intrinsics.phone_default(540, 960)
    trans = np.array([0.0, 0.85, 3.0])
    region = model.dominant_bone_mask(lambda n: n == "head").astype(np.float64)
    baker = TextureBaker(model, SIZE)
    rgb = render_textured(verts, model.faces, uv, face_uv, known_skin, 0.0, trans, intr)
    cam = to_camera(torch.as_tensor(verts), torch.tensor([0.0]), torch.as_tensor(trans, dtype=torch.float32)[None])
    cam = cam[0].numpy().astype(np.float64)
    baker.add(Shot(rgb=rgb, mask=np.ones((intr.height, intr.width), np.float32), intr=intr, to_image=np.eye(3),
                   vertices=verts, yaw=0.0, trans=trans, camera=cam, region=region))
    m = baker.map
    on_head = region[model.faces[m.face]].min(1) > 0
    assert baker.base_w[on_head].max() > SEEN_FULL
    assert baker.base_w[region[model.faces[m.face]].max(1) == 0].max() == 0
