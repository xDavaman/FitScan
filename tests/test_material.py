import json
import struct

import numpy as np
import pytest
import trimesh

from fitscan import material as MAT
from fitscan.body import BodyParams
from fitscan.export import build_rig, write_glb
from fitscan.texture import to_linear, to_srgb, uv_map


@pytest.fixture(scope="module")
def rig(model):
    return build_rig(model, BodyParams(), morphs=False)


@pytest.fixture(scope="module")
def ao(rig):
    return MAT.ambient_occlusion(rig.vertices, rig.faces.astype(np.int64), directions=64)


def test_ambient_occlusion_sphere_is_open():
    s = trimesh.creation.icosphere(subdivisions=3)
    ao = MAT.ambient_occlusion(s.vertices, s.faces, directions=48, resolution=256)
    assert ao.min() > 0.9


def test_ambient_occlusion_on_body(model, rig, ao):
    """Achseln und Schritt sehen wenig vom Raum, Scheitel und Brust viel."""
    v = rig.vertices
    top = np.argmax(v[:, 1])
    assert ao[top] > 0.9
    middle = np.flatnonzero((np.abs(v[:, 0]) < 0.02) & (v[:, 1] > 0.6) & (v[:, 1] < 1.0))
    crotch = middle[np.argmin(v[middle, 1])]  # tiefster Punkt der Körpermitte zwischen den Beinen
    assert ao[crotch] < 0.6
    assert np.median(ao) > 0.7


def test_room_light_is_removed(model, rig, ao):
    """Gleichmäßige Haut, die mit Deckenlicht und Verdeckung schattiert wurde, kommt gleichmäßig zurück."""
    size = 1024
    m = uv_map(model, size)
    f = rig.faces.astype(np.int64)
    vn = rig.normals.astype(np.float64)
    shade_v = np.exp(0.35 * vn[:, 1] + 0.5 * np.log(np.clip(ao, 0.05, 1)))
    shade_t = (shade_v[f[m.face]] * m.bary).sum(1)
    albedo = to_linear(np.array([[[205, 160, 135]]], np.uint8)).reshape(3)
    img = np.zeros((size * size, 3), np.float32)
    img[m.texel] = albedo[None] * shade_t[:, None] * 0.8
    skin = to_srgb(img).reshape(size, size, 3)
    mat = MAT.skin_material(model, rig.vertices, rig.faces, skin)
    open_t = (ao[f[m.face]] * m.bary).sum(1) > 0.3  # Augäpfel und Mundraum (fast ganz verdeckt) zählen nicht
    before = to_linear(skin).reshape(-1, 3)[m.texel][open_t] @ MAT.LUMA
    after = to_linear(mat.albedo).reshape(-1, 3)[m.texel][open_t] @ MAT.LUMA
    assert np.std(np.log(after)) < 0.35 * np.std(np.log(before)), (np.std(np.log(after)), np.std(np.log(before)))
    assert mat.normal.shape == (size, size, 3) and mat.roughness.shape == (size, size)


def test_normal_map_orientation():
    """Höhe steigt nach rechts: Normale kippt nach links (rot < 128). Steigt sie im Bild nach oben
    (v wächst), kippt die Normale nach unten (grün < 128). Flach: (128, 128, 255)."""
    h = np.zeros((64, 64), np.float32)
    flat = MAT.normal_map(h, 0.5)
    assert np.all(np.abs(flat[..., :2].astype(int) - 128) <= 1) and flat[..., 2].min() >= 254
    right = MAT.normal_map(np.tile(np.arange(64, dtype=np.float32) * 0.1, (64, 1)), 0.5)
    assert right[32, 32, 0] < 120 and abs(int(right[32, 32, 1]) - 128) <= 1
    up = MAT.normal_map(np.tile((63 - np.arange(64, dtype=np.float32))[:, None] * 0.1, (1, 64)), 0.5)
    assert up[32, 32, 1] < 120 and abs(int(up[32, 32, 0]) - 128) <= 1


def test_glb_carries_relief_and_gloss(model, tmp_path):
    skin = np.full((256, 256, 3), 180, np.uint8)
    relief = np.full((256, 256, 3), (128, 128, 255), np.uint8)
    rough = np.full((256, 256), 160, np.uint8)
    path = write_glb(build_rig(model, BodyParams(), skin=skin, relief=relief, roughness=rough), tmp_path / "k.glb")
    data = path.read_bytes()
    n = struct.unpack("<I", data[12:16])[0]
    js = json.loads(data[20:20 + n])
    mat = js["materials"][0]
    assert "normalTexture" in mat and "metallicRoughnessTexture" in mat["pbrMetallicRoughness"]
    names = [img["name"] for img in js["images"]]
    assert names == ["Haut", "Haut_Relief", "Haut_Glanz"]
