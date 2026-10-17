import math

import numpy as np

from fitscan.quality import GRUEN, ROT, SECTORS, FrameQuality, sector_name, sector_of, summarize
from fitscan.store import Store, compare


def test_sectors():
    assert sector_of(0.0) == 0
    assert sector_of(math.pi / 2) == 3
    assert sector_of(-math.pi / 2) == 9
    assert sector_name(0) == "vorn" and sector_name(6) == "Rücken"


def test_summary_levels():
    full = [FrameQuality(i, GRUEN, [], {}, 2 * math.pi * i / SECTORS) for i in range(SECTORS)]
    assert summarize(full).level == GRUEN
    none = [FrameQuality(0, ROT, [], {}, None)]
    assert summarize(none).level == ROT


def test_store_rescan_and_compare(tmp_path):
    from fitscan.body import BodyParams

    store = Store(tmp_path)
    rec = store.save_person("Sean Test", "maennlich", 30, {"height": 180, "weight": 80})
    assert store.find("sean test").id == rec.id
    for waist in (90.0, 88.5):
        folder = store.new_scan_folder(rec.id)
        store.save_scan(rec.id, folder, BodyParams(), {"height": 180}, {"waist": waist, "height": 180.0},
                        {}, "masse")
    scans = store.scans(rec.id)
    assert len(scans) == 2 and store.latest(rec.id).measures["waist"] == 88.5
    rows = compare(scans[0].measures, scans[1].measures)
    assert ("Taillenumfang", 90.0, 88.5, -1.5) in rows
    assert isinstance(scans[-1].params().phenotype, dict)


def test_live_overlay_draws():
    from fitscan.live import draw_overlay

    img = np.zeros((360, 640, 3), np.uint8)
    out = draw_overlay(img, GRUEN, ["Arme gestreckt halten."], [True] * 5 + [False] * 7, True, 3.0)
    assert out.shape == img.shape and out.any()


def _standing_frame(background=(225, 225, 225), skin=(205, 150, 120), feet=None, shine=0.0):
    """Einfaches Bild einer stehenden Person (Rumpf, Beine, Füße) mit Maske und Körperpunkten."""
    from fitscan import vision as V
    from fitscan.vision import FrameObservation

    h, w = 800, 450
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = background
    mask = np.zeros((h, w), np.float32)
    for x0, x1, y0, y1 in ((175, 275, 160, 420), (185, 215, 420, 700), (235, 265, 420, 700)):
        mask[y0:y1, x0:x1] = 1
        rgb[y0:y1, x0:x1] = skin
    for x0, x1 in ((170, 222), (228, 280)):
        mask[700:740, x0:x1] = 1
        rgb[700:740, x0:x1] = feet or skin
    if shine:
        rows = int(shine * 260)
        rgb[160 : 160 + rows, 175:275] = 245
    kp = np.zeros((33, 3), np.float32)
    kp[:, 2] = 1.0
    kp[:, :2] = (225, 300)
    kp[V.NOSE, :2] = (225, 120)
    for idx, x in ((V.L_KNEE, 200), (V.R_KNEE, 250)):
        kp[idx, :2] = (x, 560)
    for idx, x in ((V.L_ANKLE, 200), (V.R_ANKLE, 250)):
        kp[idx, :2] = (x, 700)
    for idx, x in ((V.L_HEEL, 190), (V.R_HEEL, 240)):
        kp[idx, :2] = (x, 730)
    for idx, x in ((V.L_FOOT_INDEX, 212), (V.R_FOOT_INDEX, 262)):
        kp[idx, :2] = (x, 735)
    return rgb, FrameObservation(0, 0.0, w, h, 1, kp, None, mask)


def _codes(rgb, obs):
    from fitscan.quality import check_frame

    return {hint.code for hint in check_frame(rgb, obs).hints}


def test_color_cast_hint():
    assert "farbstich" not in _codes(*_standing_frame())
    # Nachtlicht: Weiß an der Wand wird grünlich-gelb (an den Küchenvideos gemessen: Rot 0,84, Blau 0,75)
    assert "farbstich" in _codes(*_standing_frame(background=(215, 235, 190)))


def test_highlight_hint():
    assert "glanzlicht" not in _codes(*_standing_frame())
    assert "glanzlicht" in _codes(*_standing_frame(shine=0.05))


def test_socks_hint():
    assert "socken" not in _codes(*_standing_frame())
    assert "socken" in _codes(*_standing_frame(feet=(240, 240, 240)))
    assert "socken" in _codes(*_standing_frame(feet=(40, 40, 40)))
    # lange Hose: Unterschenkel ohne Haut, dann keine Aussage über die Füße
    rgb, obs = _standing_frame(feet=(240, 240, 240))
    rgb[420:700] = (60, 60, 70)
    assert "socken" not in _codes(rgb, obs)


def test_sector_names_unique():
    names = [sector_name(i) for i in range(SECTORS)]
    assert len(set(names)) == SECTORS
