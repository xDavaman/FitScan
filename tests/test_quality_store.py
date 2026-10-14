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
