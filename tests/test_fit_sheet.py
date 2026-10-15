import numpy as np

from fitscan.fit import FrameFitResult
from fitscan.pipeline import fit_sheet


def test_fit_sheet_layout():
    tall = np.full((640, 360, 3), 245, np.uint8)
    wide = np.full((600, 400, 3), 245, np.uint8)
    results = [FrameFitResult(30.0 * i, 0.9, tall if i % 2 else wide) for i in range(8)]
    sheet = fit_sheet(results, height=200, per_row=6)
    assert sheet.shape[0] == 400  # zwei Reihen
    assert sheet.dtype == np.uint8
    assert results[0].to_json() == {"yaw_deg": 0.0, "iou": 0.9}


def test_fit_sheet_without_overlays():
    assert fit_sheet([FrameFitResult(0.0, 0.5)]) is None
