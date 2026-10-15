from pathlib import Path

import pytest

from fitscan.pipeline import best_recording, check_recordings, source_groups
from fitscan.synthetic import example_body, film_video
from fitscan.video import read_frames, read_images


def test_groups_split_videos_and_photos():
    paths = [Path("a.mov"), Path("b.MP4"), Path("c.jpg"), Path("d.PNG")]
    assert source_groups(paths) == [[Path("a.mov")], [Path("b.MP4")], [Path("c.jpg"), Path("d.PNG")]]
    with pytest.raises(ValueError, match="nicht unterstützt"):
        source_groups([Path("a.mov"), Path("notiz.txt")])


def test_unreadable_files_explain_what_to_do(tmp_path):
    bad_video = tmp_path / "IMG_0001.mov"
    bad_video.write_bytes(b"kein Video")
    with pytest.raises(ValueError, match="Maximale Kompatibilität"):
        list(read_frames(bad_video))
    heic = tmp_path / "IMG_0002.HEIC"
    heic.write_bytes(b"kein Foto")
    with pytest.raises(ValueError, match="HEIC"):
        list(read_images([heic]))


def test_several_files_use_the_best_recording(tmp_path, model):
    """Wie am PC passiert: zwei Videos ausgewählt, eines davon nicht lesbar."""
    bad = tmp_path / "kaputt.mov"
    bad.write_bytes(b"kein Video")
    params, _ = example_body(1)
    good = film_video(model, params, tmp_path / "drehung.mp4", seconds=3.0, fps=4.0)
    lines = []
    recs = check_recordings([bad, good], fps=2.0, log=lines.append)
    assert recs[0].report is None and "Maximale Kompatibilität" in recs[0].error
    assert recs[1].report.frames > 0 and recs[1].frames
    assert best_recording(recs, lines.append).paths == [good]
    assert any("Verwendet wird drehung.mp4" in line for line in lines)
    with pytest.raises(ValueError, match="Maximale Kompatibilität"):
        best_recording(check_recordings([bad, bad], log=lines.append))
