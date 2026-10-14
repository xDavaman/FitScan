import numpy as np
import pytest

from fitscan import video as V

av = pytest.importorskip("av")


def _hlg_planes(lin709: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lineare BT.709-Farben (H,W,3; 1 = Referenzweiß) -> 10-Bit-YUV-Ebenen wie im iPhone-Video."""
    lin = lin709 @ np.linalg.inv(V.BT2020_TO_709.astype(np.float64)).T * V._HLG_WHITE
    lin = np.clip(lin, 0, 1)
    e = np.where(lin <= 1 / 12, np.sqrt(3 * lin), V._HLG_A * np.log(np.maximum(12 * lin - V._HLG_B, 1e-12)) + V._HLG_C)
    y = 0.2627 * e[..., 0] + 0.6780 * e[..., 1] + 0.0593 * e[..., 2]
    cb, cr = (e[..., 2] - y) / 1.8814, (e[..., 0] - y) / 1.4746
    to = lambda a, off, span: np.round(off + span * a).astype(np.uint16)  # noqa: E731
    return to(y, 64, 876), to(cb[::2, ::2], 512, 896), to(cr[::2, ::2], 512, 896)


def _expected(lin709: np.ndarray) -> np.ndarray:
    white = V.HDR_PEAK[V.HLG]
    peak = lin709.max(-1, keepdims=True)
    x = np.clip(lin709 * (1 + peak / white**2) / (1 + peak), 0, 1)
    return 255 * np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def test_hdr_to_srgb_keeps_bright_skin_and_grey():
    greys = [0.05, 0.3, 1.0, 2.0, 3.0, 3.7]
    skin = [2.0, 1.25, 1.0]  # helle Haut im Fensterlicht: über Referenzweiß, in OpenCV überstrahlt
    cols = np.array([[g, g, g] for g in greys] + [skin], np.float64)
    img = np.repeat(cols, 4, 0)[None].repeat(4, 0)  # je Farbe ein Block von 4x4 Pixeln
    out = V.hdr_to_srgb(*_hlg_planes(img)).astype(float)
    got = out[2, 2::4]  # Blockmitte
    want = _expected(cols)
    assert np.abs(got - want).max() <= 3
    grey = got[: len(greys)]
    assert np.abs(grey - grey.mean(1, keepdims=True)).max() <= 2  # neutral bleibt neutral
    assert (np.diff(grey[:, 1]) > 0).all()  # heller bleibt heller
    assert 185 <= grey[2, 1] <= 200  # Referenzweiß (Wand) hell, aber nicht abgeschnitten
    assert got[-1].max() < 245 and got[-1, 0] > got[-1, 1] > got[-1, 2]  # Haut behält Zeichnung und Farbton


def _write_hlg(path, frames: list[np.ndarray], fps: int = 10):
    out = av.open(str(path), "w")
    stream = out.add_stream("ffv1", rate=fps)
    h, w = frames[0].shape[:2]
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p10le"
    cc = stream.codec_context
    cc.color_trc, cc.color_primaries, cc.colorspace, cc.color_range = V.HLG, 9, 9, 1
    for lin in frames:
        frame = av.VideoFrame(w, h, "yuv420p10le")
        for plane, data in zip(frame.planes, _hlg_planes(lin), strict=True):
            buf = np.zeros((data.shape[0], plane.line_size // 2), np.uint16)
            buf[:, : data.shape[1]] = data
            plane.update(buf.tobytes())
        for packet in stream.encode(frame):
            out.mux(packet)
    for packet in stream.encode():
        out.mux(packet)
    out.close()


def test_read_hdr_video(tmp_path):
    path = tmp_path / "hlg.mkv"
    lin = np.full((96, 64, 3), 0.2)
    lin[:, :32] = (2.4, 1.5, 1.2)  # helle Haut links
    frames = [lin * (1 + 0.1 * k) for k in range(6)]
    _write_hlg(path, frames)
    read = list(V.read_frames(path, fps=5))
    assert [i for i, _, _ in read] == [0, 2, 4]
    rgb = read[0][2]
    assert rgb.shape == (96, 64, 3)
    assert rgb[48, 16].max() < 250  # nicht überstrahlt
    assert rgb[48, 16, 0] > rgb[48, 16, 1] > rgb[48, 16, 2]
    assert abs(int(rgb[48, 48, 0]) - int(rgb[48, 48, 2])) <= 2
    full = dict(V.read_full_frames(path, {1, 4}))
    assert sorted(full) == [1, 4]
    assert np.abs(full[4].astype(int) - read[2][2].astype(int)).max() <= 1
