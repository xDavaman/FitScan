import pytest

from fitscan.cli import main


def test_info_runs(capsys):
    main(["info"])
    assert "FitScan" in capsys.readouterr().out


def test_body_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FITSCAN_HOME", str(tmp_path))
    main(["koerper", "--name", "Test", "--geschlecht", "w", "--alter", "28", "--groesse", "170",
          "--gewicht", "62"])
    out = capsys.readouterr().out
    assert "Körpergröße" in out and "170.0" in out
    assert list(tmp_path.glob("personen/test/scans/*/koerper.glb"))


def test_ui_builds():
    pytest.importorskip("gradio")
    from fitscan.ui import build

    assert build() is not None
