"""Befehle für die Kommandozeile. Ohne Befehl startet die Oberfläche.

    fitscan                     Oberfläche im Browser (lokal, offline)
    fitscan live                Webcam-Assistent mit Ampel; Leertaste nimmt auf
    fitscan pruefen VIDEO       nur die Aufnahme bewerten (Licht, Position, Haltung, Blickwinkel)
    fitscan scan VIDEO|FOTOS…   Körper anpassen, exportieren, speichern
    fitscan koerper             Körper nur aus Maßen
    fitscan export SCANORDNER   gespeicherten Scan neu exportieren (optional andere Statur)
    fitscan fbx DATEI.glb       FBX über ein installiertes Blender erzeugen
    fitscan selbsttest          Testkörper filmen und neu anpassen (prüft Installation und GPU)
    fitscan info                Gerät, Grafikkarte, Versionen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .measure import MEASURES

GENDER_ALIASES = {"m": "maennlich", "männlich": "maennlich", "maennlich": "maennlich",
                  "w": "weiblich", "weiblich": "weiblich", "d": "divers", "divers": "divers"}
OPTIONAL_FLAGS = {"brust": "chest", "taille": "waist", "huefte": "hips", "oberschenkel": "thigh",
                  "wade": "calf", "oberarm": "upperarm", "hals": "neck", "schrittlaenge": "inseam"}


def _add_person_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--name", default="Ich", help="unter diesem Namen wird gespeichert")
    p.add_argument("--geschlecht", default="m", help="m, w oder d")
    p.add_argument("--alter", type=float, default=30.0)
    p.add_argument("--groesse", type=float, required=True, help="Körpergröße in cm")
    p.add_argument("--gewicht", type=float, required=True, help="Gewicht in kg")
    for flag, key in OPTIONAL_FLAGS.items():
        p.add_argument(f"--{flag}", type=float, help=f"{MEASURES[key][0]} in cm (optional)")


def _person(args):
    from .fit import Person

    gender = GENDER_ALIASES.get(args.geschlecht.lower())
    if gender is None:
        raise SystemExit("Geschlecht bitte als m, w oder d angeben.")
    measures = {"height": args.groesse, "weight": args.gewicht}
    for flag, key in OPTIONAL_FLAGS.items():
        if getattr(args, flag) is not None:
            measures[key] = getattr(args, flag)
    person = Person(gender, args.alter, measures)
    problems = person.validate()
    if problems:
        raise SystemExit("\n".join(problems))
    return person


def _print_result(result) -> None:
    print("\nMaße des Modells:")
    for key, (label, unit) in MEASURES.items():
        if key in result.measures:
            print(f"  {label:22s} {result.measures[key]:7.1f} {unit}")
    print("\nDateien:")
    for k, v in result.files.items():
        print(f"  {k:12s} {v}")
    for h in result.hints:
        print("Hinweis:", h)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fitscan", description="FitScan Windows-Prototyp")
    parser.add_argument("--geraet", default=None, help="cpu oder cuda (Standard: automatisch)")
    sub = parser.add_subparsers(dest="cmd")

    p_ui = sub.add_parser("ui", help="Oberfläche starten")
    p_ui.add_argument("--port", type=int, default=7860)
    p_ui.add_argument("--kein-browser", action="store_true")

    p_live = sub.add_parser("live", help="Webcam-Assistent")
    p_live.add_argument("--kamera", type=int, default=0)

    p_check = sub.add_parser("pruefen", aliases=["check"], help="Aufnahme bewerten")
    p_check.add_argument("quelle", nargs="+", type=Path)

    p_scan = sub.add_parser("scan", help="Körper aus Video oder Fotos")
    p_scan.add_argument("quelle", nargs="+", type=Path)
    p_scan.add_argument("--fps", type=float, default=4.0, help="ausgewertete Bilder pro Sekunde")
    p_scan.add_argument("--kanten-nachschaerfen", action="store_true",
                        help="Personenmaske nachschärfen (nur bei unruhigem Hintergrund sinnvoll)")
    p_scan.add_argument("--ohne-hautbild", action="store_true",
                        help="kein Hautbild (Textur) aus dem Video, Modell einfarbig (schneller)")
    p_scan.add_argument("--kopfvideo", type=Path,
                        help="Nahaufnahme des Kopfes (Kopf langsam drehen): Gesicht, Ohren, Augen genauer")
    _add_person_args(p_scan)

    p_head = sub.add_parser("kopf", aliases=["head"], help="Kopfvideo zum letzten Scan einer Person hinzunehmen")
    p_head.add_argument("video", type=Path, help="Nahaufnahme des Kopfes, Kopf langsam in alle Richtungen drehen")
    p_head.add_argument("--name", default="Ich", help="Person, deren letzter Scan ergänzt wird")

    p_body = sub.add_parser("koerper", aliases=["body"], help="Körper nur aus Maßen")
    _add_person_args(p_body)

    p_exp = sub.add_parser("export", help="gespeicherten Scan neu exportieren")
    p_exp.add_argument("scanordner", type=Path)
    p_exp.add_argument("--statur-gewicht", type=float, help="0 = sehr dünn, 0.5 = Durchschnitt, 1 = sehr kräftig")
    p_exp.add_argument("--statur-muskeln", type=float, help="0 = wenig, 0.5 = Durchschnitt, 1 = sehr muskulös")

    p_fbx = sub.add_parser("fbx", help="GLB mit Blender in FBX umwandeln")
    p_fbx.add_argument("glb", type=Path)
    p_fbx.add_argument("--ziel", type=Path)

    p_test = sub.add_parser("selbsttest", aliases=["selftest"], help="Testkörper filmen und anpassen")
    p_test.add_argument("--schnell", action="store_true")

    sub.add_parser("info", help="Gerät und Versionen anzeigen")

    args = parser.parse_args(argv)
    cmd = args.cmd or "ui"

    if cmd == "ui":
        from .ui import launch

        launch(port=getattr(args, "port", 7860), open_browser=not getattr(args, "kein_browser", False))
    elif cmd == "live":
        from .live import run

        path = run(camera=args.kamera)
        if path:
            print(f"Letzte Aufnahme: {path}\nWeiter mit: fitscan scan \"{path}\" --groesse … --gewicht …")
    elif cmd in ("pruefen", "check"):
        from .pipeline import check_recordings
        from .quality import sector_name

        for rec in check_recordings(list(args.quelle), log=print, keep_frames=False):
            if rec.report is None:
                continue
            report = rec.report
            print(f"\n{rec.label} – Ergebnis: {report.level} ({report.usable} von {report.frames} Bildern brauchbar)")
            print("Erfasste Blickwinkel:",
                  ", ".join(sector_name(i) for i, c in enumerate(report.covered) if c) or "keine")
            for line in report.summary:
                print(" -", line)
    elif cmd == "scan":
        from .pipeline import scan

        person = _person(args)
        source = args.quelle[0] if len(args.quelle) == 1 else list(args.quelle)
        _print_result(scan(source, person, args.name, device=args.geraet, fps=args.fps,
                           refine=args.kanten_nachschaerfen, texture=not args.ohne_hautbild,
                           head_video=args.kopfvideo, log=print))
    elif cmd in ("kopf", "head"):
        from .pipeline import add_head
        from .store import Store

        store = Store()
        person = store.find(args.name)
        latest = store.latest(person.id) if person else None
        if latest is None:
            raise SystemExit(f"Für „{args.name}“ gibt es noch keinen Scan. Erst „fitscan scan …“ ausführen.")
        _print_result(add_head(latest, args.video, store, device=args.geraet, log=print))
    elif cmd in ("koerper", "body"):
        from .pipeline import body_from_measurements

        _print_result(body_from_measurements(_person(args), args.name, device=args.geraet, log=print))
    elif cmd == "export":
        from .pipeline import reexport
        from .store import Store

        folder = args.scanordner.resolve()
        person_id = folder.parent.parent.name
        record = next((s for s in Store().scans(person_id) if s.folder.resolve() == folder), None)
        if record is None:
            raise SystemExit(f"Kein gespeicherter Scan in {folder}")
        changes = {}
        if args.statur_gewicht is not None:
            changes["weight"] = args.statur_gewicht
        if args.statur_muskeln is not None:
            changes["muscle"] = args.statur_muskeln
        for k, v in reexport(record, changes=changes or None, device=args.geraet, log=print).items():
            print(f"  {k:12s} {v}")
    elif cmd == "fbx":
        from .blender import convert

        target = args.ziel or args.glb.with_suffix(".fbx")
        info = convert(args.glb, target)
        print(f"FBX geschrieben: {target}\n{info}")
    elif cmd in ("selbsttest", "selftest"):
        from .selftest import run

        result = run(quick=args.schnell, device=args.geraet, log=print)
        print()
        print(result.text())
        sys.exit(0 if result.passed else 1)
    elif cmd == "info":
        import platform

        import torch

        from . import __version__

        print(f"FitScan {__version__}, Python {platform.python_version()}, PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"Grafikkarte: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
        else:
            print("Keine CUDA-Grafikkarte gefunden, es wird der Prozessor verwendet.")
        from .blender import find_blender

        print("Blender:", find_blender() or "nicht gefunden (nur für FBX nötig)")


if __name__ == "__main__":
    main()
