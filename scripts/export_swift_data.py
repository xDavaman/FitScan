"""Schreibt Annys Daten und die PC-Prüfwerte für die Swift-App (apple/FitScanCore).

    python scripts/export_swift_data.py            # beides
    python scripts/export_swift_data.py --nur-daten

Neu ausführen, wenn sich Anny, die Regler oder der Rechenweg am PC ändern.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fitscan import swiftdata
from fitscan.body import BodyModel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "apple/FitScanCore/Sources/FitScanCore/Resources/anny_v1.fsan"
REFERENCE = ROOT / "apple/FitScanCore/Tests/FitScanCoreTests/Resources/referenz.json"
SETTINGS = ROOT / "apple/FitScanCore/Sources/FitScanCore/Resources/fotoscan.json"
CLIPS = ROOT / "apple/FitScanCore/Sources/FitScanCore/Resources/bewegungen.json"
FACE = ROOT / "apple/FitScanCore/Sources/FitScanCore/Resources/mimik.json"
SCAN_RULES = ROOT / "apple/FitScanCore/Sources/FitScanCore/Resources/scanregeln.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nur-daten", action="store_true", help="ohne Prüfwerte (die brauchen ein paar Minuten)")
    args = ap.parse_args()
    model = BodyModel.get("cpu")
    path = swiftdata.write_data(model, DATA)
    print(f"Anny-Daten: {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")
    SETTINGS.write_text(json.dumps(swiftdata.photo_settings(), ensure_ascii=False, indent=1) + "\n", "utf-8")
    print(f"Fotoscan-Einstellungen: {SETTINGS.relative_to(ROOT)}")
    SCAN_RULES.write_text(json.dumps(swiftdata.scan_rules(), ensure_ascii=False, indent=1) + "\n", "utf-8")
    print(f"Regeln des Scan-Assistenten: {SCAN_RULES.relative_to(ROOT)}")
    library, face = swiftdata.clip_library(), swiftdata.face_rig_data(model)
    for target, content in ((CLIPS, library), (FACE, face)):
        target.write_text(json.dumps(content, ensure_ascii=False, separators=(",", ":")) + "\n", "utf-8")
        print(f"{target.relative_to(ROOT)} ({target.stat().st_size / 1e3:.0f} kB)")
    if not args.nur_daten:
        ref = swiftdata.reference(model)
        anny = swiftdata.AnnyData(path)
        ref["fotoscan"] = swiftdata.photo_reference(model, anny)
        ref["bewegungen"] = swiftdata.motion_reference(model, anny, library, face)
        ref["scan"] = swiftdata.scan_reference()
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE.write_text(json.dumps(ref, ensure_ascii=False, separators=(",", ":")), "utf-8")
        print(f"Prüfwerte: {REFERENCE.relative_to(ROOT)} ({REFERENCE.stat().st_size / 1e3:.0f} kB)")
        for name, fit in ref["fits"].items():
            print(f"  {name}: {fit['pc_measures']}")
        print(f"  fotoscan: {ref['fotoscan']['fit']['measures']} (wahr {ref['fotoscan']['truth']})")


if __name__ == "__main__":
    main()
