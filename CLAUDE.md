# FitScan – Hinweise für Claude

FitScan baut aus einem Handyvideo (oder Fotos) plus eingegebenen Körpermaßen ein riggbares,
wasserdichtes 1:1-Körpermodell für Blender, Cascadeur und andere 3D-Programme.
Besitzer: Sean-John Davis. Mit ihm auf Deutsch sprechen, Oberflächentexte immer auf Deutsch.

Technischer Plan: https://claude.ai/code/artifact/e834733a-d72b-4a65-bdea-881002437ef9

## Feste Regeln

- **Keine SMPL/SMPL-X-Modelle, kein `smplx`, kein `chumpy`**, auch nicht Annys „smplx“-Topologie.
  Grund: nicht kommerziell bzw. teuer lizenziert. Körpervorlage ist **Anny** (Code Apache 2.0,
  MakeHuman-Daten CC0).
- MakeHuman-/MPFB-**Code** (AGPL/GPL) nicht einbinden. Blender (GPL) nur als externes Programm aufrufen.
- Jede neue Abhängigkeit muss `python scripts/check_licenses.py` bestehen (frei, kommerziell nutzbar).
- Läuft **offline**: Netz nur bei der Einrichtung (Pakete, MediaPipe-Modell). Keine Cloud-Aufrufe zur Laufzeit.
- Nie direkt auf `main` arbeiten: eigener Branch, Pull Request, CI grün.
- Reihenfolge: erst Windows-PC-Prototyp abnehmen, dann Mac/iPhone (Swift, Core ML, Apple Vision).

## Aufbau (`src/fitscan`)

| Datei | Aufgabe |
|---|---|
| `body.py` | Anny laden (`BodyModel.get`), `BodyParams` (Phänotyp, lokale Regler, Maßstab, Feinform-Offsets) |
| `measure.py` | Maße wie mit dem Maßband (konvexe Hülle von Schnitten), differenzierbar über `MeasureProbe` |
| `fit.py` | Anpassung an Maße (`fit_to_measurements`) und an Videobilder (`fit_to_frames`) |
| `camera.py` | Lochkamera, Anny→Kamera, Silhouetten-Rendering |
| `vision.py` | MediaPipe Pose (Körperpunkte, Maske, Blickrichtung); auf dem iPhone: Apple Vision |
| `quality.py` | Scan-Assistent: Licht, Schärfe, Abstand, Position, A-Pose, Stillhalten, Blickwinkel |
| `video.py` | Video/Fotos einlesen, prüfen, bestes Bild pro Blickwinkel wählen |
| `mesh_check.py` | wasserdicht, Orientierung, kaputte oder umgeklappte Dreiecke |
| `export.py` | eigener GLB-Exporter (Skelett, Gewichte, Formregler), OBJ, Cascadeur-Prüfung, Namensschemata |
| `blender.py`, `blender_glb_to_fbx.py` | optional FBX/.blend über installiertes Blender |
| `store.py` | Personen und Scan-Verlauf unter `daten/personen/…`, Vergleich von Scans |
| `pipeline.py` | Gesamtablauf `scan()`, `body_from_measurements()`, `reexport()` |
| `synthetic.py`, `selftest.py` | künstliche Testkörper und -videos mit bekannten Maßen |
| `ui.py`, `live.py`, `cli.py` | Gradio-Oberfläche, Webcam-Ampel, Befehle |

## Konventionen

- Anny: Meter, Z oben, Person blickt nach −Y, linke Körperseite +X.
- Kamera: x rechts, y unten, z vom Objektiv weg; `to_camera`: (x, −z, y) nach Drehung um Z.
  Drehung 0 = Person schaut in die Kamera, +90° = rechte Körperseite sichtbar.
- glTF-Export: Y oben, Blick +Z, Füße auf y = 0, Maßstab 1, zusätzliches Bodengelenk als Wurzel.
- Namensschemata: `anny` (z. B. `upperarm01.L`) und `unreal` (z. B. `upperarm_l`, für Cascadeur/Unreal).
- Maße in cm, Gewicht in kg, intern Meter.

## Arbeiten

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu   # Windows: cu128
uv pip install -p .venv/bin/python -e ".[prototyp,dev]"
.venv/bin/ruff check src tests scripts
.venv/bin/python -m pytest -q -m "not slow"     # schnell
.venv/bin/python -m pytest -q                   # inkl. Videoanpassung (einige Minuten auf CPU)
.venv/bin/python -m fitscan selbsttest --schnell
.venv/bin/python scripts/check_licenses.py
```

- Unter Linux braucht MediaPipe `libegl1 libgles2 libgl1`.
- Anny legt beim ersten Start einen Zwischenspeicher unter `~/.cache/anny` an (gut 1 GB), danach schnell.
- Ohne eigenes Video testen: `synthetic.film()` (Silhouetten) oder `synthetic.film_video()` (gerendertes Drehvideo).
- Die Skills unter `.claude/skills/` beschreiben die wiederkehrenden Abläufe (Fit, Export prüfen, Scan bewerten …).

## Abnahme vor dem Wechsel auf Mac/iPhone

Modell lädt in Blender und Cascadeur (AutoPosing), Netz bei allen Testscans geschlossen, Umfänge
< ca. 2 cm vom Maßband, Scan-Prüfung erkennt schlechtes Licht/Haltung/fehlende Seiten, zwei Scans
derselben Person ergeben fast dasselbe Modell, Modell lässt sich in ein anderes Blender-Projekt
einfügen und animieren.
