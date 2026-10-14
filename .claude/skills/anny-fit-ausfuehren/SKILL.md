---
name: anny-fit-ausfuehren
description: Körpermodell aus Video, Fotos oder nur Maßen erzeugen und das Ergebnis bewerten. Verwenden, wenn ein Scan durchlaufen oder die Anpassung (fit.py) geändert werden soll.
---

# Anny-Fit ausführen

1. Umgebung prüfen: `.venv/bin/python -m fitscan info` (GPU, Blender).
2. Eingaben klären: Körpergröße und Gewicht sind Pflicht; Geschlecht (m/w/d), Alter; weitere Maße optional.
3. Ausführen:
   - Video/Fotos: `python -m fitscan scan VIDEO --name NAME --geschlecht m --alter 30 --groesse 180 --gewicht 80 [--taille 85 …]`
   - Nur Maße: `python -m fitscan koerper --name NAME --groesse … --gewicht …`
4. Ergebnis bewerten (steht im Scanordner `daten/personen/<id>/scans/<zeit>/`):
   - `scan.json`: Modellmaße gegen Eingaben. Umfänge sollen < 2 cm abweichen, Größe exakt.
   - `bilder.json`: `iou` je Bild (≥ 0,85 gut; deutlich darunter = Maske oder Haltung passt nicht), `yaw_deg` muss zur Drehung im Video passen.
   - Protokoll: „Feinform … geglättet“ heißt, umgeklappte Dreiecke wurden verhindert.
5. Wenn Änderungen an `fit.py` nötig sind: zuerst `python -m fitscan selbsttest` (bekannter Testkörper) vorher/nachher vergleichen, dann `pytest -m slow tests/test_video_fit.py`.

Stellschrauben in `fit.py`: `SILHOUETTE_WEIGHT`, `OFFSET_SMOOTH`, `OFFSET_SIZE`, `VIDEO_LOCALS`, `steps`.
Keine SMPL-Modelle verwenden; Vorlage bleibt Anny.
