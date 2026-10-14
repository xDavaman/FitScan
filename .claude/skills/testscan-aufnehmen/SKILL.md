---
name: testscan-aufnehmen
description: Anleitung für eine saubere Testaufnahme mit Handy oder Webcam geben, oder ein künstliches Testvideo mit bekannten Maßen erzeugen. Verwenden vor der Abnahme oder wenn kein echtes Video vorliegt.
---

# Testscan aufnehmen

## Echte Aufnahme (Handy)
- Checkliste aus `quality.CHECKLIST` vorlesen: eng anliegende Kleidung, ruhiger Hintergrund, Licht von vorn, etwa 2 m Abstand, Handy hochkant auf Hüfthöhe, A-Pose, einmal langsam drehen (ca. 20 s).
- Mit Maßband messen: Größe, Gewicht, dazu Brust, Taille, Hüfte, Oberschenkel, Wade, Oberarm, Hals, Schrittlänge. Diese Werte sind die Wahrheit für die Abnahme (< 2 cm).
- Video auf den PC kopieren, dann `fitscan pruefen` und `fitscan scan`.

## Webcam (live)
- `start_windows.bat live`: Ampel und Rundum-Anzeige; Leertaste startet/stoppt die Aufnahme; Datei landet in `daten/aufnahmen/`.

## Künstlich (ohne Person)
```python
from fitscan.body import BodyModel
from fitscan.synthetic import example_body, film_video
m = BodyModel.get()
p, extra = example_body(seed=3, variation=2.5)
film_video(m, p, "test.mp4", arm_deg=extra["arm_deg"])
```
Wahre Maße: `measure.measure_params(m, p)`.
