---
name: auf-iphone-uebertragen
description: Den abgenommenen PC-Rechenweg auf iPhone und Mac übertragen (Swift, Core ML, Apple Vision, offline). Verwenden erst nach bestandener Windows-Abnahme.
---

# Auf iPhone übertragen

Voraussetzung: Abnahme Bauabschnitt 1 erfüllt (siehe CLAUDE.md). Xcode läuft nur auf dem Mac.

| PC (Python) | iPhone (Swift) |
|---|---|
| `vision.PoseAnalyzer` (MediaPipe) | Apple Vision: `VNDetectHumanBodyPose3DRequest`, `GeneratePersonSegmentationRequest` |
| `quality.check_frame` | gleiche Regeln in Swift, live auf dem Kamerabild (Ampel) |
| Anny-Vorlage + Regler | Vorlage und Formbasis als Core-ML-Modell oder Metal-Rechnung exportieren |
| `fit.fit_to_frames` (PyTorch) | gleicher Ablauf mit Metal Performance Shaders / Core ML, Startwerte aus Maßen |
| `export.write_glb` | Swift-Port des GLB-Schreibers (kein Fremdcode nötig) |
| `store.Store` | lokale Dateien im App-Container, kein Server |

Schritte:
1. Anny-Daten (CC0) und Regler als Binärdatei exportieren; Testfälle aus `tests/` als JSON mitnehmen.
2. Swift-Paket „FitScanCore“ (ohne UI) mit denselben Tests; Ergebnisse gegen PC vergleichen (Maße ±0,5 cm).
3. Kamera-App: Checkliste, Live-Ampel, Aufnahme, Fit im Hintergrund, Export/Teilen.
4. Koordinaten wie in CLAUDE.md beibehalten (glTF: Y oben, Blick +Z).
