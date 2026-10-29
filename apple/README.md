# FitScan für iPhone

Das iPhone baut aus Größe, Gewicht, Maßband-Werten und nach Wunsch einem Drehvideo oder Fotos Annys
Körpermodell, zeigt es in 3D, lässt es winken, rufen, sprechen und gehen und gibt es als GLB an Blender und
Cascadeur weiter. Alles läuft offline
auf dem Gerät.

Ausbau in Etappen, je ein Pull Request, nach dem Merge per Kabel auf Sean-Johns iPhone:

| Etappe | Inhalt | Stand |
|---|---|---|
| 0 | Modell aus Maßen, 3D-Vorschau, GLB teilen, Verlauf | auf main |
| 1 | Startbildschirm mit Begrüßung und drehendem Modell, Maßbänder in der Vorschau, Scan-Anleitung, Willkommen beim ersten Start, Nachbearbeitung (Maße ändern, Form-Regler mit Live-Vorschau) | auf main |
| 2 | Bewegungen winken, rufen, sprechen, gehen in der Vorschau (Startbildschirm winkt zur Begrüßung), GLB mit Animationen und 52 Gesichtsreglern | in Arbeit |
| 3 | Scan aus Fotos und Videos der Mediathek: Apple Vision (Personenmaske, Körperpunkte), Form an die Umrisse angepasst, Kontrollbilder | auf main |
| 4 | Kamera-Scan: Live-Ampel mit den Regeln von `quality.py`, Ansage, Countdown, eine Drehung wird aufgenommen und wie in Etappe 3 berechnet | in Arbeit |
| 5 | Gesicht über Frontkamera und Face-ID-Sensor | geplant |

| Ordner | Inhalt |
|---|---|
| `FitScanCore/` | Swift-Paket ohne Oberfläche: Anny-Körper, Maße wie mit dem Maßband, Anpassung an Maße, GLB-Export, Personen und Verlauf |
| `FitScanCore/Sources/FitScanCore/Resources/anny_v1.fsan` | Annys Daten (CC0), geschrieben von `scripts/export_swift_data.py` |
| `FitScanCore/Sources/FitScanCore/Resources/fotoscan.json` | Einstellungen des Fotoscans (Gelenke, Regler, Gewichte), ebenfalls vom PC |
| `FitScanCore/Sources/FitScanCore/Resources/bewegungen.json`, `mimik.json` | Bewegungen aus der Bibliothek (Gelenkdrehungen, Hüfte, Mimik-Kurven) und Annys 52 Gesichtsregler, vom PC |
| `FitScanCore/Sources/FitScanCore/Resources/scanregeln.json` | Grenzwerte und Hinweistexte des Scan-Assistenten aus `quality.py`, dazu die Zeiten des Kamera-Scans, vom PC |
| `FitScanCore/Tests/` | Tests gegen die Prüfwerte vom PC (`referenz.json`) |
| `FitScanApp/` | iPhone-App (SwiftUI, SceneKit), Xcode-Projekt `FitScan.xcodeproj`, Schema `FitScan`. Neue Dateien in `FitScanApp/FitScan/` nimmt Xcode von selbst auf. |

Keine fremden Bibliotheken: nur Foundation, SwiftUI, SceneKit, Vision, AVFoundation und PhotosUI von Apple.

## Rechenweg wie am PC

| PC (`src/fitscan`) | iPhone (`FitScanCore`) |
|---|---|
| `body.BodyModel.rest` (PyTorch) | `BodyModel.rest`: Formen dünn besetzt, Knochenlagen per Procrustes |
| `measure.MeasureProbe` | `MeasureProbe`: gleiche Schnitte, konvexe Hülle, gleiche Suchbänder |
| `fit.fit_to_measurements` (Adam, Autograd) | `fitToMeasurements`: Adam mit von Hand abgeleitetem Gradienten |
| `photofit.fit_photos` (eingeschränkte Fassung von `fit.fit_to_frames`) | `fitPhotos`: Abstandsbild der Maske, Randpunkte, Körperpunkte, eigene kleine Kinematik, Gradienten von Hand (am PC gegen PyTorch geprüft) |
| `vision.py` (MediaPipe) | Apple Vision in der App: `VNGeneratePersonSegmentationRequest`, `VNDetectHumanBodyPoseRequest` |
| `quality.check_frame` | `ScanRules.check`: gleiche Grenzwerte und Texte (`scanregeln.json`); live ohne Farbstich, Socken, Kontrast und Hintergrund |
| `export.build_rig`, `write_glb` | `buildRig`, `writeGLB`: Bodengelenk, 4 Einflüsse je Punkt, 4 Formregler |
| `motion.play`, `motion.export_glb` | `MotionRig`, `previewFrames`, `writeGLB(…, clips:, face:)`: Gelenkdrehungen relativ zur Bezugshaltung, Hüfte in Hüfthöhen, glTF-Animationen, Gesichtsregler an die Gesichtsgröße angepasst |
| `store.Store` | `PersonStore`: Dokumente-Ordner der App, auch in der Dateien-App sichtbar |

Am PC geprüft (`tests/test_swiftdata.py`): Annys Daten in der App treffen den PC auf 0,01 mm. Die
Ableitungen der Anpassung weichen um 0,01 % von PyTorch ab, und ein numpy-Nachbau der Swift-Schleife
ergibt dieselben Maße wie der PC. Die Bewegungen treffen `motion.play` auf 0,01 mm (Knochen, Haut, Mimik).

## Auf dem Mac bauen und testen

Voraussetzung: Xcode 16 oder neuer. Zuletzt geprüft am 18.10.2026 mit Xcode 27 im Simulator: 9 von 9 Tests grün,
Anpassung 0,4 bis 0,8 s, das geteilte GLB ist gültig und lässt sich in Blender importieren und bewegen.

```bash
cd apple/FitScanCore
swift test -c release          # Tests gegen den PC, im Release-Modus deutlich schneller

cd ../FitScanApp
xcodebuild -project FitScan.xcodeproj -scheme FitScan \
  -destination 'generic/platform=iOS Simulator' build
```

Danach in Xcode ein beliebiges aktuelles iPhone als Simulator wählen (`xcrun simctl list devices available`
zeigt, welche installiert sind), Modell aus Maßen berechnen, 3D-Vorschau drehen, GLB teilen. Für das eigene iPhone
in Xcode unter *Signing & Capabilities* das Team des Apple-Entwicklerkontos wählen. Die Bundle-ID ist
`com.xdavaman.fitscan`.

Fotoscan (Etappe 3) am besten auf dem iPhone prüfen, weil die Personenmaske von Vision nicht in jedem Simulator
läuft: *Neuer Scan › Aus Fotos und Videos*, ein Drehvideo oder vier Fotos (vorn, beide Seiten, hinten) wählen,
Größe und Gewicht eintragen, berechnen. Das Schema startet die App als Release-Build; im Debug-Build dauert die
Anpassung ein Vielfaches.

Kamera-Scan (Etappe 4) nur auf dem iPhone (der Simulator hat keine Kamera): *Neuer Scan › Mit der Kamera*,
vordere oder hintere Kamera wählen, *Kamera starten*. Das Handy hochkant auf Hüfthöhe hinstellen, zwei Meter
zurücktreten. Die Ampel und die Ansage sagen, was noch fehlt (Licht, Abstand, Haltung). Bei Grün zählt die App
herunter, dann einmal langsam drehen; die Punkte vorn, Seite, Rücken, Seite, vorn werden abgehakt. Danach prüft die
App die Bilder und rechnet wie beim Fotoscan. Zu prüfen: Ampel springt bei Gegenlicht, abgeschnittenen Füßen und
anliegenden Armen um, die Drehung wird mit beiden Kameras erkannt, die Ansage ist bei stummgeschaltetem Handy zu hören.

Bewegungen (Etappe 2): Beim Start winkt das Modell einmal von vorn (mit weichem Übergang aus der Ruhepose und
zurück) und dreht sich danach. Im Modell unter
*Bewegungen* Winken, Rufen, Sprechen oder Gehen tippen (beim ersten Mal wird kurz gerechnet), nochmal tippen hält an.
Unter *Weitergeben › Dateien mit Bewegungen erstellen* entstehen `koerper_bewegungen.glb` und
`koerper_bewegungen_unreal.glb`; in Blender liegen die vier Bewegungen als Aktionen im Action Editor.

## TestFlight

Erst nach ausdrücklichem Ja von Sean-John zur Signierung und zum Hochladen mit seiner Apple-ID:

1. In App Store Connect die App „FitScan“ mit der Bundle-ID anlegen.
2. In Xcode *Product › Archive*, dann *Distribute App › TestFlight & App Store*.
3. Sean-John als internen Tester eintragen. Er installiert die App über die TestFlight-App. Jeder
   neue Build erscheint dort als Update.

Für jeden neuen Build `CURRENT_PROJECT_VERSION` erhöhen.

## Daten neu schreiben

Wenn sich Anny, die Regler oder der Rechenweg am PC ändern:

```bash
.venv/bin/python scripts/export_swift_data.py
.venv/bin/python -m pytest -q tests/test_swiftdata.py
```
