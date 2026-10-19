# FitScan für iPhone

Das iPhone baut aus Größe, Gewicht und Maßband-Werten Annys Körpermodell, zeigt es in 3D und gibt es
als GLB an Blender und Cascadeur weiter. Alles läuft offline auf dem Gerät.

Ausbau in Etappen, je ein Pull Request, nach dem Merge per Kabel auf Sean-Johns iPhone:

| Etappe | Inhalt | Stand |
|---|---|---|
| 0 | Modell aus Maßen, 3D-Vorschau, GLB teilen, Verlauf | auf main |
| 1 | Startbildschirm mit Begrüßung und drehendem Modell, Maßbänder in der Vorschau, Scan-Anleitung, Willkommen beim ersten Start, Nachbearbeitung (Maße ändern, Form-Regler mit Live-Vorschau) | in Arbeit |
| 2 | Animationen winken, rufen, sprechen, gehen (Clips aus der Bewegungsbibliothek) | geplant |
| 3 | Scan aus Fotos und Videos der Mediathek (Apple Vision, Umrisse von vorn und von der Seite) | geplant |
| 4 | Kamera-Scan mit Ampel (Regeln wie `quality.py`) | geplant |
| 5 | Gesicht über Frontkamera und Face-ID-Sensor | geplant |

| Ordner | Inhalt |
|---|---|
| `FitScanCore/` | Swift-Paket ohne Oberfläche: Anny-Körper, Maße wie mit dem Maßband, Anpassung an Maße, GLB-Export, Personen und Verlauf |
| `FitScanCore/Sources/FitScanCore/Resources/anny_v1.fsan` | Annys Daten (CC0), geschrieben von `scripts/export_swift_data.py` |
| `FitScanCore/Tests/` | Tests gegen die Prüfwerte vom PC (`referenz.json`) |
| `FitScanApp/` | iPhone-App (SwiftUI, SceneKit), Xcode-Projekt `FitScan.xcodeproj`, Schema `FitScan`. Neue Dateien in `FitScanApp/FitScan/` nimmt Xcode von selbst auf. |

Keine fremden Bibliotheken: nur Foundation, SwiftUI und SceneKit von Apple.

## Rechenweg wie am PC

| PC (`src/fitscan`) | iPhone (`FitScanCore`) |
|---|---|
| `body.BodyModel.rest` (PyTorch) | `BodyModel.rest`: Formen dünn besetzt, Knochenlagen per Procrustes |
| `measure.MeasureProbe` | `MeasureProbe`: gleiche Schnitte, konvexe Hülle, gleiche Suchbänder |
| `fit.fit_to_measurements` (Adam, Autograd) | `fitToMeasurements`: Adam mit von Hand abgeleitetem Gradienten |
| `export.build_rig`, `write_glb` | `buildRig`, `writeGLB`: Bodengelenk, 4 Einflüsse je Punkt, 4 Formregler |
| `store.Store` | `PersonStore`: Dokumente-Ordner der App, auch in der Dateien-App sichtbar |

Am PC geprüft (`tests/test_swiftdata.py`): Annys Daten in der App treffen den PC auf 0,01 mm. Die
Ableitungen der Anpassung weichen um 0,01 % von PyTorch ab, und ein numpy-Nachbau der Swift-Schleife
ergibt dieselben Maße wie der PC.

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
