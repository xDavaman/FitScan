# FitScan für iPhone

Erster TestFlight-Build: Das iPhone baut aus Größe, Gewicht und Maßband-Werten Annys Körpermodell,
zeigt es in 3D und gibt es als GLB an Blender und Cascadeur weiter. Alles läuft offline auf dem Gerät.
Der Video-Scan (Apple Vision, Anpassung an die Umrisse) kommt in einem der nächsten Builds.

| Ordner | Inhalt |
|---|---|
| `FitScanCore/` | Swift-Paket ohne Oberfläche: Anny-Körper, Maße wie mit dem Maßband, Anpassung an Maße, GLB-Export, Personen und Verlauf |
| `FitScanCore/Sources/FitScanCore/Resources/anny_v1.fsan` | Annys Daten (CC0), geschrieben von `scripts/export_swift_data.py` |
| `FitScanCore/Tests/` | Tests gegen die Prüfwerte vom PC (`referenz.json`) |
| `FitScanApp/` | iPhone-App (SwiftUI, SceneKit), Xcode-Projekt `FitScan.xcodeproj`, Schema `FitScan` |

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

Voraussetzung: Xcode 16 oder neuer.

```bash
cd apple/FitScanCore
swift test -c release          # Tests gegen den PC, im Release-Modus deutlich schneller

cd ../FitScanApp
xcodebuild -project FitScan.xcodeproj -scheme FitScan \
  -destination 'platform=iOS Simulator,name=iPhone 16' build
```

Danach im Simulator: Modell aus Maßen berechnen, 3D-Vorschau drehen, GLB teilen. Für das eigene iPhone
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
