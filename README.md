# FitScan

FitScan erstellt aus einem Handyvideo (oder Fotos) und deinen Körpermaßen ein **riggbares,
wasserdichtes 1:1-3D-Körpermodell** für Blender, Cascadeur und andere 3D-Programme.
Alles läuft lokal auf deinem Rechner, nach der Einrichtung auch ohne Internet.

Technischer Plan: https://claude.ai/code/artifact/e834733a-d72b-4a65-bdea-881002437ef9

> Stand: **Bauabschnitt 1, Windows-PC-Prototyp.** Nach bestandener Abnahme mit echten Scans geht es auf Mac und iPhone weiter.

## Schnellstart unter Windows

1. Repository herunterladen (grüner Button „Code“ → „Download ZIP“ oder `git clone`) und entpacken.
2. **`start_windows.bat` doppelklicken.**
   Beim ersten Start wird alles eingerichtet: uv, Python 3.12, PyTorch mit CUDA 12.8 (passend für die
   RTX 5060 Ti), FitScan und das Körperpunkt-Modell. Das dauert einige Minuten und braucht etwa 6 GB Platz.
   Danach öffnet sich die Oberfläche im Browser (`http://127.0.0.1:7860`).
3. Zum Ausprobieren ohne eigenes Video: Reiter **„5. Selbsttest“** → „Selbsttest starten“.
   Ein künstlicher Testkörper mit bekannten Maßen wird gefilmt und neu angepasst.

Weitere Befehle über die Startdatei, z. B. in der Eingabeaufforderung im FitScan-Ordner:

```bat
start_windows.bat live                    :: Webcam-Assistent mit Ampel, Leertaste nimmt auf
start_windows.bat pruefen video.mp4       :: nur die Aufnahme bewerten
start_windows.bat scan video.mp4 --name Sean --geschlecht m --alter 30 --groesse 182 --gewicht 84
start_windows.bat scan video.mp4 --name Sean --groesse 182 --gewicht 84 --kanten-nachschaerfen   :: unruhiger Hintergrund
start_windows.bat scan video.mp4 --kopfvideo kopf.mp4 --name Sean --groesse 182 --gewicht 84   :: mit Kopfaufnahme
start_windows.bat kopf kopf.mp4 --name Sean   :: Kopfaufnahme zum letzten Scan hinzunehmen
start_windows.bat koerper --name Sean --groesse 182 --gewicht 84 --taille 86
start_windows.bat selbsttest
start_windows.bat info                    :: Grafikkarte, Versionen, Blender gefunden?
```

## So gelingt der Scan

- Eng anliegende Kleidung, lange Haare zusammenbinden
- Ruhiger, einfarbiger Hintergrund, der sich von Kleidung und Haut abhebt
- Helles, gleichmäßiges Licht von vorn oder oben, kein Fenster hinter dir
- Etwa 2 m Abstand, Handy hochkant auf Hüfthöhe (Stativ), ganzer Körper mit Rand im Bild
- A-Pose: Arme gestreckt, etwa 30° vom Körper, Füße schulterbreit
- Einmal langsam auf der Stelle drehen (etwa 20 Sekunden)

**Kopfaufnahme (optional, für Gesicht, Ohren und Augen):** Handy hochkant etwa 40 cm vor dem
Gesicht, Kopf langsam nach links und rechts (bis die Ohren zu sehen sind), nach oben und unten
drehen, neutral schauen, Mund geschlossen, etwa 30 Sekunden. Heller, ruhiger Hintergrund hilft beim
Freistellen des Kopfumrisses.

Körpergröße und Gewicht sind Pflicht. Mit dem Maßband gemessene Umfänge (Brust, Taille, Hüfte …)
machen das Modell genauer. Der Webcam-Assistent (`live`) zeigt mit einer Ampel, ob Licht, Abstand,
Position und Haltung stimmen, und mit einem Kreis, welche Seiten schon erfasst sind.

## Was herauskommt

Jeder Scan landet in `daten/personen/<name>/scans/<datum>/`:

| Datei | Wofür |
|---|---|
| `koerper.glb` | Modell mit Skelett, Gewichten und Formreglern, Knochennamen wie Anny (`upperarm01.L`) – für Blender |
| `koerper_unreal.glb` | dasselbe mit Knochennamen im Unreal-Stil (`upperarm_l`) – für Cascadeur, Unreal, Unity |
| `koerper.obj` | nur das Netz (z. B. für 3D-Druck oder Bildhauerprogramme), mit `.mtl` und Hautbild |
| `haut.jpg` | Hautbild aus dem Video (Hautton, Tattoos, Muttermale, Kleidung) auf Annys UV-Karte, 4096 × 4096 |
| `ansichten.json` | Lage und Haltung des Körpers in jedem verwendeten Videobild |
| `feinform.json` | Feinform: Abstand der Modellränder zum Umriss im Video vorher/nachher, Verschiebung je Körperbereich |
| `kopf.json`, `kopfvergleich.jpg` | nur mit Kopfaufnahme: Gesichtsmaße, Passgenauigkeit, Foto und Modell nebeneinander |
| `augen.jpg` | nur mit Kopfaufnahme: eigenes, feines Bild der Augäpfel (zweites Material „Augen“ in GLB, FBX, OBJ) |
| `params.json`, `offsets.npy` | Körperparameter; daraus wird jeder Re-Scan weitergerechnet |
| `scan.json`, `bilder.json` | Maße, Eingaben, Qualitätsbericht, Passgenauigkeit je Bild |

Eigenschaften: echte Größe in Metern, Füße auf dem Boden, Y oben, geschlossenes Netz ohne Löcher,
A-Pose als Ruhepose, symmetrisches Skelett, Formregler `weight_up/down` und `muscle_up/down`.

**Feinform (Muskeln als Form):** Nach der groben Anpassung werden 48 Bilder rundherum genau
ausgerichtet. Dann verschiebt sich jeder Punkt des Netzes glatt entlang seiner Normalen, bis die
Randlinie des Modells in jedem Bild auf dem Umriss der Person liegt und nirgends darüber hinausragt;
Lage und Haltung jedes Bildes werden dabei mit nachgeschärft. So kommen Schulter-, Arm-, Brust- und
Rückenmuskeln, Bauch, Gesäß und Waden als echte Form ins Modell (beim Testvideo lagen die
Modellränder vorher im Mittel 8,6 mm neben dem Umriss, danach 4,7 mm). Eingegebene Maßbandwerte
bleiben dabei erhalten; Kopf, Hände und Füße bleiben unverändert. Für das Hautbild werden die Bilder
danach auf der neuen Form noch einmal ausgerichtet, damit Schrift und Tattoos nicht doppelt liegen.
Grenze: Was von keiner Seite am Rand liegt (Rinne der Wirbelsäule, Linien zwischen den
Bauchmuskeln), bleibt so glatt wie die Vorlage und ist nur als Schatten im Hautbild zu sehen. Ohne
Feinform (schneller): Häkchen entfernen oder `--ohne-feinform`.

**Hautbild:** Jedes Videobild wird einzeln ausgerichtet (Drehung, Lage, Haltung von Armen und
Beinen), dann werden die Farben auf die Körperoberfläche übertragen. Ein zweiter Durchgang schiebt
jedes Bild per optischem Fluss nach, damit Tattoos und Schrift aus allen Bildern genau
übereinanderliegen. Feinheiten kommen aus den schärfsten, frontalsten Bildern, die Grundfarbe aus
vielen. Nie gesehene Stellen (Fußsohlen, Achseln) werden aus der Umgebung aufgefüllt. Das Licht der
Aufnahme steckt im Hautbild; ein Farbstich des Lichts wird an hellen, farblosen Flächen im Hintergrund
(weiße Wand) gemessen und herausgerechnet. iPhone-Videos in HDR (Standard bei neueren iPhones)
werden in voller Farbtiefe gelesen und auf normale Helligkeit umgerechnet; sonst wären helle Haut
und Wände überstrahlt und Augenbrauen, Poren und Fältchen gingen verloren. Ohne Hautbild (schneller):
Häkchen in der Oberfläche entfernen oder `--ohne-hautbild`.

**Kopf:** Mit Kopfaufnahme sucht MediaPipe Face Landmarker (Apache 2.0) in jedem Bild 478
Gesichtspunkte. Ein Bündelausgleich über alle Bilder ergibt daraus die Gesichtsform und die Kopflage
je Bild. Annys Kopf wird glatt so verformt, dass ihr Gesicht auf diese Form passt; danach werden
Schädel und Ohren an den Kopfumriss in allen Bildern angepasst (GrabCut-Freistellung). Das Hautbild
von Gesicht, Kopfhaut und Ohren kommt scharf aus der Kopfaufnahme; nur streifend gesehene Stellen
(Hinterkopf, Scheitel) werden aus der Umgebung ergänzt. Die Augen werden mit der gemessenen Iris-
und Augenweißfarbe und in der gemessenen Blickrichtung neu gezeichnet, in einem eigenen Bild
(etwa zehnmal feiner als in Annys Hautbild). Ist der Schädel flacher oder höher als bei Anny,
gleicht der Hals aus, damit die eingegebene Körpergröße bleibt. Skelett, Netz und UV-Karte bleiben
gleich; das Modell bleibt riggbar und geschlossen. Mimik (offener Mund, Lächeln) wird übergangen; Haare sind (noch) nur Farbe
auf der Kopfhaut.

- **Blender:** Datei → Importieren → glTF 2.0 → `koerper.glb`.
- **Cascadeur:** `koerper_unreal.glb` importieren (oder FBX, siehe unten), dann Quick Rigging.
- **FBX:** Blender (kostenlos) installieren, dann im Reiter „Verlauf und Statur“ auf „FBX erstellen“
  oder `start_windows.bat fbx pfad\koerper_unreal.glb`. Liegt Blender nicht am Standardort, den Pfad zu
  `blender.exe` in der Umgebungsvariable `FITSCAN_BLENDER` angeben.
- **Statur ändern** (dünner, kräftiger, muskulöser): Reiter „Verlauf und Statur“ oder
  `start_windows.bat export <scanordner> --statur-gewicht 0.7 --statur-muskeln 0.6`.

Re-Scans unter demselben Namen starten beim letzten Scan; die Oberfläche zeigt die Veränderung der Maße.

## Stand der Genauigkeit

Gemessen an gerenderten Testvideos mit bekannten Maßen (Selbsttest, MediaPipe-Freistellung):
Umfänge im Mittel unter 1 cm daneben, Körpergröße exakt, Blickrichtung je Bild richtig erkannt.
Der Brustumfang fällt derzeit 2 bis 3 cm zu klein aus. Mit dem Maßband gemessener Brustumfang
als Eingabe gleicht das aus. Die Abnahme mit echten Handyvideos steht noch aus.

## Lizenzen

FitScan nutzt nur frei und kommerziell nutzbare Bausteine: Anny (Apache 2.0, Daten CC0),
PyTorch (BSD), MediaPipe mit Pose- und Face-Landmarker-Modell (Apache 2.0), OpenCV (Apache 2.0),
Gradio (Apache 2.0), trimesh (MIT), SciPy (BSD), PyAV (BSD) mit FFmpeg-Bibliotheken (LGPL,
unverändert als eigenständige Bibliotheken) für HDR-Videos.
Kein SMPL. Blender wird nur als eigenständiges Programm aufgerufen. Die Modelle, die du mit
FitScan erzeugst, gehören dir und können in andere Projekte übernommen werden.
Prüfung: `python scripts/check_licenses.py`.

## Entwicklung

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv/bin/python -e ".[prototyp,dev]"
.venv/bin/python -m pytest -q
```

Hinweise für Claude und die wiederkehrenden Abläufe stehen in [`CLAUDE.md`](CLAUDE.md) und `.claude/skills/`.
