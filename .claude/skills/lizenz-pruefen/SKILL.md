---
name: lizenz-pruefen
description: Lizenzen aller Abhängigkeiten und Modelle prüfen, bevor etwas Neues eingebunden wird. Verwenden bei jeder neuen Bibliothek, jedem KI-Modell und vor Releases.
---

# Lizenz prüfen

1. `python scripts/check_licenses.py` (mit `--alle` für die komplette Liste). Rückgabe 1 = Verstoß.
2. Erlaubt: MIT, BSD, Apache 2.0, ISC, Zlib, PSF, CC0, MPL 2.0 (unverändert genutzt), LGPL nur als eigenständige Bibliothek.
3. Verboten: GPL/AGPL im ausgelieferten Code, nicht-kommerzielle Lizenzen (CC-BY-NC, „research only“), SMPL/SMPL-X/SMPL-H/MANO, `smplx`, `chumpy`, MakeHuman-/MPFB-Code.
4. KI-Modelle zusätzlich prüfen: Lizenz der Gewichte UND der Trainingsdaten (z. B. MediaPipe Apache 2.0, SAM 2 Apache 2.0).
5. Neue Pakete mit unvollständigen Metadaten von Hand prüfen und mit Begründung in `KNOWN` eintragen.
6. Werkzeuge nur auf dem Entwicklerrechner (Blender, Cascadeur) werden nicht ausgeliefert; Cascadeur Free ist nur nicht-kommerziell nutzbar.
