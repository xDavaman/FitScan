---
name: release-check
description: Abschlussprüfung vor einem Merge, einer Übergabe an Sean-John oder dem Wechsel auf Mac/iPhone. Verwenden, wenn ein Bauabschnitt fertig sein soll.
---

# Release-Check

1. Branch und Pull Request, nicht `main`. CI grün (ruff, Lizenzen, Tests, glTF-Validator).
2. Lokal: `pytest -q` (inkl. `slow`), `python -m fitscan selbsttest` → BESTANDEN.
3. Export: Skill `export-pruefen` (Validator 0/0, Cascadeur-Regeln leer, Blender-Bewegungstest).
4. Offline: Skill `offline-pruefen`.
5. Lizenzen: Skill `lizenz-pruefen`.
6. Abnahme Bauabschnitt 1 (mit echten Scans von Sean-John):
   - [ ] Modell lädt in Blender und Cascadeur, AutoPosing funktioniert
   - [ ] Netz bei allen Testscans geschlossen
   - [ ] Umfänge < ca. 2 cm vom Maßband
   - [ ] Scan-Prüfung erkennt schlechtes Licht, falsche Haltung, fehlende Seiten
   - [ ] Zwei Scans derselben Person ergeben fast dasselbe Modell (Skill `modelle-vergleichen`)
   - [ ] Modell lässt sich in ein anderes Blender-Projekt einfügen und animieren
7. README und CLAUDE.md auf Stand bringen.
