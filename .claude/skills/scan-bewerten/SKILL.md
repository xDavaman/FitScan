---
name: scan-bewerten
description: Eine Videoaufnahme oder Fotos vor dem Fit auf Licht, Schärfe, Abstand, Haltung und Blickwinkel prüfen und verständliche Tipps auf Deutsch geben. Verwenden, wenn Sean-John ein Video schickt oder ein Scan schlecht wurde.
---

# Scan bewerten

1. `python -m fitscan pruefen VIDEO` (oder mehrere Fotos).
2. Bericht lesen:
   - Ergebnis `gruen` = mindestens 8 von 12 Blickwinkeln sauber, `gelb` = brauchbar mit Lücken, `rot` = nichts verwendbar.
   - „Erfasste Blickwinkel“ zeigt, welche Seiten fehlen.
   - Die häufigsten Hinweise stehen mit Anzahl darunter.
3. Rückmeldung an den Nutzer: höchstens drei konkrete Punkte, wichtigster zuerst, in Alltagssprache
   (z. B. „Licht kam von hinten: Lampe hinter die Kamera stellen“).
4. Schwellen stehen in `quality.LIMITS`. Nur ändern, wenn ein echtes Video falsch bewertet wird; dann Beispiel und Messwerte (`metrics`) festhalten.

Checkliste für die Aufnahme: `quality.CHECKLIST`.
