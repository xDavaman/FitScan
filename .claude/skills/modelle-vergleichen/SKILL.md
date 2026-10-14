---
name: modelle-vergleichen
description: Zwei Scans derselben Person oder Modell gegen Maßband vergleichen (Wiederholgenauigkeit, Veränderung über Zeit). Verwenden für Re-Scans und die Abnahme „zwei Scans ergeben fast dasselbe Modell“.
---

# Modelle vergleichen

1. Scans finden: `Store().scans(person_id)`; jeder Scan hat `measures`, `inputs`, `params()`.
2. Maße vergleichen: `store.compare(alt.measures, neu.measures)` → (Name, alt, neu, Differenz).
3. Form vergleichen (gleiche Topologie, daher Punkt für Punkt):
   ```python
   from fitscan.body import BodyModel
   m = BodyModel.get()
   a = m.rest_numpy(scan_a.params())["vertices"]; b = m.rest_numpy(scan_b.params())["vertices"]
   import numpy as np; d = np.linalg.norm(a - b, axis=1) * 100   # cm
   print(np.median(d), np.percentile(d, 95))
   ```
4. Bewertung Wiederholgenauigkeit: Umfänge innerhalb ±1 cm, 95 % der Punkte < 1,5 cm. Größere Abweichungen: Aufnahmen mit `scan-bewerten` prüfen.
5. In der Oberfläche: Reiter „Verlauf und Statur“ zeigt die Veränderung zum vorherigen Scan.
