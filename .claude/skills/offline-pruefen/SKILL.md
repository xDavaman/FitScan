---
name: offline-pruefen
description: Sicherstellen, dass FitScan nach der Einrichtung ohne Internet läuft. Verwenden nach neuen Abhängigkeiten oder Modellen und vor jeder Abnahme.
---

# Offline prüfen

1. Einmalige Downloads (nur bei der Einrichtung): Python-Pakete, MediaPipe-Modelle (`vision.ensure_pose_model`, `head.ensure_face_model`, `hands.ensure_hand_model`, liegen danach in `daten/modelle/`). Anny rechnet seinen Zwischenspeicher lokal (`~/.cache/anny`).
2. Test ohne Netz:
   - Windows: WLAN/Kabel trennen, dann `start_windows.bat selbsttest`.
   - Linux: `unshare -n .venv/bin/python -m fitscan selbsttest --schnell` (als root) oder Proxy-Variablen auf eine ungültige Adresse setzen.
3. Code durchsuchen: `rg -n "urllib|requests|http" src/` → einzig erlaubt ist der Modell-Download in `vision.py`.
4. Gradio-Oberfläche läuft nur auf `127.0.0.1`; keine Freigabe (`share=False`).
