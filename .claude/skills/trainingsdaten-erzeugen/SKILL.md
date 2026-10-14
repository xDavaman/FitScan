---
name: trainingsdaten-erzeugen
description: Lizenzfreie synthetische Trainingsdaten (Bilder, Masken, Körperpunkte, Maße) aus Anny erzeugen, um eigene KI-Modelle auf der RTX 5060 Ti zu trainieren. Verwenden, wenn ein eigenes Modell fremde, unpassend lizenzierte Modelle ersetzen soll.
---

# Trainingsdaten erzeugen

1. Zufällige Körper: `synthetic.example_body(seed, variation)`; Variation 1–3, Seeds fortlaufend.
2. Pro Körper Blickwinkel rendern:
   - Masken + Körperpunkte: `synthetic.film(model, params, arm_deg, views=12)`
   - Schattierte Bilder: `synthetic.render_shaded(...)` oder Drehvideo `film_video(...)`
3. Beschriftung je Beispiel speichern: Parameter (`params.to_json()`), Maße (`measure_params`), Drehung, Kamera.
4. Vielfalt erhöhen: Hintergründe, Licht, Kamerahöhe, Abstand, Armwinkel, Rauschen auf Körperpunkte.
5. Training lokal mit PyTorch und CUDA 12.8 (RTX 5060 Ti); Modelle danach nach Core ML konvertieren (`coremltools`).
6. Nur selbst erzeugte oder frei lizenzierte Daten verwenden (siehe `lizenz-pruefen`).
