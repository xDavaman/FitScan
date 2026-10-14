---
name: export-pruefen
description: Exportierte GLB/FBX/OBJ-Dateien auf Gültigkeit, Wasserdichtigkeit, Rig und Cascadeur-Regeln prüfen. Verwenden nach Änderungen an export.py oder vor einer Übergabe an Blender/Cascadeur.
---

# Export prüfen

1. Netz: `mesh_check.check_mesh(verts, faces, template)` → `ok`, 4 geschlossene Teile (Körper, 2 Augen, Mundraum), keine umgeklappten Dreiecke.
2. glTF-Validator (Khronos, offiziell):
   ```bash
   npm install --no-save gltf-validator@2.0.0-dev.3.10
   node scripts/validate_gltf.mjs pfad/koerper.glb pfad/koerper_unreal.glb   # 0 Fehler, 0 Warnungen
   ```
3. Cascadeur-Regeln: `export.cascadeur_check(build_rig(...))` muss leer sein (echte Größe, Füße auf y = 0, Maßstab 1, spiegelgleiche L/R-Gelenke, eindeutige Namen).
4. Blender (falls installiert): `python -m fitscan fbx koerper_unreal.glb` → Infos: 1 Armature, 105 Knochen, Formregler `weight_up/down`, `muscle_up/down`, Höhe in Metern.
   Ohne Blender-Programm geht es auch mit dem Python-Modul `bpy` (separate Python-3.11-Umgebung, nur zum Testen).
5. Bewegungstest in Blender: einen Armknochen um 60° drehen; nur Armpunkte bewegen sich, Füße und Kopf bleiben.
6. Tests: `pytest tests/test_export.py`.

Erwartung: Y oben, Blick +Z, Bodengelenk `ground` (anny) bzw. `root` (unreal) als Wurzel, vier Gewichte je Punkt, keine UVs.
