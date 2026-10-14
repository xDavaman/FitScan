"""GLB in FBX (oder .blend) umwandeln, mit Blender. Läuft in Blender, nicht in FitScan selbst.

Aufruf:
    blender -b -P src/fitscan/blender_glb_to_fbx.py -- koerper.glb koerper.fbx
    blender -b -P src/fitscan/blender_glb_to_fbx.py -- koerper.glb koerper.blend

Die FBX-Einstellungen sind auf Cascadeur und Spiele-Engines abgestimmt: Maßstab 1 an der
Wurzel, keine zusätzlichen Endknochen, Y oben, Formregler als Blendshapes.
Blender ist GPL; dieses Skript ruft Blender nur auf und wird nicht mit der App ausgeliefert.
"""

from __future__ import annotations

import json
import sys

import bpy


def convert(src: str, dst: str) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src, bone_heuristic="TEMPERANCE")
    objects = list(bpy.context.scene.objects)
    armatures = [o for o in objects if o.type == "ARMATURE"]
    meshes = [o for o in objects if o.type == "MESH"]
    info = {
        "armatures": len(armatures),
        "bones": sum(len(a.data.bones) for a in armatures),
        "meshes": len(meshes),
        "vertices": sum(len(m.data.vertices) for m in meshes),
        "shape_keys": [k.name for m in meshes if m.data.shape_keys for k in m.data.shape_keys.key_blocks],
        "height_m": max((m.dimensions.z for m in meshes), default=0.0),
    }
    if dst.lower().endswith(".fbx"):
        bpy.ops.export_scene.fbx(
            filepath=dst,
            object_types={"ARMATURE", "MESH"},
            add_leaf_bones=False,
            apply_unit_scale=True,
            apply_scale_options="FBX_SCALE_ALL",
            bake_anim=False,
            use_armature_deform_only=False,
            mesh_smooth_type="FACE",
            axis_forward="-Z",
            axis_up="Y",
            primary_bone_axis="Y",
            secondary_bone_axis="X",
        )
    elif dst.lower().endswith(".blend"):
        bpy.ops.wm.save_as_mainfile(filepath=dst)
    else:
        raise SystemExit(f"Unbekanntes Zielformat: {dst}")
    return info


def main(argv: list[str]) -> None:
    args = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    if len(args) != 2:
        raise SystemExit("Aufruf: blender -b -P blender_glb_to_fbx.py -- eingabe.glb ausgabe.fbx")
    info = convert(args[0], args[1])
    print("FITSCAN_INFO " + json.dumps(info))


if __name__ == "__main__":
    main(sys.argv)
