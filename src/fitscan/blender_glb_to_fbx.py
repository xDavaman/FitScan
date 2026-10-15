"""GLB in FBX (oder .blend) umwandeln, mit Blender. Läuft in Blender, nicht in FitScan selbst.

Aufruf:
    blender -b -P src/fitscan/blender_glb_to_fbx.py -- koerper.glb koerper.fbx
    blender -b -P src/fitscan/blender_glb_to_fbx.py -- koerper.glb koerper.blend

Die FBX-Einstellungen sind auf Cascadeur und Spiele-Engines abgestimmt: Maßstab 1 an der
Wurzel, keine zusätzlichen Endknochen, Y oben, Formregler als Blendshapes. Ein Hautbild wird in
die FBX eingebettet bzw. in die .blend gepackt; die für GLB gedoppelten Punkte an UV-Nähten
werden beim Import wieder zusammengeführt (geschlossenes Netz).
Blender ist GPL; dieses Skript ruft Blender nur auf und wird nicht mit der App ausgeliefert.
"""

from __future__ import annotations

import json
import sys

import bpy


def show_textures(meshes: list) -> None:
    """Die .blend öffnet sonst in Blenders Volumenansicht (Solid): Dort ist das Modell grau, Haut
    und Augen sieht man erst in der Materialvorschau. Deshalb gleich Materialvorschau, in der
    Volumenansicht die Bildfarben, Farben wie im Video und die Ansicht auf den Körper gerichtet."""
    for mat in bpy.data.materials:
        if mat.use_nodes:
            images = [n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE"]
            if images:
                mat.node_tree.nodes.active = images[0]  # zeigt die Volumenansicht mit „Textur“
    height = max((m.dimensions.z for m in meshes), default=1.8)
    for screen in bpy.data.screens:
        if screen.name == "Sculpting":
            continue  # Modellieren bleibt grau mit Matcap, wie gewohnt
        for area in screen.areas:
            for space in area.spaces:
                if space.type != "VIEW_3D":
                    continue
                space.shading.type = "MATERIAL"
                space.shading.color_type = "TEXTURE"
                view = space.region_3d
                if view is not None:
                    view.view_location = (0.0, 0.0, height / 2)
                    view.view_distance = 2.4 * height
    scene = bpy.context.scene
    scene.display.shading.color_type = "TEXTURE"
    # Das Hautbild ist ein Foto (Licht steckt schon drin): „Standard“ zeigt seine Farben unverändert,
    # Blenders Vorgabe AgX macht die Haut blass und rosa.
    scene.view_settings.view_transform = "Standard"


def convert(src: str, dst: str) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src, bone_heuristic="TEMPERANCE", merge_vertices=True)
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
        "textures": sorted(i.name for i in bpy.data.images if i.size[0] > 0),
        "uv_layers": sum(len(m.data.uv_layers) for m in meshes),
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
            path_mode="COPY",
            embed_textures=True,
        )
    elif dst.lower().endswith(".blend"):
        show_textures(meshes)
        bpy.ops.file.pack_all()
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
