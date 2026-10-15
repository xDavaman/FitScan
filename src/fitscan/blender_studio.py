"""Modell mit übertragenen Bewegungen im Studio-Look. Läuft in Blender, nicht in FitScan selbst.

Aufruf (von video_motion.studio):
    blender -b -P src/fitscan/blender_studio.py -- studio.json

studio.json: glb (Modell mit glTF-Animationen), folder (Ausgabe), furniture (je Bewegung erschlossene
Bänke als Boxen in Anny-Koordinaten), render (Vorschauvideo ja/nein).

Ausgabe im Ordner:
- bewegung_studio.blend: alle Bewegungen hintereinander auf der Zeitleiste (NLA), weiße glänzende
  Figur, dunkelblaues Studio, schwarze Matte; die Haut aus dem Scan bleibt als Material „Haut“ in
  der Datei und kann im Materialfeld zurückgestellt werden.
- modell_bewegt.fbx: Modell mit allen Bewegungen (eine Aufnahme je Bewegung) für Cascadeur/Unreal.
- vorschau.mp4 (hochkant 720×1280) und je Bewegung ein Standbild.
Blender ist GPL und wird nur als Programm aufgerufen.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

GAP = 10  # Bilder Pause zwischen zwei Bewegungen


def anny_to_blender(p):
    # Anny (x links, y hinten, z oben) und Blender nach glTF-Import (Blick -Y wird zu ...) teilen
    # die Z-Achse; der glTF-Import dreht Y oben -> Z oben, so dass Anny-Koordinaten direkt passen.
    return Vector((p[0], p[1], p[2]))


def principled(name, color, rough, coat=0.0, metallic=0.0, emission=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    if "Coat Weight" in b.inputs:
        b.inputs["Coat Weight"].default_value = coat
        b.inputs["Coat Roughness"].default_value = 0.15
    if emission is not None:
        key = "Emission Color" if "Emission Color" in b.inputs else "Emission"
        b.inputs[key].default_value = (*emission, 1)
        if "Emission Strength" in b.inputs:
            b.inputs["Emission Strength"].default_value = 1.0
    return mat


def area(name, loc, energy, color, size, target):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.color, data.size = energy, color, size
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = loc
    direction = Vector(target) - Vector(loc)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return obj


def build_studio(sc):
    world = bpy.data.worlds.new("Studio")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.004, 0.012, 0.045, 1)
    bg.inputs["Strength"].default_value = 1.0
    floor_mat = principled("Studioboden", (0.006, 0.014, 0.045), 0.45)
    bpy.ops.mesh.primitive_plane_add(size=40)
    floor = bpy.context.object
    floor.name = "Boden"
    floor.data.materials.append(floor_mat)
    mat_mat = principled("Matte", (0.008, 0.008, 0.01), 0.85)
    bpy.ops.mesh.primitive_cube_add(size=1)
    matte = bpy.context.object
    matte.name = "Matte"
    matte.scale = (1.0, 2.0, 0.006)
    matte.location = (0, 0, 0.003)
    matte.data.materials.append(mat_mat)
    area("Hauptlicht", (2.2, -2.6, 3.2), 220, (1.0, 0.98, 0.95), 2.5, (0, 0, 1.0))
    area("Fuelllicht", (-2.8, -1.5, 1.8), 50, (0.85, 0.9, 1.0), 3.0, (0, 0, 1.0))
    area("Kante_links", (-1.8, 2.4, 2.4), 260, (0.25, 0.45, 1.0), 1.5, (0, 0, 1.1))
    area("Kante_rechts", (1.8, 2.4, 2.4), 260, (0.25, 0.45, 1.0), 1.5, (0, 0, 1.1))
    return floor, matte


def furniture_objects(spec, ranges):
    mat = principled("Bank", (0.02, 0.02, 0.022), 0.5)
    objs = []
    for name, boxes in spec.items():
        if name not in ranges:
            continue
        start, end = ranges[name]
        for box in boxes:
            bpy.ops.mesh.primitive_cube_add(size=2)
            o = bpy.context.object
            o.name = f"{box['name']}_{name}"
            rot = Matrix([r for r in box["rotation"]]).to_4x4()
            o.matrix_world = Matrix.Translation(anny_to_blender(box["centre"])) @ rot
            o.scale = box["half"]
            o.data.materials.append(mat)
            for frame, hidden in ((0, True), (start, False), (end, True)):
                o.hide_render = hidden
                o.hide_viewport = hidden
                o.keyframe_insert("hide_render", frame=frame)
                o.keyframe_insert("hide_viewport", frame=frame)
            objs.append(o)
    return objs


def follow_camera(sc, arm, root_bone, total):
    target = bpy.data.objects.new("Kameraziel", None)
    sc.collection.objects.link(target)
    c = target.constraints.new("COPY_LOCATION")
    c.target, c.subtarget = arm, root_bone
    c.use_z = False
    cam_data = bpy.data.cameras.new("Kamera")
    cam_data.lens = 40
    cam = bpy.data.objects.new("Kamera", cam_data)
    sc.collection.objects.link(cam)
    cam.parent = target
    cam.location = (1.5, -3.6, 1.15)
    look = bpy.data.objects.new("Blickpunkt", None)
    sc.collection.objects.link(look)
    look.parent = target
    look.location = (0, 0, 0.85)
    t = cam.constraints.new("TRACK_TO")
    t.target = look
    t.track_axis, t.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    # langsame Fahrt um die Figur wie im Vorbild
    target.rotation_euler = (0, 0, math.radians(-15))
    target.keyframe_insert("rotation_euler", frame=0)
    target.rotation_euler = (0, 0, math.radians(25))
    target.keyframe_insert("rotation_euler", frame=max(total, 1))
    sc.camera = cam
    return cam


def main(argv):
    spec = json.loads(Path(argv[argv.index("--") + 1]).read_text("utf-8"))
    folder = Path(spec["folder"])
    pre = spec.get("prefix", "")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=spec["glb"], merge_vertices=True, bone_heuristic="TEMPERANCE")
    sc = bpy.context.scene
    arm = next(o for o in sc.objects if o.type == "ARMATURE")
    meshes = [o for o in sc.objects if o.type == "MESH"]
    actions = [a for a in bpy.data.actions if a.name.split("_")[0] != "" and a.users >= 0]
    # glTF-Import: Aktionen heißen wie die Animationen (bei Blender 4.0 mit Objektnamen dahinter)
    actions = sorted(actions, key=lambda a: a.name)
    names = {a.name: a.name.rsplit("_", 1)[0] if a.name.count("_") >= 2 else a.name for a in actions}
    if arm.animation_data is None:
        arm.animation_data_create()
    ad = arm.animation_data
    ad.action = None
    for track in list(ad.nla_tracks):
        ad.nla_tracks.remove(track)
    track = ad.nla_tracks.new()
    track.name = "Bewegungen"
    start = 0
    ranges = {}
    for a in actions:
        a.use_fake_user = True
        lo, hi = a.frame_range
        strip = track.strips.new(a.name, start, a)
        strip.extrapolation = "NOTHING"
        length = int(round(hi - lo))
        key = next((k for k in spec["furniture"] if a.name.startswith(k)), names[a.name])
        ranges[key] = (start, start + length)
        start += length + GAP
    total = max(start - GAP, 1)
    sc.frame_start, sc.frame_end = 0, total
    sc.render.fps = 30
    skin_mats = [m for o in meshes for m in o.data.materials if m]
    studio_mat = principled("Studio_Weiss", (0.62, 0.64, 0.68), 0.32, coat=0.35)
    for m in skin_mats:
        m.use_fake_user = True
        if m.name.lower().startswith("haut"):
            m.name = "Haut"
    for o in meshes:
        for i in range(len(o.material_slots)):
            o.material_slots[i].material = studio_mat
    build_studio(sc)
    furniture_objects(spec["furniture"], ranges)
    root_bone = "root" if "root" in arm.data.bones else arm.data.bones[0].name
    if "pelvis" in arm.data.bones:
        root_bone = "pelvis"
    follow_camera(sc, arm, root_bone, total)
    sc.render.engine = "BLENDER_EEVEE"
    ee = sc.eevee
    ee.taa_render_samples = 32
    for attr in ("use_gtao", "use_ssr", "use_soft_shadows"):
        if hasattr(ee, attr):
            setattr(ee, attr, True)
    if hasattr(ee, "use_bloom"):
        ee.use_bloom = True
        ee.bloom_threshold, ee.bloom_intensity = 1.2, 0.03
    sc.render.resolution_x, sc.render.resolution_y = 720, 1280
    sc.view_settings.view_transform = "Filmic" if "Filmic" in [
        e.identifier for e in sc.view_settings.bl_rna.properties["view_transform"].enum_items] else "Standard"
    sc.view_settings.look = "None"
    info = {"actions": [a.name for a in actions], "frames": total, "ranges": ranges}
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / f"{pre}bewegung_studio.blend"))
    # FBX: alle Bewegungen als eigene Aufnahmen
    bpy.ops.export_scene.fbx(filepath=str(folder / f"{pre}modell_bewegt.fbx"), object_types={"ARMATURE", "MESH"},
                             add_leaf_bones=False, apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL",
                             bake_anim=True, bake_anim_use_all_actions=True, bake_anim_use_nla_strips=False,
                             bake_anim_force_startend_keying=True, bake_anim_simplify_factor=0.0,
                             use_armature_deform_only=False, mesh_smooth_type="FACE", axis_forward="-Z",
                             axis_up="Y", primary_bone_axis="Y", secondary_bone_axis="X", path_mode="COPY",
                             embed_textures=False)
    if spec.get("render", True):
        for key, (a, b) in ranges.items():
            sc.frame_set((a + b) // 2)
            sc.render.image_settings.file_format = "JPEG"
            sc.render.filepath = str(folder / f"{pre}standbild_{key}.jpg")
            bpy.ops.render.render(write_still=True)
    if spec.get("render", True) and spec.get("video", True):
        sc.render.image_settings.file_format = "FFMPEG"
        sc.render.ffmpeg.format = "MPEG4"
        sc.render.ffmpeg.codec = "H264"
        sc.render.ffmpeg.constant_rate_factor = "HIGH"
        sc.render.filepath = str(folder / f"{pre}vorschau.mp4")
        bpy.ops.render.render(animation=True)
    print("FITSCAN_STUDIO " + json.dumps(info))


if __name__ == "__main__":
    main(sys.argv)
