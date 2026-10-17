"""Vorschauvideo einer Übung mit Haut und Geräten. Läuft in Blender.

Aufruf:
    blender -b -P src/fitscan/blender_preview.py -- koerper.glb uebung.npz video.mp4 [standbild.png]

uebung.npz enthält die verformten Punkte je Bild in Annys Reihenfolge (vertices, F×V×3, Boden z = 0),
die Geräte als Zylinder je Bild (cylinders, F×K×7: Mitte, Achse×halbe Länge, Radius) und Quader
(boxes, F×K×15: Mitte, Drehung 3×3, halbe Kanten; z. B. eine Bank), Bildrate (fps), Untertitel
(captions: Start, Ende, Text), optional Bilder für ein Standbild (stills) und je Bild Blickpunkt
und Kamerarichtung (views, F×2×3). Video "-": nur die Standbilder.
Die Haut kommt aus dem GLB, die Form aus den Punkten. Blender wird nur als Programm aufgerufen.
"""

from __future__ import annotations

import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector, kdtree


def load_body(glb: str, rest: np.ndarray):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb, merge_vertices=False)
    mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH")
    for o in list(bpy.context.scene.objects):
        if o.type == "ARMATURE":
            mesh.parent = None
            mesh.matrix_world = o.matrix_world @ mesh.matrix_world
            bpy.data.objects.remove(o)
    for mod in list(mesh.modifiers):
        mesh.modifiers.remove(mod)
    mesh.shape_key_clear()
    # Blender-Punkt -> Anny-Punkt über die Lage in der Ruhepose
    tree = kdtree.KDTree(len(rest))
    for i, p in enumerate(rest):
        tree.insert(Vector(p), i)
    tree.balance()
    index = np.array([tree.find(mesh.matrix_world @ v.co)[1] for v in mesh.data.vertices])
    mesh.matrix_world.identity()
    return mesh, index


def build_scene(n_cylinders: int, fps: int, frames: int, n_boxes: int = 0):
    sc = bpy.context.scene
    sc.render.fps = fps
    sc.frame_start, sc.frame_end = 0, frames - 1
    metal = bpy.data.materials.new("Metall")
    metal.use_nodes = True
    bsdf = metal.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.08, 0.08, 0.09, 1)
    bsdf.inputs["Metallic"].default_value = 0.8
    bsdf.inputs["Roughness"].default_value = 0.35
    cylinders = []
    for _ in range(n_cylinders):
        bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=2, vertices=64)
        c = bpy.context.object
        c.data.materials.append(metal)
        bpy.ops.object.shade_smooth()
        cylinders.append(c)
    pad = bpy.data.materials.new("Polster")
    pad.use_nodes = True
    pad.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.05, 0.05, 0.06, 1)
    pad.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.6
    boxes = []
    for _ in range(n_boxes):
        bpy.ops.mesh.primitive_cube_add(size=2)
        b = bpy.context.object
        b.data.materials.append(pad)
        bevel = b.modifiers.new("Kante", "BEVEL")
        bevel.width = 0.02
        bevel.segments = 3
        boxes.append(b)
    bpy.ops.mesh.primitive_plane_add(size=12)
    floor = bpy.context.object
    fm = bpy.data.materials.new("Boden")
    fm.use_nodes = True
    fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.32, 0.33, 0.35, 1)
    fm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.8
    floor.data.materials.append(fm)
    sun = bpy.data.objects.new("Sonne", bpy.data.lights.new("Sonne", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (0.7, 0.2, 0.9)
    sc.collection.objects.link(sun)
    fill = bpy.data.objects.new("Licht", bpy.data.lights.new("Licht", "AREA"))
    fill.data.energy = 400
    fill.data.size = 3
    fill.location = (-2.5, -3.0, 2.5)
    fill.rotation_euler = (0.9, 0, -0.7)
    sc.collection.objects.link(fill)
    world = bpy.data.worlds.new("Welt")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.58, 0.62, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    sc.world = world
    cam = bpy.data.objects.new("Kamera", bpy.data.cameras.new("Kamera"))
    cam.data.lens = 35
    sc.collection.objects.link(cam)
    sc.camera = cam
    sc.render.engine = "BLENDER_EEVEE"
    sc.eevee.use_soft_shadows = True
    sc.view_settings.view_transform = "Standard"
    return cylinders, boxes, cam


def add_caption(cam, captions, fps):
    body = bpy.data.curves.new("Text", "FONT")
    text = bpy.data.objects.new("Text", body)
    bpy.context.scene.collection.objects.link(text)
    text.parent = cam
    text.location = (0, -0.33, -1.0)
    body.size = 0.035
    body.align_x = "CENTER"
    mat = bpy.data.materials.new("Schrift")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (1, 1, 1, 1)
    out = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emit.outputs[0], out.inputs[0])
    body.materials.append(mat)

    def caption_at(frame):
        t = frame / fps
        for start, end, txt in captions:
            if float(start) <= t < float(end):
                return str(txt)
        return ""

    return body, caption_at


def main(argv):
    glb, data_path, video = argv[:3]
    still = argv[3] if len(argv) > 3 else None
    d = np.load(data_path, allow_pickle=True)
    verts, cyl = d["vertices"], d["cylinders"]
    box = d["boxes"] if "boxes" in d.files else np.zeros((len(verts), 0, 15))
    fps = int(d["fps"])
    mesh, index = load_body(glb, d["rest"])
    cylinders, boxes, cam = build_scene(cyl.shape[1], fps, len(verts), box.shape[1])
    caption, caption_at = add_caption(cam, list(d["captions"]), fps)
    # Ganzen Ablauf samt Geräten ins Bild nehmen: Mitte des Bereichs, Abstand nach Höhe.
    points = np.concatenate([verts.reshape(-1, 3), cyl[..., :3].reshape(-1, 3)])
    low, high = points.min(axis=0), points.max(axis=0)
    target = (low + high) / 2
    direction = np.array(d["camera"]) if "camera" in d else np.array([2.6, -3.0, 0.6])
    half_fov = np.arctan(36 * 9 / 16 / 2 / cam.data.lens)
    distance = 1.2 * (high[2] - low[2]) / 2 / np.tan(half_fov) + 0.5
    cam_offset = direction / np.linalg.norm(direction) * max(distance, float(np.linalg.norm(direction)))
    views = d["views"] if "views" in d.files else None

    def update(scene, *_):
        f = min(max(scene.frame_current, 0), len(verts) - 1)
        mesh.data.vertices.foreach_set("co", verts[f][index].astype(np.float32).ravel())
        mesh.data.update()
        for k, c in enumerate(cylinders):
            centre, axis, r = cyl[f, k, :3], cyl[f, k, 3:6], cyl[f, k, 6]
            length = float(np.linalg.norm(axis))
            c.location = Vector(centre)
            c.rotation_euler = Vector((0, 0, 1)).rotation_difference(Vector(axis / length)).to_euler()
            c.scale = (r, r, length)
        for k, b in enumerate(boxes):
            m = Matrix(box[f, k, 3:12].reshape(3, 3).tolist()).to_4x4()
            b.matrix_world = Matrix.Translation(Vector(box[f, k, :3])) @ m @ Matrix.Diagonal(
                Vector(list(box[f, k, 12:15]) + [1.0]))
        look, offset = (views[f, 0], views[f, 1]) if views is not None else (target, cam_offset)
        cam.location = Vector(look + offset)
        cam.rotation_euler = (Vector(look) - cam.location).to_track_quat("-Z", "Y").to_euler()
        caption.body = caption_at(f)

    bpy.app.handlers.frame_change_pre.append(update)
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = 1280, 720
    if still:
        stills = [int(i) for i in d["stills"]]
        for k, f in enumerate(stills):
            sc.frame_set(f)
            sc.render.filepath = f"{still}_{k}.png"
            bpy.ops.render.render(write_still=True)
    if video == "-":
        return
    sc.render.image_settings.file_format = "FFMPEG"
    sc.render.ffmpeg.format = "MPEG4"
    sc.render.ffmpeg.codec = "H264"
    sc.render.ffmpeg.constant_rate_factor = "HIGH"
    sc.render.filepath = video
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:])
