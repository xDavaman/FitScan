"""Ordnet die 478 MediaPipe-Gesichtspunkte Stellen auf Annys Kopf zu (einmalig, wird eingecheckt).

Annys Kopf wird schattiert aus mehreren Richtungen gerendert, MediaPipe sucht darauf das Gesicht,
und jeder Punkt wird auf die Oberfläche zurückgerechnet. Gemittelt über die Ansichten, in denen
die Stelle zur Kamera zeigt.

Aufruf: python scripts/calibrate_face.py
Schreibt src/fitscan/data/face_landmarks_anny.json
"""

from __future__ import annotations

import json

import numpy as np

from fitscan.body import BodyModel, BodyParams
from fitscan.camera import Intrinsics
from fitscan.head import (
    CORRESPONDENCE,
    LANDMARKS,
    FaceAnalyzer,
    closest_surface_points,
    head_vertex_mask,
    render_head,
)

VIEWS = (-20, -10, 0, 10, 20)


def main() -> None:
    model = BodyModel.get("cpu")
    verts = model.rest_numpy(BodyParams(phenotype={"gender": 0.0, "age": 0.55, "muscle": 0.5, "weight": 0.5,
                                                    "height": 0.5, "proportions": 0.5}))["vertices"]
    verts = verts.astype(np.float64)
    intr = Intrinsics.phone_default(720, 1280)
    analyzer = FaceAnalyzer(video_mode=False)
    acc = np.zeros((LANDMARKS, 3))
    weight = np.zeros(LANDMARKS)
    center = verts[head_vertex_mask(model)].mean(0)
    try:
        for yaw in VIEWS:
            img, ids, cam = render_head(model, verts, yaw, intr)
            pts, _ = analyzer.analyze(img)
            if pts is None:
                print(f"Ansicht {yaw}°: kein Gesicht erkannt")
                continue
            xs = np.clip(np.round(pts[:, 0]).astype(int), 0, intr.width - 1)
            ys = np.clip(np.round(pts[:, 1]).astype(int), 0, intr.height - 1)
            f = ids[ys, xs] - 1
            ok = f >= 0
            # Schnittpunkt des Sehstrahls mit dem getroffenen Dreieck, zurück in Anny-Koordinaten
            tri = cam[model.faces[f[ok]]]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            ray = np.stack([(pts[ok, 0] - intr.cx) / intr.focal, (pts[ok, 1] - intr.cy) / intr.focal,
                            np.ones(ok.sum())], -1)
            t = (n * tri[:, 0]).sum(1) / (n * ray).sum(1)
            hit_cam = ray * t[:, None]
            a = np.radians(yaw)
            rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            p = np.stack([hit_cam[:, 0], hit_cam[:, 2] - 0.45, -hit_cam[:, 1]], -1)  # Umkehr von render_head
            world = p @ rot + center
            facing = -(n / np.linalg.norm(n, axis=1, keepdims=True) * ray / np.linalg.norm(ray, axis=1,
                                                                                            keepdims=True)).sum(1)
            w = np.clip(facing, 0.05, 1.0) ** 2
            acc[ok] += w[:, None] * world
            weight[ok] += w
            print(f"Ansicht {yaw}°: {ok.sum()} Punkte")
    finally:
        analyzer.close()
    if (weight == 0).any():
        raise SystemExit(f"{(weight == 0).sum()} Gesichtspunkte nie getroffen")
    target = acc / weight[:, None]
    tri, bary, dist = closest_surface_points(verts, model.faces, target)
    print(f"Abstand gemittelter Punkt zur Oberfläche: Median {np.median(dist) * 1000:.1f} mm, max {dist.max() * 1000:.1f} mm")
    CORRESPONDENCE.write_text(json.dumps({
        "quelle": "scripts/calibrate_face.py, Anny 0.6.1, MediaPipe Face Landmarker",
        "dreieck": tri.astype(int).tolist(),
        "bary": np.round(bary, 5).tolist(),
    }), "utf-8")
    print(f"geschrieben: {CORRESPONDENCE}")


if __name__ == "__main__":
    main()
