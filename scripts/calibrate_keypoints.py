"""Eicht die MediaPipe-Körperpunkte auf Annys Gelenke (einmalig, Ergebnis wird eingecheckt).

MediaPipe setzt z. B. den Schulterpunkt höher als Annys Schultergelenk. Mit künstlich gerenderten
Testkörpern (bekannte Gelenke) wird je Körperpunkt ein fester Versatz im Körperkoordinatensystem
gelernt, relativ zur Körpergröße, links und rechts spiegelgleich.

Aufruf: python scripts/calibrate_keypoints.py [--koerper 6] [--ansichten 16]
Schreibt src/fitscan/data/keypoint_offsets.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from fitscan.body import BodyModel, axis_angle_to_matrix
from fitscan.camera import MEDIAPIPE_TO_BONE, Intrinsics, project, to_camera
from fitscan.synthetic import example_body, render_shaded
from fitscan.vision import PoseAnalyzer

OUT = Path(__file__).resolve().parents[1] / "src" / "fitscan" / "data" / "keypoint_offsets.json"
MIRROR = {11: 12, 13: 14, 15: 16, 23: 24, 25: 26, 27: 28}


def collect(model: BodyModel, bodies: int, views: int):
    analyzer = PoseAnalyzer(video_mode=False)
    intr = Intrinsics.phone_default(540, 960)
    rows = []
    for seed in range(100, 100 + bodies):
        params, extra = example_body(seed, 1.5)
        with torch.no_grad():
            rest = model.rest(model.phenotype_tensor(params), model.local_tensor(params), None, params.scale)
            pose = torch.eye(4)[None, None].repeat(1, len(model.bone_labels), 1, 1)
            for bone, sign in (("upperarm01.L", 1.0), ("upperarm01.R", -1.0)):
                aa = torch.tensor([[0.0, sign * math.radians(extra["arm_deg"]), 0.0]])
                pose[0, model.bone(bone)] = axis_angle_to_matrix(aa)[0]
            posed = model.posed(rest, pose, params.scale)
            verts = posed["vertices"][0].numpy()
            heads = posed["bone_heads"][0].numpy()
        height = float(verts[:, 2].max() - verts[:, 2].min())
        zc = (verts[:, 2].max() + verts[:, 2].min()) / 2
        for v in range(views):
            yaw = 2 * math.pi * v / views + 0.05
            trans = np.array([0.0, zc, 2.2])
            obs = analyzer.analyze(render_shaded(verts, model.faces, yaw, trans, intr))
            if obs.keypoints is None:
                continue
            for mp_idx, bone in MEDIAPIPE_TO_BONE.items():
                x, y, vis = obs.keypoints[mp_idx]
                if vis > 0.5:
                    rows.append((mp_idx, heads[model.bone(bone)], height, yaw, trans, (x, y)))
        print(f"Körper {seed}: {len(rows)} Punkte bisher")
    analyzer.close()
    return rows, intr


def fit_offsets(rows, intr) -> dict[int, list[float]]:
    keys = sorted({r[0] for r in rows})
    delta = torch.zeros(len(keys), 3, requires_grad=True)
    k_idx = torch.tensor([keys.index(r[0]) for r in rows])
    head = torch.tensor(np.array([r[1] for r in rows]), dtype=torch.float32)
    height = torch.tensor([r[2] for r in rows], dtype=torch.float32)[:, None]
    yaw = torch.tensor([r[3] for r in rows], dtype=torch.float32)
    trans = torch.tensor(np.array([r[4] for r in rows]), dtype=torch.float32)
    target = torch.tensor(np.array([r[5] for r in rows]), dtype=torch.float32)

    def error(d):
        pts = head + d[k_idx] * height
        cam = torch.stack([to_camera(p[None], y[None], t[None])[0, 0] for p, y, t in zip(pts, yaw, trans, strict=True)])
        return (project(cam, intr) - target).norm(dim=-1)

    before = error(torch.zeros_like(delta)).detach()
    opt = torch.optim.Adam([delta], lr=0.003)
    for _ in range(400):
        opt.zero_grad()
        loss = error(delta).pow(2).mean() + 10.0 * delta.pow(2).sum()
        loss.backward()
        opt.step()
    after = error(delta).detach()
    out = {k: delta[i].detach().numpy() for i, k in enumerate(keys)}
    # links/rechts spiegelgleich mitteln (Anny: +X ist links)
    for left, right in MIRROR.items():
        if left in out and right in out:
            mirrored = out[right] * np.array([-1.0, 1.0, 1.0])
            avg = (out[left] + mirrored) / 2
            out[left], out[right] = avg, avg * np.array([-1.0, 1.0, 1.0])
    for k in keys:
        sel = k_idx == keys.index(k)
        print(f"Punkt {k:2d} ({MEDIAPIPE_TO_BONE[k]:14s}): Abweichung {before[sel].mean():5.1f} px -> "
              f"{after[sel].mean():5.1f} px, Versatz {np.round(out[k] * 100, 2)} % der Größe")
    return {k: [round(float(x), 5) for x in v] for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--koerper", type=int, default=6)
    ap.add_argument("--ansichten", type=int, default=16)
    args = ap.parse_args()
    model = BodyModel.get("cpu")
    rows, intr = collect(model, args.koerper, args.ansichten)
    offsets = fit_offsets(rows, intr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"einheit": "Anteil der Körpergröße, Anny-Koordinaten (x links, y hinten, z oben)",
                               "versatz": {str(k): v for k, v in offsets.items()}}, indent=2) + "\n", "utf-8")
    print(f"Geschrieben: {OUT}")


if __name__ == "__main__":
    main()
