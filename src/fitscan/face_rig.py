"""Gesichtsregler (Mimik) als Formregler: Mund, Kiefer, Zunge, Lider, Brauen, Wangen, Nase, Blick.

Annys Skelett hat im Gesicht nur die beiden Augenknochen. Die Mimik kommt deshalb als Formregler
(glTF Morph Targets, in Blender Shape Keys, in FBX Blendshapes). Jeder Regler wirkt allein, lässt
sich mit anderen mischen und ist unabhängig von der Körperbewegung.

Quelle: Annys „facial actions“ (seit Anny 0.6), gebaut auf dem Face-Units-Paket von Mika Suominen
für MakeHuman (CC0). Es sind die 52 Regler von Apple ARKit (jawOpen, mouthSmileLeft,
eyeBlinkRight, ...): Die Gesichtserkennung auf dem iPhone liefert genau diese Werte, damit kann
die App das Gesicht später live steuern. „Left“/„Right“ meint wie bei ARKit die Seite der Person.

Die Regler sind für Annys Grundkopf gebaut. Für die Person werden sie mit der Größe ihres Gesichts
skaliert (Abstand der Augenwinkel, Mund bis Kinn); die eigene Kopfform aus dem Kopfvideo bleibt,
die Regler kommen als Verschiebung obendrauf.
"""

from __future__ import annotations

import threading

import numpy as np

_LOCK = threading.Lock()
_CACHE: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}

# Gesichtspunkte (MediaPipe-Nummern), mit denen die Gesichtsgröße gemessen wird
EYE_OUTER = (33, 263)
NOSE_BRIDGE, CHIN = 168, 152


def anny_facial_actions(device: str = "cpu") -> tuple[list[str], np.ndarray, np.ndarray]:
    """Namen, Verschiebungen (A, V, 3) und Grundnetz (V, 3) von Annys Gesichtsreglern,
    in Annys Koordinaten (Meter, Z oben). Einmal pro Prozess berechnet."""
    with _LOCK:
        if device not in _CACHE:
            import anny
            import torch

            m = anny.Anny(local_changes="default", facial_actions="all").to(dtype=torch.float32)
            m.eval()
            labels = list(m.facial_action_labels)
            pose = torch.eye(4)[None, None].repeat(1, m.bone_count, 1, 1)
            with torch.no_grad():
                base = m(pose_parameters=pose)["vertices"][0].cpu().numpy()
                deltas = []
                for label in labels:
                    out = m(pose_parameters=pose, facial_actions={label: 1.0})["vertices"][0].cpu().numpy()
                    deltas.append(out - base)
            _CACHE[device] = (labels, np.stack(deltas).astype(np.float32), base.astype(np.float64))
        return _CACHE[device]


def face_size(points: np.ndarray) -> np.ndarray:
    """Breite (äußere Augenwinkel) und Höhe (Nasenwurzel bis Kinn) des Gesichts in Metern."""
    width = np.linalg.norm(points[EYE_OUTER[1]] - points[EYE_OUTER[0]])
    height = np.linalg.norm(points[NOSE_BRIDGE] - points[CHIN])
    return np.array([width, height])


def face_morphs(model, verts: np.ndarray, device: str = "cpu") -> dict[str, np.ndarray]:
    """Alle Gesichtsregler für eine Person: verts ist ihr Netz in der Ruhepose (Anny-Koordinaten,
    ohne Formregler). Gibt Name -> Verschiebung (V, 3) zurück."""
    from .head import load_correspondence, surface_points

    labels, deltas, base = anny_facial_actions(device)
    if deltas.shape[1] != len(verts):
        raise ValueError(f"Gesichtsregler passen nicht zum Netz ({deltas.shape[1]} statt {len(verts)} Punkte).")
    tri, bary = load_correspondence()
    mine = face_size(surface_points(np.asarray(verts, np.float64), model.faces, tri, bary))
    theirs = face_size(surface_points(base, model.faces, tri, bary))
    scale = float(np.sqrt(np.prod(mine / theirs)))
    return {label: (d * scale).astype(np.float32) for label, d in zip(labels, deltas, strict=True)}
