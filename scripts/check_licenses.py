"""Lizenzprüfung: Alle installierten Pakete müssen frei und kommerziell nutzbar sein.

Verboten: GPL/AGPL (starkes Copyleft), nicht-kommerzielle Lizenzen, SMPL/SMPL-X und deren
Hilfspakete. Werkzeuge, die nur auf dem Entwicklerrechner laufen (z. B. Blender), sind nicht
installiert und werden nicht ausgeliefert.

Aufruf: python scripts/check_licenses.py [--alle]   (Rückgabe 1 bei Verstoß)
"""

from __future__ import annotations

import re
import sys
from importlib import metadata

FORBIDDEN_PACKAGES = {"smplx", "smpl", "chumpy", "smplpytorch", "mpfb", "makehuman"}

DENY = [
    (re.compile(r"\bA?GPL\b|GNU (Affero )?General Public", re.I), "GPL/AGPL"),
    (re.compile(r"non[- ]?commercial|\bNC\b|CC-BY-NC", re.I), "nicht kommerziell"),
    (re.compile(r"\bSMPL", re.I), "SMPL"),
]
LGPL = re.compile(r"\bLGPL\b|Lesser General Public", re.I)

# Pakete mit unvollständigen Metadaten, deren Lizenz von Hand geprüft wurde
KNOWN = {
    "anny": "Apache-2.0 (Code), CC0 (MakeHuman-Daten)",
    "roma": "BSD-3-Clause",
    "mediapipe": "Apache-2.0",
    "opencv-python": "Apache-2.0",
    "opencv-contrib-python": "Apache-2.0",
    "torch": "BSD-3-Clause",
    "torchvision": "BSD-3-Clause",
    "triton": "MIT",
    "certifi": "MPL-2.0 (schwaches Copyleft, unverändert genutzt)",
    "tqdm": "MPL-2.0 und MIT",
    "text-unidecode": "Artistic-1.0 (gewählt aus Artistic/GPL)",
    "warp-lang": "Apache-2.0",
}
# CUDA-Laufzeitbibliotheken sind laut NVIDIA-Lizenz frei weiterverteilbar
NVIDIA = re.compile(r"^(nvidia-|cuda-)")
OWN = {"fitscan"}


def license_text(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    parts = [meta.get("License-Expression") or "", meta.get("License") or ""]
    parts += [c for c in meta.get_all("Classifier") or [] if c.startswith("License")]
    return " | ".join(p for p in parts if p).replace("\n", " ")[:300]


def main(argv: list[str]) -> int:
    show_all = "--alle" in argv
    problems, warnings = [], []
    for dist in sorted(metadata.distributions(), key=lambda d: (d.metadata["Name"] or "").lower()):
        name = (dist.metadata["Name"] or "").lower()
        text = license_text(dist)
        if name in OWN:
            continue
        if name in FORBIDDEN_PACKAGES:
            problems.append(f"{name}: verbotenes Paket (SMPL/AGPL-Umfeld)")
            continue
        if name in KNOWN or NVIDIA.match(name):
            if show_all:
                print(f"ok   {name}: {KNOWN.get(name, 'NVIDIA-Laufzeit, weiterverteilbar')}")
            continue
        hit = next((label for rx, label in DENY if rx.search(text) and not LGPL.search(text)), None)
        if hit:
            problems.append(f"{name}: {hit} ({text})")
        elif LGPL.search(text):
            warnings.append(f"{name}: LGPL – nur als eigenständige Bibliothek nutzen ({text})")
        elif not text:
            warnings.append(f"{name}: keine Lizenzangabe in den Metadaten")
        elif show_all:
            print(f"ok   {name}: {text}")
    for w in warnings:
        print("HINWEIS", w)
    for p in problems:
        print("FEHLER ", p)
    print(f"{len(problems)} Verstöße, {len(warnings)} Hinweise")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
