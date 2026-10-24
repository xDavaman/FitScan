"""Die App-Clips (Winken, Rufen, Sprechen, Gehen) neu rechnen und unter src/fitscan/data/clips/ ablegen.

Aufruf:
    python scripts/make_clips.py            # alle
    python scripts/make_clips.py gehen      # nur einzelne
    python scripts/make_clips.py --glb ZIEL.glb   # dazu Annys Grundkörper mit allen Clips als GLB
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fitscan import clips
from fitscan import posing as P
from fitscan.body import BodyModel, BodyParams
from fitscan.motion import export_glb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help=", ".join(clips.NAMES))
    ap.add_argument("--glb", type=Path)
    args = ap.parse_args()
    unknown = set(args.names) - set(clips.NAMES)
    if unknown:
        ap.error(f"unbekannt: {', '.join(sorted(unknown))}")
    model = BodyModel.get()
    fig = P.Figure.from_params(model, BodyParams())
    hands = clips._hands(fig)
    for name in args.names or clips.NAMES:
        clip = getattr(clips, name)(fig, hands=hands)
        path = clip.motion.save(clips.DATA / f"{name}.npz")
        print(f"{path}: {clip.motion.frames} Bilder, {len(clip.motion.bones)} Knochen, "
              f"{len(clip.motion.face)} Gesichtsregler, Füße {clip.foot_error * 1000:.1f} mm")
    if args.glb:
        motions = [clips.load(n) for n in clips.NAMES]
        print(export_glb(model, BodyParams(), motions, args.glb))


if __name__ == "__main__":
    main()
