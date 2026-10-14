"""Optionaler FBX-Export über ein installiertes Blender (kostenlos, wird nur aufgerufen).

Blender wird gesucht über die Umgebungsvariable FITSCAN_BLENDER, den Suchpfad und die üblichen
Installationsorte unter Windows und macOS.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).with_name("blender_glb_to_fbx.py")


def find_blender() -> Path | None:
    env = os.environ.get("FITSCAN_BLENDER")
    if env and Path(env).exists():
        return Path(env)
    found = shutil.which("blender")
    if found:
        return Path(found)
    candidates: list[Path] = []
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
        if base:
            candidates += sorted(Path(base, "Blender Foundation").glob("Blender*/blender.exe"), reverse=True)
    steam = Path("C:/Program Files (x86)/Steam/steamapps/common/Blender/blender.exe")
    candidates += [steam, Path("/Applications/Blender.app/Contents/MacOS/Blender")]
    for c in candidates:
        if c.exists():
            return c
    return None


def convert(glb: Path, target: Path, blender: Path | None = None, timeout: int = 600) -> dict:
    """GLB -> FBX oder .blend. Gibt Infos zum Import zurück (Knochen, Formregler, Größe)."""
    blender = blender or find_blender()
    if blender is None:
        raise FileNotFoundError(
            "Blender nicht gefunden. Blender kostenlos von blender.org installieren oder den Pfad "
            "zu blender.exe in der Umgebungsvariable FITSCAN_BLENDER angeben."
        )
    if not SCRIPT.exists():
        raise FileNotFoundError(f"Umwandlungsskript fehlt: {SCRIPT}")
    cmd = [str(blender), "-b", "--factory-startup", "-P", str(SCRIPT), "--", str(glb), str(target)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    for line in proc.stdout.splitlines():
        if line.startswith("FITSCAN_INFO "):
            return json.loads(line[len("FITSCAN_INFO "):])
    raise RuntimeError("Blender-Umwandlung fehlgeschlagen:\n" + (proc.stderr or proc.stdout)[-2000:])
