"""Freie Motion-Capture-Aufnahmen der CMU (mocap.cs.cmu.edu) herunterladen, nur bei der Einrichtung.

Die Daten der CMU dürfen in verkaufte Produkte, aber nicht als Daten weiterverkauft werden
(„You may include this data in commercially-sold products, but you may not resell this data
directly, even in converted form.“). FitScan überträgt sie auf Annys Skelett und liefert nur
fertige Bewegungen für das eigene Modell aus.

Aufruf:
    python scripts/hole_bewegungen.py --ziel ORDNER 02_01 09_01 ...
    python scripts/hole_bewegungen.py --ziel ORDNER --index      # Liste aller Aufnahmen (HTML)

Nur Standardbibliothek, damit es vor jeder Installation läuft.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://mocap.cs.cmu.edu"
INDEX = BASE + "/search.php?subjectnumber=%25&motion=%25"
AGENT = "FitScan-Einrichtung/1.0"


def subject_of(trial: str) -> str:
    return trial.split("_")[0]


def fetch(url: str, target: Path, tries: int = 4) -> bool:
    if target.exists() and target.stat().st_size > 0:
        return True
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            time.sleep(2 ** i)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** i)
    return False


def download(trials: list[str], dest: Path) -> dict[str, bool]:
    """ASF (Skelett je Person) und AMC (Bewegung) laden. Gibt je Aufnahme zurück, ob sie da ist."""
    done: dict[str, bool] = {}
    for trial in trials:
        subject = subject_of(trial)
        asf = dest / f"{subject}.asf"
        ok = fetch(f"{BASE}/subjects/{subject}/{subject}.asf", asf)
        ok = ok and fetch(f"{BASE}/subjects/{subject}/{trial}.amc", dest / f"{trial}.amc")
        done[trial] = ok
        print(("ok    " if ok else "fehlt ") + trial, flush=True)
    return done


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ziel", type=Path, required=True)
    ap.add_argument("--index", action="store_true", help="Liste aller Aufnahmen als index.html speichern")
    ap.add_argument("trials", nargs="*", help="Aufnahmen wie 02_01")
    args = ap.parse_args(argv)
    args.ziel.mkdir(parents=True, exist_ok=True)
    if args.index and not fetch(INDEX, args.ziel / "index.html"):
        print("Index nicht erreichbar", file=sys.stderr)
        return 1
    done = download(args.trials, args.ziel)
    return 0 if all(done.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
