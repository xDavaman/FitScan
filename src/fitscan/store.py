"""Personen und ihre Scans speichern, Re-Scans vergleichen. Alles lokal, nichts geht ins Netz.

Ablage (unter FITSCAN_HOME, Standard ./daten):
    personen/<id>/person.json
    personen/<id>/scans/<zeitstempel>/params.json, offsets.npy, scan.json, koerper.glb ...
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
import unicodedata
from pathlib import Path

from .body import BodyParams
from .measure import MEASURES
from .vision import fitscan_home


@dataclasses.dataclass
class PersonRecord:
    id: str
    name: str
    gender: str
    age_years: float
    created: str
    measures: dict[str, float] = dataclasses.field(default_factory=dict)  # zuletzt eingegeben


@dataclasses.dataclass
class ScanRecord:
    person_id: str
    id: str
    created: str
    folder: Path
    source: str
    inputs: dict[str, float]
    measures: dict[str, float]
    quality: dict
    files: dict[str, str]

    def params(self) -> BodyParams:
        return BodyParams.load(self.folder)


def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "person"


class Store:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else fitscan_home() / "personen"
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Personen
    def persons(self) -> list[PersonRecord]:
        out = []
        for f in sorted(self.root.glob("*/person.json")):
            out.append(PersonRecord(**json.loads(f.read_text("utf-8"))))
        return out

    def person(self, person_id: str) -> PersonRecord:
        f = self.root / person_id / "person.json"
        if not f.exists():
            raise KeyError(f"Person nicht gefunden: {person_id}")
        return PersonRecord(**json.loads(f.read_text("utf-8")))

    def find(self, name: str) -> PersonRecord | None:
        for p in self.persons():
            if p.name.lower() == name.lower() or p.id == name:
                return p
        return None

    def save_person(self, name: str, gender: str, age_years: float,
                    measures: dict[str, float] | None = None) -> PersonRecord:
        existing = self.find(name)
        if existing:
            pid, created = existing.id, existing.created
        else:
            base, n = _slug(name), 1
            pid = base
            while (self.root / pid).exists():
                n += 1
                pid = f"{base}-{n}"
            created = dt.datetime.now().isoformat(timespec="seconds")
        rec = PersonRecord(pid, name, gender, float(age_years), created,
                           {k: float(v) for k, v in (measures or {}).items() if v})
        (self.root / pid).mkdir(parents=True, exist_ok=True)
        (self.root / pid / "person.json").write_text(
            json.dumps(dataclasses.asdict(rec), indent=2, ensure_ascii=False), "utf-8")
        return rec

    # ------------------------------------------------------------------ Scans
    def new_scan_folder(self, person_id: str) -> Path:
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = self.root / person_id / "scans" / stamp
        n = 1
        while folder.exists():
            n += 1
            folder = folder.with_name(f"{stamp}_{n}")
        folder.mkdir(parents=True)
        return folder

    def save_scan(self, person_id: str, folder: Path, params: BodyParams, inputs: dict[str, float],
                  measures: dict[str, float], quality: dict, source: str,
                  files: dict[str, str] | None = None) -> ScanRecord:
        params.save(folder)
        info = {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "inputs": inputs,
            "measures": measures,
            "quality": quality,
            "files": files or {},
        }
        (folder / "scan.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), "utf-8")
        return self._record(person_id, folder)

    def _record(self, person_id: str, folder: Path) -> ScanRecord:
        info = json.loads((folder / "scan.json").read_text("utf-8"))
        return ScanRecord(person_id=person_id, id=folder.name, folder=folder, **info)

    def scans(self, person_id: str) -> list[ScanRecord]:
        base = self.root / person_id / "scans"
        return [self._record(person_id, f.parent) for f in sorted(base.glob("*/scan.json"))]

    def latest(self, person_id: str) -> ScanRecord | None:
        scans = self.scans(person_id)
        return scans[-1] if scans else None


def compare(old: dict[str, float], new: dict[str, float]) -> list[tuple[str, float, float, float]]:
    """(deutscher Name, alt, neu, Differenz) für jedes Maß, das in beiden Scans vorkommt."""
    rows = []
    for key, (label, _unit) in MEASURES.items():
        if key in old and key in new:
            rows.append((label, old[key], new[key], round(new[key] - old[key], 1)))
    return rows
