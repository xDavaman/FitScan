"""Selbsttest: synthetischer Testkörper mit bekannten Maßen wird "gefilmt" und neu angepasst.

Damit lässt sich auf jedem Rechner prüfen, ob Installation, Grafikkarte und Anpassung
funktionieren, ohne dass schon ein eigenes Video vorliegt.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable

import torch

from .body import BodyModel, param_to_age
from .fit import Person, fit_to_frames, fit_to_measurements
from .measure import CIRCUMFERENCES, MEASURES, measure_params
from .synthetic import example_body, film


@dataclasses.dataclass
class SelfTestResult:
    device: str
    seconds: float
    rows: list[tuple[str, float, float, float]]  # Name, wahr, nur Maße, mit Video
    mean_error_measures: float
    mean_error_video: float
    min_iou: float
    yaw_errors_deg: list[float]

    @property
    def passed(self) -> bool:
        return self.mean_error_video < 2.0 and max(self.yaw_errors_deg) < 20

    def text(self) -> str:
        lines = [f"Gerät: {self.device}, Dauer {self.seconds:.0f} s"]
        lines.append(f"{'Maß':22s} {'wahr':>7s} {'nur Maße':>9s} {'mit Video':>9s}")
        for name, t, a, b in self.rows:
            lines.append(f"{name:22s} {t:7.1f} {a:9.1f} {b:9.1f}")
        lines.append(f"Mittlerer Umfangsfehler: nur Maße {self.mean_error_measures:.2f} cm, "
                     f"mit Video {self.mean_error_video:.2f} cm")
        lines.append(f"Größter Drehfehler {max(self.yaw_errors_deg):.0f}°, kleinste Deckung {self.min_iou:.2f}")
        lines.append("Ergebnis: BESTANDEN" if self.passed else "Ergebnis: NICHT BESTANDEN")
        return "\n".join(lines)


def run(seed: int = 3, variation: float = 2.5, views: int = 12, quick: bool = False,
        device: str | None = None, log: Callable[[str], None] | None = None) -> SelfTestResult:
    start = time.time()
    model = BodyModel.get(device)
    truth_params, extra = example_body(seed, variation)
    truth = measure_params(model, truth_params)
    frames = film(model, truth_params, extra["arm_deg"], views=views)
    gender = "weiblich" if truth_params.phenotype["gender"] > 0.5 else "maennlich"
    person = Person(gender, param_to_age(truth_params.phenotype["age"]),
                    {"height": truth["height"], "weight": truth["weight"]})
    p0, m0 = fit_to_measurements(model, person, log=log)
    steps = (40, 80, 60) if quick else (80, 160, 140)
    _, m1, frames_fit = fit_to_frames(model, person, frames, init=p0, steps=steps, log=log)
    rows = [(MEASURES[k][0], truth[k], m0[k], m1[k]) for k in MEASURES if k in truth]
    e0 = sum(abs(m0[k] - truth[k]) for k in CIRCUMFERENCES) / len(CIRCUMFERENCES)
    e1 = sum(abs(m1[k] - truth[k]) for k in CIRCUMFERENCES) / len(CIRCUMFERENCES)
    yaw_err = []
    for i, r in enumerate(frames_fit):
        expected = 360.0 * i / views
        d = abs((r.yaw_deg - expected + 180) % 360 - 180)
        yaw_err.append(d)
    dev = str(model.device)
    if dev.startswith("cuda"):
        dev += f" ({torch.cuda.get_device_name(model.device)})"
    return SelfTestResult(dev, time.time() - start, rows, e0, e1,
                          min(r.iou for r in frames_fit), yaw_err)
