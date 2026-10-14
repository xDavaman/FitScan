import pytest

from fitscan.body import param_to_age
from fitscan.fit import Person, fit_to_frames, fit_to_measurements
from fitscan.measure import CIRCUMFERENCES, measure_params
from fitscan.synthetic import example_body, film


@pytest.mark.slow
def test_video_fit_recovers_body(model):
    """Künstlich gefilmter Körper: Drehung je Bild und Umfänge müssen stimmen."""
    truth_params, extra = example_body(3, 2.5)
    truth = measure_params(model, truth_params)
    views = 8
    frames = film(model, truth_params, extra["arm_deg"], views=views)
    gender = "weiblich" if truth_params.phenotype["gender"] > 0.5 else "maennlich"
    person = Person(gender, param_to_age(truth_params.phenotype["age"]),
                    {"height": truth["height"], "weight": truth["weight"]})
    p0, m0 = fit_to_measurements(model, person)
    _, m1, results = fit_to_frames(model, person, frames, init=p0, steps=(40, 80, 60))
    for i, r in enumerate(results):
        expected = 360 * i / views
        assert abs((r.yaw_deg - expected + 180) % 360 - 180) < 20, (i, r.yaw_deg)
    err0 = sum(abs(m0[k] - truth[k]) for k in CIRCUMFERENCES) / len(CIRCUMFERENCES)
    err1 = sum(abs(m1[k] - truth[k]) for k in CIRCUMFERENCES) / len(CIRCUMFERENCES)
    assert err1 < 2.0
    assert err1 < err0
