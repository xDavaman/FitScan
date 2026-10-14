import numpy as np

from fitscan.body import BodyParams, age_to_param, param_to_age
from fitscan.fit import Person, fit_to_measurements
from fitscan.measure import CIRCUMFERENCES, measure_params


def test_age_mapping_roundtrip():
    for years in (1, 11, 18, 25, 40, 70, 90):
        assert abs(param_to_age(age_to_param(years)) - years) < 1e-6


def test_params_save_load(tmp_path):
    p = BodyParams(phenotype={"weight": 0.7}, local={"measure-bust-circ-incr": 0.2}, scale=1.03,
                   offsets=np.ones((5, 3), np.float32))
    p.save(tmp_path)
    q = BodyParams.load(tmp_path)
    assert q.phenotype == p.phenotype and q.local == p.local and q.scale == p.scale
    assert np.allclose(q.offsets, p.offsets)


def test_default_body_measures_plausible(model):
    m = measure_params(model, BodyParams())
    assert 150 < m["height"] < 200
    assert 40 < m["weight"] < 110
    for key in CIRCUMFERENCES:
        assert 15 < m[key] < 140, key
    assert m["hips"] > m["thigh"] > m["calf"]


def test_fit_hits_entered_measures(model):
    person = Person("weiblich", 34, {"height": 168.0, "weight": 64.0, "waist": 76.0, "hips": 100.0})
    _, m = fit_to_measurements(model, person)
    assert abs(m["height"] - 168.0) < 0.2
    assert abs(m["weight"] - 64.0) < 2.0
    assert abs(m["waist"] - 76.0) < 1.5
    assert abs(m["hips"] - 100.0) < 1.5


def test_person_validation():
    assert Person("maennlich", 30, {"height": 180, "weight": 80}).validate() == []
    assert Person("maennlich", 30, {"height": 0, "weight": 80}).validate()
