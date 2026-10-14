import pytest

from fitscan.body import BodyModel


@pytest.fixture(scope="session")
def model() -> BodyModel:
    return BodyModel.get("cpu")
