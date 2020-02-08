import numpy as np
import pytest

from src.model_code.auxiliary import gini


#########################################################################
# FIXTURES
#########################################################################


@pytest.fixture
def setup_gini():
    out = {
        "pop": np.array([1, 2, 3], dtype=float),
        "val": np.array([1, 1, 1], dtype=float),
        "makeplot": False,
    }
    return out


#########################################################################
# TESTS
#########################################################################


def test_gini_equal(setup_gini):
    expected = 0.0
    actual, _, _ = gini(**setup_gini)
    assert actual == expected


def test_gini_intermediate(setup_gini):
    setup_gini["pop"] = np.array([1, 1, 1], dtype=float)
    setup_gini["val"] = np.array([1, 8, 1], dtype=float)
    expected = 1 - 8 / 15
    actual, _, _ = gini(**setup_gini)
    assert actual == expected


def test_gini_unequal(setup_gini):
    setup_gini["pop"] = np.array([1e10, 1e10, 1e-10], dtype=float)
    setup_gini["val"] = np.array([0, 0, 5], dtype=float)
    expected = 1.0
    actual, _, _ = gini(**setup_gini)
    assert actual == expected
