import numpy as np
import pytest

from src.analysis.auxiliary import gini
from src.analysis.auxiliary import labor_input
from src.analysis.auxiliary import util_retired
from src.analysis.auxiliary import util_working


#########################################################################
# FIXTURES
#########################################################################


@pytest.fixture
def setup_gini():
    out = {}
    out["pop"] = np.array(object=[1, 2, 3], dtype=float)
    out["val"] = np.array(object=[1, 1, 1], dtype=float)
    out["makeplot"] = False
    return out


@pytest.fixture
def setup_labor_input():
    out = {}
    out["interest_rate"] = 1.0
    out["wage_rate"] = 1.0
    out["assets_this_period"] = 10.0
    out["assets_next_period"] = 10.0
    out["productivity"] = 3.0
    out["eff"] = 1.0
    out["gamma"] = 0.42
    out["tau"] = 0.11
    return out


@pytest.fixture
def setup_util_retired():
    out = {}
    out["interest_rate"] = 0.42
    out["assets_this_period"] = 0.11
    out["pension_benefits"] = 1.0
    out["assets_next_period"] = 3.0
    out["sigma"] = 1.0
    out["gamma"] = 1.0
    out["neg"] = 10.0
    return out


@pytest.fixture
def setup_util_working():
    out = {}
    out["interest_rate"] = 0.42
    out["wage_rate"] = 0.3
    out["assets_this_period"] = 0.11
    out["assets_next_period"] = 3.0
    out["productivity"] = 3.0
    out["eff"] = 1.0
    out["tau"] = 0.2
    out["neg"] = 10.0
    out["gamma"] = 1.0
    out["sigma"] = 1.0
    return out


#########################################################################
# TESTS
#########################################################################


def test_gini_equal(setup_gini):
    expected = 0.0
    actual, _, _ = gini(**setup_gini)
    assert actual == expected


def test_gini_intermediate(setup_gini):
    setup_gini["pop"] = np.array(object=[1, 1, 1], dtype=float)
    setup_gini["val"] = np.array(object=[1, 8, 1], dtype=float)
    expected = 1 - 8 / 15
    actual, _, _ = gini(**setup_gini)
    assert actual == expected


def test_gini_unequal(setup_gini):
    setup_gini["pop"] = np.array(object=[1e10, 1e10, 1e-10], dtype=float)
    setup_gini["val"] = np.array(object=[0, 0, 5], dtype=float)
    expected = 1.0
    actual, _, _ = gini(**setup_gini)
    assert actual == expected


def test_labor_input_in_range(setup_labor_input):
    expected = 0.5
    actual = labor_input(**setup_labor_input)
    assert actual == expected


def test_labor_input_too_high(setup_labor_input):
    setup_labor_input["assets_this_period"] = 1.0
    expected = 1.0
    actual = labor_input(**setup_labor_input)
    assert actual == expected


def test_labor_input_too_low(setup_labor_input):
    setup_labor_input["assets_next_period"] = 1.0
    expected = 0.0
    actual = labor_input(**setup_labor_input)
    assert actual == expected


def test_util_retired(setup_util_retired):
    expected = 0.0
    actual = util_retired(**setup_util_retired)
    assert actual == expected


def test_util_working(setup_util_working):
    expected = 0.0
    actual = util_working(**setup_util_working)
    assert actual == expected
