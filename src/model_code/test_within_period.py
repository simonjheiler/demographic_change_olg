import pytest
from numpy.testing import assert_almost_equal

from src.model_code.within_period import get_consumption
from src.model_code.within_period import get_labor_input
from src.model_code.within_period import util


#########################################################################
# FIXTURES
#########################################################################


@pytest.fixture
def setup_get_labor_input():
    out = {
        "assets_this_period": 2.0,
        "assets_next_period": 2.0,
        "interest_rate": 0.1,
        "wage_rate": 2.0,
        "income_tax_rate": 0.2,
        "productivity": 1.0,
        "efficiency": 0.5,
        "gamma": 0.5,
    }
    return out


@pytest.fixture
def setup_get_consumption():
    out = {
        "assets_this_period": 2.0,
        "assets_next_period": 2.0,
        "pension_benefit": 0.4,
        "labor_input": 0.5,
        "interest_rate": 0.1,
        "wage_rate": 2.0,
        "income_tax_rate": 0.2,
        "productivity": 1.0,
        "efficiency": 0.5,
    }
    return out


@pytest.fixture
def setup_util():
    out = {
        "consumption": 1.0,
        "labor_input": 0.5,
        "hc_effort": 0.0,
        "sigma": 2.0,
        "gamma": 2.0,
    }
    return out


#########################################################################
# TESTS
#########################################################################


def test_get_labor_input_in_range(setup_get_labor_input):
    expected = 0.375
    actual = get_labor_input(**setup_get_labor_input)
    assert_almost_equal(actual, expected, decimal=12)


def test_get_labor_input_too_high(setup_get_labor_input):
    setup_get_labor_input["gamma"] = 2.0
    expected = 1.0
    actual = get_labor_input(**setup_get_labor_input)
    assert actual == expected


def test_get_labor_input_too_low(setup_get_labor_input):
    setup_get_labor_input["assets_this_period"] = 3.0
    expected = 0.0
    actual = get_labor_input(**setup_get_labor_input)
    assert actual == expected


def test_get_consumption_working(setup_get_consumption):
    setup_get_consumption["pension_benefit"] = 0.0
    expected = 0.6
    actual = get_consumption(**setup_get_consumption)
    assert_almost_equal(actual, expected, decimal=12)


def test_get_consumption_retired(setup_get_consumption):
    setup_get_consumption["labor_input"] = 0.0
    expected = 0.6
    actual = get_consumption(**setup_get_consumption)
    assert_almost_equal(actual, expected, decimal=12)


def test_util(setup_util):
    expected = -0.5
    actual = util(**setup_util)
    assert actual == expected
