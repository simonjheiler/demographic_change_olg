import pytest

from src.model_code.within_period import get_consumption
from src.model_code.within_period import get_labor_input
from src.model_code.within_period import util


#########################################################################
# FIXTURES
#########################################################################


@pytest.fixture
def setup_get_labor_input():
    out = {
        "interest_rate": 1.0,
        "wage_rate": 1.0,
        "assets_this_period": 10.0,
        "assets_next_period": 10.0,
        "productivity": 1.0,
        "eff": 1.0,
        "gamma": 1.0,
        "tau": 0.0,
    }
    return out


@pytest.fixture
def setup_util():
    out = {
        "interest_rate": 0.42,
        "assets_this_period": 0.11,
        "pension_benefits": 1.0,
        "assets_next_period": 3.0,
        "sigma": 1.0,
        "gamma": 1.0,
        "neg": 10.0,
    }
    return out


@pytest.fixture
def setup_get_consumption():
    out = {
        "interest_rate": 0.42,
        "wage_rate": 0.3,
        "assets_this_period": 0.11,
        "assets_next_period": 3.0,
        "productivity": 3.0,
        "eff": 1.0,
        "tau": 0.2,
        "neg": 10.0,
        "gamma": 1.0,
        "sigma": 1.0,
    }
    return out


#########################################################################
# TESTS
#########################################################################


def test_get_labor_input_in_range(setup_labor_input):
    expected = 1.0
    actual = get_labor_input(**setup_labor_input)
    assert actual == expected


def test_get_labor_input_too_high(setup_labor_input):
    setup_labor_input["assets_this_period"] = 1.0
    expected = 1.0
    actual = get_labor_input(**setup_labor_input)
    assert actual == expected


def test_get_labor_input_too_low(setup_labor_input):
    setup_labor_input["assets_this_period"] = 1.0
    expected = 0.0
    actual = get_labor_input(**setup_labor_input)
    assert actual == expected


def test_get_consumption_working(setup_get_consumption):
    expected = 0.0
    actual = get_consumption(**setup_get_consumption)
    assert actual == expected


def test_get_consumption_retired(setup_get_consumption):
    expected = 0.0
    actual = get_consumption(**setup_get_consumption)
    assert actual == expected


def test_util(setup_util):
    expected = 0.0
    actual = util(**setup_util)
    assert actual == expected
