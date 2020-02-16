import numpy as np
import pytest
from numpy.testing import assert_array_equal

from src.model_code.aggregate import aggregate_hc_readable
from src.model_code.aggregate import aggregate_hc_vectorized
from src.model_code.solve import solve_by_backward_induction_hc_vectorized

#########################################################################
# PARAMETERS
#########################################################################

alpha = 0.36
beta = 0.97
gamma = 1.0
delta_k = 0.06
delta_hc = 0.0
sigma = 2.0
age_max = 66
age_retire = 46
population_growth_rate = 0.011
capital_min = 0.01
capital_max = 30.0
n_gridpoints_capital = 100
hc_min = 0.5
hc_max = 3.0
n_gridpoints_hc = 2
hc_init = 0.5
n_prod_states = 2
prod_states = np.array([0.5, 3.0], dtype=np.float64)
transition_prod_states = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
efficiency = np.ones(age_max, dtype=np.float64)
income_tax_rate = 0.11
aggregate_capital_in = 1.209551542349482
aggregate_labor_in = 0.160261889093575
neg = np.float64(-1e10)
efficiency = np.ones(age_retire, dtype=np.float64)

# calculate derived parameters
capital_grid = np.linspace(
    capital_min, capital_max, n_gridpoints_capital, dtype=np.float64
)
hc_grid = np.linspace(hc_min, hc_max, n_gridpoints_hc, dtype=np.float64)
duration_retired = age_max - age_retire + 1
duration_working = age_retire - 1

# Measure of each generation
mass = np.ones((age_max, 1), dtype=np.float64)
for j in range(1, age_max):
    mass[j] = mass[j - 1] / (1 + population_growth_rate)

# Normalized measure of each generation (sum up to 1)
mass = mass / sum(mass)

interest_rate = np.float64(
    alpha * (aggregate_capital_in ** (alpha - 1)) * (aggregate_labor_in ** (1 - alpha))
    - delta_k
)
wage_rate = np.float64(
    (1 - alpha) * (aggregate_capital_in ** alpha) * (aggregate_labor_in ** (-alpha))
)
pension_benefit = np.float64(
    income_tax_rate * wage_rate * aggregate_labor_in / np.sum(mass[age_retire - 1 :])
)


#########################################################################
# FIXTURES
#########################################################################


@pytest.fixture
def setup_solve_hc():
    out = {
        "interest_rate": interest_rate,
        "wage_rate": wage_rate,
        "capital_grid": capital_grid,
        "n_gridpoints_capital": n_gridpoints_capital,
        "hc_grid": hc_grid,
        "n_gridpoints_hc": n_gridpoints_hc,
        "sigma": sigma,
        "gamma": gamma,
        "pension_benefit": pension_benefit,
        "neg": neg,
        "age_max": age_max,
        "age_retire": age_retire,
        "income_tax_rate": income_tax_rate,
        "beta": 0.97,
        "zeta": 0.16,
        "psi": 0.65,
        "delta_hc": delta_hc,
        "efficiency": efficiency,
    }
    return out


@pytest.fixture
def setup_aggregate_hc(setup_solve_hc):
    (
        policy_capital_working,
        policy_hc_working,
        policy_labor_working,
        policy_capital_retired,
    ) = solve_by_backward_induction_hc_vectorized(**setup_solve_hc)
    out = {
        "policy_capital_working": policy_capital_working,
        "policy_hc_working": policy_hc_working,
        "policy_labor_working": policy_labor_working,
        "policy_capital_retired": policy_capital_retired,
        "age_max": age_max,
        "age_retire": age_retire,
        "n_gridpoints_capital": n_gridpoints_capital,
        "hc_init": hc_init,
        "capital_grid": capital_grid,
        "n_gridpoints_hc": n_gridpoints_hc,
        "hc_grid": hc_grid,
        "mass": mass,
        "population_growth_rate": population_growth_rate,
    }
    return out


#########################################################################
# TESTS
#########################################################################


def test_aggregate_hc_readable_equals_vectorized(setup_aggregate_hc):
    (
        aggregate_capital_out_readable,
        aggregate_labor_out_readable,
        mass_distribution_full_working_readable,
        mass_distribution_full_retired_readable,
    ) = aggregate_hc_readable(**setup_aggregate_hc)
    (
        aggregate_capital_out_vectorized,
        aggregate_labor_out_vectorized,
        mass_distribution_full_working_vectorized,
        mass_distribution_full_retired_vectorized,
    ) = aggregate_hc_vectorized(**setup_aggregate_hc)
    assert aggregate_capital_out_readable == aggregate_capital_out_vectorized
    assert aggregate_labor_out_readable == aggregate_labor_out_vectorized
    assert_array_equal(
        mass_distribution_full_working_readable,
        mass_distribution_full_working_vectorized,
    )
    assert_array_equal(
        mass_distribution_full_retired_readable,
        mass_distribution_full_retired_vectorized,
    )