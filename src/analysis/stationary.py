"""
Solve for a stationary equilibrium of an overlapping generations model
with human capital accumulation adopted from Ludwig, Schelkle, Vogel (2006).

"""
import json
import pickle
import sys

import numpy as np

from bld.project_paths import project_paths_join as ppj
from src.model_code.aggregate import aggregate_stationary
from src.model_code.auxiliary import set_continuous_point_on_grid
from src.model_code.solve import solve_by_backward_induction_hc_vectorized as solve_hc
from src.model_code.within_period import get_factor_prices

#####################################################
# PARAMETERS
######################################################

# Load general parameters
setup = json.load(open(ppj("IN_MODEL_SPECS", "setup_general.json"), encoding="utf-8"))
alpha = np.float64(setup["alpha"])
beta = np.float64(setup["beta"])
gamma = np.float64(setup["gamma"])
delta_k = np.float64(setup["delta_k"])
delta_hc = np.float64(setup["delta_hc"])
sigma = np.float64(setup["sigma"])
psi = np.float64(setup["psi"])
zeta = np.float64(setup["zeta"])
age_max = np.int32(setup["age_max"])
age_retire = np.int32(setup["age_retire"])
capital_min = np.float64(setup["capital_min"])
capital_max = np.float64(setup["capital_max"])
n_gridpoints_capital = np.int32(setup["n_gridpoints_capital"])
assets_init = np.float64(setup["assets_init"])
hc_min = np.float64(setup["hc_min"])
hc_max = np.float64(setup["hc_max"])
n_gridpoints_hc = np.int32(setup["n_gridpoints_hc"])
hc_init = np.float64(setup["hc_init"])
tolerance_capital = np.float64(setup["tolerance_capital"])
tolerance_labor = np.float64(setup["tolerance_labor"])
max_iterations_inner = np.int32(setup["max_iterations_inner"])
iteration_update_inner = np.float64(setup["iteration_update_inner"])

# Load demographic parameters
efficiency = np.loadtxt(
    ppj("IN_DATA", "efficiency_multiplier.csv"), delimiter=",", dtype=np.float64
)
fertility_rates = np.loadtxt(
    ppj("OUT_DATA", "fertility_rates.csv"), delimiter=",", dtype=np.float64
)
survival_rates_all = np.loadtxt(
    ppj("OUT_DATA", "survival_rates.csv"), delimiter=",", dtype=np.float64
)
mass_all = np.loadtxt(
    ppj("OUT_DATA", "mass_distribution.csv"), delimiter=",", dtype=np.float64
)

# Calculate derived parameters
capital_grid = np.linspace(
    capital_min, capital_max, n_gridpoints_capital, dtype=np.float64
)
hc_grid = np.linspace(hc_min, hc_max, n_gridpoints_hc, dtype=np.float64)

assets_init_gridpoints = np.zeros(2, dtype=np.int32)
assets_init_weights = np.zeros(2, dtype=np.float64)
hc_init_gridpoints = np.zeros(2, dtype=np.int32)
hc_init_weights = np.zeros(2, dtype=np.float64)

set_continuous_point_on_grid(
    assets_init, capital_grid, assets_init_gridpoints, assets_init_weights
)
set_continuous_point_on_grid(hc_init, hc_grid, hc_init_gridpoints, hc_init_weights)

duration_retired = age_max - age_retire + 1
duration_working = age_retire - 1

#####################################################
# FUNCTIONS
######################################################


def solve_stationary(model_specs):

    # Load model specifications
    setup_name = model_specs["setup_name"]

    if setup_name == "initial":
        time_idx = 0
    elif setup_name == "final":
        time_idx = -1

    population_growth_rate = fertility_rates[time_idx] - 1
    survival_rates = survival_rates_all[:, time_idx]
    mass = mass_all[:, time_idx]

    aggregate_capital_in = model_specs["aggregate_capital_init"]
    aggregate_labor_in = model_specs["aggregate_labor_init"]
    income_tax_rate = model_specs["income_tax_rate"]

    ################################################################
    # Loop over capital, labor and pension benefits
    ################################################################
    # Initialize iteration
    num_iterations_inner = 0  # Counter for iterations

    aggregate_capital_out = aggregate_capital_in + 10
    aggregate_labor_out = aggregate_labor_in + 10
    neg = np.float64(-1e10)  # very small number

    while (num_iterations_inner < max_iterations_inner) and (
        (abs(aggregate_capital_out - aggregate_capital_in) > tolerance_capital)
        or (abs(aggregate_labor_out - aggregate_labor_in) > tolerance_labor)
    ):
        num_iterations_inner += 1

        print(f"Iteration {num_iterations_inner} out of {max_iterations_inner}")

        # Calculate factor prices from aggregates
        (interest_rate, wage_rate, pension_benefit) = get_factor_prices(
            aggregate_capital=aggregate_capital_in,
            aggregate_labor=aggregate_labor_in,
            alpha=alpha,
            delta_k=delta_k,
            income_tax_rate=income_tax_rate,
            mass=mass,
            age_retire=age_retire,
        )

        ############################################################################
        # Solve for policy functions
        ############################################################################

        (
            policy_capital_working,
            policy_hc_working,
            policy_labor_working,
            policy_capital_retired,
            value_retired,
            value_working,
        ) = solve_hc(
            interest_rate=interest_rate,
            wage_rate=wage_rate,
            capital_grid=capital_grid,
            n_gridpoints_capital=n_gridpoints_capital,
            hc_grid=hc_grid,
            n_gridpoints_hc=n_gridpoints_hc,
            sigma=sigma,
            gamma=gamma,
            pension_benefit=pension_benefit,
            neg=neg,
            age_max=age_max,
            age_retire=age_retire,
            income_tax_rate=income_tax_rate,
            beta=beta,
            zeta=zeta,
            psi=psi,
            delta_hc=delta_hc,
            efficiency=efficiency,
            survival_rates=survival_rates,
        )

        ############################################################################
        # Aggregate capital stock and employment
        ############################################################################

        (
            aggregate_capital_out,
            aggregate_labor_out,
            mass_distribution_full_working,
            mass_distribution_full_retired,
        ) = aggregate_stationary(
            policy_capital_working=policy_capital_working,
            policy_hc_working=policy_hc_working,
            policy_labor_working=policy_labor_working,
            policy_capital_retired=policy_capital_retired,
            age_max=age_max,
            age_retire=age_retire,
            n_gridpoints_capital=n_gridpoints_capital,
            capital_grid=capital_grid,
            n_gridpoints_hc=n_gridpoints_hc,
            hc_grid=hc_grid,
            assets_init_gridpoints=assets_init_gridpoints,
            assets_init_weights=assets_init_weights,
            hc_init_gridpoints=hc_init_gridpoints,
            hc_init_weights=hc_init_weights,
            population_growth_rate=population_growth_rate,
            survival_rates=survival_rates,
            efficiency=efficiency,
            mass_newborns=mass[0],
        )

        # Update the guess on capital and labor
        aggregate_capital_in = (
            1 - iteration_update_inner
        ) * aggregate_capital_in + iteration_update_inner * aggregate_capital_out
        aggregate_labor_in = (
            1 - iteration_update_inner
        ) * aggregate_labor_in + iteration_update_inner * aggregate_labor_out

        # Display results
        print("capital | labor | pension ")
        print([aggregate_capital_in, aggregate_labor_in, pension_benefit])
        print("deviation capital | deviation labor")
        print(
            [
                abs(aggregate_capital_out - aggregate_capital_in),
                abs(aggregate_labor_out - aggregate_labor_in),
            ]
        )

    ################################################################
    # Display results and calculate summary statistics
    ################################################################

    # Calculate equilibrium prices and pension benefits
    (interest_rate, wage_rate, pension_benefit) = get_factor_prices(
        aggregate_capital_in,
        aggregate_labor_in,
        alpha,
        delta_k,
        income_tax_rate,
        mass,
        age_retire,
    )

    # Display equilibrium results
    print(
        "aggregate_capital | aggregate_labor | wage_rate | interest_rate | pension_benefit "
    )
    print(
        [
            aggregate_capital_in,
            aggregate_labor_in,
            wage_rate,
            interest_rate,
            pension_benefit,
        ]
    )

    # Check mass of agents at upper bound of grids
    mass_upper_bound_capital = np.sum(
        np.sum(mass_distribution_full_working, axis=1)[-1, :]
    )
    print(f"mass of agents at upper bound of asset grid = {mass_upper_bound_capital}")

    mass_upper_bound_hc = np.sum(np.sum(mass_distribution_full_working, axis=2)[-1, :])
    print(
        f"mass of agents at upper bound of human capital grid = {mass_upper_bound_hc}"
    )

    # Return results
    results = {
        "aggregate_capital": aggregate_capital_in,
        "aggregate_labor": aggregate_labor_in,
        "wage_rate": wage_rate,
        "interest_rate": interest_rate,
        "pension_benefit": pension_benefit,
        "policy_capital_working": policy_capital_working,
        "policy_hc_working": policy_hc_working,
        "policy_labor_working": policy_labor_working,
        "policy_capital_retired": policy_capital_retired,
        "value_retired": value_retired,
        "value_working": value_working,
        "mass_distribution_full_working": mass_distribution_full_working,
        "mass_distribution_full_retired": mass_distribution_full_retired,
    }

    return results


#####################################################
# SCRIPT
######################################################


if __name__ == "__main__":

    model_name = sys.argv[1]

    model_specs = json.load(
        open(ppj("IN_MODEL_SPECS", f"stationary_{model_name}.json"), encoding="utf-8")
    )
    results_stationary = solve_stationary(model_specs)

    with open(ppj("OUT_ANALYSIS", f"stationary_{model_name}.pickle"), "wb") as out_file:
        pickle.dump(results_stationary, out_file)
