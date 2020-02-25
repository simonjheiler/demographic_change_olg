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
from src.model_code.solve import solve_retired
from src.model_code.solve import solve_working
from src.model_code.within_period import get_factor_prices


#####################################################
# PARAMETERS
#####################################################

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
tolerance_capital = np.float64(setup["tolerance_capital_stationary"])
tolerance_labor = np.float64(setup["tolerance_labor_stationary"])
max_iterations = np.int32(setup["max_iterations_stationary"])
iteration_update = np.float64(setup["iteration_update_stationary"])

# Load demographic parameters
efficiency = np.loadtxt(
    ppj("IN_DATA", "efficiency_multiplier.csv"), delimiter=",", dtype=np.float64
)

with open(ppj("OUT_DATA", "simulated_demographics.pickle"), "rb") as in_file:
    demographics = pickle.load(in_file)

# Calculate derived parameters
capital_grid = np.linspace(
    capital_min, capital_max, n_gridpoints_capital, dtype=np.float64
)
hc_grid = np.logspace(np.log(hc_min), np.log(hc_max), n_gridpoints_hc, base=np.exp(1))

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
#####################################################


def solve_stationary(params):

    # Load model specifications
    setup_name = params["setup_name"]

    survival_rates = demographics[f"survival_rates_{setup_name}"]
    mass = demographics[f"mass_{setup_name}"]

    aggregate_capital_in = params["aggregate_capital_init"]
    aggregate_labor_in = params["aggregate_labor_init"]
    income_tax_rate = params["income_tax_rate"]

    ################################################################
    # Loop over capital, labor and pension benefits
    ################################################################
    # Initialize iteration
    num_iterations = 0  # Counter for iterations

    aggregate_capital_out = aggregate_capital_in + 10
    aggregate_labor_out = aggregate_labor_in + 10
    neg = np.float64(-1e10)  # very small number

    while (num_iterations < max_iterations) and (
        (abs(aggregate_capital_out - aggregate_capital_in) > tolerance_capital)
        or (abs(aggregate_labor_out - aggregate_labor_in) > tolerance_labor)
    ):
        num_iterations += 1

        print(f"Iteration {num_iterations} out of {max_iterations}")

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

        # Initialize objects for backward iteration
        duration_retired = age_max - age_retire + 1  # length of retirement
        duration_working = age_retire - 1  # length of working life

        value_working = np.zeros(
            (n_gridpoints_capital, n_gridpoints_hc, duration_working), dtype=np.float64
        )
        value_retired = np.zeros(
            (n_gridpoints_capital, duration_retired), dtype=np.float64
        )
        policy_capital_working = np.zeros(
            (n_gridpoints_capital, n_gridpoints_hc, duration_working), dtype=np.int32
        )
        policy_capital_retired = np.zeros(
            (n_gridpoints_capital, duration_retired), dtype=np.int32
        )
        policy_hc_working = np.zeros(
            (n_gridpoints_capital, n_gridpoints_hc, duration_working), dtype=np.int32
        )
        policy_labor_working = np.zeros(
            (n_gridpoints_capital, n_gridpoints_hc, duration_working), dtype=np.float64,
        )

        ############################################################
        # BACKWARD INDUCTION
        ############################################################

        # Retired agents

        # Last period utility
        consumption_last = (1 + interest_rate) * capital_grid + pension_benefit
        flow_utility_last = (consumption_last ** ((1 - sigma) * gamma)) / (1 - sigma)
        value_retired[:, -1] = flow_utility_last

        # Create meshes for assets this period and assets next period
        assets_next_period, assets_this_period = np.meshgrid(capital_grid, capital_grid)

        # Initiate objects to store temporary policy and value functions
        policy_capital_retired_tmp = np.zeros(n_gridpoints_capital, dtype=np.int32)
        value_retired_tmp = np.zeros(n_gridpoints_capital, dtype=np.float64)

        # Iterate backwards through retirement period
        for age_idx in range(duration_retired - 2, -1, -1):
            # Look up continuation values for assets_next_period
            value_next_period = value_retired[:, age_idx + 1]
            # Replicate in assets_this_period dimension
            continuation_value = np.repeat(
                value_next_period[np.newaxis, :], n_gridpoints_capital, axis=0
            )

            # Solve for policy and value function
            value_retired_tmp, policy_capital_retired_tmp = solve_retired(
                assets_this_period=assets_this_period,
                assets_next_period=assets_next_period,
                interest_rate=interest_rate,
                pension_benefit=pension_benefit,
                beta=beta,
                gamma=gamma,
                sigma=sigma,
                neg=neg,
                continuation_value=continuation_value,
                survival_rate=survival_rates[age_idx],
            )

            # Store results
            policy_capital_retired[:, age_idx] = policy_capital_retired_tmp
            value_retired[:, age_idx] = value_retired_tmp

        # Working agents

        # Create meshes for assets_this_period, assets_next_period, hc_this_period
        # and hc_next_period
        (
            assets_next_period,
            assets_this_period,
            hc_this_period,
            hc_next_period,
        ) = np.meshgrid(capital_grid, capital_grid, hc_grid, hc_grid,)

        # Initiate objects to store temporary policy and value functions
        policy_capital_working_tmp = np.zeros(
            (n_gridpoints_capital, n_gridpoints_hc), dtype=np.int32
        )
        policy_hc_working_tmp = np.zeros(
            (n_gridpoints_capital, n_gridpoints_hc), dtype=np.int32
        )
        policy_labor_working_tmp = np.zeros(
            (n_gridpoints_capital, n_gridpoints_hc), dtype=np.float64
        )
        value_working_tmp = np.zeros(
            (n_gridpoints_capital, n_gridpoints_hc), dtype=np.float64
        )

        # Iterate backwards through working period
        for age_idx in range(duration_working - 1, -1, -1):

            # Look up continuation values for combinations of assets_next_period
            # and hc_next_period
            if age_idx == duration_working - 1:  # retired next period
                value_next_period = np.repeat(
                    value_retired[:, 0, np.newaxis], n_gridpoints_hc, axis=1
                )
            else:
                value_next_period = value_working[:, :, age_idx + 1]

            # Replicate continuation value in assets_this_period and hc_this_period dimension
            continuation_value = np.repeat(
                value_next_period[np.newaxis, :, :], n_gridpoints_capital, axis=0
            )
            continuation_value = np.repeat(
                continuation_value[:, :, np.newaxis, :], n_gridpoints_hc, axis=2
            )

            # Solve for policy and value function
            (
                policy_capital_working_tmp,
                policy_hc_working_tmp,
                policy_labor_working_tmp,
                value_working_tmp,
            ) = solve_working(
                assets_this_period=assets_this_period,
                assets_next_period=assets_next_period,
                hc_this_period=hc_this_period,
                hc_next_period=hc_next_period,
                interest_rate=interest_rate,
                wage_rate=wage_rate,
                income_tax_rate=income_tax_rate,
                beta=beta,
                gamma=gamma,
                sigma=sigma,
                neg=neg,
                continuation_value=continuation_value,
                delta_hc=delta_hc,
                zeta=zeta,
                psi=psi,
                n_gridpoints_capital=n_gridpoints_capital,
                n_gridpoints_hc=n_gridpoints_hc,
                efficiency=np.float64(efficiency[age_idx]),
                survival_rate=survival_rates[age_idx],
            )

            # Store results
            policy_capital_working[:, :, age_idx] = policy_capital_working_tmp
            policy_hc_working[:, :, age_idx] = policy_hc_working_tmp
            policy_labor_working[:, :, age_idx] = policy_labor_working_tmp
            value_working[:, :, age_idx] = value_working_tmp

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
            survival_rates=survival_rates,
            efficiency=efficiency,
            mass_newborns=mass[0],
        )

        # Update the guess on capital and labor
        aggregate_capital_in = (
            1 - iteration_update
        ) * aggregate_capital_in + iteration_update * aggregate_capital_out
        aggregate_labor_in = (
            1 - iteration_update
        ) * aggregate_labor_in + iteration_update * aggregate_labor_out

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
#####################################################


if __name__ == "__main__":

    model_name = sys.argv[1]
    # model_name = "final"

    model_specs = json.load(
        open(ppj("IN_MODEL_SPECS", f"stationary_{model_name}.json"), encoding="utf-8")
    )
    results_stationary = solve_stationary(model_specs)

    with open(ppj("OUT_ANALYSIS", f"stationary_{model_name}.pickle"), "wb") as out_file:
        pickle.dump(results_stationary, out_file)
