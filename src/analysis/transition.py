import json  # noqa:F401
import pickle  # noqa:F401
import sys  # noqa:F401

import numpy as np  # noqa:F401
import pandas as pd  # noqa:F401

from bld.project_paths import project_paths_join as ppj
from src.model_code.aggregate import aggregate_hc_readable as aggregate_hc
from src.model_code.solve import solve_retired
from src.model_code.solve import solve_working


#####################################################
# PARAMETERS
######################################################

setup = json.load(open(ppj("IN_MODEL_SPECS", "setup.json"), encoding="utf-8"))
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
capital_init = np.float64(setup["capital_init"])
hc_min = np.float64(setup["hc_min"])
hc_max = np.float64(setup["hc_max"])
n_gridpoints_hc = np.int32(setup["n_gridpoints_hc"])
hc_init = np.float64(setup["hc_init"])
tolerance_capital = np.float64(setup["tolerance_capital"])
tolerance_labor = np.float64(setup["tolerance_labor"])
max_iterations_inner = np.int32(setup["max_iterations_inner"])
max_iterations_outer = np.int32(setup["max_iterations_outer"])

# Load demographic parameters
efficiency = np.squeeze(
    np.array(
        pd.read_csv(ppj("IN_DATA", "efficiency_multiplier.csv")).values,
        dtype=np.float64,
    )
)
fertility_rates = np.array(
    pd.read_csv(ppj("OUT_DATA", "fertility_rates.csv")).values, dtype=np.float64
)
survival_rates = np.array(
    pd.read_csv(ppj("OUT_DATA", "survival_rates.csv")).values, dtype=np.float64
)
mass = np.array(
    pd.read_csv(ppj("OUT_DATA", "mass_distribution.csv")).values, dtype=np.float64
)

# Calculate derived parameters
capital_grid = np.linspace(
    capital_min, capital_max, n_gridpoints_capital, dtype=np.float64
)
hc_grid = np.linspace(hc_min, hc_max, n_gridpoints_hc, dtype=np.float64)
duration_retired = age_max - age_retire + 1
duration_working = age_retire - 1

#####################################################
# FUNCTIONS
######################################################


def solve_transition(
    results_initial, results_final, transition_params,
):

    # Load final steady state
    aggregate_capital_final = results_final["aggregate_capital_in"]
    aggregate_labor_final = results_final["aggregate_labor_in"]
    value_retired_final = results_final["value_retired"]
    value_working_final = results_final["value_working"]
    policy_capital_retired_final = results_final["policy_capital_retired"]
    policy_capital_working_final = results_final["policy_capital_working"]
    policy_hc_working_final = results_final["policy_hc_working"]
    policy_labor_working_final = results_final["policy_labor_working"]

    # Load initial steady state
    aggregate_capital_initial = results_initial["aggregate_capital_in"]
    aggregate_labor_initial = results_initial["aggregate_labor_in"]
    income_tax_rate_initial = results_initial["income_tax_rate"]
    mass_distribution_full_working_init = results_initial[
        "mass_distribution_full_working"
    ]
    mass_distribution_full_retired_init = results_initial[
        "mass_distribution_full_retired"
    ]

    # Basic parameters
    duration_transition = transition_params["duration_transition"]
    population_growth_rate = transition_params["population_growth_rate"]

    # Set policy experiment
    # reform = 0(reform effective at t = 1)
    # reform = 1(reform effective at t = 21)
    reform = 0

    # # Initial guesses on the paths of K and L
    aggregate_capital_in = np.zeros((duration_transition, 1), dtype=np.float64)
    aggregate_labor_in = np.zeros((duration_transition, 1), dtype=np.float64)

    # Check whether we can use the paths for K and L from previously run executions
    try:
        with open(ppj("OUT_ANALYSIS", "stationary_initial.pickle"), "wb") as out_file:
            results_transition = pickle.load(None, out_file)
        duration_transition_aux = results_transition["duration_transition"]
        aggregate_capital_aux = results_transition["aggregate_capital"]
        aggregate_labor_aux = results_transition["aggregate_labor"]
        # Need to be careful because previously duration_transition could have been different
        if (
            duration_transition_aux > duration_transition
        ):  # If T_old > duration_transition,
            # just cut the vector of K_t and L_t
            aggregate_capital_in = aggregate_capital_aux[:duration_transition]
            aggregate_labor_in = aggregate_labor_aux[:duration_transition]
        elif (
            duration_transition_aux < duration_transition
        ):  # If T_old < duration_transition, then take the full vectors and fill the remaining
            # elements with the last available value
            aggregate_capital_in[:duration_transition_aux] = aggregate_capital_aux[
                :duration_transition_aux
            ]
            aggregate_capital_in[duration_transition_aux:] = np.repeat(
                aggregate_capital_aux[duration_transition_aux],
                duration_transition - duration_transition_aux,
                1,
            )
            aggregate_labor_in[:duration_transition_aux] = aggregate_labor_aux[
                :duration_transition_aux
            ]
            aggregate_labor_in[:duration_transition_aux] = np.repeat(
                aggregate_labor_aux[duration_transition_aux],
                duration_transition - duration_transition_aux,
                1,
            )
        else:
            aggregate_capital = aggregate_capital_aux
            aggregate_labor = aggregate_labor_aux
    except FileNotFoundError:
        for time_idx in range(duration_transition + 1):
            aggregate_capital_in[time_idx] = aggregate_capital_initial + (
                aggregate_capital_final - aggregate_capital_initial
            ) / duration_transition * (time_idx - 1)
            aggregate_labor_in[time_idx] = aggregate_labor_initial + (
                aggregate_labor_final - aggregate_labor_initial
            ) / duration_transition * (time_idx - 1)

    aggregate_capital[1] = aggregate_capital_initial
    aggregate_labor[-1] = aggregate_labor_final

    aggregate_capitalnew = aggregate_capital
    aggregate_labornew = aggregate_labor

    ################################################################
    # Loop over capital and labor
    ################################################################

    # Initialize objects for iteration
    value_retired = np.zeros(
        (n_gridpoints_capital, duration_retired, duration_transition), dtype=np.float64
    )
    value_working = np.zeros(
        (n_gridpoints_capital, n_gridpoints_hc, duration_working, duration_transition),
        dtype=np.float64,
    )
    policy_capital_retired = np.zeros(
        (n_gridpoints_capital, duration_retired, duration_transition), dtype=np.int32
    )
    policy_capital_working = np.zeros(
        (n_gridpoints_capital, n_gridpoints_hc, duration_working, duration_transition),
        dtype=np.int32,
    )
    policy_hc_working = np.zeros(
        (n_gridpoints_capital, n_gridpoints_hc, duration_working, duration_transition),
        dtype=np.int32,
    )
    policy_labor_working = np.zeros(
        (n_gridpoints_hc, n_gridpoints_capital, duration_retired, duration_transition),
        dtype=np.float64,
    )

    # Store final steady state values and policies as last period in transition duration
    value_retired[:, :, -1] = value_retired_final
    value_working[:, :, :, -1] = value_working_final
    policy_capital_retired[:, :, -1] = policy_capital_retired_final
    policy_capital_working[:, :, :, -1] = policy_capital_working_final
    policy_hc_working[:, :, :, -1] = policy_hc_working_final
    policy_labor_working[:, :, :, -1] = policy_labor_working_final

    aggregate_capital_out = np.zeros((duration_transition, 1), dtype=np.float64)
    aggregate_labor_out = np.zeros((duration_transition, 1), dtype=np.float64)

    # Construct policy rate path
    income_tax_rate = np.zeros((duration_transition, 1), dtype=np.float64)

    if reform == 0:
        income_tax_rate[1] = income_tax_rate_initial
    else:
        income_tax_rate[1:20] = income_tax_rate_initial

    # Initialize iteration
    num_iterations_outer = 0
    deviation_capital = 10
    deviation_labor = 10
    neg = np.float64(-1e10)

    ################################################################
    # Loop over path for capital, labor and pension benefits
    ################################################################

    while (num_iterations_outer < max_iterations_outer) and (
        (abs(deviation_capital) > tolerance_capital)
        or (abs(deviation_labor) > tolerance_labor)
    ):
        num_iterations_outer += 1

        print(f"Iteration {num_iterations_outer} out of {max_iterations_outer}")

        for time_idx in range(duration_transition - 2, -1, -1):

            # Load aggregate variables and policy variables
            income_tax_rate = income_tax_rate[time_idx]
            aggregate_capital_in = aggregate_capital[time_idx]
            aggregate_labor_in = aggregate_labor[time_idx]

            # Calculate factor prices from aggregates
            interest_rate = np.float64(
                alpha
                * (aggregate_capital_in ** (alpha - 1))
                * (aggregate_labor_in ** (1 - alpha))
                - delta_k
            )
            wage_rate = np.float64(
                (1 - alpha)
                * (aggregate_capital_in ** alpha)
                * (aggregate_labor_in ** (-alpha))
            )
            pension_benefit = np.float64(
                income_tax_rate
                * wage_rate
                * aggregate_labor_in
                / np.sum(mass[age_retire - 1 :])
            )

            ############################################################
            # BACKWARD INDUCTION
            ############################################################

            # Retired agents

            # Last period utility
            consumption_last = (1 + interest_rate) * capital_grid + pension_benefit
            flow_utility_last = (consumption_last ** ((1 - sigma) * gamma)) / (
                1 - sigma
            )
            value_retired[:, -1, time_idx] = flow_utility_last

            # Create meshes for assets this period and assets next period
            assets_next_period, assets_this_period = np.meshgrid(
                capital_grid, capital_grid
            )

            # Initiate objects to store temporary policy and value functions
            policy_capital_retired_tmp = np.zeros(n_gridpoints_capital, dtype=np.int32)
            value_retired_tmp = np.zeros(n_gridpoints_capital, dtype=np.float64)

            # Iterate backwards through retirement period
            for age_idx in range(duration_retired - 2, -1, -1):
                # Look up continuation values for assets_next_period and replicate in
                # assets_this_period dimension
                continuation_value = np.repeat(
                    value_retired[np.newaxis, :, age_idx + 1, time_idx + 1],
                    n_gridpoints_capital,
                    axis=0,
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
                    n_gridpoints_capital=n_gridpoints_capital,
                    survival_rate=survival_rates[age_idx],
                    policy_capital_retired_tmp=n_gridpoints_hc,
                    value_retired_tmp=value_retired_tmp,
                )

                # Store results
                policy_capital_retired[
                    :, age_idx, time_idx
                ] = policy_capital_retired_tmp
                value_retired[:, age_idx, time_idx] = value_retired_tmp

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

                # Replicate continuation value in assets_this_period and
                # hc_this_period dimension
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
                    policy_capital_working_tmp=policy_capital_working_tmp,
                    policy_hc_working_tmp=policy_hc_working_tmp,
                    policy_labor_working_tmp=policy_labor_working_tmp,
                    value_working_tmp=value_working_tmp,
                )

                # Store results
                policy_capital_working[:, :, age_idx] = policy_capital_working_tmp
                policy_hc_working[:, :, age_idx] = policy_hc_working_tmp
                policy_labor_working[:, :, age_idx] = policy_labor_working_tmp
                value_working[:, :, age_idx] = value_working_tmp

        ############################################################################
        # Iterating over the distribution
        ############################################################################

        # Initializations
        mass_distribution_full_working = mass_distribution_full_working_init
        mass_distribution_full_retired = mass_distribution_full_retired_init

        for time_idx in range(duration_transition - 1):

            policy_capital_working = policy_capital_working[:, :, :, time_idx]
            policy_capital_retired = policy_capital_retired[:, :, time_idx]
            policy_labor_working = policy_labor_working[:, :, :, time_idx]

            ############################################################################
            # Aggregate capital stock and employment
            ############################################################################

            (
                aggregate_capital_out_tmp,
                aggregate_labor_out_tmp,
                mass_distribution_full_working_tmp,
                mass_distribution_full_retired_tmp,
            ) = aggregate_hc(
                policy_capital_working=policy_capital_working,
                policy_hc_working=policy_hc_working,
                policy_labor_working=policy_labor_working,
                policy_capital_retired=policy_capital_retired,
                age_max=age_max,
                age_retire=age_retire,
                n_gridpoints_capital=n_gridpoints_capital,
                hc_init=hc_init,
                capital_grid=capital_grid,
                n_gridpoints_hc=n_gridpoints_hc,
                hc_grid=hc_grid,
                mass=mass,
                population_growth_rate=population_growth_rate,
                survival_rates=survival_rates,
            )

            aggregate_capital_out[time_idx + 1] = aggregate_capital_out_tmp
            aggregate_labor_out[time_idx] = aggregate_labor_out_tmp

            mass_distribution_full_working[
                :, :, 0
            ] = mass_distribution_full_working_init[:, :, 0]
            mass_distribution_full_working = mass_distribution_full_working_tmp
            mass_distribution_full_retired = mass_distribution_full_retired_tmp

        # Display results
        deviation_capital = max(abs(aggregate_capitalnew - aggregate_capital))
        deviation_labor = max(abs(aggregate_labornew - aggregate_labor))

        # Update the guess on capital and labor
        aggregate_capital = 0.8 * aggregate_capital + 0.2 * aggregate_capitalnew
        aggregate_labor = 0.8 * aggregate_labor + 0.2 * aggregate_labornew
        print("deviation-capital deviation-labor")
        print([deviation_capital, deviation_labor])

    results = {
        "mass_distribution_full_working": mass_distribution_full_working,
        "mass_distribution_full_retired": mass_distribution_full_retired,
    }

    return results


#####################################################
# SCRIPT
######################################################


if __name__ == "__main__":

    # model_name = sys.argv[1]
    transition_params = json.load(
        open(ppj("IN_MODEL_SPECS", "transition.json"), encoding="utf-8")
    )

    with open(ppj("OUT_ANALYSIS", "stationary_initial.pickle"), "rb") as in_file:
        results_stationary_initial = pickle.load(in_file)
    with open(ppj("OUT_ANALYSIS", "stationary_final.pickle"), "rb") as in_file:
        results_stationary_final = pickle.load(in_file)

    results_transition = solve_transition(
        results_stationary_initial, results_stationary_final, transition_params
    )

    with open(ppj("OUT_ANALYSIS", f"transition.pickle"), "wb") as out_file:
        pickle.dump(results_transition, out_file)
