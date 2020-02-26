"""
Solve for the transitional dynamics between two stationary equilibria of an
overlapping generations model with human capital accumulation adopted from
Ludwig, Schelkle, Vogel (2006).

"""
import json
import pickle

import numpy as np

from bld.project_paths import project_paths_join as ppj
from src.model_code.aggregate import aggregate_step
from src.model_code.auxiliary import set_continuous_point_on_grid
from src.model_code.solve import solve_retired
from src.model_code.solve import solve_working
from src.model_code.within_period import get_factor_prices


#####################################################
# PARAMETERS
#####################################################

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
tolerance_capital = np.float64(setup["tolerance_capital_transition"])
tolerance_labor = np.float64(setup["tolerance_labor_transition"])
max_iterations = np.int32(setup["max_iterations_transition"])
iteration_update = np.float64(setup["iteration_update_transition"])

# Load demographic parameters
efficiency = np.loadtxt(
    ppj("IN_DATA", "efficiency_multiplier.csv"), delimiter=",", dtype=np.float64
)
with open(ppj("OUT_DATA", "simulated_demographics.pickle"), "rb") as in_file:
    demographics = pickle.load(in_file)

survival_rates = demographics["survival_rates_transition"]
mass = demographics["mass_transition"]
fertility_rates = demographics["fertility_transition"]

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


def solve_transition(
    results_initial, results_final, transition_params,
):
    """Solve for equilibrium transitional dynamics.

    Solve for sets of value functions, policy functions and cross-sectional distributions
    of agents. First, guess initial path for aggregate capital and aggregate labor. Second,
    iterate backwards over time with the following steps: calculate factor prices, solve
    for current period optimal choices for assets and human capital (for all ages) given
    next periods value function (for all ages) by calculate sum of flow utility and expected
    discounted continuation value on meshgrids for all possible combinations for assets and
    human capital this period  and next period, then find maximal value over assets and human
    capital next period. After deriving policies for all time periods, initiate with cross-
    sectional distribution of initial stationary equilibrium and iterate forward over policy
    functions to obtain cross-sectional distributions for all time periods. Finally, aggregate
    over households to obtain transition paths for aggregate variables and verify initial guess.
    If tolerance for deviation is exceeded, update guess and repeat.

    Arguments:
        results_initial: dictionary
            Equilibrium results for the initial stationary equilibrium containing
            the variables "aggregate_capital", "aggregate_labor", "wage_rate", "interest_rate",
            "pension_benefit", "policy_capital_working", "policy_hc_working",
            "policy_labor_working", "policy_capital_retired", "value_retired", "value_working",
            "mass_distribution_full_working", and "mass_distribution_full_retired"
        results_final: dictionary
            Equilibrium results for the final stationary equilibrium containing
            the variables "aggregate_capital", "aggregate_labor", "wage_rate", "interest_rate",
            "pension_benefit", "policy_capital_working", "policy_hc_working",
            "policy_labor_working", "policy_capital_retired", "value_retired", "value_working",
            "mass_distribution_full_working", and "mass_distribution_full_retired"
        transition_params: dictionary
            Model specifications containing the variables "duration_transition",
            "mortality_adjustment_start_time", "mortality_adjustment_end_time",
            "mortality_adjustment_factor", "aggregate_capital_init", and "aggregate_labor_init"
    Returns:
        results: dictionary
            Transitional equilibrium results containing the variables "aggregate_capital",
            "aggregate_labor", "value_working", "value_retired", "policy_capital_working",
            "policy_hc_working", "policy_labor_working", "policy_capital_retired",
            "mass_distribution_full_working", and "mass_distribution_full_retired"
    """

    # Load final steady state
    aggregate_capital_final = results_final["aggregate_capital"]
    aggregate_labor_final = results_final["aggregate_labor"]
    value_retired_final = results_final["value_retired"]
    value_working_final = results_final["value_working"]
    policy_capital_retired_final = results_final["policy_capital_retired"]
    policy_capital_working_final = results_final["policy_capital_working"]
    policy_hc_working_final = results_final["policy_hc_working"]
    policy_labor_working_final = results_final["policy_labor_working"]

    # Load initial steady state
    aggregate_capital_initial = results_initial["aggregate_capital"]
    aggregate_labor_initial = results_initial["aggregate_labor"]
    mass_distribution_full_working_init = results_initial[
        "mass_distribution_full_working"
    ]
    mass_distribution_full_retired_init = results_initial[
        "mass_distribution_full_retired"
    ]

    # Load initial and final income tax rate
    model_specs_initial = json.load(
        open(ppj("IN_MODEL_SPECS", "stationary_initial.json"), encoding="utf-8")
    )
    income_tax_rate_initial = model_specs_initial["income_tax_rate"]
    model_specs_final = json.load(
        open(ppj("IN_MODEL_SPECS", "stationary_final.json"), encoding="utf-8")
    )
    income_tax_rate_final = model_specs_final["income_tax_rate"]

    # Basic parameters
    duration_transition = transition_params["duration_transition"]

    # # Initial guesses on the paths of K and L
    aggregate_capital_in = np.zeros((duration_transition + 1), dtype=np.float64)
    aggregate_labor_in = np.zeros((duration_transition + 1), dtype=np.float64)

    # Check whether we can use the paths for K and L from previously run executions
    try:
        with open(ppj("OUT_ANALYSIS", "transition.pickle"), "rb") as in_file:
            results_transition_aux = pickle.load(in_file)
        duration_transition_aux = len(results_transition_aux["aggregate_capital"]) - 1
        aggregate_capital_aux = results_transition_aux["aggregate_capital"]
        aggregate_labor_aux = results_transition_aux["aggregate_labor"]
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
            aggregate_capital_in[: duration_transition_aux + 1] = aggregate_capital_aux
            aggregate_capital_in[duration_transition_aux + 1 :] = np.repeat(
                aggregate_capital_aux[-1], duration_transition - duration_transition_aux
            )
            aggregate_labor_in[: duration_transition_aux + 1] = aggregate_labor_aux
            aggregate_labor_in[duration_transition_aux + 1 :] = np.repeat(
                aggregate_labor_aux[-1], duration_transition - duration_transition_aux,
            )
        else:
            aggregate_capital_in = aggregate_capital_aux
            aggregate_labor_in = aggregate_labor_aux
    except FileNotFoundError:
        try:
            aggregate_capital_in = np.array(transition_params["aggregate_capital_init"])
            aggregate_labor_in = np.array(transition_params["aggregate_labor_init"])
        except KeyError:
            aggregate_capital_in = np.linspace(
                aggregate_capital_initial,
                aggregate_capital_final,
                duration_transition + 1,
            )
            aggregate_labor_in = np.linspace(
                aggregate_labor_initial, aggregate_labor_final, duration_transition + 1,
            )

    # Construct policy rate path
    income_tax_rate = np.linspace(
        income_tax_rate_initial, income_tax_rate_final, duration_transition + 1
    )

    # Initialize iteration
    num_iterations = 0
    deviation_capital = 10
    deviation_labor = 10
    neg = np.float64(-1e10)

    ################################################################
    # Loop over path for capital, labor and pension benefits
    ################################################################

    # Initialize objects for iteration
    value_retired = np.zeros(
        (n_gridpoints_capital, duration_retired, duration_transition + 1),
        dtype=np.float64,
    )
    value_working = np.zeros(
        (
            n_gridpoints_capital,
            n_gridpoints_hc,
            duration_working,
            duration_transition + 1,
        ),
        dtype=np.float64,
    )
    policy_capital_retired = np.zeros(
        (n_gridpoints_capital, duration_retired, duration_transition + 1),
        dtype=np.int32,
    )
    policy_capital_working = np.zeros(
        (
            n_gridpoints_capital,
            n_gridpoints_hc,
            duration_working,
            duration_transition + 1,
        ),
        dtype=np.int32,
    )
    policy_hc_working = np.zeros(
        (
            n_gridpoints_capital,
            n_gridpoints_hc,
            duration_working,
            duration_transition + 1,
        ),
        dtype=np.int32,
    )
    policy_labor_working = np.zeros(
        (
            n_gridpoints_capital,
            n_gridpoints_hc,
            duration_working,
            duration_transition + 1,
        ),
        dtype=np.float64,
    )

    # Store final steady state values and policies as last period in transition duration
    value_retired[:, :, -1] = value_retired_final
    value_working[:, :, :, -1] = value_working_final
    policy_capital_retired[:, :, -1] = policy_capital_retired_final
    policy_capital_working[:, :, :, -1] = policy_capital_working_final
    policy_hc_working[:, :, :, -1] = policy_hc_working_final
    policy_labor_working[:, :, :, -1] = policy_labor_working_final

    # Initializations
    mass_distribution_full_working = np.zeros(
        (
            n_gridpoints_capital,
            n_gridpoints_hc,
            duration_working,
            duration_transition + 1,
        ),
        dtype=np.float64,
    )
    mass_distribution_full_retired = np.zeros(
        (n_gridpoints_capital, duration_retired, duration_transition + 1),
        dtype=np.float64,
    )

    # Initial distribution is stationary distribution of initial equilibrium
    mass_distribution_full_working[:, :, :, 0] = mass_distribution_full_working_init
    mass_distribution_full_retired[:, :, 0] = mass_distribution_full_retired_init

    aggregate_capital_out = np.zeros((duration_transition + 1), dtype=np.float64)
    aggregate_labor_out = np.zeros((duration_transition + 1), dtype=np.float64)
    aggregate_capital_out[0] = aggregate_capital_in[0]

    # Iterate
    while (num_iterations < max_iterations) and (
        (abs(deviation_capital) > tolerance_capital)
        or (abs(deviation_labor) > tolerance_labor)
    ):
        num_iterations += 1

        print(f"Iteration {num_iterations} out of {max_iterations}")

        for time_idx in range(duration_transition - 1, -1, -1):

            # Load aggregate variables
            aggregate_capital_tmp = aggregate_capital_in[time_idx]
            aggregate_labor_tmp = aggregate_labor_in[time_idx]
            income_tax_rate_tmp = income_tax_rate[time_idx]
            mass_tmp = mass[:, time_idx]

            # Load next period value functions
            value_retired_current = value_retired[:, :, time_idx + 1]
            value_working_current = value_working[:, :, :, time_idx + 1]

            # Calculate factor prices from aggregates
            (interest_rate, wage_rate, pension_benefit) = get_factor_prices(
                aggregate_capital=aggregate_capital_tmp,
                aggregate_labor=aggregate_labor_tmp,
                alpha=alpha,
                delta_k=delta_k,
                income_tax_rate=income_tax_rate_tmp,
                mass=mass_tmp,
                age_retire=age_retire,
            )

            ############################################################
            # BACKWARD INDUCTION
            ############################################################

            # Retired agents
            # Initiate objects to store temporary policy and value functions
            policy_capital_retired_tmp = np.zeros(n_gridpoints_capital, dtype=np.int32)
            value_retired_tmp = np.zeros(n_gridpoints_capital, dtype=np.float64)

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

            # Iterate backwards through retirement period
            for age_idx in range(duration_retired - 2, -1, -1):
                # Look up continuation values for assets_next_period
                value_next_period = value_retired_current[:, age_idx + 1]
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
                    survival_rate=survival_rates[age_idx, time_idx],
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
                        value_retired_current[:, 0, np.newaxis], n_gridpoints_hc, axis=1
                    )
                else:
                    value_next_period = value_working_current[:, :, age_idx + 1]

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
                    income_tax_rate=income_tax_rate[time_idx],
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
                    survival_rate=survival_rates[age_idx, time_idx],
                )

                # Store results
                policy_capital_working[
                    :, :, age_idx, time_idx
                ] = policy_capital_working_tmp
                policy_hc_working[:, :, age_idx, time_idx] = policy_hc_working_tmp
                policy_labor_working[:, :, age_idx, time_idx] = policy_labor_working_tmp
                value_working[:, :, age_idx, time_idx] = value_working_tmp

        ############################################################################
        # Iterating over the distribution
        ############################################################################

        for time_idx in range(duration_transition):

            # Load current demographic parameters
            population_growth_rate_current = fertility_rates[time_idx] - 1
            survival_rates_current = survival_rates[:, time_idx]

            # Load current policy functions
            policy_capital_retired_current = policy_capital_retired[:, :, time_idx]
            policy_capital_working_current = policy_capital_working[:, :, :, time_idx]
            policy_hc_working_current = policy_hc_working[:, :, :, time_idx]
            policy_labor_working_current = policy_labor_working[:, :, :, time_idx]

            # Load current mass distribution
            mass_distribution_full_working_in = mass_distribution_full_working[
                :, :, :, time_idx
            ]
            mass_distribution_full_retired_in = mass_distribution_full_retired[
                :, :, time_idx
            ]

            ############################################################################
            # Aggregate capital stock and employment
            ############################################################################

            (
                aggregate_capital_out_tmp,
                aggregate_labor_out_tmp,
                mass_distribution_full_working_tmp,
                mass_distribution_full_retired_tmp,
            ) = aggregate_step(
                mass_distribution_full_working_in=mass_distribution_full_working_in,
                mass_distribution_full_retired_in=mass_distribution_full_retired_in,
                policy_capital_working=policy_capital_working_current,
                policy_hc_working=policy_hc_working_current,
                policy_labor_working=policy_labor_working_current,
                policy_capital_retired=policy_capital_retired_current,
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
                population_growth_rate=population_growth_rate_current,
                survival_rates=survival_rates_current,
                efficiency=efficiency,
            )

            # Store results
            aggregate_capital_out[time_idx + 1] = aggregate_capital_out_tmp
            aggregate_labor_out[time_idx] = aggregate_labor_out_tmp

            mass_distribution_full_working[
                :, :, :, time_idx + 1
            ] = mass_distribution_full_working_tmp
            mass_distribution_full_retired[
                :, :, time_idx + 1
            ] = mass_distribution_full_retired_tmp

        # Display results
        labor_distribution_age_tmp = np.zeros(age_retire, dtype=np.float64)
        for age_idx in range(duration_working):
            for assets_this_period_idx in range(n_gridpoints_capital):
                for hc_this_period_idx in range(n_gridpoints_hc):

                    labor_distribution_age_tmp[age_idx] += (
                        policy_labor_working[
                            assets_this_period_idx, hc_this_period_idx, age_idx, -1
                        ]
                        * hc_grid[hc_this_period_idx]
                        * efficiency[age_idx]
                        * mass_distribution_full_working[
                            assets_this_period_idx, hc_this_period_idx, age_idx, -1
                        ]
                    )

        aggregate_labor_out[-1] = np.sum(labor_distribution_age_tmp)

        deviation_capital = max(abs(aggregate_capital_in - aggregate_capital_out))
        deviation_labor = max(abs(aggregate_labor_in - aggregate_labor_out))

        # Update the guess on capital and labor
        aggregate_capital_in = (
            1 - iteration_update
        ) * aggregate_capital_in + iteration_update * aggregate_capital_out
        aggregate_labor_in = (
            1 - iteration_update
        ) * aggregate_labor_in + iteration_update * aggregate_labor_out

        print("deviation-capital deviation-labor")
        print([deviation_capital, deviation_labor])

    results = {
        "aggregate_capital": aggregate_capital_in,
        "aggregate_labor": aggregate_labor_in,
        "value_working": value_working,
        "value_retired": value_retired,
        "policy_capital_working": policy_capital_working,
        "policy_hc_working": policy_hc_working,
        "policy_labor_working": policy_labor_working,
        "policy_capital_retired": policy_capital_retired,
        "mass_distribution_full_working": mass_distribution_full_working,
        "mass_distribution_full_retired": mass_distribution_full_retired,
    }

    return results


#####################################################
# SCRIPT
#####################################################


if __name__ == "__main__":

    transition_params_model = json.load(
        open(
            ppj("IN_MODEL_SPECS", "transition_constant_tax_rate.json"), encoding="utf-8"
        )
    )

    with open(ppj("OUT_ANALYSIS", "stationary_initial.pickle"), "rb") as in_file:
        results_stationary_initial = pickle.load(in_file)
    with open(ppj("OUT_ANALYSIS", "stationary_final.pickle"), "rb") as in_file:
        results_stationary_final = pickle.load(in_file)

    # results_transition = solve_transition(
    #     results_stationary_initial, results_stationary_final, transition_params_model
    # )
    with open(ppj("OUT_ANALYSIS", "transition.pickle"), "rb") as in_file:
        results_transition = pickle.load(in_file)

    with open(ppj("OUT_ANALYSIS", f"transition.pickle"), "wb") as out_file:
        pickle.dump(results_transition, out_file)
