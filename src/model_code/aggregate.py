"""Conduct forward iteration of cross-sectional distributions of agents given
    policy functions.
"""
import numpy as np

#########################################################################
# ADAPTED MODEL WITHOUT IDIOSYNCRATIC RISK AND WITH HUMAN CAPITAL
#########################################################################


def aggregate_stationary(
    policy_capital_working,
    policy_hc_working,
    policy_labor_working,
    policy_capital_retired,
    age_max,
    age_retire,
    n_gridpoints_capital,
    capital_grid,
    n_gridpoints_hc,
    hc_grid,
    assets_init_gridpoints,
    assets_init_weights,
    hc_init_gridpoints,
    hc_init_weights,
    survival_rates,
    efficiency,
    mass_newborns,
):
    """ Calculate stationary cross-sectional (asset and human capital) distribution,
        aggregate labor and aggregate saving from household policy functions.

        Start from a given initial mass of households and initial asset and human capital
        levels, apply constant population growth rate to calculate mass of past newborns,
        constant survival rates to simulate transition of past newborn to current age-t agents
        and apply policy functions for asset holdings, human capital levels and labor
        supply to obtain current period aggregates (assets, human capital, labor supply) and
        cross-sectional distribution over state variables (assets, human capital).

    Arguments:
        policy_capital_working: np.array(n_gridpoints_capital, n_gridpoints_hc, age_max)
            Savings policy function for working age agents (storing optimal asset choices
            by index on asset grid as int)
        policy_hc_working: np.array(n_gridpoints_capital, n_gridpoints_hc, age_max)
            Human capital policy function (storing optimal human capital choices by
            index on human capital grid as int)
        policy_labor_working: np.array(n_gridpoints_capital, n_gridpoints_hc, age_max)
            Labor supply policy function (storing optimal hours worked as np.float64)
        policy_capital_retired: np.array(n_gridpoints_hc, duration_retired)
            Savings policy function for retired agents (storing optimal asset choices
            by index on asset grid as int)
        age_max: np.int32
            Maximum age of agents
        age_retire: np.int32
            Retirement age of agents
        n_gridpoints_capital: np.int32
            Number of grid points of capital grid
        capital_grid: np.array(n_gridpoints_capital)
            Asset grid
        n_gridpoints_hc: np.int32
            Number of grid points of human capital grid
        hc_grid: np.array(n_gridpoints_capital)
            Human capital grid
        assets_init_gridpoints: np.array(2)
            Interpolation indices (int) of initial asset level on capital grid
        assets_init_weights: np.array(2)
            Interpolation weights (float) of initial asset level on capital grid
        hc_init_gridpoints: np.array(2)
            Interpolation indices (int) of initial human capital level on human capital grid
        hc_init_weights: np.array(2)
            Interpolation weights (float) of initial human capital level on human capital grid
        mass_newborns: np.float64
            Mass of newborn agents in stationary equilibrium
        survival_rates: np.array(age_max)
            Vector of conditional year-to-year survival rates
        efficiency: np.array(age_max)
            Vector of age-dependent labor efficiency multipliers
    Returns:
        aggregate_capital_out: np.float64
            Aggregate capital stock derived from household policy functions and
            cross-sectional distribution
        aggregate_labor_out: np.float64
            Aggregate labor supply derived from household policy functions and
            cross-sectional distribution
        mass_distribution_full_working: np.array(n_gridpoints_capital, n_gridpoints_hc,
            duration_working)
            Cross-sectional distribution of working age agents by asset holdings, human capital
            level and age
        mass_distribution_full_retired: np.array(n_gridpoints_capital, duration_retired)
            Cross-sectional distribution of retired agents by asset holdings and age
    """
    # Initialize objects for forward iteration
    duration_retired = age_max - age_retire + 1  # length of retirement
    duration_working = age_retire - 1  # length of working life

    # Aggregate variables by age (i.e. for each generation)
    asset_distribution_age = np.zeros(age_max, dtype=np.float64)
    hc_distribution_age = np.zeros(age_max, dtype=np.float64)
    labor_distribution_age = np.zeros(age_max, dtype=np.float64)

    # Distributions of agents
    mass_distribution_full_working = np.zeros(
        (n_gridpoints_capital, n_gridpoints_hc, duration_working), dtype=np.float64,
    )
    mass_distribution_full_retired = np.zeros(
        (n_gridpoints_capital, duration_retired), dtype=np.float64
    )

    # Store mass of newborn agents at initial node
    for i in range(2):
        for j in range(2):
            mass_distribution_full_working[
                assets_init_gridpoints[i], hc_init_gridpoints[j], 0
            ] = (mass_newborns * assets_init_weights[i] * hc_init_weights[j])

    ############################################################################
    # Iterating over the distribution
    ############################################################################
    # Workers
    for age_idx in range(duration_working):
        for assets_this_period_idx in range(n_gridpoints_capital):
            for hc_this_period_idx in range(n_gridpoints_hc):

                assets_next_period_idx = policy_capital_working[
                    assets_this_period_idx, hc_this_period_idx, age_idx
                ]
                hc_next_period_idx = policy_hc_working[
                    assets_this_period_idx, hc_this_period_idx, age_idx
                ]

                if age_idx < duration_working - 1:
                    mass_distribution_full_working[
                        assets_next_period_idx, hc_next_period_idx, age_idx + 1
                    ] += (
                        mass_distribution_full_working[
                            assets_this_period_idx, hc_this_period_idx, age_idx
                        ]
                        * survival_rates[age_idx]
                    )
                elif age_idx == duration_working - 1:
                    mass_distribution_full_retired[assets_next_period_idx, 0] += (
                        mass_distribution_full_working[
                            assets_this_period_idx, hc_this_period_idx, age_idx
                        ]
                        * survival_rates[age_idx]
                    )

        for assets_this_period_idx in range(n_gridpoints_capital):
            for hc_this_period_idx in range(n_gridpoints_hc):
                labor_distribution_age[age_idx] += (
                    policy_labor_working[
                        assets_this_period_idx, hc_this_period_idx, age_idx
                    ]
                    * hc_grid[hc_this_period_idx]
                    * efficiency[age_idx]
                    * mass_distribution_full_working[
                        assets_this_period_idx, hc_this_period_idx, age_idx
                    ]
                )

        # Aggregate variables
        if age_idx < duration_working - 1:
            asset_distribution_age[age_idx + 1] = np.dot(
                capital_grid,
                np.sum(mass_distribution_full_working, axis=1)[:, age_idx + 1],
            )
            hc_distribution_age[age_idx + 1] = np.dot(
                hc_grid, np.sum(mass_distribution_full_working, axis=0)[:, age_idx + 1]
            )
        elif age_idx == duration_working - 1:
            asset_distribution_age[age_idx + 1] = np.dot(
                capital_grid, mass_distribution_full_retired[:, 0]
            )

    # Retirees
    for age_idx in range(duration_retired - 1):
        for assets_this_period_idx in range(n_gridpoints_capital):

            assets_next_period_idx = policy_capital_retired[
                assets_this_period_idx, age_idx
            ]
            mass_distribution_full_retired[assets_next_period_idx, age_idx + 1] += (
                mass_distribution_full_retired[assets_this_period_idx, age_idx]
                * survival_rates[age_retire - 1 + age_idx]
            )

        # Aggregate assets and human capital
        asset_distribution_age[duration_working + age_idx + 1] = np.dot(
            capital_grid, mass_distribution_full_retired[:, age_idx + 1]
        )

    aggregate_capital_out = np.sum(asset_distribution_age)
    aggregate_labor_out = np.sum(labor_distribution_age)

    return (
        aggregate_capital_out,
        aggregate_labor_out,
        mass_distribution_full_working,
        mass_distribution_full_retired,
    )


def aggregate_step(
    mass_distribution_full_working_in,
    mass_distribution_full_retired_in,
    policy_capital_working,
    policy_hc_working,
    policy_labor_working,
    policy_capital_retired,
    age_max,
    age_retire,
    n_gridpoints_capital,
    capital_grid,
    n_gridpoints_hc,
    hc_grid,
    assets_init_gridpoints,
    assets_init_weights,
    hc_init_gridpoints,
    hc_init_weights,
    population_growth_rate,
    survival_rates,
    efficiency,
):
    """ Calculate 1-period evolution of cross-sectional (asset and human capital) distribution,
        aggregate labor and aggregate saving from household policy functions.

        Start from a given cross-sectional distribution and, iterating through all possible
        states, apply policy functions for asset holdings, human capital levels and labor
        supply to obtain current period aggregates (assets, human capital, labor supply) and
        next periods cross-sectional distribution over state variables (assets, human capital).

    Arguments:
        mass_distribution_full_working_in: np.array(n_gridpoints_capital, n_gridpoints_hc,
            duration_working)
            Current period cross-sectional distribution of working age households by assets,
            human capital and age
        mass_distribution_full_retired_in: np.array(n_gridpoints_capital, duration_retired)
            Current period cross-sectional distribution of retired households by assets and age
        policy_capital_working: np.array(n_gridpoints_capital, n_gridpoints_hc, age_max)
            Savings policy function for working age agents (storing optimal asset choices
            by index on asset grid as int)
        policy_hc_working: np.array(n_gridpoints_capital, n_gridpoints_hc, age_max)
            Human capital policy function (storing optimal human capital choices by
            index on human capital grid as int)
        policy_labor_working: np.array(n_gridpoints_capital, n_gridpoints_hc, age_max)
            Labor supply policy function (storing optimal hours worked as np.float64)
        policy_capital_retired: np.array(n_gridpoints_hc, duration_retired)
            Savings policy function for retired agents (storing optimal asset choices
            by index on asset grid as int)
        age_max: np.int32
            Maximum age of agents
        age_retire: np.int32
            Retirement age of agents
        n_gridpoints_capital: np.int32
            Number of grid points of capital grid
        capital_grid: np.array(n_gridpoints_capital)
            Asset grid
        n_gridpoints_hc: np.int32
            Number of grid points of human capital grid
        hc_grid: np.array(n_gridpoints_capital)
            Human capital grid
        assets_init_gridpoints: np.array(2)
            Interpolation indices (int) of initial asset level on capital grid
        assets_init_weights: np.array(2)
            Interpolation weights (float) of initial asset level on capital grid
        hc_init_gridpoints: np.array(2)
            Interpolation indices (int) of initial human capital level on human capital grid
        hc_init_weights: np.array(2)
            Interpolation weights (float) of initial human capital level on human capital grid
        population_growth_rate: np.float64
            Growth rate of newborns from current period to next period
        survival_rates: np.array(age_max)
            Vector of conditional year-to-year survival rates
        efficiency: np.array(age_max)
            Vector of age-dependent labor efficiency multipliers
    Returns:
        aggregate_capital_out: np.float64
            Current period aggregate savings (pre interest payment)
        aggregate_labor_out: np.float64
            Current period aggregate labor supply
        mass_distribution_full_working_out: np.array(n_gridpoints_capital, n_gridpoints_hc,
            age_max)
            Next periods distribution of working age agents by asset holdings, human capital
            level and age
        mass_distribution_full_retired_out: np.array(n_gridpoints_capital, age_max)
            Next periods distribution of retired agents by asset holdings and age
    """
    # Initialize objects for forward iteration
    duration_retired = age_max - age_retire + 1  # length of retirement
    duration_working = age_retire - 1  # length of working life

    # Aggregate variables by age (i.e. for each generation)
    asset_distribution_age = np.zeros(age_max, dtype=np.float64)
    hc_distribution_age = np.zeros(age_max, dtype=np.float64)
    labor_distribution_age = np.zeros(age_max, dtype=np.float64)

    # Distributions of agents
    mass_distribution_full_working_out = np.zeros(
        (n_gridpoints_capital, n_gridpoints_hc, duration_working), dtype=np.float64,
    )
    mass_distribution_full_retired_out = np.zeros(
        (n_gridpoints_capital, duration_retired), dtype=np.float64
    )

    # Store mass of newborn agents at initial node
    for i in range(2):
        for j in range(2):
            mass_distribution_full_working_out[
                assets_init_gridpoints[i], hc_init_gridpoints[j], 0
            ] = (
                np.sum(mass_distribution_full_working_in[:, :, 0])
                * (1 + population_growth_rate)
                * assets_init_weights[i]
                * hc_init_weights[j]
            )

    ############################################################################
    # Iterating over the distribution
    ############################################################################
    # Workers
    for age_idx in range(duration_working):
        for assets_this_period_idx in range(n_gridpoints_capital):
            for hc_this_period_idx in range(n_gridpoints_hc):

                assets_next_period_idx = policy_capital_working[
                    assets_this_period_idx, hc_this_period_idx, age_idx
                ]
                hc_next_period_idx = policy_hc_working[
                    assets_this_period_idx, hc_this_period_idx, age_idx
                ]

                if age_idx < duration_working - 1:
                    mass_distribution_full_working_out[
                        assets_next_period_idx, hc_next_period_idx, age_idx + 1
                    ] += (
                        mass_distribution_full_working_in[
                            assets_this_period_idx, hc_this_period_idx, age_idx
                        ]
                        * survival_rates[age_idx]
                    )
                elif age_idx == duration_working - 1:
                    mass_distribution_full_retired_out[assets_next_period_idx, 0] += (
                        mass_distribution_full_working_in[
                            assets_this_period_idx, hc_this_period_idx, age_idx
                        ]
                        * survival_rates[age_idx]
                    )

                labor_distribution_age[age_idx] += (
                    policy_labor_working[
                        assets_this_period_idx, hc_this_period_idx, age_idx
                    ]
                    * hc_grid[hc_this_period_idx]
                    * efficiency[age_idx]
                    * mass_distribution_full_working_in[
                        assets_this_period_idx, hc_this_period_idx, age_idx
                    ]
                )

        # Aggregate variables
        if age_idx < duration_working - 1:
            asset_distribution_age[age_idx + 1] = np.dot(
                capital_grid,
                np.sum(mass_distribution_full_working_out, axis=1)[:, age_idx + 1],
            )
            hc_distribution_age[age_idx + 1] = np.dot(
                hc_grid,
                np.sum(mass_distribution_full_working_out, axis=0)[:, age_idx + 1],
            )
        elif age_idx == duration_working - 1:
            asset_distribution_age[age_idx + 1] = np.dot(
                capital_grid, mass_distribution_full_retired_out[:, 0]
            )

    # Retirees
    for age_idx in range(duration_retired - 1):
        for assets_this_period_idx in range(n_gridpoints_capital):

            assets_next_period_idx = policy_capital_retired[
                assets_this_period_idx, age_idx
            ]
            mass_distribution_full_retired_out[assets_next_period_idx, age_idx + 1] += (
                mass_distribution_full_retired_in[assets_this_period_idx, age_idx]
                * survival_rates[age_idx]
            )

        # Aggregate assets and human capital
        asset_distribution_age[duration_working + age_idx + 1] = np.dot(
            capital_grid, mass_distribution_full_retired_out[:, age_idx + 1]
        )

    aggregate_capital_out = np.sum(asset_distribution_age)
    aggregate_labor_out = np.sum(labor_distribution_age)

    return (
        aggregate_capital_out,
        aggregate_labor_out,
        mass_distribution_full_working_out,
        mass_distribution_full_retired_out,
    )
