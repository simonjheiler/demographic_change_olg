import numba as nb  # noqa:F401
import numpy as np

#########################################################################
# STANDARD MODEL WITH IDIOSYNCRATIC RISK AND NO HUMAN CAPITAL
#########################################################################


# @nb.njit
def aggregate_baseline_readable(
    policy_capital_working,
    policy_capital_retired,
    policy_labor_working,
    age_max,
    n_prod_states,
    n_gridpoints_capital,
    duration_working,
    productivity_init,
    transition_prod_states,
    efficiency,
    capital_grid,
    mass,
    duration_retired,
    population_growth_rate,
    prod_states,
):
    """ Calculate aggregate variables and cross-sectional distribution from HH policy functions.

    Arguments
    ---------
        policy_capital_working: np.array(n_prod_states, n_gridpoints_capital, duration_working)
            Savings policy function for working age agents (storing optimal asset choices by
            index on asset grid as int)
        policy_capital_retired: np.array(n_gridpoints_capital, duration_retired)
            Savings policy function for retired agents  (storing optimal asset choices by
            index on asset grid as int)
        policy_labor_working: np.array(n_prod_states, n_gridpoints_capital, duration_working)
            Labor supply policy function (storing optimal hours worked as np.float64)
        age_max: int
            Maximum age of agents
        n_prod_states: int
            Number of idiosyncratic productivity states
        n_gridpoints_capital: int
            Number of grid points of capital grid
        duration_working: int
            Length of working period
        productivity_init: np.array(2, 1)
            Initial distribution of idiosyncratic productivity states
        transition_prod_states: np.array(n_prod_states, n_prod_states)
            Transition probabilities for idiosyncratic productivity states
        efficiency: np.array(age_retire)
            Profile of age-dependent labor efficiency multipliers
        capital_grid: np.array(n_gridpoints_capital)
            Asset grid
        mass: np.array(age_max, 1)
            Vector of relative shares of agents by age
        duration_retired: int
            Length of retirement period
        population_growth_rate: np.float64
            Annual population growth rate
        prod_states: np.array(n_prod_states, 1)
            Current idiosyncratic productivity state
    Returns
    -------
        aggregate_capital_out: np.float64
            Aggregate capital stock derived from household policy functions and
            cross-sectional distribution
        aggregate_labor_out: np.float64
            Aggregate labor supply derived from household policy functions and
            cross-sectional distribution
        mass_distribution_age_assets: np.array(n_gridpoints_capital, age_max)
            Distribution of agents by asset holdings and age
        mass_distribution_working: np.array(
            n_prod_states, n_gridpoints_capital, duration_working
        )
            Distribution of working age agents by productivity state, asset holdings
            and age
        mass_distribution_retired np.array(n_gridpoints_capital, duration_retired)
            Distribution of retired agents by asset holdings and age
    """

    # Aggregate capital for each generation
    asset_distribution_age = np.zeros((age_max, 1), dtype=np.float64)

    # Distribution of workers over capital and shocks for each working cohort
    mass_distribution_working = np.zeros(
        (n_prod_states, n_gridpoints_capital, duration_working), dtype=np.float64,
    )

    # Newborns
    mass_distribution_working[0, 0, 0] = productivity_init[0] * mass[0]
    mass_distribution_working[1, 0, 0] = productivity_init[1] * mass[0]

    # Distribution of agents over capital for each cohort (pooling together
    # both productivity shocks). This would be useful when computing total
    # capital
    mass_distribution_age_assets = np.zeros(
        (n_gridpoints_capital, age_max), dtype=np.float64
    )
    mass_distribution_age_assets[0, 0] = mass[
        0
    ]  # Distribution of newborns over capital

    # Aggregate labor supply by generation
    labor_distribution_age = np.zeros((duration_working, 1), dtype=np.float64)

    # Distribution of retirees over capital
    mass_distribution_retired = np.zeros(
        (n_gridpoints_capital, duration_retired), dtype=np.float64
    )

    ############################################################################
    # Iterating over the distribution
    ############################################################################
    # Workers
    for age_idx in range(duration_working):  # iterations over cohorts
        for assets_this_period_idx in range(
            n_gridpoints_capital
        ):  # current asset holdings
            for productivity_idx in range(n_prod_states):  # current shock

                assets_next_period_idx = policy_capital_working[
                    productivity_idx, assets_this_period_idx, age_idx
                ]  # optimal saving (as index in asset grid)

                for productivity_next_period_idx in range(
                    n_prod_states
                ):  # tomorrow's shock

                    if age_idx < duration_working - 1:
                        mass_distribution_working[
                            productivity_next_period_idx,
                            assets_next_period_idx,
                            age_idx + 1,
                        ] += (
                            transition_prod_states[
                                productivity_idx, productivity_next_period_idx
                            ]
                            * mass_distribution_working[
                                productivity_idx, assets_this_period_idx, age_idx
                            ]
                            / (1 + population_growth_rate)
                        )
                    elif (
                        age_idx == duration_working - 1
                    ):  # need to be careful because workers transition to retirees
                        mass_distribution_retired[assets_next_period_idx, 0] += (
                            transition_prod_states[
                                productivity_idx, productivity_next_period_idx
                            ]
                            * mass_distribution_working[
                                productivity_idx, assets_this_period_idx, age_idx
                            ]
                            / (1 + population_growth_rate)
                        )

        # Aggregate labor by age
        labor_distribution_age[age_idx] = 0

        for assets_this_period_idx in range(n_gridpoints_capital):
            for productivity_idx in range(n_prod_states):
                labor_distribution_age[age_idx] += (
                    prod_states[productivity_idx]
                    * efficiency[age_idx]
                    * policy_labor_working[
                        productivity_idx, assets_this_period_idx, age_idx
                    ]
                    * mass_distribution_working[
                        productivity_idx, assets_this_period_idx, age_idx
                    ]
                )

        # Aggregate capital by age
        for assets_this_period_idx in range(n_gridpoints_capital):
            if age_idx < duration_working - 1:
                mass_distribution_age_assets[
                    assets_this_period_idx, age_idx + 1
                ] = np.sum(
                    mass_distribution_working[:, assets_this_period_idx, age_idx + 1]
                )
            else:
                mass_distribution_age_assets[
                    assets_this_period_idx, age_idx + 1
                ] = mass_distribution_retired[assets_this_period_idx, 0]
        asset_distribution_age[age_idx + 1] = np.dot(
            capital_grid, mass_distribution_age_assets[:, age_idx + 1]
        )

    # Retirees
    for age_idx in range(duration_retired - 1):  # iterations over cohort

        for assets_this_period_idx in range(
            n_gridpoints_capital
        ):  # current asset holdings
            assets_next_period_idx = policy_capital_retired[
                assets_this_period_idx, age_idx
            ]  # optimal saving (as index in asset grid)
            mass_distribution_retired[
                assets_next_period_idx, age_idx + 1
            ] += mass_distribution_retired[assets_this_period_idx, age_idx] / (
                1 + population_growth_rate
            )

        # Distribution by capital and age
        mass_distribution_age_assets[
            :, duration_working + age_idx
        ] = mass_distribution_retired[:, age_idx + 1]
        # Aggregate capital by age
        asset_distribution_age[duration_working + age_idx + 1] = np.dot(
            capital_grid,
            mass_distribution_age_assets[:, duration_working + age_idx + 1],
        )

    aggregate_capital_out = np.sum(asset_distribution_age)
    aggregate_labor_out = np.sum(labor_distribution_age)

    return (
        aggregate_capital_out,
        aggregate_labor_out,
        mass_distribution_age_assets,
        mass_distribution_working,
        mass_distribution_retired,
    )


#########################################################################
# ADAPTED MODEL WITHOUT IDIOSYNCRATIC RISK AND WITH HUMAN CAPITAL
#########################################################################


# @nb.njit
def aggregate_hc_readable(
    policy_capital_working,
    policy_hc_working,
    policy_labor_working,
    policy_capital_retired,
    age_max,
    age_retire,
    n_gridpoints_capital,
    hc_init,
    capital_grid,
    n_gridpoints_hc,
    hc_grid,
    mass,
    population_growth_rate,
    survival_rates,
):
    """ Calculate aggregate variables and cross-sectional distribution from HH policy functions.

    Arguments
    ---------
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
        hc_init: np.float64
            Initial level of human capital
        capital_grid: np.array(n_gridpoints_capital)
            Asset grid
        n_gridpoints_hc: np.int32
            Number of grid points of human capital grid
        hc_grid: np.array(n_gridpoints_capital)
            Human capital grid
        mass: np.array(age_max, 1)
            Vector of relative shares of agents by age
        population_growth_rate: np.float64
            Annual population growth rate
        survival_rates: np.array(age_max)
            Vector of conditional year-to-year survival rates
    Returns
    -------
        aggregate_capital_out: np.float64
            Aggregate capital stock derived from household policy functions and
            cross-sectional distribution
        aggregate_labor_out: np.float64
            Aggregate labor supply derived from household policy functions and
            cross-sectional distribution
        mass_distribution_full: np.array(n_gridpoints_capital, n_gridpoints_hc, age_max)
            Distribution of agents by asset holdings, human capital level and age
        mass_distribution_capital: np.array(n_gridpoints_capital, age_max)
            Distribution of agents by asset holdings and age
        mass_distribution_hc: np.array(n_gridpoints_hc, age_max)
            Distribution of agents by human capital level and age
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

    # make sure that all agents start with  correct initial level, i.e. hc = 1 and assets = 0
    mass_distribution_full_working[0, 0, 0] = mass[0]

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
                        / (1 + population_growth_rate)
                        * survival_rates[age_idx]
                    )
                elif age_idx == duration_working - 1:
                    mass_distribution_full_retired[assets_next_period_idx, 0] += (
                        mass_distribution_full_working[
                            assets_this_period_idx, hc_this_period_idx, age_idx
                        ]
                        / (1 + population_growth_rate)
                        * survival_rates[age_idx]
                    )

        for assets_this_period_idx in range(n_gridpoints_capital):
            for hc_this_period_idx in range(n_gridpoints_hc):
                labor_distribution_age[age_idx] += (
                    policy_labor_working[
                        assets_this_period_idx, hc_this_period_idx, age_idx
                    ]
                    * hc_grid[hc_this_period_idx]
                    * mass_distribution_full_working[
                        assets_this_period_idx, hc_this_period_idx, age_idx
                    ]
                )

        # Aggregate variables
        asset_distribution_age[age_idx] = np.dot(
            capital_grid, np.sum(mass_distribution_full_working, axis=1)[:, age_idx]
        )
        hc_distribution_age[age_idx] = np.dot(
            hc_grid, np.sum(mass_distribution_full_working, axis=0)[:, age_idx]
        )

    # Retirees
    for age_idx in range(duration_retired - 1):
        for assets_this_period_idx in range(n_gridpoints_capital):

            assets_next_period_idx = policy_capital_retired[
                assets_this_period_idx, age_idx
            ]
            mass_distribution_full_retired[assets_next_period_idx, age_idx + 1] += (
                mass_distribution_full_retired[assets_this_period_idx, age_idx]
                / (1 + population_growth_rate)
                * survival_rates[age_idx]
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


def aggregate_hc_vectorized(
    policy_capital_working,
    policy_hc_working,
    policy_labor_working,
    policy_capital_retired,
    age_max,
    age_retire,
    n_gridpoints_capital,
    hc_init,
    capital_grid,
    n_gridpoints_hc,
    hc_grid,
    mass,
    population_growth_rate,
    survival_rates,
):
    """ Calculate aggregate variables and cross-sectional distribution from HH policy functions.

    Arguments
    ---------
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
        hc_init: np.float64
            Initial level of human capital
        capital_grid: np.array(n_gridpoints_capital)
            Asset grid
        n_gridpoints_hc: np.int32
            Number of grid points of human capital grid
        hc_grid: np.array(n_gridpoints_capital)
            Human capital grid
        mass: np.array(age_max, 1)
            Vector of relative shares of agents by age
        population_growth_rate: np.float64
            Annual population growth rate
        survival_rates: np.array(age_max)
            Vector of conditional year-to-year survival rates
    Returns
    -------
        aggregate_capital_out: np.float64
            Aggregate capital stock derived from household policy functions and
            cross-sectional distribution
        aggregate_labor_out: np.float64
            Aggregate labor supply derived from household policy functions and
            cross-sectional distribution
        mass_distribution_full: np.array(n_gridpoints_capital, n_gridpoints_hc, age_max)
            Distribution of agents by asset holdings, human capital level and age
        mass_distribution_capital: np.array(n_gridpoints_capital, age_max)
            Distribution of agents by asset holdings and age
        mass_distribution_hc: np.array(n_gridpoints_hc, age_max)
            Distribution of agents by human capital level and age
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

    # make sure that all agents start with  correct initial level, i.e. hc = 1 and assets = 0
    mass_distribution_full_working[0, 0, 0] = mass[0]

    ############################################################################
    # Iterating over the distribution
    ############################################################################
    # Workers
    mass_distribution_full_next_period = np.zeros(
        (n_gridpoints_capital, n_gridpoints_hc), dtype=np.float64
    )

    for age_idx in range(duration_working):

        for assets_next_period_idx in range(n_gridpoints_capital):
            for hc_next_period_idx in range(n_gridpoints_hc):
                mass_distribution_full_next_period[
                    assets_next_period_idx, hc_next_period_idx
                ] = np.sum(
                    np.where(
                        np.logical_and(
                            policy_capital_working[:, :, age_idx]
                            == assets_next_period_idx,
                            policy_hc_working[:, :, age_idx] == hc_next_period_idx,
                        ),
                        mass_distribution_full_working[:, :, age_idx],
                        0.0,
                    )
                )

        if age_idx < duration_working - 1:
            mass_distribution_full_working[:, :, age_idx + 1] = (
                mass_distribution_full_next_period
                / (1 + population_growth_rate)
                * survival_rates[age_idx]
            )
        elif age_idx == duration_working - 1:
            mass_distribution_full_retired[:, 0] = (
                np.sum(mass_distribution_full_next_period, axis=1)
                / (1 + population_growth_rate)
                * survival_rates[age_idx]
            )

        for assets_this_period_idx in range(n_gridpoints_capital):
            for hc_this_period_idx in range(n_gridpoints_hc):
                labor_distribution_age[age_idx] += (
                    policy_labor_working[
                        assets_this_period_idx, hc_this_period_idx, age_idx
                    ]
                    * hc_grid[hc_this_period_idx]
                    * mass_distribution_full_working[
                        assets_this_period_idx, hc_this_period_idx, age_idx
                    ]
                )

        # Aggregate variables
        asset_distribution_age[age_idx] = np.dot(
            capital_grid, np.sum(mass_distribution_full_working, axis=1)[:, age_idx]
        )
        hc_distribution_age[age_idx] = np.dot(
            hc_grid, np.sum(mass_distribution_full_working, axis=0)[:, age_idx]
        )

    # Retirees
    for age_idx in range(duration_retired - 1):
        for assets_this_period_idx in range(n_gridpoints_capital):

            assets_next_period_idx = policy_capital_retired[
                assets_this_period_idx, age_idx
            ]
            mass_distribution_full_retired[assets_next_period_idx, age_idx + 1] += (
                mass_distribution_full_retired[assets_this_period_idx, age_idx]
                / (1 + population_growth_rate)
                * survival_rates[age_idx]
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
