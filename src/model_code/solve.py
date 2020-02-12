import numba as nb
import numpy as np

from src.model_code.within_period import get_consumption
from src.model_code.within_period import get_consumption_hc
from src.model_code.within_period import get_hc_effort
from src.model_code.within_period import get_labor_input
from src.model_code.within_period import get_labor_input_hc
from src.model_code.within_period import util


#########################################################################
# STANDARD MODEL WITH IDIOSYNCRATIC RISK AND NO HUMAN CAPITAL
#########################################################################


@nb.njit
def solve_by_backward_induction_baseline_readable(
    interest_rate,
    wage_rate,
    capital_grid,
    n_gridpoints_capital,
    sigma,
    gamma,
    pension_benefit,
    neg,
    duration_retired,
    duration_working,
    n_prod_states,
    income_tax_rate,
    beta,
    prod_states,
    efficiency,
    transition_prod_states,
):
    """ Calculate household policy functions.

    Arguments
    ---------
        interest_rate: np.float64
            Current interest rate on capital holdings
        wage_rate: np.float64
            Current wage rate on effective labor input
        capital_grid: np.array(n_gridpoints_capital)
            Asset grid
        n_gridpoints_capital: int
            Number of grid points of capital grid
        sigma: np.float64
            Inverse of inter-temporal elasticity of substitution
        gamma: np.float64
            Weight of consumption utility vs. leisure utility
        pension_benefit: np.float64
            Income from pension benefits
        neg: np.float64
            Very small number
        duration_working: int
            Length of working period
        duration_retired: int
            Length of retirement period
        n_prod_states: int
            Number of idiosyncratic productivity states
        income_tax_rate: np.float64
            Tax rate on labor income
        beta: np.float64
            Time discount factor
        prod_states: np.array(2, 1)
            Current idiosyncratic productivity state
        efficiency: np.array(age_retire, 1)
            Age-dependent labor efficiency multiplier
        transition_prod_states: np.array(n_prod_states, n_prod_states)
            Transition probabilities for idiosyncratic productivity states
    Returns
    -------
        policy_capital_working: np.array(n_prod_states, n_gridpoints_capital, duration_working)
            Savings policy function for working age agents (storing optimal asset choices by
            index on asset grid as int)
        policy_capital_retired: np.array(n_gridpoints_capital, duration_retired)
            Savings policy function for retired agents  (storing optimal asset choices by
            index on asset grid as int)
        policy_labor_working: np.array(n_prod_states, n_gridpoints_capital, duration_working)
            Labor supply policy function (storing optimal hours worked as np.float64)
    """

    # Initialize objects for backward induction
    value_retired = np.zeros((n_gridpoints_capital, duration_retired), dtype=np.float64)
    policy_capital_retired = np.zeros(
        (n_gridpoints_capital, duration_retired), dtype=np.int8
    )
    value_working = np.zeros(
        (n_prod_states, n_gridpoints_capital, duration_working), dtype=np.float64,
    )
    policy_capital_working = np.zeros(
        (n_prod_states, n_gridpoints_capital, duration_working), dtype=np.int8
    )
    policy_labor_working = np.zeros(
        (n_prod_states, n_gridpoints_capital, duration_working), dtype=np.float64,
    )

    ############################################################
    # BACKWARD INDUCTION
    ############################################################

    # Retired households

    # Last period utility
    consumption_last = (1 + interest_rate) * capital_grid + pension_benefit
    flow_utility_last = (consumption_last ** ((1 - sigma) * gamma)) / (1 - sigma)
    value_retired[:, duration_retired - 1] = flow_utility_last

    for age_idx in range(duration_retired - 2, -1, -1):  # age

        for assets_this_period_idx in range(n_gridpoints_capital):  # assets today

            # Initialize right-hand side of Bellman equation
            value_current_max = neg
            assets_next_period_idx = -1
            # More efficient is to use:
            # l = min(policy_capital_retired(max(j-1,1),i),n_gridpoints_capital-1)-1;

            # Loop over all k's in the capital grid to find the value,
            # which gives max of the right-hand side of Bellman equation

            while assets_next_period_idx < n_gridpoints_capital - 1:

                assets_next_period_idx += 1
                assets_this_period = capital_grid[assets_this_period_idx]
                assets_next_period = capital_grid[assets_next_period_idx]

                # Instantaneous utility
                consumption = get_consumption(
                    assets_this_period=assets_this_period,
                    assets_next_period=assets_next_period,
                    pension_benefit=pension_benefit,
                    labor_input=np.float64(0.0),
                    interest_rate=interest_rate,
                    wage_rate=wage_rate,
                    income_tax_rate=income_tax_rate,
                    productivity=np.float64(0.0),
                    efficiency=np.float64(0.0),
                )

                if consumption <= 0.0:
                    flow_utility = neg
                    assets_next_period_idx = n_gridpoints_capital - 1
                else:
                    flow_utility = util(
                        consumption=consumption,
                        labor_input=np.float64(0.0),
                        hc_effort=np.float64(0.0),
                        gamma=gamma,
                        sigma=sigma,
                    )

                # Right-hand side of Bellman equation
                value_current = (
                    flow_utility
                    + beta * value_retired[assets_next_period_idx, age_idx + 1]
                )

                # Store indirect utility and optimal saving
                if value_current > value_current_max:
                    value_retired[assets_this_period_idx, age_idx] = value_current
                    policy_capital_retired[
                        assets_this_period_idx, age_idx
                    ] = assets_next_period_idx
                    value_current_max = value_current

    # Working households
    for age_idx in range(duration_working - 1, -1, -1):
        for productivity_idx in range(n_prod_states):
            for assets_this_period_idx in range(n_gridpoints_capital):

                # Initialize right-hand side of Bellman equation
                value_current_max = neg
                assets_next_period_idx = -1
                # More efficient is to use:
                # l=min(policy_capital_working(e,max(j-1,1),i),n_gridpoints_capital-1)-1;

                while (
                    assets_next_period_idx < n_gridpoints_capital - 1
                ):  # assets tomorrow
                    assets_next_period_idx += 1
                    assets_this_period = capital_grid[assets_this_period_idx]
                    assets_next_period = capital_grid[assets_next_period_idx]

                    # Optimal labor supply
                    lab = get_labor_input(
                        assets_this_period=assets_this_period,
                        assets_next_period=assets_next_period,
                        interest_rate=interest_rate,
                        wage_rate=wage_rate,
                        income_tax_rate=income_tax_rate,
                        productivity=np.float64(prod_states[productivity_idx]),
                        efficiency=np.float64(efficiency[age_idx]),
                        gamma=gamma,
                    )

                    # Instantaneous utility
                    consumption = get_consumption(
                        assets_this_period=assets_this_period,
                        assets_next_period=assets_next_period,
                        pension_benefit=np.float64(0.0),
                        labor_input=lab,
                        interest_rate=interest_rate,
                        wage_rate=wage_rate,
                        income_tax_rate=income_tax_rate,
                        productivity=np.float64(prod_states[productivity_idx]),
                        efficiency=np.float64(efficiency[age_idx]),
                    )

                    if consumption <= 0.0:
                        flow_utility = neg
                        assets_next_period_idx = n_gridpoints_capital - 1
                    else:
                        flow_utility = util(
                            consumption=consumption,
                            labor_input=lab,
                            hc_effort=np.float64(0.0),
                            gamma=gamma,
                            sigma=sigma,
                        )

                    # Right-hand side of Bellman equation
                    if age_idx == duration_working - 1:  # retired next period
                        value_current = (
                            flow_utility
                            + beta * value_retired[assets_next_period_idx, 0]
                        )
                    else:
                        value_current = flow_utility + beta * (
                            transition_prod_states[productivity_idx, 0]
                            * value_working[0, assets_next_period_idx, age_idx + 1]
                            + transition_prod_states[productivity_idx, 1]
                            * value_working[1, assets_next_period_idx, age_idx + 1]
                        )

                    # Store indirect utility, optimal saving and labor
                    if value_current > value_current_max:
                        value_working[
                            productivity_idx, assets_this_period_idx, age_idx
                        ] = value_current
                        policy_labor_working[
                            productivity_idx, assets_this_period_idx, age_idx
                        ] = lab
                        policy_capital_working[
                            productivity_idx, assets_this_period_idx, age_idx
                        ] = assets_next_period_idx
                        value_current_max = value_current

    return policy_capital_working, policy_labor_working, policy_capital_retired


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
        efficiency: np.array(age_retire, 1)
            Age-dependent labor efficiency multiplier
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


@nb.njit
def solve_by_backward_induction_hc_readable(
    interest_rate,
    wage_rate,
    capital_grid,
    n_gridpoints_capital,
    hc_grid,
    n_gridpoints_hc,
    sigma,
    gamma,
    pension_benefit,
    neg,
    age_max,
    age_retire,
    income_tax_rate,
    beta,
    zeta,
    psi,
    delta_hc,
):
    """ Calculate household policy functions.

    Arguments
    ---------
        interest_rate: np.float64
            Current interest rate on capital holdings
        wage_rate: np.float64
            Current wage rate on effective labor input
        capital_grid: np.array(n_gridpoints_capital)
            Asset grid
        n_gridpoints_capital: np.int32
            Number of grid points of capital grid
        hc_grid: np.array(n_gridpoints_hc)
            Human capital grid
        n_gridpoints_hc: np.int32
            Number of grid points of human capital grid
        sigma: np.float64
            Inverse of inter-temporal elasticity of substitution
        gamma: np.float64
            Weight of consumption utility vs. leisure utility
        pension_benefit: np.float64
            Income from pension benefits
        neg: np.float64
            Very small number
        age_max: np.int32
            Maximum age of agents
        age_retire: np.int32
            Retirement age of agents
        income_tax_rate: np.float64
            Tax rate on labor income
        beta: np.float64
            Time discount factor
        zeta: np.float64
            Scaling factor (average learning ability)
        psi: np.float64
            Curvature parameter of hc formation technology
        delta_hc: np.float64
            Depreciation rate on human capital
    Returns
    -------
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
    """

    # Initialize objects for backward iteration
    duration_retired = age_max - age_retire + 1  # length of retirement
    duration_working = age_retire - 1  # length of working life

    value_working = np.zeros(
        (n_gridpoints_capital, n_gridpoints_hc, duration_working), dtype=np.float64
    )
    value_retired = np.zeros((n_gridpoints_capital, duration_retired), dtype=np.float64)
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

    # Last period utility
    consumption_last = (1 + interest_rate) * capital_grid + pension_benefit
    flow_utility_last = (consumption_last ** ((1 - sigma) * gamma)) / (1 - sigma)
    value_retired[:, -1] = flow_utility_last

    # Retired agents
    # iterate backwards through T-1 to zero
    for age_idx in range(duration_retired - 2, -1, -1):
        for assets_this_period_idx in range(n_gridpoints_capital):

            # Initialize right-hand side of Bellman equation
            value_current_max = neg
            assets_next_period_idx = -1

            while assets_next_period_idx < n_gridpoints_capital - 1:  # assets tomorrow
                assets_next_period_idx += 1
                assets_this_period = capital_grid[assets_this_period_idx]
                assets_next_period = capital_grid[assets_next_period_idx]

                # Optimal labor supply
                lab = 0.0

                # Implied hc effort
                hc_effort = 0.0

                # Implied consumption
                consumption = get_consumption_hc(
                    assets_this_period=assets_this_period,
                    assets_next_period=assets_next_period,
                    pension_benefit=pension_benefit,
                    labor_input=lab,
                    interest_rate=interest_rate,
                    wage_rate=wage_rate,
                    income_tax_rate=income_tax_rate,
                    hc_this_period=np.float64(0.0),
                )

                # Instantaneous utility
                if consumption <= 0.0:
                    flow_utility = neg
                    assets_next_period_idx = n_gridpoints_capital - 1
                else:
                    flow_utility = util(
                        consumption=consumption,
                        labor_input=lab,
                        hc_effort=hc_effort,
                        gamma=gamma,
                        sigma=sigma,
                    )

                value_current = (
                    flow_utility
                    + beta * value_retired[assets_next_period_idx, age_idx + 1]
                )

                # Store indirect utility, optimal saving and labor
                if value_current > value_current_max:
                    value_retired[assets_this_period_idx, age_idx] = value_current
                    policy_capital_retired[
                        assets_this_period_idx, age_idx
                    ] = assets_next_period_idx
                    value_current_max = value_current

    # working agents
    for age_idx in range(duration_working - 1, -1, -1):
        for assets_this_period_idx in range(n_gridpoints_capital):
            for hc_this_period_idx in range(n_gridpoints_hc):

                # Initialize right-hand side of Bellman equation
                value_current_max = neg
                assets_next_period_idx = -1

                while assets_next_period_idx < n_gridpoints_capital - 1:
                    assets_next_period_idx += 1
                    assets_this_period = capital_grid[assets_this_period_idx]
                    assets_next_period = capital_grid[assets_next_period_idx]

                    hc_next_period_idx = -1
                    while hc_next_period_idx < n_gridpoints_hc - 1:  # assets tomorrow
                        hc_next_period_idx += 1
                        hc_this_period = hc_grid[hc_this_period_idx]
                        hc_next_period = hc_grid[hc_next_period_idx]

                        # Optimal labor supply
                        lab = get_labor_input_hc(
                            assets_this_period=assets_this_period,
                            assets_next_period=assets_next_period,
                            interest_rate=interest_rate,
                            wage_rate=wage_rate,
                            income_tax_rate=income_tax_rate,
                            hc_this_period=hc_this_period,
                            gamma=gamma,
                        )

                        # Implied hc effort
                        hc_effort = get_hc_effort(
                            hc_this_period=hc_this_period,
                            hc_next_period=hc_next_period,
                            zeta=zeta,
                            psi=psi,
                            delta_hc=delta_hc,
                        )

                        # Implied consumption
                        consumption = get_consumption_hc(
                            assets_this_period=assets_this_period,
                            assets_next_period=assets_next_period,
                            pension_benefit=np.float64(0.0),
                            labor_input=lab,
                            interest_rate=interest_rate,
                            wage_rate=wage_rate,
                            income_tax_rate=income_tax_rate,
                            hc_this_period=hc_this_period,
                        )

                        # Instantaneous utility
                        if consumption <= 0.0:
                            flow_utility = neg
                            assets_next_period_idx = n_gridpoints_capital - 1
                        else:
                            flow_utility = util(
                                consumption=consumption,
                                labor_input=lab,
                                hc_effort=np.float64(0.0),
                                gamma=gamma,
                                sigma=sigma,
                            )

                        # Right-hand side of Bellman equation
                        if age_idx == duration_working - 1:  # retired next period
                            value_current = (
                                flow_utility
                                + beta * value_retired[assets_next_period_idx, 0]
                            )
                        else:
                            value_current = (
                                flow_utility
                                + beta
                                * value_working[
                                    assets_next_period_idx,
                                    hc_next_period_idx,
                                    age_idx + 1,
                                ]
                            )

                        # Store indirect utility, optimal saving and labor
                        if value_current > value_current_max:
                            value_working[
                                assets_this_period_idx, hc_this_period_idx, age_idx
                            ] = value_current
                            policy_labor_working[
                                assets_this_period_idx, hc_this_period_idx, age_idx
                            ] = lab
                            policy_capital_working[
                                assets_this_period_idx, hc_this_period_idx, age_idx
                            ] = assets_next_period_idx
                            policy_hc_working[
                                assets_this_period_idx, hc_this_period_idx, age_idx
                            ] = hc_next_period_idx
                            value_current_max = value_current

    return (
        policy_capital_working,
        policy_hc_working,
        policy_labor_working,
        policy_capital_retired,
    )


def solve_by_backward_induction_hc_vectorized(
    interest_rate,
    wage_rate,
    capital_grid,
    n_gridpoints_capital,
    hc_grid,
    n_gridpoints_hc,
    sigma,
    gamma,
    pension_benefit,
    neg,
    age_max,
    age_retire,
    income_tax_rate,
    beta,
    zeta,
    psi,
    delta_hc,
):
    """ Calculate household policy functions.

    Arguments
    ---------
        interest_rate: np.float64
            Current interest rate on capital holdings
        wage_rate: np.float64
            Current wage rate on effective labor input
        capital_grid: np.array(n_gridpoints_capital)
            Asset grid
        n_gridpoints_capital: np.int32
            Number of grid points of capital grid
        hc_grid: np.array(n_gridpoints_hc)
            Human capital grid
        n_gridpoints_hc: np.int32
            Number of grid points of human capital grid
        sigma: np.float64
            Inverse of inter-temporal elasticity of substitution
        gamma: np.float64
            Weight of consumption utility vs. leisure utility
        pension_benefit: np.float64
            Income from pension benefits
        neg: np.float64
            Very small number
        age_max: np.int32
            Maximum age of agents
        age_retire: np.int32
            Retirement age of agents
        income_tax_rate: np.float64
            Tax rate on labor income
        beta: np.float64
            Time discount factor
        zeta: np.float64
            Scaling factor (average learning ability)
        psi: np.float64
            Curvature parameter of hc formation technology
        delta_hc: np.float64
            Depreciation rate on human capital
    Returns
    -------
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
    """

    # Initialize objects for backward iteration
    duration_retired = age_max - age_retire + 1  # length of retirement
    duration_working = age_retire - 1  # length of working life

    value_working = np.zeros(
        (n_gridpoints_capital, n_gridpoints_hc, duration_working), dtype=np.float64
    )
    value_retired = np.zeros((n_gridpoints_capital, duration_retired), dtype=np.float64)
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

    # Last period utility
    consumption_last = (1 + interest_rate) * capital_grid + pension_benefit
    flow_utility_last = (consumption_last ** ((1 - sigma) * gamma)) / (1 - sigma)
    value_retired[:, -1] = flow_utility_last

    # Retired agents
    # iterate backwards through T-1 to zero
    for age_idx in range(duration_retired - 2, -1, -1):
        for assets_this_period_idx in range(n_gridpoints_capital):

            # Initialize right-hand side of Bellman equation
            value_current_max = neg
            assets_next_period_idx = -1

            while assets_next_period_idx < n_gridpoints_capital - 1:  # assets tomorrow
                assets_next_period_idx += 1
                assets_this_period = capital_grid[assets_this_period_idx]
                assets_next_period = capital_grid[assets_next_period_idx]

                # Optimal labor supply
                lab = 0.0

                # Implied hc effort
                hc_effort = 0.0

                # Implied consumption
                consumption = get_consumption_hc(
                    assets_this_period=assets_this_period,
                    assets_next_period=assets_next_period,
                    pension_benefit=pension_benefit,
                    labor_input=lab,
                    interest_rate=interest_rate,
                    wage_rate=wage_rate,
                    income_tax_rate=income_tax_rate,
                    hc_this_period=np.float64(0.0),
                )

                # Instantaneous utility
                if consumption <= 0.0:
                    flow_utility = neg
                    assets_next_period_idx = n_gridpoints_capital - 1
                else:
                    flow_utility = util(
                        consumption=consumption,
                        labor_input=lab,
                        hc_effort=hc_effort,
                        gamma=gamma,
                        sigma=sigma,
                    )

                value_current = (
                    flow_utility
                    + beta * value_retired[assets_next_period_idx, age_idx + 1]
                )

                # Store indirect utility, optimal saving and labor
                if value_current > value_current_max:
                    value_retired[assets_this_period_idx, age_idx] = value_current
                    policy_capital_retired[
                        assets_this_period_idx, age_idx
                    ] = assets_next_period_idx
                    value_current_max = value_current

    # working agents
    for age_idx in range(duration_working - 1, -1, -1):
        for assets_this_period_idx in range(n_gridpoints_capital):
            for hc_this_period_idx in range(n_gridpoints_hc):

                # Initialize right-hand side of Bellman equation
                value_current_max = neg
                assets_next_period_idx = -1

                while assets_next_period_idx < n_gridpoints_capital - 1:
                    assets_next_period_idx += 1
                    assets_this_period = capital_grid[assets_this_period_idx]
                    assets_next_period = capital_grid[assets_next_period_idx]

                    hc_next_period_idx = -1
                    while hc_next_period_idx < n_gridpoints_hc - 1:  # assets tomorrow
                        hc_next_period_idx += 1
                        hc_this_period = hc_grid[hc_this_period_idx]
                        hc_next_period = hc_grid[hc_next_period_idx]

                        # Optimal labor supply
                        lab = get_labor_input_hc(
                            assets_this_period=assets_this_period,
                            assets_next_period=assets_next_period,
                            interest_rate=interest_rate,
                            wage_rate=wage_rate,
                            income_tax_rate=income_tax_rate,
                            hc_this_period=hc_this_period,
                            gamma=gamma,
                        )

                        # Implied hc effort
                        hc_effort = get_hc_effort(
                            hc_this_period=hc_this_period,
                            hc_next_period=hc_next_period,
                            zeta=zeta,
                            psi=psi,
                            delta_hc=delta_hc,
                        )

                        # Implied consumption
                        consumption = get_consumption_hc(
                            assets_this_period=assets_this_period,
                            assets_next_period=assets_next_period,
                            pension_benefit=np.float64(0.0),
                            labor_input=lab,
                            interest_rate=interest_rate,
                            wage_rate=wage_rate,
                            income_tax_rate=income_tax_rate,
                            hc_this_period=hc_this_period,
                        )

                        # Instantaneous utility
                        if consumption <= 0.0:
                            flow_utility = neg
                            assets_next_period_idx = n_gridpoints_capital - 1
                        else:
                            flow_utility = util(
                                consumption=consumption,
                                labor_input=lab,
                                hc_effort=np.float64(0.0),
                                gamma=gamma,
                                sigma=sigma,
                            )

                        # Right-hand side of Bellman equation
                        if age_idx == duration_working - 1:  # retired next period
                            value_current = (
                                flow_utility
                                + beta * value_retired[assets_next_period_idx, 0]
                            )
                        else:
                            value_current = (
                                flow_utility
                                + beta
                                * value_working[
                                    assets_next_period_idx,
                                    hc_next_period_idx,
                                    age_idx + 1,
                                ]
                            )

                        # Store indirect utility, optimal saving and labor
                        if value_current > value_current_max:
                            value_working[
                                assets_this_period_idx, hc_this_period_idx, age_idx
                            ] = value_current
                            policy_labor_working[
                                assets_this_period_idx, hc_this_period_idx, age_idx
                            ] = lab
                            policy_capital_working[
                                assets_this_period_idx, hc_this_period_idx, age_idx
                            ] = assets_next_period_idx
                            policy_hc_working[
                                assets_this_period_idx, hc_this_period_idx, age_idx
                            ] = hc_next_period_idx
                            value_current_max = value_current

    return (
        policy_capital_working,
        policy_hc_working,
        policy_labor_working,
        policy_capital_retired,
    )


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
    mass_distribution_capital_working = np.zeros(
        (n_gridpoints_capital, duration_working), dtype=np.float64
    )
    mass_distribution_hc_working = np.zeros(
        (n_gridpoints_hc, duration_working), dtype=np.float64
    )
    mass_distribution_full_retired = np.zeros(
        (n_gridpoints_capital, duration_retired), dtype=np.float64
    )

    # make sure that all agents start with  correct initial level, i.e. hc = 1 and assets = 0
    mass_distribution_full_working[0, 0, 0] = mass[0]
    mass_distribution_capital_working[0, 0] = mass[0]
    mass_distribution_hc_working[0, 0] = mass[0]

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
                    ] += mass_distribution_full_working[
                        assets_this_period_idx, hc_this_period_idx, age_idx
                    ] / (
                        1 + population_growth_rate
                    )
                elif age_idx == duration_working - 1:
                    mass_distribution_full_retired[
                        assets_next_period_idx, 0
                    ] += mass_distribution_full_working[
                        assets_this_period_idx, hc_this_period_idx, age_idx
                    ] / (
                        1 + population_growth_rate
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

        # Aggregate assets by age
        for assets_this_period_idx in range(n_gridpoints_capital):
            mass_distribution_capital_working[assets_this_period_idx, age_idx] = np.sum(
                mass_distribution_full_working[assets_this_period_idx, :, age_idx]
            )
        # Aggregate human capital by age
        for hc_this_period_idx in range(n_gridpoints_hc):
            mass_distribution_hc_working[hc_this_period_idx, age_idx] = np.sum(
                mass_distribution_full_working[:, hc_this_period_idx, age_idx]
            )

        # Aggregate variables
        asset_distribution_age[age_idx] = np.dot(
            capital_grid, mass_distribution_capital_working[:, age_idx]
        )
        hc_distribution_age[age_idx] = np.dot(
            hc_grid, mass_distribution_hc_working[:, age_idx]
        )

    # Retirees
    for age_idx in range(duration_retired - 1):
        for assets_this_period_idx in range(n_gridpoints_capital):

            assets_next_period_idx = policy_capital_retired[
                assets_this_period_idx, age_idx
            ]
            mass_distribution_full_retired[
                assets_next_period_idx, age_idx + 1
            ] += mass_distribution_full_retired[assets_this_period_idx, age_idx] / (
                1 + population_growth_rate
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
        mass_distribution_capital_working,
        mass_distribution_hc_working,
        mass_distribution_full_retired,
    )
