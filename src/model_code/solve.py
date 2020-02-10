import numba as nb
import numpy as np

from src.model_code.within_period import get_consumption
from src.model_code.within_period import get_labor_input
from src.model_code.within_period import util


@nb.njit
def solve_by_backward_induction(
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
        prod_states: np.array(n_prod_states)
            Vector of idiosyncratic productivity states
        efficiency: np.array(age_max)
            Vector of age-dependent labor efficiency multiplier
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
        policy_labor:  np.array(n_prod_states, n_gridpoints_capital, duration_working)
            Labor supply policy function (storing optimal hours worked as np.float64)
    """

    # Initializations for backward induction
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
    policy_labor = np.zeros(
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

    for age in range(duration_retired - 2, -1, -1):  # age

        for assets_this_period_idx in range(n_gridpoints_capital):  # assets today

            # Initialize right-hand side of Bellman equation
            vmin = neg
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
                v0 = (
                    flow_utility + beta * value_retired[assets_next_period_idx, age + 1]
                )

                # Store indirect utility and optimal saving
                if v0 > vmin:
                    value_retired[assets_this_period_idx, age] = v0
                    policy_capital_retired[
                        assets_this_period_idx, age
                    ] = assets_next_period_idx
                    vmin = v0

    # Working households
    for age in range(duration_working - 1, -1, -1):
        for e in range(n_prod_states):
            for assets_this_period_idx in range(n_gridpoints_capital):

                # Initialize right-hand side of Bellman equation
                vmin = neg
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
                        productivity=np.float64(prod_states[e]),
                        efficiency=np.float64(efficiency[age]),
                        gamma=gamma,
                    )

                    # Instantaneous utility
                    consumption = get_consumption(
                        assets_this_period=assets_this_period,
                        assets_next_period=assets_next_period,
                        pension_benefit=0,
                        labor_input=lab,
                        interest_rate=interest_rate,
                        wage_rate=wage_rate,
                        income_tax_rate=income_tax_rate,
                        productivity=np.float64(prod_states[e]),
                        efficiency=np.float64(efficiency[age]),
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
                    if age == duration_working - 1:  # retired next period
                        v0 = (
                            flow_utility
                            + beta * value_retired[assets_next_period_idx, 0]
                        )
                    else:
                        v0 = flow_utility + beta * (
                            transition_prod_states[e, 0]
                            * value_working[0, assets_next_period_idx, age + 1]
                            + transition_prod_states[e, 1]
                            * value_working[1, assets_next_period_idx, age + 1]
                        )

                    # Store indirect utility, optimal saving and labor
                    if v0 > vmin:
                        value_working[e, assets_this_period_idx, age] = v0
                        policy_labor[e, assets_this_period_idx, age] = lab
                        policy_capital_working[
                            e, assets_this_period_idx, age
                        ] = assets_next_period_idx
                        vmin = v0

    return policy_capital_working, policy_capital_retired, policy_labor


# @nb.njit
def aggregate_readable(
    policy_capital_working,
    policy_capital_retired,
    policy_labor,
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
        policy_labor:  np.array(n_prod_states, n_gridpoints_capital, duration_working)
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
        efficiency: np.array(age_max)
            Vector of age-dependent labor efficiency multiplier
        capital_grid: np.array(n_gridpoints_capital)
            Asset grid
        mass: np.array(age_max, 1)
            Vector of relative shares of agents by age
        duration_retired: int
            Length of retirement period
        population_growth_rate: np.float64
            Annual population growth rate
        prod_states: np.array(n_prod_states)
            Vector of idiosyncratic productivity states
    Returns
    -------
        aggregate_capital_out: np.float64
            Aggregate capital stock derived from household policy functions and
            cross-sectional distribution
        aggregate_labor_out: np.float64
            Aggregate labor supply derived from household policy functions and
            cross-sectional distribution
        gk: np.array( , )
            ...
        gkW: np.array(n_prod_states, duration_working, n_gridpoints_capital)
            ...
        gkR np.array(duration_retired, n_gridpoints_capital)
            ...
    """

    # Aggregate capital for each generation
    kgen = np.zeros((age_max, 1), dtype=np.float64)

    # Distribution of workers over capital and shocks for each working cohort
    gkW = np.zeros(
        (n_prod_states, n_gridpoints_capital, duration_working), dtype=np.float64,
    )

    # Newborns
    gkW[0, 0, 0] = productivity_init[0] * mass[0]
    gkW[1, 0, 0] = productivity_init[1] * mass[0]

    # Distribution of agents over capital for each cohort (pooling together
    # both productivity shocks). This would be useful when computing total
    # capital
    gk = np.zeros((n_gridpoints_capital, age_max), dtype=np.float64)
    gk[0, 0] = mass[0]  # Distribution of newborns over capital

    # Aggregate labor supply by generation
    labgen = np.zeros((duration_working, 1), dtype=np.float64)

    # Distribution of retirees over capital
    gkR = np.zeros((n_gridpoints_capital, duration_retired), dtype=np.float64)

    ############################################################################
    # Iterating over the distribution
    ############################################################################
    # Workers
    for ind_age in range(duration_working):  # iterations over cohorts
        for ind_k in range(n_gridpoints_capital):  # current asset holdings
            for ind_e in range(n_prod_states):  # current shock

                ind_kk = policy_capital_working[
                    ind_e, ind_k, ind_age
                ]  # optimal saving (as index in asset grid)

                for ind_ee in range(n_prod_states):  # tomorrow's shock

                    if ind_age < duration_working - 1:
                        gkW[ind_ee, ind_kk, ind_age + 1] += (
                            transition_prod_states[ind_e, ind_ee]
                            * gkW[ind_e, ind_k, ind_age]
                            / (1 + population_growth_rate)
                        )
                    elif (
                        ind_age == duration_working - 1
                    ):  # need to be careful because workers transition to retirees
                        gkR[ind_kk, 0] += (
                            transition_prod_states[ind_e, ind_ee]
                            * gkW[ind_e, ind_k, ind_age]
                            / (1 + population_growth_rate)
                        )

        # Aggregate labor by age
        labgen[ind_age] = 0

        for ind_k in range(n_gridpoints_capital):
            for ind_e in range(n_prod_states):
                labgen[ind_age] += (
                    prod_states[ind_e]
                    * efficiency[ind_age]
                    * policy_labor[ind_e, ind_k, ind_age]
                    * gkW[ind_e, ind_k, ind_age]
                )

        # Aggregate capital by age
        for ind_k in range(n_gridpoints_capital):
            if ind_age < duration_working - 1:
                gk[ind_k, ind_age + 1] = np.sum(gkW[:, ind_k, ind_age + 1])
            else:
                gk[ind_k, ind_age + 1] = gkR[ind_k, 0]
        kgen[ind_age + 1] = np.dot(capital_grid, gk[:, ind_age + 1])

    # Retirees
    for ind_age in range(duration_retired - 1):  # iterations over cohort

        for ind_k in range(n_gridpoints_capital):  # current asset holdings
            ind_kk = policy_capital_retired[
                ind_k, ind_age
            ]  # optimal saving (as index in asset grid)
            gkR[ind_kk, ind_age + 1] += gkR[ind_k, ind_age] / (
                1 + population_growth_rate
            )

        # Distribution by capital and age
        gk[:, duration_working + ind_age] = gkR[:, ind_age + 1]
        # Aggregate capital by age
        kgen[duration_working + ind_age + 1] = np.dot(
            capital_grid, gk[:, duration_working + ind_age]
        )

    aggregate_capital_out = np.sum(kgen)
    aggregate_labor_out = np.sum(labgen)

    return aggregate_capital_out, aggregate_labor_out, gk, gkW, gkR
