"""Solve for household policy functions."""
import numpy as np

from src.model_code.within_period import get_consumption
from src.model_code.within_period import get_hc_effort
from src.model_code.within_period import util


#########################################################################
# ADAPTED MODEL WITHOUT IDIOSYNCRATIC RISK AND WITH HUMAN CAPITAL
#########################################################################


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
    efficiency,
    survival_rates,
):
    """Calculate household policy functions.

    The *hc_vectorized* function is an adapted implementation of the backward induction
    solution algorithm for a model with assets and human capital but without
    idiosyncratic productivity states. Other than the *hc_readable* function, it
    operates on meshgrids for every combination of current assets, current human capital,
    next period assets and next period human capital on the assets / human capital grid,
    and simultaneously calculates within-period quantities for all combinations, backing
    out hc_effort, labor input and consumption implied by the choices and calculating
    optimal choices and corresponding value functions by maximizing over the meshgrids
    along the corresponding dimensions.

    Arguments:
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
        efficiency: np.array(age_retire)
            Profile of age-dependent labor efficiency multipliers
    Returns:
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

        # Look up continuation values for assets_next_period and replicate in
        # assets_this_period dimension
        continuation_value = np.repeat(
            value_retired[np.newaxis, :, age_idx + 1], n_gridpoints_capital, axis=0
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

        # Look up continuation values for combinations of assets_next_period and hc_next_period
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

    return (
        policy_capital_working,
        policy_hc_working,
        policy_labor_working,
        policy_capital_retired,
        value_retired,
        value_working,
    )


#  nb.njit
def solve_retired(
    assets_this_period,
    assets_next_period,
    interest_rate,
    pension_benefit,
    beta,
    gamma,
    sigma,
    neg,
    continuation_value,
    n_gridpoints_capital,
    survival_rate,
    policy_capital_retired_tmp,
    value_retired_tmp,
):
    """Calculate optimal policy and value function for retired agents of a given age.

    Calculate within-period quantities of retired agents implied by state / choice
    combinations on meshes and derive optimal choices and corresponding value
    functions by maximizing sums of flow utility and discounted continuation value
    over possible choices given states.

    Arguments:
        assets_this_period:
            ...
        assets_next_period:
            ...
        interest_rate:
            ...
        pension_benefit:
            ...
        beta:
            ...
        gamma:
            ...
        sigma:
            ...
        neg:
            ...
        continuation_value:
            ...
        n_gridpoints_capital:
            ...
        survival_rate:
            ...
        policy_capital_retired_tmp:
            ...
        value_retired_tmp:
            ...
    Returns:
        value_retired_tmp:
            ...
        policy_capital_retired_tmp:
            ...
    """
    # Consumption
    consumption = get_consumption(
        assets_this_period=assets_this_period,
        assets_next_period=assets_next_period,
        pension_benefit=pension_benefit,
        labor_input=np.float64(0.0),
        interest_rate=interest_rate,
        wage_rate=np.float64(0.0),
        income_tax_rate=np.float64(0.0),
        productivity=np.float64(0.0),
        efficiency=np.float64(0.0),
    )

    # Flow utility
    flow_utility = util(
        consumption=consumption,
        labor_input=np.float64(0.0),
        hc_effort=np.float64(0.0),
        gamma=gamma,
        sigma=sigma,
    )
    flow_utility = np.where(consumption < 0.0, neg, flow_utility)

    # Calculate value on meshes (i.e. for all choices)
    value_full = flow_utility + beta * survival_rate * continuation_value

    # Derive optimal policies and store value function given optimal choices

    # Numba implementation
    # for assets_this_period_idx in range(n_gridpoints_capital):
    #     policy_capital_retired_tmp[assets_this_period_idx] = np.argmax(
    #         value_full[assets_this_period_idx, :]
    #     )
    #     value_retired_tmp[assets_this_period_idx] = np.max(
    #         value_full[assets_this_period_idx, :]
    #     )

    # No-numba implementation
    policy_capital_retired_tmp = np.argmax(value_full, axis=1)
    value_retired_tmp = np.max(value_full, axis=1)

    return value_retired_tmp, policy_capital_retired_tmp


def solve_working(
    assets_this_period,
    assets_next_period,
    hc_this_period,
    hc_next_period,
    interest_rate,
    wage_rate,
    income_tax_rate,
    beta,
    gamma,
    sigma,
    neg,
    continuation_value,
    delta_hc,
    zeta,
    psi,
    n_gridpoints_capital,
    n_gridpoints_hc,
    efficiency,
    survival_rate,
    policy_capital_working_tmp,
    policy_hc_working_tmp,
    policy_labor_working_tmp,
    value_working_tmp,
):
    """Calculate optimal policy and value function for working agents of a given age.

    Calculate within-period quantities of working agents implied by states / choices
    combinations on meshes for current assets, next period assets, current human
    capital and next period human capital and derive optimal choices and corresponding
    value functions by maximizing sums of flow utility and discounted continuation value
    over possible choices for assets and human capital next period given states.

    Arguments:
        assets_this_period: ...
            ...
        assets_next_period: ...
            ...
        hc_this_period: ...
            ...
        hc_next_period: ...
            ...
        interest_rate: ...
            ...
        wage_rate: ...
            ...
        income_tax_rate: ...
            ...
        beta: ...
            ...
        gamma: ...
            ...
        sigma: ...
            ...
        neg: ...
            ...
        continuation_value: ...
            ...
        delta_hc: ...
            ...
        zeta: ...
            ...
        psi: ...
            ...
        n_gridpoints_capital: ...
            ...
        n_gridpoints_hc: ...
            ...
        efficiency: ...
            ...
        survival_rate: ...
            ...
        policy_capital_working_tmp: ...
            ...
        policy_hc_working_tmp: ...
            ...
        policy_labor_working_tmp: ...
            ...
        value_working_tmp: ...
            ...
    Returns:
        policy_capital_working_tmp: ...
            ...
        policy_hc_working_tmp: ...
            ...
        policy_labor_working_tmp: ...
            ...
        value_working_tmp: ...
            ...
    """
    # Implied hc effort
    hc_effort = get_hc_effort(
        hc_this_period=hc_this_period,
        hc_next_period=hc_next_period,
        zeta=zeta,
        psi=psi,
        delta_hc=delta_hc,
    )

    hc_effort = np.where(np.isnan(hc_effort), 0.0, hc_effort)

    # Implied labor supply
    labor_input = 1 - hc_effort

    # Consumption
    consumption = get_consumption(
        assets_this_period=assets_this_period,
        assets_next_period=assets_next_period,
        pension_benefit=np.float64(0.0),
        labor_input=labor_input,
        interest_rate=interest_rate,
        wage_rate=wage_rate,
        income_tax_rate=income_tax_rate,
        productivity=hc_this_period,
        efficiency=efficiency,
    )

    # Flow utility
    flow_utility = util(
        consumption=consumption,
        labor_input=labor_input,
        hc_effort=hc_effort,
        gamma=gamma,
        sigma=sigma,
    )

    flow_utility = np.where(consumption < 0.0, neg, flow_utility)
    flow_utility = np.where(hc_effort > 1.0, neg, flow_utility)

    # Value function on the complete mesh (i.e. for all possible choices)
    value_full = flow_utility + beta * survival_rate * continuation_value

    # Derive optimal policies and store value function given optimal choices
    policy_capital_working_tmp = np.argmax(np.max(value_full, axis=3), axis=1)
    policy_hc_working_tmp = np.argmax(np.max(value_full, axis=1), axis=2)
    value_working_tmp = np.max(np.max(value_full, axis=3), axis=1)
    for assets_this_period_idx in range(n_gridpoints_capital):
        for hc_this_period_idx in range(n_gridpoints_hc):
            assets_next_period_idx = policy_capital_working_tmp[
                assets_this_period_idx, hc_this_period_idx
            ]
            hc_next_period_idx = policy_hc_working_tmp[
                assets_this_period_idx, hc_this_period_idx
            ]
            policy_labor_working_tmp[
                assets_this_period_idx, hc_this_period_idx
            ] = labor_input[
                assets_this_period_idx,
                assets_next_period_idx,
                hc_this_period_idx,
                hc_next_period_idx,
            ]

    return (
        policy_capital_working_tmp,
        policy_hc_working_tmp,
        policy_labor_working_tmp,
        value_working_tmp,
    )
