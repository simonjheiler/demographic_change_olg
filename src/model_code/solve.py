"""Solve for household policy functions."""
import numpy as np

from src.model_code.within_period import get_consumption
from src.model_code.within_period import get_hc_effort
from src.model_code.within_period import util


#########################################################################
# ADAPTED MODEL WITHOUT IDIOSYNCRATIC RISK AND WITH HUMAN CAPITAL
#########################################################################


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
    survival_rate,
):
    """Calculate optimal policy and value function for retired agents of a given age.

    Calculate within-period quantities of retired agents implied by state / choice
    combinations on meshes and derive optimal choices and corresponding value
    functions by maximizing sums of flow utility and discounted continuation value
    over possible choices given states.

    Arguments:
        assets_this_period: np.array(n_gridpoints_capital, n_gridpoints_capital)
            Meshgrid of this periods asset holdings
        assets_next_period: np.array(n_gridpoints_capital, n_gridpoints_capital)
            Meshgrid of next periods asset holdings
        interest_rate: np.float64
            Current interest rate on capital holdings
        pension_benefit: np.float64
            Income from pension benefits
        beta: np.float64
            Time discount factor
        gamma: np.float64
            Weight of consumption utility vs. leisure utility
        sigma: np.float64
            Inverse of inter-temporal elasticity of substitution
        neg: np.float64
            Very small number
        continuation_value: np.array(n_gridpoints_capital, n_gridpoints_capital)
            Meshgrid of continuation value from next period asset and human capital
            level
        survival_rate: np.float64
            Conditional year-to-year survival rate for the current age
    Returns:
        value_retired_tmp: np.array(n_gridpoints_capital)
            Value function for agents of given age induced by optimal choices
        policy_capital_retired_tmp: np.array(n_gridpoints_hc, duration_retired)
            Savings policy function for retired agents (storing optimal asset choices
            by index on asset grid as int)
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
):
    """Calculate optimal policy and value function for working agents of a given age.

    Calculate within-period quantities of working agents implied by states / choices
    combinations on meshes for current assets, next period assets, current human
    capital and next period human capital and derive optimal choices and corresponding
    value functions by maximizing sums of flow utility and discounted continuation value
    over possible choices for assets and human capital next period given states.

    Arguments:
        assets_this_period: np.array(n_gridpoints_capital, n_gridpoints_capital,
            n_gridpoints_hc, n_gridpoints_hc)
            Meshgrid of this periods asset holdings
        assets_next_period: np.array(n_gridpoints_capital, n_gridpoints_capital,
            n_gridpoints_hc, n_gridpoints_hc)
            Meshgrid of next periods asset holdings
        hc_this_period: np.array(n_gridpoints_capital, n_gridpoints_capital,
            n_gridpoints_hc, n_gridpoints_hc)
            Meshgrid of this periods human capital level
        hc_next_period: np.array(n_gridpoints_capital, n_gridpoints_capital,
            n_gridpoints_hc, n_gridpoints_hc)
            Meshgrid of next periods human capital level
        interest_rate: np.float64
            Current interest rate on capital holdings
        wage_rate: np.float64
            Current wage rate on effective labor supply
        income_tax_rate: np.float64
            Tax rate on labor income
        beta: np.float64
            Time discount factor
        gamma: np.float64
            Weight of consumption utility vs. leisure utility
        sigma: np.float64
            Inverse of inter-temporal elasticity of substitution
        neg: np.float64
            Very small number
        continuation_value: np.array(n_gridpoints_capital, n_gridpoints_capital,
            n_gridpoints_hc, n_gridpoints_hc)
            Meshgrid of continuation value from next period asset and human capital
            level
        delta_hc: np.float64
            Depreciation rate on human capital
        zeta: np.float64
            Scaling factor (average learning ability)
        psi: np.float64
            Curvature parameter of hc formation technology
        n_gridpoints_capital: np.int32
            Number of grid points of asset grid
        n_gridpoints_hc: np.int32
            Number of grid points of human capital grid
        efficiency: np.float64
            Current level of age-dependent labor efficiency multiplier
        survival_rate: np.float64
            Conditional year-to-year survival rate for the current age
    Returns:
        policy_capital_working_tmp: np.array(n_gridpoints_capital, n_gridpoints_hc)
            Savings policy function for agents of given age (storing optimal asset
            choices by index on asset grid as int)
        policy_hc_working_tmp: np.array(n_gridpoints_capital, n_gridpoints_hc)
            Human capital policy function for agents of given age (storing optimal
            human capital choices by index on human capital grid as int)
        policy_labor_working_tmp: np.array(n_gridpoints_capital, n_gridpoints_hc)
            Labor supply policy function for agents of given age (storing optimal
            hours worked as np.float64)
        value_working_tmp: np.array(n_gridpoints_capital, n_gridpoints_hc)
            Value function for agents of given age induced by optimal choices
    """
    # Initialize objects
    policy_labor_working_tmp = np.zeros(
        (n_gridpoints_capital, n_gridpoints_hc), dtype=np.float64
    )

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
