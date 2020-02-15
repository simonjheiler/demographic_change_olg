import numba as nb
import numpy as np


#########################################################################
# STANDARD MODEL WITH IDIOSYNCRATIC RISK AND NO HUMAN CAPITAL
#########################################################################


@nb.njit
def get_labor_input_baseline(
    assets_this_period,
    assets_next_period,
    interest_rate,
    wage_rate,
    income_tax_rate,
    productivity,
    efficiency,
    gamma,
):
    """ Calculate optimal household labor input.

    Arguments
    ---------
        assets_this_period: np.float64
            Current asset holdings (pre interest payment)
        assets_next_period: np.float64
            Savings for asset holdings next period (pre interest payment)
        interest_rate: np.float64
            Current interest rate on capital holdings
        wage_rate: np.float64
            Current wage rate on effective labor input
        income_tax_rate: np.float64
            Tax rate on labor income
        productivity: np.float64
            Current household productivity level (shock)
        efficiency: np.float64
            Age-dependent labor efficiency multiplier
        gamma: np.float64
            Weight of consumption utility vs. leisure utility
    Returns
    -------
        labor_input: np.float64
            Optimal hours worked
    """
    if efficiency == 0.0:
        labor_input = 0.0
    elif gamma == 1.0:
        labor_input = 1.0
    else:
        labor_input = (
            gamma * (1.0 - income_tax_rate) * productivity * efficiency * wage_rate
            - (1.0 - gamma)
            * ((1.0 + interest_rate) * assets_this_period - assets_next_period)
        ) / ((1.0 - income_tax_rate) * productivity * efficiency * wage_rate)

    if labor_input > 1.0:
        labor_input = 1.0
    elif labor_input < 0.0:
        labor_input = 0.0

    return labor_input


@nb.njit
def get_consumption(
    assets_this_period,
    assets_next_period,
    pension_benefit,
    labor_input,
    interest_rate,
    wage_rate,
    income_tax_rate,
    productivity,
    efficiency,
):
    """ Calculate consumption level via household budget constraint.

    Arguments
    ---------
        assets_this_period: np.float64
            Current asset holdings (pre interest payment)
        assets_next_period: np.float64
            Savings for asset holdings next period (pre interest payment)
        pension_benefit: np.float64
            Income from pension benefits
        labor_input: np.float64
            Hours worked
        interest_rate: np.float64
            Current interest rate on capital holdings
        wage_rate: np.float64
            Current wage rate on effective labor input
        income_tax_rate: np.float64
            Tax rate on labor income
        productivity: np.float64
            Current idiosyncratic productivity state or level of human capital
        efficiency: np.float64
            Age-dependent labor efficiency multiplier
    Returns
    -------
        consumption: np.float64
            Household consumption in the current period
    """
    consumption = (
        (1 + interest_rate) * assets_this_period
        + (1 - income_tax_rate) * wage_rate * productivity * efficiency * labor_input
        + pension_benefit
        - assets_next_period
    )

    return consumption


@nb.njit
def util(consumption, labor_input, hc_effort, gamma, sigma):
    """ Calculate per-period flow utility of household.

    Arguments
    ---------
        consumption: np.float64
            Household consumption in the current period
        labor_input: np.float64
            Hours worked
        hc_effort: np.float64
            Time invested in human capital accumulation
        gamma: np.float64
            Weight of consumption utility vs. leisure utility
        sigma: np.float64
            Inverse of inter-temporal elasticity of substitution
    Returns
    -------
        flow_utility: np.float64
            Household flow utility for the current period
    """
    flow_utility = (
        ((consumption ** gamma) * (1 - labor_input - hc_effort) ** (1 - gamma))
        ** (1 - sigma)
    ) / (1 - sigma)
    return flow_utility


#########################################################################
# ADAPTED MODEL WITHOUT IDIOSYNCRATIC RISK AND WITH HUMAN CAPITAL
#########################################################################


@nb.njit
def hc_accumulation(hc_this_period, hc_effort, zeta, psi, delta_hc):
    """ Calculate new (post-investment and depreciation) level of human capital.

    Human capital formation technology taken from Ludwig, Schelkle, Vogel (2012),
    adopted rom Ben-Porath (1967).

    Arguments
    ---------
        hc_this_period: np.float64
            Current level of human capital
        hc_effort: np.float64
            Time spent acquiring new human capital
        zeta: np.float64
            scaling factor (average learning ability)
        psi: np.float64
            curvature parameter of hc formation technology
        delta_hc: np.float64
            depreciation rate on human capital
    Returns
    -------
        hc_next_period: np.float64
            New level of human capital
    """
    hc_next_period = (
        hc_this_period * (1 - delta_hc) + zeta * (hc_this_period * hc_effort) ** psi
    )

    return hc_next_period


@nb.njit
def get_hc_effort(hc_this_period, hc_next_period, zeta, psi, delta_hc):
    """ Calculate (implied) effort invested in human capital accumulation.

    Human capital formation technology taken from Ludwig, Schelkle, Vogel (2012),
    adopted rom Ben-Porath (1967).

    Arguments
    ---------
        hc_this_period: np.float64
            Current level of human capital
        hc_next_period: np.float64
            Future level of human capital
        zeta: np.float64
            Scaling factor (average learning ability)
        psi: np.float64
            Curvature parameter of hc formation technology
        delta_hc: np.float64
            Depreciation rate on human capital
    Returns
    -------
        hc_effort: np.float64
            Effort invested in human capital accumulation
    """
    hc_effort = zeta ** (-1) * (hc_next_period - hc_this_period * (1 - delta_hc)) ** (
        1 / psi
    )

    return hc_effort


@nb.njit
def get_factor_prices(
    aggregate_capital,
    aggregate_labor,
    alpha,
    delta_k,
    income_tax_rate,
    mass,
    age_retire,
):
    interest_rate = np.float64(
        alpha * (aggregate_capital ** (alpha - 1)) * (aggregate_labor ** (1 - alpha))
        - delta_k
    )
    wage_rate = np.float64(
        (1 - alpha) * (aggregate_capital ** alpha) * (aggregate_labor ** (-alpha))
    )
    pension_benefit = np.float64(
        income_tax_rate * wage_rate * aggregate_labor / np.sum(mass[age_retire - 1 :])
    )

    return interest_rate, wage_rate, pension_benefit
