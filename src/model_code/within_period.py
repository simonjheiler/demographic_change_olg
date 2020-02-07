import numba as nb


@nb.njit
def _get_labor_input(
    assets_this_period,
    assets_next_period,
    interest_rate,
    wage_rate,
    tax_rate,
    productivity,
    efficiency,
    gamma,
):
    labor_input = (
        gamma * (1 - tax_rate) * productivity * efficiency * wage_rate
        - (1 - gamma) * ((1 + interest_rate) * assets_this_period - assets_next_period)
    ) / ((1 - tax_rate) * wage_rate * productivity * efficiency)

    return labor_input


@nb.njit
def util(
    consumption, labor_input, gamma, sigma,
):
    flow_utility = (
        ((consumption ** gamma) * (1 - labor_input) ** (1 - gamma)) ** (1 - sigma)
    ) / (1 - sigma)

    return flow_utility


@nb.njit
def _get_consumption(
    assets_this_period,
    assets_next_period,
    pension_benefit,
    labor_input,
    interest_rate,
    wage_rate,
    tax_rate,
    productivity,
    efficiency,
):
    consumption = (
        (1 + interest_rate) * assets_this_period
        + (1 - tax_rate) * wage_rate * productivity * efficiency * labor_input
        + pension_benefit
        - assets_next_period
    )

    return consumption
