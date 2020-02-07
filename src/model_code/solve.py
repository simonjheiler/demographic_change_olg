import numpy as np

from src.model_code.within_period import _get_consumption
from src.model_code.within_period import _get_labor_input
from src.model_code.within_period import util


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
    n_states_productivity,
    tax_rate,
    beta,
    z,
    eff,
    tran,
):

    # Initializations for backward induction
    value_retired = np.zeros((n_gridpoints_capital, duration_retired), dtype=np.float64)
    policy_capital_retired = np.zeros(
        (n_gridpoints_capital, duration_retired), dtype=np.int8
    )

    value_working = np.zeros(
        (n_states_productivity, n_gridpoints_capital, duration_working),
        dtype=np.float64,
    )
    policy_capital_working = np.zeros(
        (n_states_productivity, n_gridpoints_capital, duration_working), dtype=np.int8
    )
    policy_labor = np.zeros(
        (n_states_productivity, n_gridpoints_capital, duration_working),
        dtype=np.float64,
    )

    ############################################################
    # BACKWARD INDUCTION
    ############################################################

    # Retired households

    # Last period utility
    consumption = (1 + interest_rate) * capital_grid + pension_benefit
    flow_utility = (consumption ** ((1 - sigma) * gamma)) / (1 - sigma)
    value_retired[:, duration_retired - 1] = flow_utility

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
                consumption = _get_consumption(
                    assets_this_period=assets_this_period,
                    assets_next_period=assets_next_period,
                    pension_benefit=pension_benefit,
                    labor_input=np.float64(0.0),
                    interest_rate=interest_rate,
                    wage_rate=wage_rate,
                    tax_rate=tax_rate,
                    productivity=np.float64(0.0),
                    efficiency=np.float64(0.0),
                )

                if consumption <= 0:
                    flow_utility = neg
                    assets_next_period_idx = n_gridpoints_capital - 1
                else:
                    flow_utility = util(
                        consumption=consumption,
                        labor_input=0,
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
        for e in range(n_states_productivity):
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
                    lab = _get_labor_input(
                        assets_this_period=assets_this_period,
                        assets_next_period=assets_next_period,
                        interest_rate=interest_rate,
                        wage_rate=wage_rate,
                        tax_rate=tax_rate,
                        productivity=z[e],
                        efficiency=eff[age],
                        gamma=gamma,
                    )

                    # Check feasibility of labor supply
                    if lab > 1:
                        lab = 1
                    elif lab < 0:
                        lab = 0

                    # Instantaneous utility
                    consumption = _get_consumption(
                        assets_this_period=assets_this_period,
                        assets_next_period=assets_next_period,
                        pension_benefit=0,
                        labor_input=lab,
                        interest_rate=interest_rate,
                        wage_rate=wage_rate,
                        tax_rate=tax_rate,
                        productivity=z[e],
                        efficiency=eff[age],
                    )

                    if consumption <= 0:
                        flow_utility = neg
                        assets_next_period_idx = n_gridpoints_capital - 1
                    else:
                        flow_utility = util(
                            consumption=consumption,
                            labor_input=lab,
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
                            tran[e, 0]
                            * value_working[0, assets_next_period_idx, age + 1]
                            + tran[e, 1]
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
def aggregate(
    policy_capital_working,
    policy_capital_retired,
    policy_labor,
    age_max,
    n_states_productivity,
    n_gridpoints_capital,
    duration_working,
    z_init,
    tran,
    eff,
    capital_grid,
    mass,
    duration_retired,
    n,
    z,
):
    ############################################################################
    # Aggregate capital stock and employment
    ############################################################################

    # Aggregate capital for each generation
    kgen = np.zeros((age_max, 1), dtype=np.float64)

    # Distribution of workers over capital and shocks for each working cohort
    gkW = np.zeros(
        (n_states_productivity, n_gridpoints_capital, duration_working),
        dtype=np.float64,
    )

    # Newborns
    gkW[0, 0, 0] = z_init[0] * mass[0]
    gkW[1, 0, 0] = z_init[1] * mass[0]

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
            for ind_e in range(n_states_productivity):  # current shock

                ind_kk = policy_capital_working[
                    ind_e, ind_k, ind_age
                ]  # optimal saving (as index in asset grid)

                for ind_ee in range(n_states_productivity):  # tomorrow's shock

                    if ind_age < duration_working - 1:
                        gkW[ind_ee, ind_kk, ind_age + 1] += (
                            tran[ind_e, ind_ee] * gkW[ind_e, ind_k, ind_age] / (1 + n)
                        )
                    elif (
                        ind_age == duration_working - 1
                    ):  # need to be careful because workers transit to retirees
                        gkR[ind_kk, 0] += (
                            tran[ind_e, ind_ee] * gkW[ind_e, ind_k, ind_age] / (1 + n)
                        )

        # Aggregate labor by age
        labgen[ind_age] = 0

        for ind_k in range(n_gridpoints_capital):
            for ind_e in range(n_states_productivity):
                labgen[ind_age] += (
                    z[ind_e]
                    * eff[ind_age]
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
            gkR[ind_kk, ind_age + 1] += gkR[ind_k, ind_age] / (1 + n)

        # Distribution by capital and age
        gk[:, duration_working + ind_age] = gkR[:, ind_age + 1]
        # Aggregate capital by age
        kgen[duration_working + ind_age + 1] = np.dot(
            capital_grid, gk[:, duration_working + ind_age]
        )

    k1 = np.sum(kgen)
    l1 = np.sum(labgen)

    return k1, l1, gk, gkW, gkR
