import json
import pdb  # noqa:F401

import numba as nb  # noqa:F401
import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.analysis.auxiliary import gini
from src.analysis.auxiliary import labor_input
from src.analysis.auxiliary import util_retired
from src.analysis.auxiliary import util_working

#######################################################
# FUNCTIONS
#######################################################


#####################################################
# SCRIPT
######################################################

if __name__ == "__main__":

    # Load parameters
    eff = pd.read_csv(ppj("IN_DATA", "ef.csv"))
    eff = np.array(object=eff.values, dtype=float)
    sr = pd.read_csv(ppj("IN_DATA", "sr.csv"))
    sr = np.array(object=sr.values, dtype=float)

    with open(ppj("IN_MODEL_SPECS", "setup.json")) as json_file:
        params = json.load(json_file)

    alpha = params["alpha"]
    beta = params["beta"]
    sigma = params["sigma"]
    delta = params["delta"]
    max_age = params["max_age"]
    retirement_age = params["retirement_age"]
    beta = params["beta"]
    sigma = params["sigma"]
    alpha = params["alpha"]
    delta = params["delta"]
    num_income_states = params["num_income_states"]
    reform = params["reform"]
    J = params["J"]
    JR = params["JR"]
    n = params["n"]
    N = params["N"]

    tR = J - JR + 1  # length of retirement
    tW = JR - 1  # length of working life

    # Distribution of newborns over shocks
    z_init = [0.2037, 0.7963]

    tran = np.array(object=[[0.9261, 1.0 - 0.9261], [1.0 - 0.9811, 0.9811]])

    # Measure of each generation
    mass = np.ones((J, 1))
    for j in range(1, J):
        mass[j] = mass[j - 1] / (1 + n)

    # Normalized measure of each generation (sum up to 1)
    mass = mass / sum(mass)

    maxkap = 30  # maximum value of capital grid
    minkap = 0.001  # minimum value of capital grid
    nk = 10  # number of grid points
    kap = np.linspace(minkap, maxkap, nk)

    # Capital grid
    capital_min = 0.001  # minimum value of capital grid
    capital_max = 30  # maximum value of capital grid
    n_gridpoints_capital = 10  # number of grid points
    capital_grid = np.linspace(capital_min, capital_max, n_gridpoints_capital)

    # Social Security tax rate and initial guesses
    if reform == 0:
        tau = 0.11
        k0 = 3.3254
        l0 = 0.3414
        z = [3.0, 0.5]
        gamma = 0.42
    elif reform == 1:
        tau = 0.0
        k0 = 4.244
        l0 = 0.3565
        z = [3.0, 0.5]
        k0 = 1.0792
        l0 = 0.1616
        z = [0.5, 0.5]
        gamma = 0.42
    elif reform == 3:
        tau = 0.0
        k0 = 1.343
        l0 = 0.1691
        z = [0.5, 0.5]
        gamma = 0.42
    elif reform == 4:
        tau = 0.11
        k0 = 5.4755
        l0 = 0.7533
        z = [3.0, 0.5]
        gamma = 0.999
    elif reform == 5:
        tau = 0.0
        k0 = 6.845
        l0 = 0.7535
        z = [3.0, 0.5]
        gamma = 0.999

    # Tolpension_benefits ance levels for capital, labor and pension benefits
    tolk = 1e-4
    tollab = 1e-4

    nq = 5  # Max number of iterations
    q = 0  # Counter for iterations

    k1 = k0 + 10
    l1 = l0 + 10
    # Initializations for backward induction
    vR = np.zeros((nk, tR), dtype=float)  # value function of retired agents
    kapRopt = np.zeros((nk, tR), dtype=int)  # optimal savings of retired agents
    # (store INDEX of k' in capital grid, not k' itself!)

    vW = np.zeros((N, nk, tW), dtype=float)  # value function of workers
    kapWopt = np.zeros((N, nk, tW), dtype=int)  # optimal savings of workers
    labopt = np.zeros((N, nk, tW), dtype=int)  # optimal labor supply
    neg = -1e10  # very small number

    ################################################################
    # Loop over capital, labor and pension benefits
    ################################################################
    while (q < nq) and ((abs(k1 - k0) > tolk) or (abs(l1 - l0) > tollab)):

        q = q + 1

        print(f"Iteration {q} out of {nq}")

        # Prices
        r0 = alpha * (k0 ** (alpha - 1)) * (l0 ** (1 - alpha)) - delta
        w0 = (1 - alpha) * (k0 ** (alpha)) * (l0 ** (-alpha))
        # Pension benefit
        pension_benefits = tau * w0 * l0 / np.sum(mass[JR - 1 :])

        ############################################################
        # BACKWARD INDUCTION
        ############################################################

        # Retired households

        # Last period utili)yc
        consumption = (
            1 + r0
        ) * kap + pension_benefits  # last period consumption (vector!)
        flow_utility = (consumption ** ((1 - sigma) * gamma)) / (
            1 - sigma
        )  # last period utility (vector!)
        vR[:, tR - 1] = flow_utility  # last period indirect utility (vector!)

        for age in range(tR - 2, -1, -1):  # age

            for assets_this_period_idx in range(nk):  # assets today

                # Initialize right-hand side of Bellman equation
                vmin = neg
                assets_next_period_idx = -1
                # More efficient is to use:
                # l = min(kapRopt(max(j-1,1),i),nk-1)-1;

                # Loop over all k's in the capital grid to find the value,
                # which gives max of the right-hand side of Bellman equation

                while assets_next_period_idx < nk - 1:  # assets tomorrow
                    assets_next_period_idx += 1
                    assets_this_period = kap[
                        assets_this_period_idx
                    ]  # current asset holdings
                    assets_next_period = kap[
                        assets_next_period_idx
                    ]  # future asset holdings

                    # pdb.set_trace()

                    # Instantaneous utility
                    flow_utility = util_retired(
                        r0,
                        assets_this_period,
                        pension_benefits,
                        assets_next_period,
                        sigma,
                        gamma,
                        neg,
                    )

                    if flow_utility == neg:
                        assets_next_period_idx = nk - 1

                    # Right-hand side of Bellman equation
                    v0 = flow_utility + beta * vR[assets_next_period_idx, age + 1]

                    # Store indirect utility and optimal saving
                    if v0 > vmin:
                        vR[assets_this_period_idx, age] = v0
                        kapRopt[assets_this_period_idx, age] = assets_next_period_idx
                        vmin = v0

        # Working households
        for age in range(tW - 1, -1, -1):  # age
            for e in range(N):  # productivity shock
                for assets_this_period_idx in range(nk):  # assets today

                    # Initialize right-hand side of Bellman equation
                    vmin = neg
                    assets_next_period_idx = -1
                    # More efficient is to use:
                    # l=min(kapWopt(e,max(j-1,1),i),nk-1)-1;

                    while assets_next_period_idx < nk - 1:  # assets tomorrow

                        assets_next_period_idx += 1

                        assets_this_period = kap[assets_this_period_idx]
                        assets_next_period = kap[assets_next_period_idx]

                        # Optimal labor supply
                        lab = labor_input(
                            r0,
                            w0,
                            assets_this_period,
                            assets_next_period,
                            z[e],
                            eff[age],
                            gamma,
                            tau,
                        )

                        # Instantaneous utility
                        flow_utility = util_working(
                            r0,
                            w0,
                            assets_this_period,
                            assets_next_period,
                            lab,
                            z[e],
                            eff[age],
                            tau,
                            neg,
                            gamma,
                            sigma,
                        )

                        if flow_utility == neg:
                            assets_next_period_idx = nk - 1

                        # Right-hand side of Bellman equation
                        if age == tW - 1:  # retired next period
                            v0 = flow_utility + beta * vR[assets_next_period_idx, 0]
                        else:
                            v0 = flow_utility + beta * (
                                tran[e, 0] * vW[0, assets_next_period_idx, age + 1]
                                + tran[e, 1] * vW[1, assets_next_period_idx, age + 1]
                            )

                        # Store indirect utility, optimal saving and labor
                        if v0 > vmin:
                            vW[e, assets_this_period_idx, age] = v0
                            kapWopt[
                                e, assets_this_period_idx, age
                            ] = assets_next_period_idx
                            labopt[e, assets_this_period_idx, age] = lab
                            vmin = v0

        ############################################################################
        # Aggregate capital stock and employment
        ############################################################################

        # Initializations
        kgen = np.zeros((J, 1))  # Aggregate capital for each generation

        # Distribution of workers over capital and shocks for each working cohort
        gkW = np.zeros((N, nk, tW))

        # Newborns
        gkW[0, 0, 0] = z_init[0] * mass[0]  # Mass of high shock agents at age 1
        # (recall that agents are born with zero assets = 1st element in the asset grid!)
        gkW[1, 0, 0] = z_init[1] * mass[0]  # Mass of low shock agents at age 1

        # Distribution of agents over capital for each cohort (pooling together
        # both productivity shocks). This would be useful when computing total
        # capital
        gk = np.zeros((nk, J))
        gk[0, 0] = mass[0]  # Distribution of newborns over capital

        # Aggregate labor supply by generation
        labgen = np.zeros((tW, 1))

        # Distribution of retirees over capital
        gkR = np.zeros((nk, tR))

        ############################################################################
        # Iterating over the distribution
        ############################################################################
        # Workers
        for ind_age in range(tW):  # iterations over cohorts
            for ind_k in range(nk):  # current asset holdings
                for ind_e in range(N):  # current shock

                    ind_kk = kapWopt[
                        ind_e, ind_k, ind_age
                    ]  # optimal saving (as index in asset grid)

                    for ind_ee in range(N):  # tomorrow's shock

                        if ind_age < tW - 1:
                            gkW[ind_ee, ind_kk, ind_age + 1] += (
                                tran[ind_e, ind_ee]
                                * gkW[ind_e, ind_k, ind_age]
                                / (1 + n)
                            )
                        elif (
                            ind_age == tW - 1
                        ):  # need to be careful because workers transit to retirees
                            gkR[ind_kk, 0] += (
                                tran[ind_e, ind_ee]
                                * gkW[ind_e, ind_k, ind_age]
                                / (1 + n)
                            )

            # Aggregate labor by age
            labgen[ind_age] = 0

            for ind_k in range(nk):
                for ind_e in range(N):
                    labgen[ind_age] += (
                        z[ind_e]
                        * eff[ind_age]
                        * labopt[ind_e, ind_k, ind_age]
                        * gkW[ind_e, ind_k, ind_age]
                    )

            # Aggregate capital by age
            for ind_k in range(nk):
                if ind_age < tW - 1:
                    gk[ind_k, ind_age + 1] = np.sum(gkW[:, ind_k, ind_age + 1])
                else:
                    gk[ind_k, ind_age + 1] = gkR[ind_k, 1]
            kgen[ind_age + 1] = np.dot(kap, gk[:, ind_age + 1])

        # Retirees
        for ind_age in range(tR - 1):  # iterations over cohort

            for ind_k in range(nk):  # current asset holdings
                ind_kk = kapRopt[
                    ind_k, ind_age
                ]  # optimal saving (as index in asset grid)
                gkR[ind_kk, ind_age + 1] += gkR[ind_k, ind_age] / (1 + n)

            # Distribution by capital and age
            gk[:, tW + ind_age] = gkR[:, ind_age + 1]
            # Aggregate capital by age
            kgen[tW + ind_age] = np.dot(kap, gk[:, tW + ind_age])

        pdb.set_trace()
        k1 = np.sum(kgen)
        l1 = np.sum(labgen)

        # Update the guess on capital and labor
        k0 = 0.95 * k0 + 0.05 * k1
        l0 = 0.95 * l0 + 0.05 * l1

        # Display results
        print("capital | labor | pension")
        print([k0, l0, pension_benefits])
        print("deviation-capital | deviation-labor")
        print([abs(k1 - k0), abs(l1 - l0)])

    # Display equilibrium results
    print("k0 | l0 | w | r | b ")
    print([k0, l0, w0, r0, pension_benefits])

    # Prices
    r0 = alpha * (k0 * (alpha - 1)) * (l0 ** (1 - alpha)) - delta
    w0 = (1 - alpha) * (k0 ** (alpha)) * (l0 ** (-alpha))

    # Mass of agent at upper bound of assets
    np.sum(gk[:, :])

    # Average hours worked
    h = np.zeros((tW, 1))

    for d in range(tW):
        for ii in range(nk):
            for jj in range(N):
                h[d] += labopt[jj, ii, d] * gkW[jj, ii, d]
        h[d] = h[d] / sum(sum(gkW[:, :, d]))

    # Gini disposable income
    incomeW = np.zeros((N, nk, tW))

    for d in range(tW):
        for ii in range(nk):
            for jj in range(N):
                incomeW[jj, ii, d] = r0 * kap[ii] + z[jj] * eff[d] * labopt[jj, ii, d]

    incomeR = (r0 * kap + pension_benefits) * tR

    pop = [gkW[:], gkR[:]]
    income = [incomeW[:], incomeR[:]]
    gini_index = gini(pop, income)
