import json
import pdb  # noqa:F401

import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.analysis.auxiliary import gini

num_income_states = 2
reform = 1
J = 66  # life-span
JR = 46  # age of retirement
tR = J - JR + 1  # length of retirement
tW = JR - 1  # length of working life
n = 0.011  # Population growth
N = 2  # number of shock realizations

eff = pd.read_csv(ppj("IN_DATA", "ef.csv"))
sr = pd.read_csv(ppj("IN_DATA", "sr.csv"))

params = json.load("setup.json")

alpha = params["alpha"]
beta = params["beta"]
sigma = params["sigma"]
delta = params["delta"]

print(eff)
print(sr)

# Distribution of newborns over shocks
z_init = [0.2037, 0.7963]

tran = pd.DataFrame(
    data=[[0.9261, 1.0 - 0.9261], [1.0 - 0.9811, 0.9811]],
    index=list(range(num_income_states)),
    columns=list(range(num_income_states)),
)

# Measure of each generation
mass = np.ones(J, 1)
for j in range(2, J):
    mass[j] = mass[j - 1] / (1 + n)

# Normalized measure of each generation (sum up to 1)
mass = mass / sum(mass)


maxkap = 30  # maximum value of capital grid
minkap = 0.001  # minimum value of capital grid
nk = 1000  # number of grid points
inckap = (maxkap - minkap) / (nk - 1)  # distance between points
aux = range(nk)
kap = minkap + inckap * (aux - 1)


# Capital grid
capital_min = 0.001  # minimum value of capital grid
capital_max = 30  # maximum value of capital grid
n_gridpoints_capital = 1000  # number of grid points
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
    gamma = 0.42
elif reform == 2:
    tau = 0.11
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


# Auxiliary parameters
# Tolerance levels for capital, labor and pension benefits
tolk = 1e-4
tollab = 1e-4

nq = 5  # Max number of iterations
q = 0  # Counter for iterations

k1 = k0 + 10
l1 = l0 + 10

# Initializations for backward induction
vR = np.zeros(nk, tR)  # value function of retired agents
kapRopt = np.ones(nk, tR)  # optimal savings of retired agents
# (store INDEX of k' in capital grid, not k' itself!)

vW = np.zeros(N, nk, tW)  # value function of workers
kapWopt = np.ones(N, nk, tW)  # optimal savings of workers

labopt = np.ones(N, nk, tW)  # optimal labor supply

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
    b = tau * w0 * l0 / np.sum(mass[JR:])

    ############################################################
    # BACKWARD INDUCTION
    ############################################################

    # Retired households

    # Last period utility
    consumption = (1 + r0) * kap + b  # last period consumption (vector!)
    flow_utility = (consumption ** ((1 - sigma) * gamma)) / (
        1 - sigma
    )  # last period utility (vector!)
    vR[:, tR] = flow_utility  # last period indirect utility (vector!)

    for age in range(tR, -1, -1, -1):  # age
        for assets_this_period_idx in range(nk):  # assets today

            # Initialize right-hand side of Bellman equation
            vmin = neg
            assets_next_period_idx = 0
            # More efficient is to use:
            # l = min(kapRopt(max(j-1,1),i),nk-1)-1;

            # Loop over all k's in the capital grid to find the value,
            # which gives max of the right-hand side of Bellman equation

            while assets_next_period_idx < nk:  # assets tomorrow
                assets_next_period_idx += 1
                assets_this_period = kap[
                    assets_this_period_idx
                ]  # current asset holdings
                assets_next_period = kap[
                    assets_next_period_idx
                ]  # future asset holdings

                # Instantaneous utility
                consumption = (1 + r0) * assets_this_period + b - assets_next_period

                if consumption <= 0:
                    flow_utility = neg
                    assets_next_period_idx = nk
                else:
                    flow_utility = (consumption ** ((1 - sigma) * gamma)) / (1 - sigma)

                # Right-hand side of Bellman equation
                v0 = flow_utility + beta * vR[assets_next_period_idx, age + 1]

                # Store indirect utility and optimal saving
                if v0 > vmin:
                    vR[assets_this_period_idx, age] = v0
                    kapRopt[assets_this_period_idx, age] = assets_next_period_idx
                    vmin = v0

    # Working households
    for age in range(tW, -1, -1, -1):  # age
        for e in range(N):  # productivity shock
            for assets_this_period_idx in range(nk):  # assets today

                # Initialize right-hand side of Bellman equation
                vmin = neg
                assets_next_period_idx = 0
                # More efficient is to use:
                # l=min(kapWopt(e,max(j-1,1),i),nk-1)-1;

                # Loop over all k's in the capital grid to find the value,
                # which gives max of the right-hand side of Bellman equation

                while assets_next_period_idx < nk:  # assets tomorrow
                    assets_next_period_idx += 1

                    assets_this_period = kap[
                        assets_this_period_idx
                    ]  # current asset holdings
                    assets_next_period = kap[
                        assets_next_period_idx
                    ]  # future asset holdings

                    # Optimal labor supply
                    lab = (
                        gamma * (1 - tau) * z[e] * eff[age] * w0
                        - (1 - gamma)
                        * ((1 + r0) * assets_this_period - assets_next_period)
                    ) / ((1 - tau) * w0 * z[e] * eff[age])

                    # Check feasibility of labor supply
                    if lab > 1:
                        lab = 1
                    elif lab < 0:
                        lab = 0

                    # Instantaneous utility
                    consumption = (
                        (1 + r0) * assets_this_period
                        + (1 - tau) * w0 * z[e] * eff[age] * lab
                        - assets_next_period
                    )

                    if consumption <= 0:
                        flow_utility = neg
                        assets_next_period_idx = nk
                    else:
                        flow_utility = (
                            ((consumption ** gamma) * (1 - lab) ** (1 - gamma))
                            ** (1 - sigma)
                        ) / (1 - sigma)

                    # Right-hand side of Bellman equation

                    if age == tW:  # retired next period
                        v0 = flow_utility + beta * vR[assets_next_period_idx, 1]
                    else:
                        v0 = flow_utility + beta * (
                            tran[e, 1] * vW[1, assets_next_period_idx, age + 1]
                            + tran[e, 2] * vW[2, assets_next_period_idx, age + 1]
                        )

                    # Store indirect utility, optimal saving and labor
                    if v0 > vmin:
                        vW[e, assets_this_period_idx, age] = v0
                        kapWopt[e, assets_this_period_idx, age] = assets_next_period_idx
                        labopt[e, assets_this_period_idx, age] = lab
                        vmin = v0

############################################################################
# Aggregate capital stock and employment
############################################################################

# Initializations
kgen = np.zeros(J, 1)  # Aggregate capital for each generation

# Distribution of workers over capital and shocks for each working cohort
gkW = np.zeros(N, nk, tW)

# Newborns
gkW[1, 1, 1] = z_init[1] * mass[1]  # Mass of high shock agents at age 1
# (recall that agents are born with zero assets = 1st element in the asset grid!)
gkW[2, 1, 1] = z_init[2] * mass[1]  # Mass of low shock agents at age 1

# Distribution of agents over capital for each cohort (pooling together
# both productivity shocks). This would be useful when computing total
# capital
gk = np.zeros(nk, J)
gk[1, 1] = mass[1]  # Distribution of newborns over capital

# Aggregate labor supply by generation
labgen = np.zeros(tW, 1)

# Distribution of retirees over capital
gkR = np.zeros(nk, tR)

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

                if ind_age < tW:
                    gkW[ind_ee, ind_kk, ind_age + 1] += (
                        tran[ind_e, ind_ee] * gkW[ind_e, ind_k, ind_age] / (1 + n)
                    )
                elif (
                    ind_age == tW
                ):  # need to be careful because workers transit to retirees
                    gkR[ind_kk, 1] += (
                        tran[ind_e, ind_ee] * gkW[ind_e, ind_k, ind_age] / (1 + n)
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
        if ind_age < tW:
            gk[ind_k, ind_age + 1] = np.sum(gkW[:, ind_k, ind_age + 1])
        else:
            gk[ind_k, ind_age + 1] = gkR[ind_k, 1]

    kgen[ind_age + 1] = kap * gk[:, ind_age + 1]


# Retirees
for ind_age in range(tR, -1, -1, -1):  # iterations over cohort

    for ind_k in range(nk):  # current asset holdings
        ind_kk = kapRopt[ind_k, ind_age]  # optimal saving (as index in asset grid)
        gkR[ind_kk, ind_age + 1] = gkR[ind_kk, ind_age + 1] + gkR[ind_k, ind_age] / (
            1 + n
        )

    # Distribution by capital and age
    gk[:, tW + ind_age + 1] = gkR[:, ind_age + 1]
    # Aggregate capital by age
    kgen[tW + ind_age + 1] = kap * gk[:, tW + ind_age + 1]

k1 = np.sum(kgen)
l1 = np.sum(labgen)

# Update the guess on capital and labor
k0 = 0.95 * k0 + 0.05 * k1
l0 = 0.95 * l0 + 0.05 * l1

# Display results
print("capital | labor | pension")
print([k0, l0, b])
print("deviation-capital | deviation-labor")
print([abs(k1 - k0), abs(l1 - l0)])

# Display equilibrium results
print("k0 | l0 | w | r | b ")
print([k0, l0, w0, r0, b])

# Prices
r0 = alpha * (k0 * (alpha - 1)) * (l0 ** (1 - alpha)) - delta
w0 = (1 - alpha) * (k0 ** (alpha)) * (l0 ** (-alpha))

# Mass of agent at upper bound of assets
np.sum(gk[:, :])

# Average hours worked
h = np.zeros(tW, 1)

for d in range(tW):
    for ii in range(nk):
        for jj in range(N):
            h[d] += labopt[jj, ii, d] * gkW[jj, ii, d]
    h[d] = h[d] / sum(sum(gkW[:, :, d]))

# Gini disposable income
incomeW = np.zeros(N, nk, tW)

for d in range(tW):
    for ii in range(nk):
        for jj in range(N):
            incomeW[jj, ii, d] = r0 * kap[ii] + z[jj] * eff[d] * labopt[jj, ii, d]

incomeR = [r0 * kap + b] * tR

pop = [gkW[:], gkR[:]]
income = [incomeW[:], incomeR[:]]
gini_index = gini(pop, income)
