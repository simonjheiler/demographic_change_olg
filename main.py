import json
import pdb  # noqa:F401

import numba as nb  # noqa:F401
import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.model_code.auxiliary import gini
from src.model_code.solve import _aggregate
from src.model_code.solve import solve_by_backward_induction

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
    num_income_states = params["num_income_states"]
    reform = params["reform"]
    J = params["J"]
    JR = params["JR"]
    n = params["n"]
    N = params["N"]

    tR = J - JR + 1  # length of retirement
    tW = JR - 1  # length of working life

    # Distribution of newborns over shocks
    z_init = np.array(object=[0.2037, 0.7963], dtype=np.float64)

    tran = np.array(object=[[0.9261, 1.0 - 0.9261], [1.0 - 0.9811, 0.9811]], dtype=np.float64)

    # Measure of each generation
    mass = np.ones((J, 1), dtype=np.float64)
    for j in range(1, J):
        mass[j] = mass[j - 1] / (1 + n)

    # Normalized measure of each generation (sum up to 1)
    mass = mass / sum(mass)

    # Capital grid
    capital_min = 0.001  # minimum value of capital grid
    capital_max = 30  # maximum value of capital grid
    n_gridpoints_capital = 10  # number of grid points
    capital_grid = np.linspace(capital_min, capital_max, n_gridpoints_capital, dtype=np.float64)

    # Social Security tax rate and initial guesses
    if reform == 0:
        tau = np.float64(0.11)
        k0 = np.float64(3.3254)
        l0 = np.float64(0.3414)
        z = np.array(object=[3.0, 0.5], dtype=np.float64)
        gamma = np.float64(0.42)
    elif reform == 1:
        tau = np.float64(0.0)
        k0 = np.float64(4.244)
        l0 = np.float64(0.3565)
        z = np.array(object=[3.0, 0.5], dtype=np.float64)
        gamma = np.float64(0.42)
    elif reform == 2:
        tau = np.float64(0.11)
        k0 = np.float64(1.0792)
        l0 = np.float64(0.1616)
        z = np.array(object=[0.5, 0.5], dtype=np.float64)
        gamma = np.float64(0.42)
    elif reform == 3:
        tau = np.float64(0.0)
        k0 = np.float64(1.343)
        l0 = np.float64(0.1691)
        z = np.array(object=[0.5, 0.5], dtype=np.float64)
        gamma = np.float64(0.42)
    elif reform == 4:
        tau = np.float64(0.11)
        k0 = np.float64(5.4755)
        l0 = np.float64(0.7533)
        z = np.array(object=[3.0, 0.5], dtype=np.float64)
        gamma = np.float64(0.999)
    elif reform == 5:
        tau = np.float64(0.0)
        k0 = np.float64(6.845)
        l0 = np.float64(0.7535)
        z = np.array(object=[3.0, 0.5], dtype=np.float64)
        gamma = np.float64(0.999)

    # Tolerance levels for capital, labor and pension benefits
    tolk = 1e-4
    tollab = 1e-4

    nq = 5  # Max number of iterations
    q = 0  # Counter for iterations

    k1 = k0 + 10
    l1 = l0 + 10
    neg = -1e10  # very small number

    ################################################################
    # Loop over capital, labor and pension benefits
    ################################################################


    while (q < nq) and ((abs(k1 - k0) > tolk) or (abs(l1 - l0) > tollab)):

        q = q + 1

        print(f"Iteration {q} out of {nq}")

        # Calculate factor prices from aggregates
        r0 = np.float64(alpha * (k0 ** (alpha - 1)) * (l0 ** (1 - alpha)) - delta)
        w0 = np.float64((1 - alpha) * (k0 ** (alpha)) * (l0 ** (-alpha)))
        pension_benefit = np.float64(tau * w0 * l0 / np.sum(mass[JR - 1 :]))

        ############################################################################
        # Solve for policy functions
        ############################################################################

        kapWopt, kapRopt, labopt = solve_by_backward_induction(
            interest_rate=r0,
            wage_rate=w0,
            capital_grid=capital_grid,
            n_gridpoints_capital=n_gridpoints_capital,
            sigma=sigma,
            gamma=gamma,
            pension_benefit=pension_benefit,
            neg=neg,
            tR=tR,
            tW=tW,
            N=N,
            tax_rate=tau,
            beta=beta,
            z=z,
            eff=eff,
            tran=tran
        )

        ############################################################################
        # Aggregate capital stock and employment
        ############################################################################

        k1, l1, gk, gkW, gkR = _aggregate(
            kapWopt=kapWopt,
            kapRopt=kapRopt,
            labopt=labopt,
            J=J,
            N=N,
            n_gridpoints_capital=n_gridpoints_capital,
            tW=tW,
            z_init=z_init,
            tran=tran,
            eff=eff,
            capital_grid=capital_grid,
            mass=mass,
            tR=tR,
            n=n,
            z=z,
        )

        # Update the guess on capital and labor
        k0 = 0.95 * k0 + 0.05 * k1
        l0 = 0.95 * l0 + 0.05 * l1

        # Display results
        print("capital | labor | pension")
        print([k0, l0, pension_benefit])
        print("deviation-capital | deviation-labor")
        print([abs(k1 - k0), abs(l1 - l0)])

    # Display equilibrium results
    print("k0 | l0 | w | r | b ")
    print([k0, l0, w0, r0, pension_benefit])

    # Prices
    r0 = alpha * (k0 ** (alpha - 1)) * (l0 ** (1 - alpha)) - delta
    w0 = (1 - alpha) * (k0 ** (alpha)) * (l0 ** (-alpha))

    # Mass of agent at upper bound of assets
    mass_upper_bound = np.sum(gk[-1, :])
    print(f"mass of agents at upper bound of asset grid = {mass_upper_bound}")

    # Average hours worked
    h = np.zeros((tW, 1))

    for d in range(tW):
        for ii in range(n_gridpoints_capital):
            for jj in range(N):
                h[d] += labopt[jj, ii, d] * gkW[jj, ii, d]
        h[d] = h[d] / sum(sum(gkW[:, :, d]))

    # Gini disposable income
    incomeW = np.zeros((N, n_gridpoints_capital, tW))

    for d in range(tW):
        for ii in range(n_gridpoints_capital):
            for jj in range(N):
                incomeW[jj, ii, d] = (
                    r0 * capital_grid[ii] + z[jj] * eff[d] * labopt[jj, ii, d]
                )

    incomeR = r0 * capital_grid + pension_benefit
    incomeR = np.tile(incomeR, (tR, 1)).T

    pop = np.zeros((((N - 1) * JR + J - (N - 1)) * n_gridpoints_capital, 1))
    pop[: (N * (JR - 1) * n_gridpoints_capital)] = gkW.reshape(((N * (JR - 1) * n_gridpoints_capital), 1), order="F")
    pop[(N * (JR - 1) * n_gridpoints_capital) :] = gkR.reshape(((J - JR + 1) * n_gridpoints_capital, 1), order="F")
    income = np.zeros((((N - 1) * JR + J - (N - 1)) * n_gridpoints_capital, 1))
    income[: (N * (JR - 1) * n_gridpoints_capital)] = incomeW.reshape(((N * (JR - 1) * n_gridpoints_capital), 1), order="F")
    income[(N * (JR - 1) * n_gridpoints_capital) :] = incomeR.reshape(((J - JR + 1) * n_gridpoints_capital, 1), order="F")

    gini_index, _, _ = gini(pop, income)
    print(f"gini_index = {gini_index}")
