import json

import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.model_code.auxiliary import gini
from src.model_code.auxiliary import reshape_as_vector
from src.model_code.solve import aggregate_readable as aggregate
from src.model_code.solve import solve_by_backward_induction

#####################################################
# SCRIPT
######################################################

if __name__ == "__main__":

    # Load parameters
    efficiency = pd.read_csv(ppj("IN_DATA", "ef.csv"))
    efficiency = np.array(efficiency.values, dtype=float)
    sr = pd.read_csv(ppj("IN_DATA", "sr.csv"))
    sr = np.array(sr.values, dtype=float)

    with open(ppj("IN_MODEL_SPECS", "setup.json")) as json_file:
        params = json.load(json_file)

    alpha = params["alpha"]
    beta = params["beta"]
    sigma = params["sigma"]
    delta = params["delta"]
    reform = params["reform"]
    age_max = params["age_max"]
    age_retire = params["age_retire"]
    population_growth_rate = params["population_growth_rate"]
    n_prod_states = params["n_prod_states"]

    duration_retired = age_max - age_retire + 1  # length of retirement
    duration_working = age_retire - 1  # length of working life

    # Distribution of newborns over shocks
    productivity_init = np.array(params["z_init"], dtype=np.float64)

    transition_prod_states = np.array(
        params["transition_prod_states"], dtype=np.float64
    )

    # Capital grid
    capital_min = params["capital_min"]  # minimum value of capital grid
    capital_max = params["capital_max"]  # maximum value of capital grid
    n_gridpoints_capital = params["n_gridpoints_capital"]  # number of grid points
    capital_grid = np.linspace(
        capital_min, capital_max, n_gridpoints_capital, dtype=np.float64
    )

    # Measure of each generation
    mass = np.ones((age_max, 1), dtype=np.float64)
    for j in range(1, age_max):
        mass[j] = mass[j - 1] / (1 + population_growth_rate)

    # Normalized measure of each generation (sum up to 1)
    mass = mass / sum(mass)

    # Social Security tax rate and initial guesses
    if reform == 0:
        income_tax_rate = np.float64(0.11)
        aggregate_capital_in = np.float64(3.3254)
        aggregate_labor_in = np.float64(0.3414)
        prod_states = np.array([3.0, 0.5], dtype=np.float64)
        gamma = np.float64(0.42)
    elif reform == 1:
        income_tax_rate = np.float64(0.0)
        aggregate_capital_in = np.float64(4.244)
        aggregate_labor_in = np.float64(0.3565)
        prod_states = np.array([3.0, 0.5], dtype=np.float64)
        gamma = np.float64(0.42)
    elif reform == 2:
        income_tax_rate = np.float64(0.11)
        aggregate_capital_in = np.float64(1.0792)
        aggregate_labor_in = np.float64(0.1616)
        prod_states = np.array([0.5, 0.5], dtype=np.float64)
        gamma = np.float64(0.42)
    elif reform == 3:
        income_tax_rate = np.float64(0.0)
        aggregate_capital_in = np.float64(1.343)
        aggregate_labor_in = np.float64(0.1691)
        prod_states = np.array([0.5, 0.5], dtype=np.float64)
        gamma = np.float64(0.42)
    elif reform == 4:
        income_tax_rate = np.float64(0.11)
        aggregate_capital_in = np.float64(5.4755)
        aggregate_labor_in = np.float64(0.7533)
        prod_states = np.array([3.0, 0.5], dtype=np.float64)
        gamma = np.float64(0.999)
    elif reform == 5:
        income_tax_rate = np.float64(0.0)
        aggregate_capital_in = np.float64(6.845)
        aggregate_labor_in = np.float64(0.7535)
        prod_states = np.array([3.0, 0.5], dtype=np.float64)
        gamma = np.float64(0.999)

    # Tolerance levels for capital, labor and pension benefits
    tolerance_capital = 1e-4
    tolerance_labor = 1e-4

    nq = 5  # Max number of iterations
    q = 0  # Counter for iterations

    aggregate_capital_out = aggregate_capital_in + 10
    aggregate_labor_out = aggregate_labor_in + 10
    neg = np.float64(-1e10)  # very small number

    ################################################################
    # Loop over capital, labor and pension benefits
    ################################################################

    while (q < nq) and (
        (abs(aggregate_capital_out - aggregate_capital_in) > tolerance_capital)
        or (abs(aggregate_labor_out - aggregate_labor_in) > tolerance_labor)
    ):

        q = q + 1

        print(f"Iteration {q} out of {nq}")

        # Calculate factor prices from aggregates
        interest_rate = np.float64(
            alpha
            * (aggregate_capital_in ** (alpha - 1))
            * (aggregate_labor_in ** (1 - alpha))
            - delta
        )
        wage_rate = np.float64(
            (1 - alpha)
            * (aggregate_capital_in ** alpha)
            * (aggregate_labor_in ** (-alpha))
        )
        pension_benefit = np.float64(
            income_tax_rate
            * wage_rate
            * aggregate_labor_in
            / np.sum(mass[age_retire - 1 :])
        )

        ############################################################################
        # Solve for policy functions
        ############################################################################

        (
            policy_capital_working,
            policy_capital_retired,
            policy_labor,
        ) = solve_by_backward_induction(
            interest_rate=interest_rate,
            wage_rate=wage_rate,
            capital_grid=capital_grid,
            n_gridpoints_capital=n_gridpoints_capital,
            sigma=sigma,
            gamma=gamma,
            pension_benefit=pension_benefit,
            neg=neg,
            duration_working=duration_working,
            duration_retired=duration_retired,
            n_prod_states=n_prod_states,
            income_tax_rate=income_tax_rate,
            beta=beta,
            prod_states=prod_states,
            efficiency=efficiency,
            transition_prod_states=transition_prod_states,
        )

        ############################################################################
        # Aggregate capital stock and employment
        ############################################################################

        aggregate_capital_out, aggregate_labor_out, gk, gkW, gkR = aggregate(
            policy_capital_working=policy_capital_working,
            policy_capital_retired=policy_capital_retired,
            policy_labor=policy_labor,
            age_max=age_max,
            n_prod_states=n_prod_states,
            n_gridpoints_capital=n_gridpoints_capital,
            duration_working=duration_working,
            productivity_init=productivity_init,
            transition_prod_states=transition_prod_states,
            efficiency=efficiency,
            capital_grid=capital_grid,
            mass=mass,
            duration_retired=duration_retired,
            population_growth_rate=population_growth_rate,
            prod_states=prod_states,
        )

        # Update the guess on capital and labor
        aggregate_capital_in = (
            0.95 * aggregate_capital_in + 0.05 * aggregate_capital_out
        )
        aggregate_labor_in = 0.95 * aggregate_labor_in + 0.05 * aggregate_labor_out

        # Display results
        print("capital | labor | pension")
        print([aggregate_capital_in, aggregate_labor_in, pension_benefit])
        print("deviation capital | deviation labor")
        print(
            [
                abs(aggregate_capital_out - aggregate_capital_in),
                abs(aggregate_labor_out - aggregate_labor_in),
            ]
        )

    ################################################################
    # Display results and calculate summary statistics
    ################################################################

    # Display equilibrium results
    print(
        "aggregate_capital | aggregate_labor | wage_rate | interest_rate | pension_benefit "
    )
    print(
        [
            aggregate_capital_in,
            aggregate_labor_in,
            wage_rate,
            interest_rate,
            pension_benefit,
        ]
    )

    # Calculate equilibrium prices
    interest_rate = (
        alpha
        * (aggregate_capital_in ** (alpha - 1))
        * (aggregate_labor_in ** (1 - alpha))
        - delta
    )
    wage_rate = (
        (1 - alpha) * (aggregate_capital_in ** alpha) * (aggregate_labor_in ** (-alpha))
    )

    # Check mass of agents at upper bound of asset grid
    mass_upper_bound = np.sum(gk[-1, :])
    print(f"mass of agents at upper bound of asset grid = {mass_upper_bound}")

    # Average hours worked
    h = np.zeros((duration_working, 1))
    for d in range(duration_working):
        for ii in range(n_gridpoints_capital):
            for jj in range(n_prod_states):
                h[d] += policy_labor[jj, ii, d] * gkW[jj, ii, d]
        h[d] = h[d] / sum(sum(gkW[:, :, d]))

    # Gini disposable income
    incomeR = interest_rate * capital_grid + pension_benefit
    incomeR = np.tile(incomeR, (duration_retired, 1)).T

    incomeW = np.zeros((n_prod_states, n_gridpoints_capital, duration_working))
    for d in range(duration_working):
        for ii in range(n_gridpoints_capital):
            for jj in range(n_prod_states):
                incomeW[jj, ii, d] = (
                    interest_rate * capital_grid[ii]
                    + prod_states[jj] * efficiency[d] * policy_labor[jj, ii, d]
                )

    pop = reshape_as_vector(
        gkW, gkR, n_prod_states, age_max, age_retire, n_gridpoints_capital
    )
    income = reshape_as_vector(
        incomeW, incomeR, n_prod_states, age_max, age_retire, n_gridpoints_capital
    )

    gini_index, _, _ = gini(pop, income)
    print(f"gini_index = {gini_index}")
