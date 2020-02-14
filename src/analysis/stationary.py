import json
import pickle
import sys

import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.model_code.aggregate import aggregate_hc_readable as aggregate_hc
from src.model_code.auxiliary import get_average_hours_worked
from src.model_code.auxiliary import get_income
from src.model_code.auxiliary import gini
from src.model_code.auxiliary import reshape_as_vector
from src.model_code.solve import solve_by_backward_induction_hc_vectorized as solve_hc


#####################################################
# FUNCTIONS
######################################################


def solve_stationary(setup):

    # Load parameters
    efficiency = np.squeeze(
        np.array(
            pd.read_csv(ppj("IN_DATA", "efficiency_multiplier.csv")).values,
            dtype=np.float64,
        )
    )

    fertility_rates = np.array(
        pd.read_csv(ppj("OUT_DATA", "fertility_rates.csv")).values, dtype=np.float64
    )
    survival_rates = np.array(
        pd.read_csv(ppj("OUT_DATA", "survival_rates.csv")).values, dtype=np.float64
    )

    population_growth_rate = fertility_rates[0]
    survival_rates = survival_rates[0, :]

    with open(ppj("IN_MODEL_SPECS", f"stationary_{setup}.json")) as json_file:
        params = json.load(json_file)

    alpha = np.float64(params["alpha"])
    beta = np.float64(params["beta"])
    sigma = np.float64(params["sigma"])
    age_max = np.int32(params["age_max"])
    age_retire = np.int32(params["age_retire"])
    zeta = np.float64(params["zeta"])
    psi = np.float64(params["psi"])
    delta_k = np.float64(params["delta_k"])
    delta_hc = np.float64(params["delta_hc"])
    capital_min = np.float64(params["capital_min"])
    capital_max = np.float64(params["capital_max"])
    n_gridpoints_capital = np.int32(params["n_gridpoints_capital"])
    hc_min = np.float64(params["hc_min"])
    hc_max = np.float64(params["hc_max"])
    n_gridpoints_hc = np.int32(params["n_gridpoints_hc"])
    hc_init = np.float64(params["hc_init"])
    income_tax_rate = np.float64(params["income_tax_rate"])
    aggregate_capital_in = np.float64(params["aggregate_capital_init"])
    aggregate_labor_in = np.float64(params["aggregate_labor_init"])
    gamma = np.float64(params["gamma"])
    tolerance_capital = np.float64(params["tolerance_capital"])
    tolerance_labor = np.float64(params["tolerance_labor"])
    max_iterations_inner = np.int32(params["max_iterations_inner"])

    # calculate derived parameters
    capital_grid = np.linspace(
        capital_min, capital_max, n_gridpoints_capital, dtype=np.float64
    )
    hc_grid = np.linspace(hc_min, hc_max, n_gridpoints_hc, dtype=np.float64)
    duration_retired = age_max - age_retire + 1
    duration_working = age_retire - 1

    # Measure of each generation
    mass = np.ones((age_max, 1), dtype=np.float64)
    for j in range(1, age_max):
        mass[j] = mass[j - 1] / (1 + population_growth_rate)

    # Normalized measure of each generation (sum up to 1)
    mass = mass / sum(mass)

    num_iterations_inner = 0  # Counter for iterations

    aggregate_capital_out = aggregate_capital_in + 10
    aggregate_labor_out = aggregate_labor_in + 10
    neg = np.float64(-1e10)  # very small number

    ################################################################
    # Loop over capital, labor and pension benefits
    ################################################################

    while (num_iterations_inner < max_iterations_inner) and (
        (abs(aggregate_capital_out - aggregate_capital_in) > tolerance_capital)
        or (abs(aggregate_labor_out - aggregate_labor_in) > tolerance_labor)
    ):
        num_iterations_inner += 1

        print(f"Iteration {num_iterations_inner} out of {max_iterations_inner}")

        # Calculate factor prices from aggregates
        interest_rate = np.float64(
            alpha
            * (aggregate_capital_in ** (alpha - 1))
            * (aggregate_labor_in ** (1 - alpha))
            - delta_k
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
            policy_hc_working,
            policy_labor_working,
            policy_capital_retired,
        ) = solve_hc(
            interest_rate=interest_rate,
            wage_rate=wage_rate,
            capital_grid=capital_grid,
            n_gridpoints_capital=n_gridpoints_capital,
            hc_grid=hc_grid,
            n_gridpoints_hc=n_gridpoints_hc,
            sigma=sigma,
            gamma=gamma,
            pension_benefit=pension_benefit,
            neg=neg,
            age_max=age_max,
            age_retire=age_retire,
            income_tax_rate=income_tax_rate,
            beta=beta,
            zeta=zeta,
            psi=psi,
            delta_hc=delta_hc,
            efficiency=efficiency,
            survival_rates=survival_rates,
        )

        ############################################################################
        # Aggregate capital stock and employment
        ############################################################################

        (
            aggregate_capital_out,
            aggregate_labor_out,
            mass_distribution_full_working,
            mass_distribution_full_retired,
        ) = aggregate_hc(
            policy_capital_working=policy_capital_working,
            policy_hc_working=policy_hc_working,
            policy_labor_working=policy_labor_working,
            policy_capital_retired=policy_capital_retired,
            age_max=age_max,
            age_retire=age_retire,
            n_gridpoints_capital=n_gridpoints_capital,
            hc_init=hc_init,
            capital_grid=capital_grid,
            n_gridpoints_hc=n_gridpoints_hc,
            hc_grid=hc_grid,
            mass=mass,
            population_growth_rate=population_growth_rate,
            survival_rates=survival_rates,
        )

        # Update the guess on capital and labor
        aggregate_capital_in = (
            0.95 * aggregate_capital_in + 0.05 * aggregate_capital_out
        )
        aggregate_labor_in = 0.95 * aggregate_labor_in + 0.05 * aggregate_labor_out

        # Display results
        print("capital | labor | pension ")
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

    # Calculate equilibrium prices and pension benefits
    interest_rate = (
        alpha
        * (aggregate_capital_in ** (alpha - 1))
        * (aggregate_labor_in ** (1 - alpha))
        - delta_k
    )
    wage_rate = (
        (1 - alpha) * (aggregate_capital_in ** alpha) * (aggregate_labor_in ** (-alpha))
    )
    pension_benefit = np.float64(
        income_tax_rate
        * wage_rate
        * aggregate_labor_in
        / np.sum(mass[age_retire - 1 :])
    )

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

    # Check mass of agents at upper bound of asset grid
    mass_upper_bound = np.sum(np.sum(mass_distribution_full_working, axis=1)[-1, :])
    print(f"mass of agents at upper bound of asset grid = {mass_upper_bound}")

    # Average hours worked
    hours = get_average_hours_worked(
        policy_labor_working, mass_distribution_full_working
    )

    # Calculate Gini coefficient for disposable income
    # Calculate disposable income
    income_working, income_retired = get_income(
        interest_rate,
        capital_grid,
        pension_benefit,
        duration_retired,
        n_gridpoints_capital,
        duration_working,
        n_gridpoints_hc,
        hc_grid,
        efficiency,
        policy_labor_working,
    )
    # Reshape mass distribution and income arrays
    mass_distribution = reshape_as_vector(
        mass_distribution_full_working, mass_distribution_full_retired
    )
    income = reshape_as_vector(income_working, income_retired)
    # Calculate Gini coefficient
    gini_index, _, _ = gini(mass_distribution, income)
    print(f"gini_index = {gini_index}")

    results = {
        "aggregate_capital_in": aggregate_capital_in,
        "aggregate_labor_in": aggregate_labor_in,
        "wage_rate": wage_rate,
        "interest_rate": interest_rate,
        "pension_benefit": pension_benefit,
        "policy_capital_working": policy_capital_working,
        "policy_hc_working": policy_hc_working,
        "policy_labor_working": policy_labor_working,
        "policy_capital_retired": policy_capital_retired,
        "mass_distribution_full_working": mass_distribution_full_working,
        "mass_distribution_full_retired": mass_distribution_full_retired,
        "average_hours_worked": hours,
    }

    return results


#####################################################
# SCRIPT
######################################################


if __name__ == "__main__":
    model_name = sys.argv[1]
    model = json.load(
        open(ppj("IN_MODEL_SPECS", model_name + ".json"), encoding="utf-8")
    )
    results = solve_stationary(model_name)

    with open(ppj("OUT_ANALYSIS", f"stationary_{model_name}.pickle"), "wb") as out_file:
        pickle.dump(results, out_file)
