"""Create tables for the term paper."""
import json
import pickle

import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj


#####################################################
# PARAMETERS & RESULTS
#####################################################


with open(ppj("IN_MODEL_SPECS", "setup_general.json")) as json_file:
    params_general = json.load(json_file)

with open(ppj("IN_MODEL_SPECS", "transition_constant_tax_rate.json")) as json_file:
    params_transition = json.load(json_file)

with open(ppj("OUT_DATA", "simulated_demographics.pickle"), "rb") as pickle_file:
    demographics = pickle.load(pickle_file)

with open(ppj("OUT_ANALYSIS", "stationary_initial.pickle"), "rb") as pickle_file:
    results_initial = pickle.load(pickle_file)

with open(ppj("OUT_ANALYSIS", "stationary_final.pickle"), "rb") as pickle_file:
    results_final = pickle.load(pickle_file)


#####################################################
# FUNCTIONS
#####################################################


def create_table_life_expectancy():

    # Load model parameters
    age_min = params_general["age_min"]
    age_max = params_general["age_max"]
    age_retire = params_general["age_retire"]
    duration_transition = params_transition["duration_transition"]
    survival_rates_conditional = demographics["survival_rates_transition"]
    mass_distribution = demographics["mass_transition"]

    # Calculate derived parameters
    duration_retire = age_max - age_retire + 1
    mortality_rates = 1.0 - survival_rates_conditional

    # Set parameters for summary statistics
    time_indices = [0, 10, 20, 30, 40, 50, 55]
    row_names = [f"at $t={idx}$" for idx in time_indices]
    row_names[0] = "before transition"
    row_names[-1] = "after transition"

    age_indices = [0, age_retire]
    col_names = ["life expectancy at birth", "life expectancy at retirement"]

    # Initiate objects to store results
    survival_rates_unconditional_at_birth = np.ones(
        (age_max, duration_transition + 1), dtype=np.float64
    )
    survival_rates_unconditional_at_retirement = np.ones(
        (duration_retire, duration_transition + 1), dtype=np.float64
    )
    life_expectancy = np.full(
        (len(time_indices), len(age_indices)), age_min, dtype=np.float64
    )

    # Calculate unconditional survival rates
    for age_idx in range(age_max):
        survival_rates_unconditional_at_birth[age_idx, :] = np.prod(
            survival_rates_conditional[:age_idx, :], axis=0
        )
    for age_idx in range(duration_retire):
        survival_rates_unconditional_at_retirement[age_idx, :] = np.prod(
            survival_rates_conditional[age_retire - 1 : age_retire - 1 + age_idx, :],
            axis=0,
        )

    # Calculate unconditional probability to die at a given age
    unconditional_mortality_rates_at_birth = np.multiply(
        survival_rates_unconditional_at_birth, mortality_rates
    )
    unconditional_mortality_rates_at_retirement = np.multiply(
        survival_rates_unconditional_at_retirement, mortality_rates[age_retire - 1 :, :]
    )

    # Calculate life expectancies as weighted average age at death
    life_expectancy[:, 0] = np.array(
        [
            np.dot(
                unconditional_mortality_rates_at_birth[:, time_idx],
                np.arange(age_max) + age_min,
            )
            for time_idx in time_indices
        ]
    )
    life_expectancy[:, 1] = np.array(
        [
            np.dot(
                unconditional_mortality_rates_at_retirement[:, time_idx],
                np.arange(age_retire - 1, age_max, 1) + age_min,
            )
            for time_idx in time_indices
        ]
    )

    # Calculate average age as weighted average of household age
    average_age = np.array(
        [
            np.dot(mass_distribution[:, time_idx], np.arange(age_max) + age_min)
            for time_idx in time_indices
        ]
    )

    # Pass results to DataFrame and format
    out = pd.DataFrame(data=life_expectancy, index=row_names, columns=col_names)
    out.insert(2, "average age", average_age)
    out = out.round(1)

    # Save results
    out.to_csv(ppj("OUT_TABLES", "life_expectancy.csv"))


def create_table_calibration():

    calibration = pd.DataFrame(
        data=[
            [
                r"$\alpha$",
                "capital weight in production function",
                params_general["alpha"],
            ],
            [r"$\beta$", "household time discount factor", params_general["beta"]],
            [
                r"$\delta_k$",
                "depreciation rate on physical capital",
                params_general["delta_k"],
            ],
            [
                r"$\delta_h$",
                "depreciation rate on human capital",
                params_general["delta_hc"],
            ],
            [r"$\zeta$", "average learning ability", params_general["zeta"]],
            [
                r"$\psi$",
                "curvature of human capital formation technology",
                params_general["psi"],
            ],
            [r"$\sigma$", "...", params_general["sigma"]],
            ["$J$", "maximum age of households", int(params_general["age_max"] - 1)],
            [
                "$JR$",
                "retirement age of households",
                int(params_general["age_retire"] - 1),
            ],
            [
                "$a_0$",
                "asset holdings of newborn agents",
                params_general["assets_init"],
            ],
            [
                "$h_0$",
                "human capital level of newborn agents",
                params_general["hc_init"],
            ],
        ],
        index=range(11),
        columns=["variable", "description", "calibrated value"],
    )

    calibration.to_csv(ppj("OUT_TABLES", "calibration.csv"))


def create_table_results_stationary():

    # Set row and column names
    row_names = ["initial steady state", "final steady state"]
    col_names = ["aggregate capital", "aggregate labor", "OADR", "pension benefits"]

    # Collect data
    out = pd.DataFrame(
        data=[
            [
                results_initial["aggregate_capital"],
                results_initial["aggregate_labor"],
                np.sum(results_initial["mass_distribution_full_retired"])
                / np.sum(results_initial["mass_distribution_full_working"]),
                results_initial["pension_benefit"],
            ],
            [
                results_final["aggregate_capital"],
                results_final["aggregate_labor"],
                np.sum(results_final["mass_distribution_full_retired"])
                / np.sum(results_final["mass_distribution_full_working"]),
                results_final["pension_benefit"],
            ],
        ],
        index=row_names,
        columns=col_names,
    )

    # Format output
    out = out.round(2)

    # Save results
    out.to_csv(ppj("OUT_TABLES", "results_stationary.csv"))


#####################################################
# SCRIPT
#####################################################


if __name__ == "__main__":

    create_table_life_expectancy()
    create_table_calibration()
    create_table_results_stationary()
