"""Create tables for the term paper."""
import json
import pickle

import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj


#####################################################
# PARAMETERS
#####################################################


with open(ppj("IN_MODEL_SPECS", "setup_general.json")) as json_file:
    params_general = json.load(json_file)

with open(ppj("OUT_DATA", "simulated_demographics.pickle"), "rb") as pickle_file:
    demographics = pickle.load(pickle_file)


#####################################################
# FUNCTIONS
#####################################################


def create_table_life_expectancy():

    survival_rates_conditional = demographics["survival_rates_transition"]
    age_min = params_general["age_min"]
    age_max = params_general["age_max"]
    age_retire = params_general["age_retire"]

    time_indices = [0, 10, 20, 30, 40, 55]
    row_names = [f"t={idx}" for idx in time_indices]
    age_indices = [0, age_retire]
    col_names = ["at birth", "at retirement"]
    life_expectancy = np.full(
        (len(time_indices), len(age_indices)), age_min, dtype=np.float64
    )
    survival_rates_unconditional = np.ones(
        survival_rates_conditional.shape, dtype=np.float64
    )

    for age_idx in range(survival_rates_conditional.shape[0]):
        survival_rates_unconditional[age_idx, :] = np.prod(
            survival_rates_conditional[:age_idx, :], axis=0
        )

    for base_time_idx, base_time in enumerate(time_indices):
        for base_age_idx, base_age in enumerate(age_indices):
            for age in range(base_age, age_max, 1):
                life_expectancy[base_time_idx, base_age_idx] += (
                    survival_rates_unconditional[age, base_time]
                    * (1 - survival_rates_conditional[age, base_time])
                    * age
                )

    life_expectancy = pd.DataFrame(
        data=life_expectancy, index=row_names, columns=col_names
    )
    life_expectancy = life_expectancy.round(1)

    life_expectancy.to_csv(ppj("OUT_DATA", "life_expectancy.csv"))


def create_table_calibration():

    calibration = pd.DataFrame(
        data=[
            [
                "$\alpha$",
                "capital weight in production function",
                params_general["alpha"],
            ],
            ["$\beta$", "household time discount factor", params_general["beta"]],
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
            ["$J$", "maximum age of households", params_general["age_max"]],
            ["$JR$", "retirement age of households", params_general["age_retire"]],
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

    calibration.to_csv(ppj("OUT_DATA", "calibration.csv"))


#####################################################
# SCRIPT
#####################################################


if __name__ == "__main__":

    create_table_life_expectancy()
    create_table_calibration()
