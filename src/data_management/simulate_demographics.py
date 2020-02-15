""" Simulate fertility and survival rates to mimic demographic development
    depicted in Ludwig, Schelkle, Vogel (2006).
"""
import json

import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj


#####################################################
# PARAMETERS
######################################################

with open(ppj("IN_MODEL_SPECS", "transition.json")) as json_file:
    params = json.load(json_file)

population_growth_rate = np.float64(params["population_growth_rate"])
transition_duration = np.int32(params["transition_duration"])

age_max = 66


#####################################################
# FUNCTIONS
######################################################


def extrapolate_survival():

    survival_rates = np.squeeze(
        np.array(pd.read_csv(ppj("IN_DATA", "survival_rates.csv")).values, dtype=float)
    )
    survival_rates = np.repeat(
        survival_rates[:, np.newaxis], transition_duration, axis=1
    )

    return survival_rates


def extrapolate_fertility():

    fertility_rates = np.full((transition_duration), population_growth_rate)

    return fertility_rates


def simulate_mass():
    # Measure of each generation
    mass = np.ones((age_max, 1), dtype=np.float64)
    for j in range(1, age_max):
        mass[j] = mass[j - 1] / (1 + population_growth_rate)

    # Normalized measure of each generation (sum up to 1)
    mass = mass / sum(mass)

    return mass


def save_data(sample_1, sample_2):
    np.savetxt(ppj("OUT_DATA", f"survival_rates.csv"), sample_1, delimiter=",")
    np.savetxt(ppj("OUT_DATA", f"fertility_rates.csv"), sample_2, delimiter=",")


#####################################################
# SCRIPT
######################################################

if __name__ == "__main__":

    survival_rates = extrapolate_survival()
    fertility_rates = extrapolate_fertility()

    save_data(survival_rates, fertility_rates)
