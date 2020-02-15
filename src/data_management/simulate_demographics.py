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

with open(ppj("IN_MODEL_SPECS", "setup.json")) as json_file:
    params_general = json.load(json_file)

age_max = np.int32(params_general["age_max"])
age_min = np.int32(params_general["age_min"])

with open(ppj("IN_MODEL_SPECS", "transition.json")) as json_file:
    params_transition = json.load(json_file)

transition_duration = np.int32(params_transition["transition_duration"])


#####################################################
# FUNCTIONS
######################################################


def extrapolate_survival():

    # Read in raw data
    survival_rates_raw = np.squeeze(
        np.array(
            pd.read_csv(ppj("IN_DATA", "survival_rates_raw.csv")).values, dtype=float
        )
    )

    # Adjustment factors to be changed for simulated change in survival probabilities
    adjustment = np.ones((age_max, transition_duration), dtype=np.float64)

    # Initiate object to store simulated survival probabilities
    survival_rates_sim = np.ones((age_max, transition_duration), dtype=np.float64)

    # Initial survival probabilities are empirical data
    survival_rates_sim[: age_max - 1, 0] = survival_rates_raw[
        age_min + 1 : age_min + age_max
    ]

    # Probability to survive after max age is zero
    survival_rates_sim[-1, 0] = 0.0

    # Simulate over transition period by iterating over adjustment factors
    for time_idx in range(1, transition_duration):
        survival_rates_sim[:, time_idx] = (
            survival_rates_sim[:, time_idx - 1] * adjustment[:, time_idx]
        )

    return survival_rates_sim


def extrapolate_fertility():

    # Read in raw data
    fertility_rates_in = np.squeeze(
        np.array(
            pd.read_csv(ppj("IN_DATA", "fertility_rates_raw.csv")).values, dtype=float
        )
    )

    # Adjustment factors to be changed for simulated change in survival probabilities
    adjustment = np.ones(transition_duration, dtype=np.float64)

    # Initiate object to store simulated survival probabilities
    fertility_rates_sim = np.ones(transition_duration, dtype=np.float64)

    # Initial fertility rate is empirical data
    fertility_rates_sim[0] = fertility_rates_in

    # Simulate over transition period by iterating over adjustment factors
    for time_idx in range(1, transition_duration):
        fertility_rates_sim[time_idx] = (
            fertility_rates_sim[time_idx - 1] * adjustment[time_idx]
        )

    return fertility_rates_sim


def simulate_mass(fertility_rates, survival_rates):

    # Initialize object to store mass distribution
    mass_sim = np.ones((age_max, transition_duration), dtype=np.float64)

    # Simulate mass starting with mass 1 agents of age zero at time zero
    mass_sim[0, 0] = 1.0
    # Simulate initial distribution based on constant fertility and survival rates
    # prior to modelling horizon
    for age_idx in range(1, age_max):
        mass_sim[age_idx, 0] = (
            mass_sim[0, 0]
            / ((fertility_rates[0]) ** age_idx)
            * survival_rates[age_idx - 1, 0]
        )
    # Simulate mass distribution throughout modelling horizon by iterating over
    # simulated fertility and survival rates
    for time_idx in range(1, transition_duration):
        mass_sim[0, time_idx] = (
            mass_sim[0, time_idx - 1] * fertility_rates[time_idx - 1]
        )
        for age_idx in range(1, age_max):
            mass_sim[age_idx, time_idx] = (
                mass_sim[age_idx - 1, time_idx - 1]
                * survival_rates[age_idx - 1, time_idx - 1]
            )

    # Normalize such that initial generation has measure one
    mass_distribution = mass_sim / np.sum(mass_sim[:, 0])

    return mass_distribution


def save_data(sample_1, sample_2, sample_3):
    np.savetxt(ppj("OUT_DATA", "survival_rates.csv"), sample_1, delimiter=",")
    np.savetxt(ppj("OUT_DATA", "fertility_rates.csv"), sample_2, delimiter=",")
    np.savetxt(ppj("OUT_DATA", "mass_distribution.csv"), sample_3, delimiter=",")


#####################################################
# SCRIPT
######################################################

if __name__ == "__main__":

    survival_rates = extrapolate_survival()
    fertility_rates = extrapolate_fertility()
    mass = simulate_mass(fertility_rates, survival_rates)

    save_data(survival_rates, fertility_rates, mass)
