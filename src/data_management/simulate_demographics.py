""" Simulate fertility and survival rates to mimic demographic development
    depicted in Ludwig, Schelkle, Vogel (2006).
"""
import json

import numpy as np

from bld.project_paths import project_paths_join as ppj


#####################################################
# PARAMETERS
######################################################

with open(ppj("IN_MODEL_SPECS", "setup_general.json")) as json_file:
    params_general = json.load(json_file)

age_max = np.int32(params_general["age_max"])
age_min = np.int32(params_general["age_min"])

with open(ppj("IN_MODEL_SPECS", "transition_constant_tax_rate.json")) as json_file:
    params_transition = json.load(json_file)
projection_length = np.int32(params_transition["duration_transition"])


#####################################################
# FUNCTIONS
######################################################


def extrapolate_survival():
    """ Extrapolate empirical survival rates

    Extrapolate empirical survival rates taken from the Human Mortality
    Database (HMD) to qualitatively replicate the change in the old age dependency ratio
    that is at the center of the analyses in Ludwig, Schelkle, Vogel (2006).

    This is a short-cut that abstracts from actual modelling of changes in fertility
    and is solely designed to generate survival rates that mimic the most important
    qualitative features of realistic projections of future mortality rates.

    Start from the observed survival rates and iteratively adjust future survival
    rates with a fixed and exogenous matrix of adjustment factors (which is, in turn,
    chosen such that basic features of the population dynamics of Ludwig, Schelkle,
    Vogel (2006) are obtained).

    """

    # Read in raw data
    survival_rates_raw = np.loadtxt(
        ppj("IN_DATA", "survival_rates_raw_test.csv"), delimiter=",", dtype=np.float64,
    )

    # Adjustment factors to be changed for simulated change in survival probabilities
    adjustment = np.ones((age_max, projection_length), dtype=np.float64)

    # Initiate object to store simulated survival probabilities
    survival_rates_sim = np.ones((age_max, projection_length), dtype=np.float64)

    # Initial survival probabilities are empirical data
    survival_rates_sim[: age_max - 1, 0] = survival_rates_raw[
        age_min + 1 : age_min + age_max
    ]

    # Probability to survive after max age is zero
    survival_rates_sim[-1, 0] = 0.0

    # Simulate over transition period by iterating over adjustment factors
    for time_idx in range(1, projection_length):
        survival_rates_sim[:, time_idx] = (
            survival_rates_sim[:, time_idx - 1] * adjustment[:, time_idx]
        )

    return survival_rates_sim


def extrapolate_fertility():
    """ Extrapolate empirical fertility rates

    Extrapolate empirical fertility rates taken from the Human Mortality
    Database (HMD) to qualitatively replicate the change in the old age dependency ratio
    that is at the center of the analyses in Ludwig, Schelkle, Vogel (2006).

    This is a short-cut that abstracts from actual modelling of changes in fertility
    and is solely designed to generate survival rates that mimic the most important
    qualitative features of realistic projections of future mortality rates.

    Start from the observed fertility rate and iteratively adjust future fertility
    rates with a fixed and exogenous vector of adjustment factors (which is, in turn,
    chosen such that basic features of the population dynamics of Ludwig, Schelkle,
    Vogel (2006) are obtained).

    """

    # Read in raw data
    fertility_rates_in = np.loadtxt(
        ppj("IN_DATA", "fertility_rates_raw.csv"), delimiter=",", dtype=np.float64,
    )

    # Adjustment factors to be changed for simulated change in survival probabilities
    adjustment = np.ones(projection_length, dtype=np.float64)

    # Initiate object to store simulated survival probabilities
    fertility_rates_sim = np.ones(projection_length + 1, dtype=np.float64)

    # Initial fertility rate is empirical data
    fertility_rates_sim[0] = fertility_rates_in

    # Simulate over transition period by iterating over adjustment factors
    for time_idx in range(1, projection_length):
        fertility_rates_sim[time_idx] = (
            fertility_rates_sim[time_idx - 1] * adjustment[time_idx]
        )

    return fertility_rates_sim


def simulate_mass(fertility_rates, survival_rates):
    """ Simulate the mass distribution by age of an economy over time

    Simulate population dynamics over time by iterating over fertility rates to generate
    and survival rates

    Time 0 distribution obtained by iterating backward over constant (initial) population
    growth rate to obtain mass of newborns in -t and then multiplying by unconditional
    survival probability to survive from -t to 0 (based on constant initial survival
    probabilities).

    Time t newborns obtained by iterating forward over fertility rates. Time t distribution
    obtained by iterating forward t-1 distribution given time t-1 conditional survival
    probabilities.

    Arguments
    ---------
        fertility_rates: np.array(projection_length)
            Year-to-year growth rate of newborn agents by time (cohort)
        survival_rates: np.array(age_max, projection_length)
            Conditional year-to-year survival probabilities by age and time (cohort)
    Returns
    -------
        mass_distribution: np.array(age_max, projection_length)
            Distribution of agents by age and time (cohort), normalized s.th. year-0
            cohort has mass 1
    """

    # Initialize object to store mass distribution
    mass_sim = np.ones((age_max, projection_length), dtype=np.float64)

    # Simulate mass starting with mass 1 agents of age zero at time zero
    mass_sim[0, 0] = 1.0

    # Simulate initial distribution based on constant fertility and survival rates
    # prior to modelling horizon
    for age_idx in range(1, age_max):
        mass_sim[age_idx, 0] = (
            mass_sim[0, 0]
            / ((fertility_rates[0]) ** age_idx)
            * np.prod(survival_rates[: age_idx - 1, 0])
        )

    # Simulate mass distribution throughout modelling horizon by iterating over
    # simulated fertility and survival rates
    for time_idx in range(1, projection_length):
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

    survival_rates_sim = extrapolate_survival()
    fertility_rates_sim = extrapolate_fertility()
    mass = simulate_mass(fertility_rates_sim, survival_rates_sim)

    save_data(survival_rates_sim, fertility_rates_sim, mass)
