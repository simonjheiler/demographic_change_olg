""" Simulate fertility and survival rates to mimic demographic development
    depicted in Ludwig, Schelkle, Vogel (2006).
"""
import copy
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
        Database (HMD) to qualitatively replicate the change in the old age dependency
        ratio that is at the center of the analyses in Ludwig, Schelkle, Vogel (2006).

    This is a short-cut that abstracts from actual modelling of changes in fertility
        and is solely designed to generate survival rates that mimic the most important
        qualitative features of realistic projections of future mortality rates.

    Start from the observed survival rates, truncate such that they are consistent with
        minimum and maximum model age. Then, iteratively convert to mortality rates,
        adjust future mortality rates with a fixed and exogenous matrix of adjustment
        factors (which is, in turn, chosen such that basic features of the population
        dynamics of Ludwig, Schelkle, Vogel (2006) are obtained), convert back to survival
        rates (making sure that mortality at maximum age is 1.0).

    Returns
    -------
        survival_rates: np.array(age_max, projection_length + 1)
            Matrix of simulated conditional year-to-year survival rates by age and time
    """

    # Read in raw data
    survival_rates_raw = np.loadtxt(
        ppj("IN_DATA", "survival_rates_raw.csv"), delimiter=",", dtype=np.float64,
    )

    # Adjustment factors to be changed for simulated change in survival probabilities
    adjustment = np.ones((age_max, projection_length), dtype=np.float64)
    adjustment[:, 10:50] = 0.96

    # Initiate object to store simulated survival probabilities
    survival_rates = np.ones((age_max, projection_length + 1), dtype=np.float64)

    # Initial survival probabilities are empirical data
    survival_rates[: age_max - 1, 0] = survival_rates_raw[
        age_min + 1 : age_min + age_max
    ]

    # Probability to survive after max age is zero
    survival_rates[-1, 0] = 0.0

    # Simulate over transition period by iterating over adjustment factors
    for time_idx in range(projection_length):
        mortality_rates_tmp = 1.0 - survival_rates[:, time_idx]
        mortality_rates_next = mortality_rates_tmp * adjustment[:, time_idx]
        mortality_rates_next[-1] = 1.0
        survival_rates_next = 1.0 - mortality_rates_next
        survival_rates[:, time_idx + 1] = survival_rates_next

    return survival_rates


def simulate_mass(survival_rates):
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
        survival_rates: np.array(age_max, projection_length)
            Conditional year-to-year survival probabilities by age and time (cohort)
    Returns
    -------
        mass_distribution: np.array(age_max, projection_length)
            Distribution of agents by age and time (cohort), normalized s.th. year-0
            cohort has mass 1
        fertility_rates: np.array(projection_length)
            Year-to-year growth rate of newborn agents by time (cohort)
    """
    # Initialize object to store mass distribution
    mass_distribution = np.ones((age_max, projection_length + 1), dtype=np.float64)
    fertility_rates = np.ones(projection_length + 1, dtype=np.float64)

    # Simulate mass starting with mass 1 agents of age zero at time zero
    mass_distribution[:, 0], fertility_rates[0] = find_stationary_population(
        survival_rates[:, 0]
    )

    # Simulate mass distribution throughout modelling horizon by iterating over
    # simulated fertility and survival rates
    for time_idx in range(projection_length):
        # iterate forward on initial population distribution
        mass_new = mass_distribution[:, time_idx] * survival_rates[:, time_idx]
        mass_distribution[1:, time_idx + 1] = mass_new[:-1]
        # find fertility rates to keep population constant
        mass_distribution[0, time_idx + 1] = 1 - sum(mass_new)
        fertility_rates[time_idx + 1] = (
            mass_distribution[0, time_idx] / mass_distribution[0, time_idx + 1]
        )

    # Normalize such that initial generation has measure one
    mass_distribution = mass_distribution / np.sum(mass_distribution[:, 0])

    return mass_distribution, fertility_rates


def find_stationary_population(survival_rates):
    """ Find the stationary population distribution and implied fertility rate given
        survival rates.

    Iteratively simulate population dynamics over time for given fixed survival rates,
        replacing mass of deceased agents in all periods. Stationary distribution is
        achieved by convergence and assumed to be met after sufficiently many iterations.

    Arguments
    ---------
        survival_rates: np.array(age_max, projection_length)
            Conditional year-to-year survival probabilities by age
    Returns
    -------
        mass_stationary: np.array(age_max)
            Stationary distribution of agents by age
        fertility_rates: np.float64
            Year-to-year growth rate of newborn agents consistent with stationary population
    """
    max_iter = 10000

    mass_in = np.zeros(age_max, dtype=np.float64)
    mass_in[0] = 1.0
    mass_out = np.zeros(age_max, dtype=np.float64)
    fertility = np.float64(1.0)

    for _ in range(max_iter):
        mass_new = mass_in * survival_rates
        mass_out[1:] = mass_new[:-1]
        mass_out[0] = 1 - sum(mass_new)
        fertility = mass_out[0] / mass_in[0]
        mass_in = copy.deepcopy(mass_out)

    fertility_stationary = fertility
    mass_stationary = mass_out

    return mass_stationary, fertility_stationary


def save_data(sample_1, sample_2, sample_3):
    np.savetxt(ppj("OUT_DATA", "survival_rates.csv"), sample_1, delimiter=",")
    np.savetxt(ppj("OUT_DATA", "fertility_rates.csv"), sample_2, delimiter=",")
    np.savetxt(ppj("OUT_DATA", "mass_distribution.csv"), sample_3, delimiter=",")


#####################################################
# SCRIPT
######################################################

if __name__ == "__main__":

    survival_rates_sim = extrapolate_survival()
    mass_sim, fertility_rates_sim = simulate_mass(survival_rates_sim)

    save_data(survival_rates_sim, fertility_rates_sim, mass_sim)
