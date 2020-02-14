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

with open(ppj("IN_MODEL_SPECS", "stationary_initial.json")) as json_file:
    params = json.load(json_file)

population_growth_rate = params["population_growth_rate"]
transition_duration = params["transition_duration"]


#####################################################
# FUNCTIONS
######################################################


def extrapolate_survival():

    survival_rates = np.array(pd.read_csv(ppj("IN_DATA", "sr.csv")).values, dtype=float)

    return survival_rates


def extrapolate_fertility():

    fertility_rates = np.full(population_growth_rate, transition_duration)

    return fertility_rates


def save_data(sample):
    sample.tofile(ppj("OUT_DATA", f"{sample}.csv"), sep=",")


#####################################################
# SCRIPT
######################################################

if __name__ == "__main__":
    survival_rates = extrapolate_survival()
    fertility_rates = extrapolate_fertility()
    save_data(survival_rates)
    save_data(fertility_rates)
