"""Produce plots for initial and final stationary equilibrium   ."""
import json
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from bld.project_paths import project_paths_join as ppj

matplotlib.use("TkAgg")


#####################################################
# PARAMETERS
######################################################

with open(ppj("IN_MODEL_SPECS", "setup_general.json")) as json_file:
    params_general = json.load(json_file)

age_max = np.int32(params_general["age_max"])
age_retire = np.int32(params_general["age_retire"])


#####################################################
# FUNCTIONS
######################################################


def plot_stationary(results):
    """Plot results for stationary equilibrium."""

    fig, ax = plt.subplots()
    # ax.plot(plot_x, plot_y)
    #
    ax.set(
        xlabel="x-axis",
        xbound=[0.0, 1.0],
        ylabel="y-axis",
        ybound=[0.0, 1.0],
        title=f"Results {model_name}",
    )
    #
    fig.savefig(ppj("OUT_FIGURES", f"results_stationary_{model_name}.png"))


#####################################################
# SCRIPT
######################################################


if __name__ == "__main__":

    model_name = sys.argv[1]

    # Load data
    with open(ppj("OUT_ANALYSIS", f"stationary_{model_name}.pickle"), "rb") as in_file:
        results_stationary = pickle.load(in_file)

    # Produce plots
    plot_stationary(results_stationary)
