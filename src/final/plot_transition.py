"""Produce plots for the transitional dynamics."""
import json
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from bld.project_paths import project_paths_join as ppj

matplotlib.use("TkAgg")


#####################################################
# PARAMETERS & RESULTS
######################################################


with open(ppj("IN_MODEL_SPECS", "setup_general.json")) as json_file:
    params_general = json.load(json_file)

with open(ppj("IN_MODEL_SPECS", "transition_constant_tax_rate.json")) as json_file:
    params_transition = json.load(json_file)

with open(ppj("OUT_ANALYSIS", "transition.pickle"), "rb") as pickle_file:
    results_transition = pickle.load(pickle_file)

age_max = np.int32(params_general["age_max"])
age_retire = np.int32(params_general["age_retire"])
duration_transition = np.int32(params_transition["duration_transition"])


#####################################################
# FUNCTIONS
######################################################


def plot_transition():
    """Plot results for transitional dynamics."""

    # Load results
    plot_x = np.arange(duration_transition + 1)
    plot_y = np.array(
        [results_transition["aggregate_capital"], results_transition["aggregate_labor"]]
    )
    legend = ["aggregate capital", "aggregate labor"]

    # Create figure and plot
    fig, ax = plt.subplots()
    for series in range(plot_y.shape[0]):
        ax.plot(plot_x, plot_y[series, :])

    # Format
    ax.set(xlabel="transition period",)
    ax.legend(legend)

    # Save figure
    fig.savefig(ppj("OUT_FIGURES", "results_transition.png"))


#####################################################
# SCRIPT
######################################################


if __name__ == "__main__":

    plot_transition()
