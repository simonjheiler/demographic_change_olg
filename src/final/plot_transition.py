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

with open(ppj("OUT_ANALYSIS", f"transition.pickle"), "rb") as pickle_file:
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

    # Create figure and plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1 = ax1.plot(plot_x, plot_y[0, :], color="tab:blue", label="assets")
    ax2 = ax1.twinx()
    line2 = ax2.plot(plot_x, plot_y[1, :], color="tab:orange", label="human capital")
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, (labels), loc=0)
    ax1.set(xlabel="transition period", ylabel="assets", ybound=[0, 15.0])
    ax2.set(ylabel="human capital", ybound=[0, 2.5])

    # Save figure
    fig.savefig(ppj("OUT_FIGURES", "results_transition.png"))


#####################################################
# SCRIPT
######################################################


if __name__ == "__main__":

    plot_transition()
