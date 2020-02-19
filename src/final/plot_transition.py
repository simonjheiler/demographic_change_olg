"""Produce plots for the transitional dynamics."""
import json
import pickle

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


with open(ppj("IN_MODEL_SPECS", "transition_constant_tax_rate.json")) as json_file:
    params_transition = json.load(json_file)

duration_transition = np.int32(params_transition["duration_transition"])


#####################################################
# FUNCTIONS
######################################################


def plot_transition(results):
    """Plot results for transitional dynamics."""

    # projection_length = mass_distribution.shape[1]
    #
    # plot_x = np.arange(0, projection_length)
    # plot_y = np.sum(mass_distribution[age_retire:, :], axis=0) / np.sum(
    #     mass_distribution[: age_retire - 1, :], axis=0
    # )
    fig, ax = plt.subplots()
    # ax.plot(plot_x, plot_y)
    #
    ax.set(
        xlabel="x-axis",
        xbound=[0.0, 1.0],
        ylabel="y-axis",
        ybound=[0.0, 1.0],
        title="Results transition",
    )

    fig.savefig(ppj("OUT_FIGURES", "results_transition.png"))

    pass


#####################################################
# SCRIPT
######################################################


if __name__ == "__main__":

    # Load transitional dynamics
    with open(ppj("OUT_ANALYSIS", "transition.pickle"), "rb") as in_file:
        results_transition = pickle.load(in_file)

    # plot results
    plot_transition(results_transition)
