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

with open(ppj("OUT_DATA", "simulated_demographics.pickle"), "rb") as in_file:
    demographics = pickle.load(in_file)


#####################################################
# FUNCTIONS
######################################################


def plot_dependency_ratio():
    """Plot the development of the old-age dependency ratio over time."""

    mass_distribution = demographics["mass_transition"]

    projection_length = mass_distribution.shape[1]

    plot_x = np.arange(0, projection_length)
    plot_y = np.sum(mass_distribution[age_retire:, :], axis=0) / np.sum(
        mass_distribution[: age_retire - 1, :], axis=0
    )
    fig, ax = plt.subplots()
    ax.plot(plot_x, plot_y)

    ax.set(
        xlabel="time index", ylabel="old-age dependency ratio", ybound=[0, 0.6],
    )

    fig.savefig(ppj("OUT_FIGURES", "dependency_ratio.png"))


#####################################################
# SCRIPT
######################################################


if __name__ == "__main__":

    plot_dependency_ratio()
