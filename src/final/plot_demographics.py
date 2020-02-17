import json

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


def plot_dependency_ratio(mass_distribution):
    """Plot the development of the old-age dependency ratio over time."""

    projection_length = mass_distribution.shape[1]

    plot_x = np.arange(0, projection_length)
    plot_y = np.sum(mass_distribution[age_retire:, :], axis=0) / np.sum(
        mass_distribution[: age_retire - 1, :], axis=0
    )
    fig, ax = plt.subplots()
    ax.plot(plot_x, plot_y)

    ax.set(
        xlabel="time index",
        ylabel="old-age dependency ratio",
        ybound=[0, 0.4],
        title="Old-age dependency ratio over time",
    )

    fig.savefig(ppj("OUT_FIGURES", "dependency_ratio.png"))


#####################################################
# SCRIPT
######################################################


if __name__ == "__main__":

    # Load simulated mass distribution
    mass_distribution_sim = np.loadtxt(
        ppj("OUT_DATA", "mass_distribution.csv"), delimiter=",", dtype=np.float64,
    )

    plot_dependency_ratio(mass_distribution_sim)
