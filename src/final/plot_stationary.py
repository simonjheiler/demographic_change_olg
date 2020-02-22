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

capital_min = np.float64(params_general["capital_min"])
capital_max = np.float64(params_general["capital_max"])
n_gridpoints_capital = np.int32(params_general["n_gridpoints_capital"])
hc_min = np.float64(params_general["hc_min"])
hc_max = np.float64(params_general["hc_max"])
n_gridpoints_hc = np.int32(params_general["n_gridpoints_hc"])

capital_grid = np.linspace(
    capital_min, capital_max, n_gridpoints_capital, dtype=np.float64
)
hc_grid = np.logspace(np.log(hc_min), np.log(hc_max), n_gridpoints_hc, base=np.exp(1))


#####################################################
# FUNCTIONS
######################################################


def plot_stationary(results):
    """Plot results for stationary equilibrium."""

    # Load results
    mass_distribution_capital = np.zeros(
        (n_gridpoints_capital, age_max), dtype=np.float64
    )
    mass_distribution_capital[:, : age_retire - 1] = np.sum(
        results["mass_distribution_full_working"], axis=1
    )
    mass_distribution_capital[:, age_retire - 1 :] = results[
        "mass_distribution_full_retired"
    ]

    mass_distribution_hc = np.zeros((n_gridpoints_hc, age_max), dtype=np.float64)
    mass_distribution_hc[:, : age_retire - 1] = np.sum(
        results["mass_distribution_full_working"], axis=0
    )
    mass_distribution_hc[0, age_retire - 1 :] = np.sum(
        results["mass_distribution_full_retired"], axis=0
    )

    capital_distribution_age = np.array(
        [
            np.dot(mass_distribution_capital[:, age], capital_grid)
            for age in range(age_max)
        ]
    )
    hc_distribution_age = np.array(
        [np.dot(mass_distribution_hc[:, age], hc_grid) for age in range(age_max)]
    )

    profile_capital = np.sum(
        np.multiply(
            np.repeat(capital_grid[:, np.newaxis], age_max, axis=1),
            mass_distribution_capital > 0,
        ),
        axis=0,
    )
    profile_hc = np.sum(
        np.multiply(
            np.repeat(hc_grid[:, np.newaxis], age_max, axis=1), mass_distribution_hc > 0
        ),
        axis=0,
    )

    plot_x = np.arange(age_max)
    plot_y_1 = np.array([capital_distribution_age, hc_distribution_age])
    plot_y_2 = np.array([profile_capital, profile_hc])

    # Create figure and plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1 = ax1.plot(plot_x, plot_y_1[0, :], color="tab:blue", label="assets")
    ax2 = ax1.twinx()
    line2 = ax2.plot(plot_x, plot_y_1[1, :], color="tab:orange", label="human capital")
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, (labels), loc=0)
    ax1.set(xlabel="age", ylabel="assets", ybound=[0, 0.5])
    ax2.set(ylabel="human capital", ybound=[0, 0.2])

    # Save figure
    fig.savefig(ppj("OUT_FIGURES", f"aggregates_by_age_{model_name}.png"))

    # Create figure and plot
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(plot_x, plot_y_2[0, :], color="tab:blue", label="assets")
    ax2 = ax1.twinx()
    line2 = ax2.plot(plot_x, plot_y_2[1, :], color="tab:orange", label="human capital")
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, (labels), loc=0)
    ax1.set(xlabel="age", ylabel="assets", ybound=[0, 30.0])
    ax2.set(ylabel="human capital", ybound=[0, 5.0])

    # Save figure
    fig.savefig(ppj("OUT_FIGURES", f"lifecycle_profiles_{model_name}.png"))


#####################################################
# SCRIPT
######################################################


if __name__ == "__main__":

    model_name = sys.argv[1]

    # Load data
    with open(
        ppj("OUT_ANALYSIS", f"stationary_{model_name}.pickle"), "rb"
    ) as pickle_file:
        results_stationary = pickle.load(pickle_file)

    # Produce plots
    plot_stationary(results_stationary)
