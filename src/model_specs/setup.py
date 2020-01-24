import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj

num_income_states = 2
reform = 1
J = 100
n = 2
nk = 2
tR = 25
tW = 30
N = 1

productivity_rates = pd.read_csv(ppj("IN_DATA", "ef.csv"))
survival_rates = pd.read_csv(ppj("IN_DATA", "sr.csv"))


print(productivity_rates)
print(survival_rates)

# Distribution of newborns over shocks
income_distribution_init = [0.2037, 0.7963]

income_state_transition = pd.DataFrame(
    data=[[0.9261, 1.0 - 0.9261], [1.0 - 0.9811, 0.9811]],
    index=list(range(num_income_states)),
    columns=list(range(num_income_states)),
)

# Measure of each generation
mass = np.ones(J, 1)
for j in range(2, J):
    mass[j] = mass[j - 1] / (1 + n)

# Normalized measure of each generation (sum up to 1)
mass = mass / sum(mass)

# Capital grid
capital_min = 0.001  # minimum value of capital grid
capital_max = 30  # maximum value of capital grid
n_gridpoints_capital = 1000  # number of grid points
capital_grid = np.linspace(capital_min, capital_max, n_gridpoints_capital)

# Social Security tax rate and initial guesses
if reform == 0:
    tau = 0.11
    k0 = 3.3254
    l0 = 0.3414
    z = [3.0, 0.5]
    gamma = 0.42
elif reform == 1:
    tau = 0.0
    k0 = 4.244
    l0 = 0.3565
    z = [3.0, 0.5]
    gamma = 0.42
elif reform == 2:
    tau = 0.11
    k0 = 1.0792
    l0 = 0.1616
    z = [0.5, 0.5]
    gamma = 0.42
elif reform == 3:
    tau = 0.0
    k0 = 1.343
    l0 = 0.1691
    z = [0.5, 0.5]
    gamma = 0.42
elif reform == 4:
    tau = 0.11
    k0 = 5.4755
    l0 = 0.7533
    z = [3.0, 0.5]
    gamma = 0.999
elif reform == 5:
    tau = 0.0
    k0 = 6.845
    l0 = 0.7535
    z = [3.0, 0.5]
    gamma = 0.999


# Auxiliary parameters
# Tolerance levels for capital, labor and pension benefits
tolk = 1e-4
tollab = 1e-4

nq = 5  # Max number of iterations
q = 0  # Counter for iterations

k1 = k0 + 10
l1 = l0 + 10

# Initializations for backward induction
vR = np.zeros(nk, tR)  # value function of retired agents
kapRopt = np.ones(nk, tR)  # optimal savings of retired agents
# (store INDEX of k' in capital grid, not k' itself!)

vW = np.zeros(N, nk, tW)  # value function of workers
kapWopt = np.ones(N, nk, tW)  # optimal savings of workers

labopt = np.ones(N, nk, tW)  # optimal labor supply

neg = -1e10  # very small number
