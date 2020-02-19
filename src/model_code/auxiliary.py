"""Conduct auxiliary operations throughout the project."""
import numba as nb
import numpy as np


def gini(pop, val, make_plot=False):
    """Computes Gini coefficient and Lorentz curve.

    The vectors pop and val must be equally long and must contain
    only positive values (zeros are also acceptable). A typical application
    would be that pop represents population sizes of some subgroups (e.g.
    different countries or states), and val represents the income per
    capita in this different subgroups. The Gini coefficient is a measure
    of how unequally income is distributed between these classes. A
    coefficient of zero means that all subgroups have exactly the same
    income per capital, so there is no dispesion of income; A very large
    coefficient would result if all the income accrues only to one subgroup
    and all the remaining groups have zero income. In the limit, when the
    total population size approaches infinity, but all the income accrues
    only to one individual, the Gini coefficient approaches unity.

    The Lorenz curve is a graphical representation of the distribution. If
    (x,y) is a point on the Lorenz curve, then the poorest x-share of the
    population has the y-share of total income. By definition, (0,0) and
    (1,1) are points on the Lorentz curve (the poorest 0% have 0% of total
    income, and the poorest 100% [ie, everyone] have 100% of total income).
    Equal distribution implies that the Lorentz curve is the 45 degree line.
    Any inequality manifests itself as deviation of the Lorentz curve from
    the 45 degree line. By construction, the Lorenz curve is weakly convex
    and increasing.

    The two concepts are related as follows: The Gini coefficient is twice
    the area between the 45 degree line and the Lorentz curve.

    Author : Yvan Lengwiler
    Release: $1.0$
    Date   : $2010-06-27$

    Arguments:
        pop: np.array(length)
            Population sizes of the different classes
        val: np.array(length)
            measurement variable (e.g. income per capita) in the different classes.
        make_plot: bool
            indicator whether a figure of the Lorentz curve should be produced
            or not. Default is false.

    Returns:
        gini: np.float64
            Gini coefficient
        lorentz_rel: np.array(num_classes)
            Lorentz curve (relative): two-column array, with the left column
            representing cumulative population shares of the different classes,
            sorted according to val, and the right column representing the cumulative
            value share that belongs to the population up to the given class. The
            Lorentz curve is a scatter plot of the left vs the right column.
        lorentz_abs: np.array(num_classes)
            Lorentz curve (absolute): Same as lorentz_rel, except that the
            components are not normalized to range in the unit interval. Thus,
            the left column of a is the absolute cumulative population sizes of
            the classes, and the right column is the absolute cumulative value of
            all classes up to the given one.
    """
    # check arguments
    assert np.prod(pop.shape) == np.prod(
        val.shape
    ), "gini expects two equally long vectors (#d ~= #d).".format(
        np.prod(pop.shape), np.prod(val.shape)
    )

    # pre-append a zero
    pop = np.pad(pop.T, (1, 0), "constant", constant_values=0)
    val = np.pad(val.T, (1, 0), "constant", constant_values=0)

    # filter out NaNs
    pop = pop[~np.isnan(pop) & ~np.isnan(val)]
    val = val[~np.isnan(pop) & ~np.isnan(val)]

    if len(pop) < 2:
        print("gini:lacking_data", "not enough data")
        gini_coefficient = np.nan
        lorentz_rel = np.nan(1, 4)
        lorentz_abs = np.nan(1, 4)
        return gini_coefficient, lorentz_rel, lorentz_abs

    assert np.all(pop >= 0) and np.all(
        val >= 0
    ), "gini expects non-negative vectors (neg elements in pop = {}, in val = {}).".format(
        sum(pop < 0), sum(val < 0)
    )

    # process input
    z = val * pop
    val_index = val.argsort()
    pop = pop[val_index]
    z = z[val_index]
    pop = np.cumsum(pop)
    z = np.cumsum(z)
    rel_pop = pop / pop[-1]
    rel_z = z / z[-1]

    # Gini coefficient

    # We compute the area below the Lorentz curve. We do this by
    # computing the average of the left and right Riemann-like sums.
    # (I say Riemann-'like' because we evaluate not on a uniform grid, but
    # on the points given by the pop data). These are the two Riemann-like sums:
    #    left_sum  = sum(rel_z[:-1] * diff(rel_pop))
    #    right_sum = sum(rel_z[1:] * diff(rel_pop))
    # The Gini coefficient is one minus twice the average of left_sum and
    # right_sum. We can put all of this into one line.
    gini_coefficient = 1 - sum((rel_z[:-1] + rel_z[1:]) * np.diff(rel_pop))

    # Lorentz curve
    lorentz_rel = [rel_pop, rel_z]
    lorentz_abs = [pop, z]
    if make_plot:
        # area(relpop, relz, 'FaceColor', [0.5, 0.5, 1.0])  # the Lorentz curve
        # plot([0, 1], [0, 1], '--k')  # 45 degree line
        # axis tight  # ranges of abscissa and ordinate are by definition exactly [0,1]
        # axis square  # both axes should be equally long
        # set(gca, 'XTick', get(gca, 'YTick'))  # ensure equal ticking
        # set(gca, 'Layer', 'top')  # grid above the shaded area
        # title(['\bfGini coefficient = ', str(g)])
        # xlabel('share of population')
        # ylabel('share of value')
        pass

    return gini_coefficient, lorentz_rel, lorentz_abs


def reshape_as_vector(in_1, in_2):
    """Cast input matrices in single vector for function "gini".

    Arguments:
        in_1: np.array(n_gridpoints_dim_1, n_gridpoints_dim_2, length_1)
            Input for working age agents
        in_2: np.array(n_gridpoints_dim_1, length_2)
            Input for retired agents
    Returns:
        out: np.array(np.prod(in_1.shape) + np.prod(in_2.shape))
            Combined input in vector shape
    """
    out = np.zeros(np.prod(in_1.shape) + np.prod(in_2.shape))
    out[: np.prod(in_1.shape)] = in_1.reshape((np.prod(in_1.shape)), order="F")
    out[np.prod(in_1.shape) :] = in_2.reshape((np.prod(in_2.shape)), order="F")

    return out


def get_average_hours_worked(policy, mass_distribution):
    """Compute average hours worked from labor input policy and mass distribution of
        working age households.

    ...

    Arguments:
        policy: np.array(n_gridpoints_dim_1, n_gridpoints_dim_2, length_1)
            Labor input policy function
        mass_distribution: np.array(n_gridpoints_dim_1, n_gridpoints_dim_2, length_1)
            Mass distribution of working age households
    Returns:
        out: np.array(length)
            Average hours worked by age
    """
    hours = np.multiply(policy, mass_distribution)
    hours_by_age = np.sum(np.sum(hours, axis=1), axis=0)
    hours_average = hours_by_age / np.sum(np.sum(mass_distribution, axis=1), axis=0)

    return hours_average


def get_income(
    interest_rate,
    capital_grid,
    pension_benefit,
    duration_retired,
    n_gridpoints_capital,
    duration_working,
    n_gridpoints_hc,
    hc_grid,
    efficiency,
    policy_labor_working,
):
    """Compute household income during working age and during retirement.

    ...

    Arguments:
        interest_rate: ---
            ...
        capital_grid: ---
            ...
        pension_benefit: ---
            ...
        duration_retired: ---
            ...
        n_gridpoints_capital: ---
            ...
        duration_working: ---
            ...
        n_gridpoints_hc: ---
            ...
        hc_grid: ---
            ...
        efficiency: ---
            ...
        policy_labor_working: ---
            ...
    Returns:
        income_retired: np.array(n_gridpoints_capital, duration_retired)
            Total household income during retirement by current asset level and age
        income_working: np.array(n_gridpoints_capital, n_gridpoints_hc, duration_working)
            Total household income during working age by current assets, current human capital
            level and age
    """
    # Repeat retirement income for all ages in retirement period
    income_retired = interest_rate * capital_grid + pension_benefit
    income_retired = np.repeat(income_retired[:, np.newaxis], duration_retired, axis=1)

    # Calculate working age income from states and labor input policy
    assets_this_period, hc_this_period = np.meshgrid(capital_grid, hc_grid)
    income_working = np.zeros((n_gridpoints_capital, n_gridpoints_hc, duration_working))

    for age_idx in range(duration_working):
        labor_input = policy_labor_working[:, :, age_idx].T
        income_working[:, :, age_idx] = (
            interest_rate * assets_this_period
            + hc_this_period * efficiency[age_idx] * labor_input
        ).T

    return income_retired, income_working


@nb.njit
def set_continuous_point_on_grid(value, grid, gp_ids, gp_weights):
    """Linearly interpolate a continuous value on a given grid.

    Linearly interpolate a continuous value on a discrete grid by first finding
    the indices of the values on *grid* circumjacent to *value* and then calculating
    linear interpolation weights.

    Arguments:
        value: np.float64
            Value to be interpolated
        grid: np.array(n_gridpoints)
            Interpolation grid
        gp_ids: np.array(2)
            Indices of interpolation nodes
        gp_weights: np.array(2)
            Interpolation weights
    """
    n = grid.size
    if value < grid[0]:
        gp_ids[0] = 0
        gp_ids[1] = 1
    elif value < grid[n - 1]:
        for i in range(1, n):
            if value < grid[i]:
                gp_ids[0] = i - 1
                gp_ids[1] = i
                break
    else:
        gp_ids[0] = n - 2
        gp_ids[1] = n - 1

    gp_weights[0] = (grid[gp_ids[1]] - value) / (grid[gp_ids[1]] - grid[gp_ids[0]])
    gp_weights[1] = np.float64(1.0) - gp_weights[0]
