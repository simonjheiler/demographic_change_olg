import pdb  # noqa:F401

import numba as nb
import numpy as np


def gini(pop, val, makeplot=False):
    """ GINI computes the Gini coefficient and the Lorentz curve.

    Usage:
        gini = gini(pop, val)
        gini, lorentz_rel = gini(pop, val)
        gini, lorentz_rel, lorantz_abs = gini(pop, val)
        ... = gini(pop, val, makeplot)

    Args:
        pop (np.array): population sizes of the different classes
        val (np.array): measurement variable (e.g. income per capita) in the
            diffrerent classes.
        makeplot (boolean): indicator whether a figure of the Lorentz curve
            should be produced or not. Default is false.

    Outputs:
        gini (float): Gini coefficient
        lorentz_rel (np.array): Lorentz curve (relative): two-column array, with
            the left column representing cumulative population shares of the
            different classes, sorted according to val, and the right column
            representing the cumulative value share that belongs to the
            population up to the given class. The Lorentz curve is a scatter
            plot of the left vs the right column.
        lorentz_abs (np.array): Lorentz curve (absolute): Same as lorentz_rel,
            except that the components are not normalized to range in the unit
            interval. Thus, the left column of a is the absolute cumulative
            population sizes of the classes, and the right column is the
            absolute cumulative value of all classes up to the given one.

    Example:
        x = rand(100,1);
        y = rand(100,1);
        gini(x,y,true);  # random populations with random incomes figure;
        gini(x,ones(100,1),true);  # perfect equality

    Explanation: The vectors pop and val must be equally long and must contain
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

    """
    # check arguments
    assert np.prod(pop.shape) == np.prod(
        val.shape
    ), "gini expects two equally long vectors (#d ~= #d).".format(
        np.prod(pop.shape), np.prod(val.shape)
    )

    # pre-append a zero
    pop = np.pad(pop.T, ((0, 0), (1, 0)), "constant", constant_values=(0))
    val = np.pad(val.T, ((0, 0), (1, 0)), "constant", constant_values=(0))

    # filter out NaNs
    pop = pop[~np.isnan(pop) & ~np.isnan(val)]
    val = val[~np.isnan(pop) & ~np.isnan(val)]

    if len(pop) < 2:
        print("gini:lacking_data", "not enough data")
        gini = np.nan
        lorentz_rel = np.nan(1, 4)
        lorentz_abs = np.nan(1, 4)
        return

    assert np.all(pop >= 0) and np.all(
        val >= 0
    ), "gini expects nonnegative vectors (neg elements in pop = {}, in val = {}).".format(
        sum(pop < 0), sum(val < 0)
    )

    # process input
    z = val * pop
    val_index = val.argsort()
    pop = pop[val_index]
    z = z[val_index]
    pop = np.cumsum(pop)
    z = np.cumsum(z)
    relpop = pop / pop[-1]
    relz = z / z[-1]

    # Gini coefficient

    # We compute the area below the Lorentz curve. We do this by
    # computing the average of the left and right Riemann-like sums.
    # (I say Riemann-'like' because we evaluate not on a uniform grid, but
    # on the points given by the pop data). These are the two Rieman-like sums:
    #    leftsum  = sum(relz[:-1] * diff(relpop))
    #    rightsum = sum(relz[1:] * diff(relpop))
    # The Gini coefficient is one minus twice the average of leftsum and
    # rightsum. We can put all of this into one line.
    gini = 1 - sum((relz[:-1] + relz[1:]) * np.diff(relpop))

    # Lorentz curve
    lorentz_rel = [relpop, relz]
    lorentz_abs = [pop, z]
    if makeplot:
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

    return gini, lorentz_rel, lorentz_abs


@nb.njit
def _get_labor_input(
    assets_this_period,
    assets_next_period,
    interest_rate,
    wage_rate,
    tax_rate,
    productivity,
    efficiency,
    gamma,
):
    labor_input = (
        gamma * (1 - tax_rate) * productivity * efficiency * wage_rate
        - (1 - gamma) * ((1 + interest_rate) * assets_this_period - assets_next_period)
    ) / ((1 - tax_rate) * wage_rate * productivity * efficiency)

    return labor_input


@nb.njit
def util(
    consumption, labor_input, gamma, sigma,
):
    flow_utility = (
        ((consumption ** gamma) * (1 - labor_input) ** (1 - gamma)) ** (1 - sigma)
    ) / (1 - sigma)

    return flow_utility


@nb.njit
def _get_consumption(
    assets_this_period,
    assets_next_period,
    pension_benefit,
    labor_input,
    interest_rate,
    wage_rate,
    tax_rate,
    productivity,
    efficiency,
):
    consumption = (
        (1 + interest_rate) * assets_this_period
        + (1 - tax_rate) * wage_rate * productivity * efficiency * labor_input
        + pension_benefit
        - assets_next_period
    )

    return consumption
