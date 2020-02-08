import numpy as np


def gini(pop, val, makeplot=False):
    """ GINI computes the Gini coefficient and the Lorentz curve.

    Usage:
        gini = gini(pop, val)
        gini, lorentz_rel = gini(pop, val)
        gini, lorentz_rel, lorentz_abs = gini(pop, val)
        ... = gini(pop, val, makeplot)

    Arguments
    ---------
        pop (np.array): population sizes of the different classes
        val (np.array): measurement variable (e.g. income per capita) in the
            different classes.
        makeplot (boolean): indicator whether a figure of the Lorentz curve
            should be produced or not. Default is false.

    Returns
    -------
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
    pop = np.pad(pop.T, ((0, 0), (1, 0)), "constant", constant_values=0)
    val = np.pad(val.T, ((0, 0), (1, 0)), "constant", constant_values=0)

    # filter out NaNs
    pop = pop[~np.isnan(pop) & ~np.isnan(val)]
    val = val[~np.isnan(pop) & ~np.isnan(val)]

    if len(pop) < 2:
        print("gini:lacking_data", "not enough data")
        gini_coefficient = np.nan
        lorentz_rel = np.nan(1, 4)
        lorentz_abs = np.nan(1, 4)
        return

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

    return gini_coefficient, lorentz_rel, lorentz_abs


def reshape_as_vector(
    in_working, in_retired, n_prod_states, age_max, age_retire, n_gridpoints_capital,
):
    """ Cast input matrices in single vector for function "gini".

    Arguments
    ---------
        in_working: np.array(n_productivity_states, duration_working, n_gridpoints_capital)
            Input for working age agents
        in_retired: np.array(duration_retired, n_gridpoints_capital)
            Input for retired agents
        n_prod_states: int
            Number of idiosyncratic productivity states
        age_max: int
            Maximum age of agents
        age_retire: int
            Retirement age
        n_gridpoints_capital: int
            Number of grid points of capital grid
    Returns
    -------
        out: np.array(np.prod(in_1.shape) + np.prod(in_2.shape), 1)
            Combined input in vector shape
    """
    out = np.zeros(
        (
            ((n_prod_states - 1) * age_retire + age_max - (n_prod_states - 1))
            * n_gridpoints_capital,
            1,
        )
    )
    out[
        : (n_prod_states * (age_retire - 1) * n_gridpoints_capital)
    ] = in_working.reshape(
        ((n_prod_states * (age_retire - 1) * n_gridpoints_capital), 1), order="F",
    )
    out[
        (n_prod_states * (age_retire - 1) * n_gridpoints_capital) :
    ] = in_retired.reshape(
        ((age_max - age_retire + 1) * n_gridpoints_capital, 1), order="F"
    )

    return out
