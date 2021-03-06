.. _model_specifications:

********************
Model specifications
********************

The directory *src.model_specs* contains `JSON <http://www.json.org/>`_ files with model specifications.

There are three types of specifications that can be passed to the code: *setup_general*, *stationary* and *transition* specifications.

*setup_general* contains basic parameters of the model. Required inputs are:

    * *beta*: time discount factor of households
    * *sigma*: inverse elasticity of substitution
    * *alpha*: capital weight in the production function
    * *delta_k*: depreciation rate on physical assets
    * *age_min*: real age corresponding to model age zero
    * *age_max*: maximum model age
    * *age_retire*: model retirement age
    * *delta_hc*: depreciation rate on human capital
    * *psi*: curvature parameter of human capital formation technology
    * *zeta*: scaling factor of the human capital formation technology (average learning ability)
    * *gamma*: weight of consumption utility
    * *capital_min*: lower bound of the asset grid
    * *capital_max*: upper bound of the asset grid
    * *n_gridpoints_capital*: number of grid points of the asset grid
    * *assets_init*: initial asset holdings of agents entering the model
    * *hc_min*: lower bound of the human capital grid
    * *hc_max*: upper bound of the human capital grid
    * *n_gridpoints_hc*: number of grid points of the human capital grid
    * *hc_init*: initial human capital level of agents entering the model
    * *tolerance_capital_stationary*: tolerance level for aggregate capital for the stationary solution algorithm
    * *tolerance_labor_stationary*: tolerance level for aggregate labor for the stationary solution algorithm
    * *tolerance_capital_transition*: tolerance level for aggregate capital for the transition solution algorithm
    * *tolerance_labor_transition*: tolerance level for aggregate labor for the transition solution algorithm
    * *max_iterations_stationary*: maximum number of iterations for the calculation of stationary equilibria
    * *max_iterations_transition*: maximum number of iterations for the calculation of transitional dynamics
    * *iteration_update_stationary*: size of the update step for the calculation of stationary equilibria
    * *iteration_update_transition*: size of the update step for the calculation of transitional dynamics

*stationary* specifications are used to calculate stationary equilibria for a given set of inputs. The required inputs are

    * *setup_name*: Name of the specification
    * *income_tax_rate*: constant labor income tax used to finance the public pension system
    * *aggregate_capital_init*: starting value for aggregate capital used in the solution algorithm
    * *aggregate_labor_init*: starting value for aggregate labor used in the solution algorithm

*transition* specifications are used to compute transitional dynamics in between two stationary equilibria. In the current setup, the required inputs are:

    * *duration_transition*: length of the transition period in between the stationary equilibria
    * *mortality_adjustment_start_time*: first time index for which adjustment of mortality rates is applied
    * *mortality_adjustment_end_time*: last time index for which adjustment of mortality rates is applied
    * *mortality_adjustment_factor*: (multiplicative) adjustment factor for mortality rates
    * *aggregate_capital_init* (optional input): starting values for aggregate capital path during transition (important: must be of length *duration_transition* + 1); if no starting value are provided, the algorithm starts with a linearly interpolated path from initial to final stationary equilibrium
    * *aggregate_labor_init* (optional input): starting values for aggregate labor path during transition (important: must be of length *duration_transition* + 1); if no starting value are provided, the algorithm starts with a linearly interpolated path from initial to final stationary equilibrium