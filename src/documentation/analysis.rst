.. _analysis:

************************************
Main model estimations / simulations
************************************

The directory *src.analysis* contains the code used to solve for stationary equilibria and transitional dynamics. This is the core of the project.

The routines contained in this directory are wrappers that call all required steps to solve a model for a given specification. Code that is called by both functions is contained in the directory *src.model_code* and documented in section :ref:`model_code`.


Stationary equilibrium
======================

.. automodule:: src.analysis.stationary
    :members:

.. automodule:: src.analysis.transition
    :members:
