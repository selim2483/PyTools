.. role:: hidden
    :class: hidden-section

pytools.options
===============
.. automodule:: pytools.options

Options tools.

.. contents:: pytools.options
    :depth: 2 
    :local:
    :backlinks: top

.. currentmodule:: pytools

Options
-------

Basic ``Options`` modules following a ``dataclasses`` formalism that can be
overrided for custumized usage.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    options.Options
    options.DataOptions
    options.LoggingOptions
    options.OptimizerOptions
    options.TrainingOptions

Head Options 
------------

Head options modules embedding multiple ``Options`` modules and other
logging/saving functionnalities.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    options.HeadOptions
    options.CoachOptions