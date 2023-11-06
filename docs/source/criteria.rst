.. role:: hidden
    :class: hidden-section

pytools.criteria
================
.. automodule:: pytools.criteria

A bunch of criteria for (RGB and multi/hyperspectral) imaging tools.

.. contents:: pytools.criteria
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: pytools

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    criteria.Loss
    criteria.reduce_loss

Reconstruction
--------------

Sliced Losses
^^^^^^^^^^^^^

Tools used to wrap univariate distances into multivariate sliced distances.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    criteria.SliceLoss
    criteria.band_slice
    criteria.stochastic_slice

Histogram Loss
^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    criteria.HistogramLoss
    criteria.histogram_loss1D
    criteria.sliced_histogram_loss

Spectral Losses
^^^^^^^^^^^^^^^

Gram Matrices
^^^^^^^^^^^^^

Self Supervided Learning
------------------------

GAN
---

