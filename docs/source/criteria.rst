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

.. autosummary::
    :toctree: generated
    :nosignatures:
    
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

.. autosummary::
    :toctree: generated
    :nosignatures:
    
    criteria.band_slice
    criteria.stochastic_slice
    criteria.sliced_function

Histogram Loss
^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    criteria.HistogramLoss

.. autosummary::
    :toctree: generated
    :nosignatures:

    criteria.histogram_loss1D
    criteria.sliced_histogram_loss

Spectral Losses
^^^^^^^^^^^^^^^
Following objects are used to compute a spectral loss that consists of a Lp distance in the image space between y and the set of images having the same spectrum (modulus of the 2D Fourier transform) as x.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    criteria.SpectralOrthoLoss

.. autosummary::
    :toctree: generated
    :nosignatures:
    
    criteria.spectrum_ortho_loss1D
    criteria.sliced_spectrum_ortho_loss

Following objects are used to compute a basic spectral loss that consists in a Lp distance between PSD of the two images.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    criteria.SpectralLoss

.. autosummary::
    :toctree: generated
    :nosignatures:
    
    criteria.spectral_loss1D
    criteria.sliced_spectral_loss

Following objects are used to compute a radial spectral loss that consists in a Lp distance between the azimuted PSD of the two images.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    criteria.RadialSpectralLoss

.. autosummary::
    :toctree: generated
    :nosignatures:
    
    criteria.radial_profile
    criteria.radial_spectral_loss1D
    criteria.sliced_radial_spectral_loss

Gram Matrices
^^^^^^^^^^^^^

These objects allows one to compute the texture distance described by Gatys et al. using Gram matrices extracted from a pretrained extractor net.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    criteria.GramLoss

.. autosummary::
    :toctree: generated
    :nosignatures:
    
    criteria.gram_loss_mse
    criteria.gram_loss_mse_layer

Self Supervided Learning
------------------------

GAN
---

