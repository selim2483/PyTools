from .loss import Loss, reduce_loss
from .slice import SliceLoss, band_slice, stochastic_slice, sliced_function
from .gram import gram_loss_mse, gram_loss_mse_layer, GramLoss
from .histograms import histogram_loss1D, sliced_histogram_loss, HistogramLoss
from .spectrum import (
    spectrum_ortho_loss1D, sliced_spectrum_ortho_loss, SpectralOrthoLoss,
    spectral_loss1D, sliced_spectral_loss, SpectralLoss,
    radial_profile, radial_spectral_loss1D, sliced_radial_spectral_loss, 
    RadialSpectralLoss)