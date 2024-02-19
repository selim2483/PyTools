from .loss import (
    Loss, 
    reduce_loss)
from .slice import (
    SliceLoss, 
    slice_tensors, sliced_function, sliced, sliced_distance)
from .gradients import image_gradient
from .gram import (
    GramLoss, GatysLoss, GatysStochasticLoss, GatysStochasticLossAdvanced, 
    GatysStochasticColorLoss,
    gram_loss_mse, gram_loss_mse_layer)
from .histograms import (
    HistogramLoss, 
    histogram_loss1D, sliced_histogram_loss)
from .spectrum import (
    spectrum_ortho_loss1D, sliced_spectrum_ortho_loss, SpectralOrthoLoss,
    spectral_loss1D, sliced_spectral_loss, SpectralLoss,
    radial_profile, radial_spectral_loss1D, sliced_radial_spectral_loss, 
    RadialSpectralLoss)
from .sifid import SIFID
from .correlation_length import corr_length, sliced_corr_length