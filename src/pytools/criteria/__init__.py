from .loss import Loss, reduce_loss
from .slice import SliceLoss, band_slice, stochastic_slice
from . import gram
from .histograms import histogram_loss1D, sliced_histogram_loss, HistogramLoss
from . import spectrum