from dataclasses import dataclass, field
from typing import Optional, Union

from . import head_options, options

BANDS_GROUPS = {
    "global": "mean_pool",
    "rgb":    [1, 2, 3],
    "nir":    [4, 6, 7],
    "swir":   [8, 9, 10] 
}

@dataclass(repr=False)
class MetricsOptions(options.MetricsOptions):
    # Style
    style_loss:           str               = "gatys"
    style_stochastic:     bool              = True
    nstyle:               int               = 1000
    style_band:           bool              = True
    style_groups:         Union[dict, bool] = field(
        default_factory=BANDS_GROUPS)
    center_gram:          bool              = True
    # Spectre
    spectral_loss:        str               = "ortho"
    spectre_stochastic:   bool              = True
    nspectre:             int               = 100
    spectre_mean:         bool              = True
    spectre_band:         bool              = True
    remove_cross:         bool              = True
    # Histogram
    histogram_stochastic: bool              = True
    nhist:                int               = 100
    histogram_band:       bool              = True
    # SIFID
    sifid:                bool              = True
    # Correlation length
    correlation_length:   bool              = True
    # Gradients
    gradients:            bool              = True

class InferenceOptions(head_options.InferenceOptions):

    def get_general_options_from_config(self):
        self.data_options    = options.DataOptions(**self.cfg.get("data")) 
        self.metric_options  = MetricsOptions(**self.cfg.get("metrics"))
        self.vgg_options     = options.VGGOptions(**self.cfg.get("vgg"))
        self.logging_options = options.LoggingOptions(
            **self.cfg.get("logging"), 
            default_logdir=getattr(self, "default_logdir", default=None)
        )
