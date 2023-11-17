from math import ceil
import os
import random
from typing import Optional, Tuple, Union

import torch

from .inference import Inference
from ..criteria import (
    GramLoss, GatysStochasticLoss, SIFID, 
    histogram_loss1D, sliced_histogram_loss, 
    spectrum_ortho_loss1D, sliced_spectrum_ortho_loss,
    radial_profile, 
    corr_length, sliced_corr_length)
from ..data import periodic_smooth_decomposition
from ..nn import MeanProjector, initialize_vgg_rgb
from ..logging.images import log_images, make_grid, log_grid_plt
from ..logging.misc import console_print
from ..options.texture import InferenceOptions
from ..utils.misc import slice_tensors, tensor2list


class TextureReconstructionInference(Inference, InferenceOptions):

    xrange: int = (0, 255)
    yrange: int = (-1., 1.)

    def __init__(self, options: str | dict):
        super().__init__(options)
        self.nbands = self.data_options.nchannels
    
    @console_print("Initializing metrics")
    def _initialize_metrics(self):
        print(self.metric_options)
        
        # Seeds
        random.seed(self.metric_options.seed)
        torch.random.manual_seed(self.metric_options.seed)

        # Style
        self.vgg, self.outputs, self.get_features_vgg = initialize_vgg_rgb(
            self.vgg_options, device=self.device)
        self.style_loss_rgb = GramLoss(
            weights     = self.vgg_options.layers_weights,
            center_gram = self.metric_options.center_gram,
            reduction   = 'none',
            device      = self.device
        )
        self.style_loss_stochastic = GatysStochasticLoss(
            vgg_options = self.vgg_options, 
            nstyle      = self.metric_options.nstyle,
            inchannels  = self.data_options.nchannels,
            outchannels = 3,
            vgg_fn      = self.get_features_vgg,
            center_gram = self.metric_options.center_gram,
            reduction   = 'none',
            device      = self.device
        )
        self.mean_projector = MeanProjector(self.nbands, 3)

        # SIFID
        self.sifid = SIFID(inception_dims=64, device=self.device)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def fn_metrics(self, x:torch.Tensor, y:torch.Tensor) -> dict:
        metric_dict = {}
        random.seed(self.metric_options.seed)
        torch.random.manual_seed(self.metric_options.seed)

        # Style
        if self.metric_options.style_stochastic :
            metric_dict["style_loss_stochastic"] = tensor2list(
                self.style_loss_stochastic(x, y))
            
            if self.metric_options.style_band:
                metric_dict = {
                    **metric_dict, **self.style_loss_stochastic.loss_dict}
        
        if self.metric_options.style_groups:
            for name, bands in self.metric_options.style_groups.items():
                if bands=="mean_pool":
                    metric_dict[f"style_loss_{name}"] = tensor2list(
                        self.style_loss_rgb(
                            self.get_features_vgg(self.mean_projector(x)), 
                            self.get_features_vgg(self.mean_projector(y))
                        )
                    )
                else:
                    metric_dict[f"style_loss_{name}"] = tensor2list(
                        self.style_loss_rgb(
                            self.get_features_vgg(x[:, bands, ...]), 
                            self.get_features_vgg(y[:, bands, ...])
                        )
                    )

        # Spectrum
        if self.metric_options.spectre_stochastic:
            metric_dict["spectral_loss_stochastic"] = tensor2list(
                sliced_spectrum_ortho_loss(
                    x, y, 
                    remove_cross=self.metric_options.remove_cross, 
                    nslice=self.metric_options.nspectre,
                    device=self.device
                )
            )
        if self.metric_options.spectre_mean :
            metric_dict["spectral_loss_mean"] = tensor2list(
                spectrum_ortho_loss1D(
                    x.mean(dim=1), y.mean(dim=1), 
                    remove_cross=self.metric_options.remove_cross
                )
            )
        if self.metric_options.spectre_band :
            for i in range(self.nbands) :
                metric_dict[f"spectral_loss_band_{i}"] = tensor2list(
                    spectrum_ortho_loss1D(
                        x[:,i,...], y[:,i,...], 
                        remove_cross=self.metric_options.remove_cross
                )
            )
                
        # Histogram
        if self.metric_options.histogram_stochastic :
            metric_dict["hist_loss"] = tensor2list(sliced_histogram_loss(
                    x, y, 
                    nslice=self.metric_options.nhist, 
                    device=self.device
            ))
        if self.metric_options.histogram_band :
            for i in range(self.nbands) :
                metric_dict[f"hist_loss_band_{i}"] = tensor2list(
                    histogram_loss1D(x[:,i,...], y[:,i,...]))

        # SIFID
        if self.metric_options.sifid :
            metric_dict["SIFID_global"] = self.sifid(
                self.mean_projector(x), self.mean_projector(y))
            
        if self.metric_options.correlation_length:
            lx, ly = corr_length(x.mean(dim=1)), corr_length(x.mean(dim=1))
            metric_dict["L_corr_diff"] = tensor2list((lx - ly).abs())
            metric_dict["L_corr_gt"] = tensor2list(lx)
            metric_dict["L_corr_rec"] = tensor2list(ly)

        if self.metric_options.gradients:
            dx_x, dy_x = x.diff(dim=-1), y.diff(dim=-1)
            dx_y, dy_y = x.diff(dim=-2), y.diff(dim=-2)
            metric_dict["gradients_x_loss_sliced"] = tensor2list(
                sliced_histogram_loss(
                    dx_x, dy_x, 
                    nslice=self.metric_options.nhist, 
                    device=self.device
                )
            )
            metric_dict["gradients_y_loss_sliced"] = tensor2list(
                sliced_histogram_loss(
                    dx_y, dy_y, 
                    nslice=self.metric_options.nhist, 
                    device=self.device
                )
            )
            metric_dict["gradients_x_loss_mean"] = tensor2list(
                histogram_loss1D(dx_x[:,i,...], dy_x[:,i,...]))
            metric_dict["gradients_y_loss_mean"] = tensor2list(
                histogram_loss1D(dx_y[:,i,...], dy_y[:,i,...]))

        return metric_dict
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Images ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def parse_reconstruction(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    @console_print("Plotting images")
    def parse_and_log_images(self) :
        x, y = self.parse_reconstruction()
        nfiles = ceil(len(x)/self.logging_options.nimg)

        def _log_images(
                _x:  torch.Tensor, 
                _y:  torch.Tensor, 
                idx: Optional[Union[int, str]] = None
        ):
            if self.logging_options.reconstruction :
                log_images(
                    make_grid(
                        x, y, 
                        xrange     = self.xrange, yrange=self.yrange, 
                        format     = self.data_options.format, 
                        resolution = self.data_options.resolution
                    ),
                    logdir = self.logging_options.logdir,
                    name   = "reconstruction",
                    idx    = idx
                )
            if self.logging_options.fft2D :
                fft2D_fn = lambda img: periodic_smooth_decomposition(
                    torch.mean(img, dim=1))[0]
                log_images(
                    make_grid(
                        x, y, 
                        fn         = fft2D_fn,
                        xrange     = self.xrange, yrange=self.yrange, 
                        format     = self.data_options.format, 
                        resolution = self.data_options.resolution
                    ),
                    logdir = self.logging_options.logdir,
                    name   = "fft2D_mean",
                    idx    = idx
                )
            if self.logging_options.fft_multiband:
                for band in range(self.nbands):
                    fft2D_fn = lambda img: periodic_smooth_decomposition(
                        img[:, band, ...])[0]
                    log_images(
                        make_grid(
                            x, y, 
                            fn         = fft2D_fn,
                            xrange     = self.xrange, yrange=self.yrange, 
                            format     = self.data_options.format, 
                            resolution = self.data_options.resolution
                        ),
                        logdir = self.logging_options.logdir,
                        name   = f"fft2D_band_{band}",
                        idx    = idx
                    )
            if self.logging_options.fft_rad:
                log_grid_plt(
                    radial_profile(x.mean(dim=1), device=self.device),
                    radial_profile(y.mean(dim=1), device=self.device),
                    savefig=os.path.join(self.logdir, "fft_rad"))

