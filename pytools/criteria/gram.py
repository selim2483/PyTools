import itertools
import random
from typing import Callable, Iterable, List, Optional, Union
import time

import torch
import torch.nn as nn

from ..criteria.loss import reduce_loss, Loss
from ..criteria.histograms import sliced_histogram_loss
from ..nn import RandomProjector, initialize_vgg
from ..nn.functionnal import compute_gram_matrix
from ..options import VGGOptions
from ..utils.checks import assert_shape, type_check
from ..utils.color import transform_color_statistics
from ..utils.misc import map_dict


LAYERS         = [1, 6, 11, 20, 29]
LAYERS_WEIGHTS = [1/n**2 for n in [64,128,256,512,512]]
NSLICES        = [n for n in [64,128,256,512,512]]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Gram matrices ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
@type_check
def feature_loss_layer(
        x:torch.Tensor, x_hat:torch.Tensor, 
        feature:   str                     = "gram", 
        nslice:    int                     = 1,
        reduction: str                     = 'mean',
        device:    Union[torch.device, str] = 'cpu'
) -> torch.Tensor:
    """Computes L2 distances between Gram matrices extracted from provided
    feature maps.

    Args:
        x (torch.Tensor): reference feature maps.
        x_hat (torch.Tensor): synthetic feature maps.
        feature (bool, optional): Whether to center feature maps to
            compute Gram matrices or not. 
            Defaults to True.
        reduction (str, optional): reduction to use for MSE computation. 
            If `'none'` : no reduction will be applied ((B) size tensor will
            be returned).
            If `'mean'` : mean loss will be returned.
            If `'sum'` : sumed loss will be returned.
            Defaults to `'mean'`.

    Returns:
        torch.Tensor: L2 distances between Gram matrices.
    """
    assert_shape(x_hat, x.shape)
    if feature == "swd":
        return sliced_histogram_loss(x, x_hat, nslice=nslice, device=device)
    if feature == "mean":
        g = x.mean(dim=(-1,-2))
        g_hat = x_hat.mean(dim=(-1,-2))
    elif feature == "covariance":
        g = compute_gram_matrix(x, center_gram=True)
        g_hat = compute_gram_matrix(x_hat, center_gram=True)
    elif feature == "gram":
        g = compute_gram_matrix(x, center_gram=False)
        g_hat = compute_gram_matrix(x_hat, center_gram=False)

    return reduce_loss(
        ((g - g_hat)**2).mean(dim=-1 if feature=="mean" else (-1,-2)), 
        reduction=reduction
    )

def feature_loss(
        F_real:    List[torch.Tensor], 
        F_hat:     List[torch.Tensor], 
        weights:   Union[Iterable, torch.Tensor] = LAYERS_WEIGHTS,
        feature:   str                           = "gram", 
        nslices:   List[int]                     = NSLICES,
        reduction: str                           = 'mean',
        device:    Union[torch.device, str]      = 'cpu',
) -> torch.Tensor:
    """Computes L2 distances between Gram matrices extracted from provided
    feature maps as described by Gatys et al.: Each layer of the extractor net
    provides feature maps for both original and reconstructed images. These
    are compared in L2 space. The contribution of each layer is finally
    wheighted and summed.

    Args:
        f_real (List[torch.Tensor]): reference feature maps.
        f_hat (List[torch.Tensor]): synthetic feature maps.
        weights (Union[Iterable, torch.Tensor]): weights to use when summing
            layer contributions.
            Default to ``LAYERS_WEIGHTS``
        feature (bool, optional): Whether to center feature maps to
            compute Gram matrices or not. 
            Defaults to True.
        reduction (str, optional): reduction to use for MSE computation. 
            If `'none'` : no reduction will be applied ((B) size tensor will
            be returned).
            If `'mean'` : mean loss will be returned.
            If `'sum'` : sumed loss will be returned.
            Defaults to `'mean'`.
        device (Union[torch.device, str], optional): device on which to place
            the tensors. 
            Defaults to "cpu".

    Returns:
        torch.Tensor: weighted over layers sum of L2 distances between Gram
            matrices extracted from feature maps.
    """
    if isinstance(weights, Iterable):
        weights = torch.tensor(weights, device=device) / len(weights)
    elif isinstance(weights, torch.Tensor):
        weights = weights / len(weights)
    else:
        raise TypeError(f"'weights' argument should be a tensor or an \
iterable, got {type(weights)}.")

    layer_losses = [
        feature_loss_layer(
            f_real.detach(), f_hat, 
            feature=feature, nslice=nslice, reduction=reduction, device=device
        ) for f_hat, f_real, nslice in zip(F_hat, F_real, nslices)
    ]

    return torch.stack(layer_losses, dim=-1) @ weights

class GramLoss(nn.Module):
    """Style Loss module.

    Computes L2 distances between Gram matrices extracted from provided
    feature maps as described by Gatys et al.: Each layer of the extractor net
    provides feature maps for both original and reconstructed images. These
    are compared in L2 space. The contribution of each layer is finally
    wheighted and summed.

    Args:
        weights (Union[Iterable, torch.Tensor]): weights to use when summing
            layer contributions.
            Default to ``LAYERS_WEIGHTS``
        feature (bool, optional): Whether to center feature maps to
            compute Gram matrices or not. 
            Defaults to True.
        reduction (str, optional): reduction to use for MSE computation.
        
            - If `'none'` : no reduction will be applied.
            - If `'mean'` : mean loss will be returned.
            - If `'sum'` : sumed loss will be returned.

            Defaults to `'mean'`.

    """
    def __init__(
            self, 
            weights   :Union[Iterable, torch.Tensor] = LAYERS_WEIGHTS,
            feature   :str                           = "gram",
            reduction :str                           = 'mean', 
            device    :Union[torch.device, str]      = 'cpu'
    ):
        super().__init__()
        self.weights     = weights
        self.feature        = feature
        self.device      = device
        self.reduction   = reduction

    def forward(
            self, 
            f_real:List[torch.Tensor], 
            f_hat:List[torch.Tensor]
    ) -> torch.Tensor :
        return feature_loss(
            f_real, 
            f_hat, 
            self.weights, 
            feature=self.feature, 
            reduction=self.reduction,
            device=self.device
        )
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Gatys RGB style loss ~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class GatysLoss(Loss):

    def __init__(
            self, 
            vgg_fn:      Optional[Callable]            = None,
            weights:     Union[Iterable, torch.Tensor] = LAYERS_WEIGHTS, 
            vgg_options: VGGOptions                    = None,               
            feature:     str                           = "gram",
            nslices:     Union[Iterable, torch.Tensor] = NSLICES,
            reduction:   str                           = 'mean', 
            device:      Union[torch.device, str]      = 'cpu'
    ):
        super().__init__(reduction, device)
        self.feature = feature
        self.nslices = nslices

        if vgg_fn is None:
            if vgg_options is None:
                raise AttributeError(
                    "Either 'vgg_fn' or 'vgg_options' arguments should be\
    provided to provide a way to extract vgg19 feature maps.")
            else:
                _, _, self.vgg_fn = initialize_vgg(
                    vgg_options, device=self.device)
                self.weigths = vgg_options.layers_weights
        else:
            self.vgg_fn, self.weights = vgg_fn, weights
        
    def loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return feature_loss(
            self.vgg_fn(x), 
            self.vgg_fn(y), 
            weights     = self.weights, 
            feature     = self.feature, 
            reduction   = self.reduction,
            device      = self.device
        )
    
    def metric_fn(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        features = kwargs.get("features", [self.feature])
        return map_dict(
            lambda key, feature: feature_loss(
                self.vgg_fn(x), 
                self.vgg_fn(y), 
                weights     = self.weights, 
                feature     = feature, 
                reduction   = self.reduction,
                device      = self.device
            ),
            dict(zip(features, features))
        )

# ~~~~~~~~~~~~~~~~~~~~~~~ Gatys stochastic style loss ~~~~~~~~~~~~~~~~~~~~~~ #

class GatysStochasticLoss(GatysLoss):

    def __init__(
            self, 
            nstyle:      int,
            inchannels:  int,
            outchannels: int,      
            vgg_fn:      Optional[Callable]            = None,
            weights:     Union[Iterable, torch.Tensor] = LAYERS_WEIGHTS, 
            vgg_options: VGGOptions                    = None,                   
            feature:     str                           = "gram",
            nslices:     Union[Iterable, torch.Tensor] = NSLICES,
            reduction:   str                           = 'mean', 
            device:      Union[torch.device, str]      = 'cpu',
            **kwargs:    dict
    ):
        super().__init__(
            vgg_fn      = vgg_fn, 
            weights     = weights, 
            vgg_options = vgg_options,
            feature     = feature,
            reduction   = reduction,
            device      = device
        )
        self.inchannels = inchannels
        self.nslices = nslices
        ntriplets_max = inchannels * (inchannels - 1) * (inchannels - 2)
        self.nstyle = min(nstyle, ntriplets_max)
        self.random_projector = RandomProjector(
            inchannels  = inchannels, 
            outchannels = outchannels, 
            batch_size  = min(nstyle, kwargs.get("batch_size", nstyle)), 
            determinist = (nstyle >= ntriplets_max)
        )
        self.loss_dict = dict()  
    
    def loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape if x.ndim==4 else 1, *x.shape
        style_losses = []

        while sum([l.shape[1] for l in style_losses]) < self.nstyle:
            style_losses.append(
                feature_loss(
                    self.vgg_fn(self.random_projector(x).reshape(-1, 3, h, w)), 
                    self.vgg_fn(self.random_projector(y).reshape(-1, 3, h, w)), 
                    weights=self.weights, 
                    feature=self.feature,
                    nslices=self.nslices,
                    reduction='none',
                    device=self.device
                ).reshape(b, -1)
            )

            self.random_projector.generate()

        return torch.cat(style_losses, dim=1).mean(dim=-1)
    
    def metric_fn(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        b, c, h, w = x.shape if x.ndim==4 else 1, *x.shape
        features = kwargs.get("features", [self.feature])
        style_losses = dict(zip(features, [[] for _ in features]))

        # For logging/inferences purposes
        for feature in features:
            for band in range(self.inchannels) :
                self.loss_dict[f"{feature}_{band}"] = torch.Tensor().to(
                    self.device)

        while (sum([l.shape[1] for l in list(style_losses.values())[0]]) 
               < self.nstyle):
            fx = self.vgg_fn(self.random_projector(x).reshape(-1, 3, h, w))
            fy = self.vgg_fn(self.random_projector(y).reshape(-1, 3, h, w))
            for feature in features:
                style_loss = feature_loss(
                    fx, 
                    fy, 
                    weights=self.weights, 
                    feature=feature,
                    nslices=self.nslices,
                    reduction='none',
                    device=self.device
                ).reshape(b, -1)
                style_losses[feature].append(style_loss)

                # Add style loss to the concerned bands
                for bands in self.random_projector.channels :
                    for band in bands:
                        self.loss_dict[f"{feature}_{band}"] = torch.cat([
                            self.loss_dict[f"{feature}_{band}"],
                            style_loss.detach()
                        ])
                
            self.random_projector.generate()

        return map_dict(
            lambda key, value: torch.cat(value, dim=1).mean(dim=-1), style_losses)
    
class GatysStochasticLossAdvanced(GatysLoss):

    def __init__(
            self, 
            nstyle:      int,
            inchannels:  int,
            outchannels: int,      
            vgg_fn:      Optional[Callable]            = None,
            weights:     Union[Iterable, torch.Tensor] = LAYERS_WEIGHTS, 
            vgg_options: VGGOptions                    = None,                   
            feature:        str                        = "gram",
            reduction:   str                           = 'mean', 
            device:      Union[torch.device, str]      = 'cpu'
    ):
        super().__init__(
            vgg_fn      = vgg_fn, 
            weights     = weights, 
            vgg_options = vgg_options,
            feature        = feature,
            reduction   = reduction,
            device      = device
        )
        self.inchannels = inchannels
        ntriplets_max = inchannels * (inchannels - 1) * (inchannels - 2)
        self.nstyle = min(nstyle, ntriplets_max)
        self.triplets = list(itertools.permutations(
            range(self.inchannels), outchannels))
        self.loss_dict = dict()
        self.new_channels()

    def new_channels(self):
        self.channels = random.choices(self.triplets, k=self.nstyle)

    def loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        style_loss_stochastic = torch.zeros(
            x.shape[0] if x.ndim == 4 else 1, device=self.device)

        # For logging/inferences purposes
        for band in range(self.inchannels) :
            self.loss_dict[f"style_loss_band_{band}"] = torch.Tensor().to(
                self.device)

        for channels in self.channels:
            style_loss = feature_loss(
                self.vgg_fn(x[..., channels, :, :]), 
                self.vgg_fn(y[..., channels, :, :]), 
                weights=self.weights, 
                feature=self.feature,
                reduction='none',
                device=self.device
            )

            # Add style loss to the concerned bands
            for band in channels :
                self.loss_dict[f"style_loss_band_{band}"] = torch.cat([
                    self.loss_dict[f"style_loss_band_{band}"],
                    style_loss.detach().view(1)
                ])
                
            style_loss_stochastic += style_loss

        return style_loss_stochastic / self.nstyle
    
class GatysStochasticColorLoss(GatysLoss):

    def __init__(
            self, 
            nstyle:       int,
            inchannels:   int,
            outchannels:  int,      
            color_target: torch.Tensor,
            vgg_fn:       Optional[Callable]            = None,
            weights:      Union[Iterable, torch.Tensor] = LAYERS_WEIGHTS, 
            vgg_options:  VGGOptions                    = None,                   
            feature:         str                           = "gram",
            reduction:    str                           = 'mean', 
            device:       Union[torch.device, str]      = 'cpu'
    ):
        super().__init__(
            vgg_fn      = vgg_fn, 
            weights     = weights, 
            vgg_options = vgg_options,
            feature        = feature,
            reduction   = reduction,
            device      = device
        )
        self.inchannels = inchannels
        self.color_target = color_target
        ntriplets_max = inchannels * (inchannels - 1) * (inchannels - 2)
        self.nstyle = min(nstyle, ntriplets_max)
        self.random_projector = RandomProjector(
            inchannels, outchannels, determinist=(nstyle >= ntriplets_max))
        self.loss_dict = dict()
    
    def loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        style_loss_stochastic = torch.zeros(
            x.shape[0] if x.ndim == 4 else 1, device=self.device)

        # For logging/inferences purposes
        for band in range(self.inchannels) :
            self.loss_dict[f"style_loss_band_{band}"] = torch.Tensor().to(
                self.device)
        
        for _ in range(self.nstyle):
            color_transform, _ = transform_color_statistics(
                self.random_projector(y), self.color_target)
            style_loss = feature_loss(
                self.vgg_fn(color_transform(self.random_projector(x))), 
                self.vgg_fn(color_transform(self.random_projector(y))), 
                weights=self.weights, 
                feature=self.feature,
                reduction='none',
                device=self.device
            )

            # Add style loss to the concerned bands
            for band in self.random_projector.channels :
                self.loss_dict[f"style_loss_band_{band}"] = torch.cat([
                    self.loss_dict[f"style_loss_band_{band}"],
                    style_loss.detach().view(1)
                ])
                
            style_loss_stochastic += style_loss
            self.random_projector.generate()

        return style_loss_stochastic / self.nstyle