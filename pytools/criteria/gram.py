import itertools
import random
from typing import Callable, Iterable, List, Optional, Union

import torch
import torch.nn as nn

from ..criteria.loss import reduce_loss, Loss
from ..nn import RandomProjector, initialize_vgg
from ..nn.functionnal import compute_gram_matrix
from ..options import VGGOptions
from ..utils.checks import assert_shape, type_check
from ..utils.color import transform_color_statistics
from ..utils.misc import tensor2list, unsqueeze_squeeze


LAYERS         = [1, 6, 11, 20, 29]
LAYERS_WEIGHTS = [1/n**2 for n in [64,128,256,512,512]]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Gram matrices ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
@type_check
def gram_loss_mse_layer(
        x:torch.Tensor, x_hat:torch.Tensor, 
        center_gram:bool=True, reduction:str='mean'
) -> torch.Tensor:
    """Computes L2 distances between Gram matrices extracted from provided
    feature maps.

    Args:
        x (torch.Tensor): reference feature maps.
        x_hat (torch.Tensor): synthetic feature maps.
        center_gram (bool, optional): Whether to center feature maps to
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
    g = compute_gram_matrix(x, center_gram=center_gram)
    g_hat = compute_gram_matrix(x_hat, center_gram=center_gram)
    return reduce_loss(
        ((g - g_hat)**2).mean(dim=(-1,-2)), reduction=reduction)

def gram_loss_mse(
        F_real      :List[torch.Tensor], 
        F_hat       :List[torch.Tensor], 
        weights     :Union[Iterable, torch.Tensor] = LAYERS_WEIGHTS,
        center_gram :bool                          = True, 
        reduction   :str                           = 'mean',
        device      :Union[torch.device, str]      = 'cpu'
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
        center_gram (bool, optional): Whether to center feature maps to
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
        gram_loss_mse_layer(
            f_hat, 
            f_real.detach(), 
            center_gram=center_gram, 
            reduction=reduction
        ) 
        for f_hat, f_real in zip(F_hat, F_real)
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
        center_gram (bool, optional): Whether to center feature maps to
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
            weights     :Union[Iterable, torch.Tensor] = LAYERS_WEIGHTS,
            center_gram :bool                          = True,
            reduction   :str                           = 'mean', 
            device      :Union[torch.device, str]      = 'cpu'
    ):
        super().__init__()
        self.weights     = weights
        self.center_gram = center_gram
        self.device      = device
        self.reduction   = reduction

    def forward(
            self, 
            f_real:List[torch.Tensor], 
            f_hat:List[torch.Tensor]
    ) -> torch.Tensor :
        return gram_loss_mse(
            f_real, 
            f_hat, 
            self.weights, 
            center_gram=self.center_gram, 
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
            center_gram: bool                          = True,
            reduction:   str                           = 'mean', 
            device:      Union[torch.device, str]      = 'cpu'
    ):
        super().__init__(reduction, device)
        self.center_gram = center_gram

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
        return gram_loss_mse(
            self.vgg_fn(x), 
            self.vgg_fn(y), 
            weights=self.weights, 
            center_gram=self.center_gram, 
            reduction=self.reduction,
            device=self.device
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
            center_gram: bool                          = True,
            reduction:   str                           = 'mean', 
            device:      Union[torch.device, str]      = 'cpu'
    ):
        super().__init__(
            vgg_fn      = vgg_fn, 
            weights     = weights, 
            vgg_options = vgg_options,
            center_gram = center_gram,
            reduction   = reduction,
            device      = device
        )
        self.inchannels = inchannels
        ntriplets_max = inchannels * (inchannels - 1) * (inchannels - 2)
        self.ntriplets = min(nstyle, ntriplets_max)
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

        for _ in range(self.ntriplets):
            style_loss = gram_loss_mse(
                self.vgg_fn(self.random_projector(x)), 
                self.vgg_fn(self.random_projector(y)), 
                weights=self.weights, 
                center_gram=self.center_gram,
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

        return style_loss_stochastic / self.ntriplets
    
class GatysStochasticLossAdvanced(GatysLoss):

    def __init__(
            self, 
            nstyle:      int,
            inchannels:  int,
            outchannels: int,      
            vgg_fn:      Optional[Callable]            = None,
            weights:     Union[Iterable, torch.Tensor] = LAYERS_WEIGHTS, 
            vgg_options: VGGOptions                    = None,                   
            center_gram: bool                          = True,
            reduction:   str                           = 'mean', 
            device:      Union[torch.device, str]      = 'cpu'
    ):
        super().__init__(
            vgg_fn      = vgg_fn, 
            weights     = weights, 
            vgg_options = vgg_options,
            center_gram = center_gram,
            reduction   = reduction,
            device      = device
        )
        self.inchannels = inchannels
        ntriplets_max = inchannels * (inchannels - 1) * (inchannels - 2)
        self.ntriplets = min(nstyle, ntriplets_max)
        self.triplets = list(itertools.permutations(
            range(self.inchannels), outchannels))
        self.loss_dict = dict()
        self.new_channels()

    def new_channels(self):
        self.channels = random.choices(self.triplets, k=self.ntriplets)

    def loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        style_loss_stochastic = torch.zeros(
            x.shape[0] if x.ndim == 4 else 1, device=self.device)

        # For logging/inferences purposes
        for band in range(self.inchannels) :
            self.loss_dict[f"style_loss_band_{band}"] = torch.Tensor().to(
                self.device)

        for channels in self.channels:
            style_loss = gram_loss_mse(
                self.vgg_fn(x[..., channels, :, :]), 
                self.vgg_fn(y[..., channels, :, :]), 
                weights=self.weights, 
                center_gram=self.center_gram,
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

        return style_loss_stochastic / self.ntriplets
    
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
            center_gram:  bool                          = True,
            reduction:    str                           = 'mean', 
            device:       Union[torch.device, str]      = 'cpu'
    ):
        super().__init__(
            vgg_fn      = vgg_fn, 
            weights     = weights, 
            vgg_options = vgg_options,
            center_gram = center_gram,
            reduction   = reduction,
            device      = device
        )
        self.inchannels = inchannels
        self.color_target = color_target
        ntriplets_max = inchannels * (inchannels - 1) * (inchannels - 2)
        self.ntriplets = min(nstyle, ntriplets_max)
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

        for _ in range(self.ntriplets):
            color_transform, _ = transform_color_statistics(
                self.random_projector(x), self.color_target)
            style_loss = gram_loss_mse(
                self.vgg_fn(color_transform(self.random_projector(x))), 
                self.vgg_fn(color_transform(self.random_projector(y))), 
                weights=self.weights, 
                center_gram=self.center_gram,
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

        return style_loss_stochastic / self.ntriplets