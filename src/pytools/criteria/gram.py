from typing import Iterable, List, Union

import torch
import torch.nn as nn

from ..criteria.loss import reduce_loss
from ..nn.functionnal import compute_gram_matrix
from ..utils.checks import assert_shape, type_check


LAYERS         = [1, 6, 11, 20, 29]
LAYERS_WEIGHTS = [1/n**2 for n in [64,128,256,512,512]]

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
        f_real      :List[torch.Tensor], 
        f_hat       :List[torch.Tensor], 
        weights     :Union[Iterable, torch.Tensor] = LAYERS_WEIGHTS,
        center_gram :bool                          = True, 
        reduction   :str                           = 'mean'
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
        weights = torch.tensor(weights) / len(weights)
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
        for f_hat, f_real in zip(f_hat, f_real)
    ]
    return torch.stack(layer_losses, dim=1) @ weights

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
            reduction=self.reduction
        )