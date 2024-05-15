from typing import Union
import torch
import torch.nn.functional as F

from ..utils.checks import type_check
from ..utils.misc import unsqueeze_squeeze
from .slice import SliceLoss, sliced_distance

@unsqueeze_squeeze(ndim=3)
def interpolate_histogram(x_sorted: torch.Tensor, y_indices: torch.Tensor):
    return F.interpolate(
        input                  = x_sorted, 
        size                   = y_indices.shape[-1],
        mode                   = 'nearest', 
        recompute_scale_factor = False
    )

@type_check
@unsqueeze_squeeze(ndim=3, ntensors=2)
def histogram_loss1D(x:torch.Tensor, y:torch.Tensor, p:int=2) -> torch.Tensor:
    """Computes Lp distance between sorted histograms of two images.

    Args:
        x (torch.Tensor): first image.
        y (torch.Tensor): second image.
        p (int, optional): distance order.

    Returns:
        torch.Tensor: histogram distances.
    """
    x_sorted, x_indices = torch.sort(x.flatten(start_dim=-2))
    y_sorted, y_indices = torch.sort(y.flatten(start_dim=-2))

    if x_indices.shape[-1] > y_indices.shape[-1]:
        y_sorted = interpolate_histogram(y_sorted, x_indices)
    elif x_indices.shape[-1] < y_indices.shape[-1]:
        x_sorted = interpolate_histogram(x_sorted, y_indices)

    return .5 * torch.mean((y_sorted - x_sorted)**p, dim=-1)

def sliced_histogram_loss(
        x:torch.Tensor, y:torch.Tensor, p:int=2,
        nslice:Union[int, None]=None, 
        device:Union[torch.device, str]="cpu",
        **kwargs
) -> torch.Tensor:
    """Computes sliced histogram distance between two images :

    - Projects images in a random direction in the color space (if nslice is
      int), or on a chosen band (if nslice is not int and band is int).
    - Then, computes the Lp distance between sorted histograms of the two
      images.

    Args:
        x (torch.Tensor): first image
        y (torch.Tensor): second image
        p (int, optional): distance order.
        nslice (Union[int, None], optional): Number of random slice to
            perform.
            If ``int``, computes random sliced distance.
            Defaults to None.
        device (Union[torch.device, str], optional): device on which to place
            the tensors. 
            Defaults to "cpu".

    Returns:
        torch.Tensor: sliced histogram distances.
    """
    return sliced_distance(
        histogram_loss1D, x, y, p=p, nslice=nslice, device=device, **kwargs)

class HistogramLoss(SliceLoss):
    """Histogram Loss module.

    Computes sliced histogram distance between two images :
        - Projects images in a random direction in the color space.
        - Then, computes the Lp distance between sorted histograms of the two
        images.

    Args:
        reduction (str, optional): reduction method to perform. 
            Should be 'none', 'mean' or 'sum'.
            Defaults to 'mean'.
        device (Union[torch.device, str], optional): device on which to place
            the tensors. 
            Defaults to "cpu".
    """
    def loss_fn(
            self, x:torch.Tensor, y:torch.Tensor, p:int=2) -> torch.Tensor:
        return histogram_loss1D(x, y, p=p)