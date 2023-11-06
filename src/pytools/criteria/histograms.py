from typing import Union
import torch
import torch.nn.functional as F

from ..utils.checks import type_check
from ..utils.misc import unsqueeze_squeeze
from .slice import SliceLoss, stochastic_slice, band_slice

@type_check
@unsqueeze_squeeze(ndim=3, ntensors=2)
def histogram_loss1D(x:torch.Tensor, y:torch.Tensor):
    """Computes L2 loss between sorted histograms of two images.

    Args:
        x (torch.Tensor): first image
        y (torch.Tensor): second image

    Returns:
        torch.Tensor: histogram distances.
    """
    x_sorted, x_indices = torch.sort(x[..., 0])
    y_sorted, y_indices = torch.sort(y[..., 0])

    stretched_y_proj = F.interpolate(
        input                  = y_sorted.unsqueeze(1), 
        size                   = y_indices.shape[-1],
        mode                   = 'nearest', 
        recompute_scale_factor = False
    ) # handles a generated image larger than the ground truth 
    diff = (stretched_y_proj[:, 0] - x_sorted) #[inv(indices)]

    return .5 * torch.mean(diff**2, dim=-1)

def sliced_histogram_loss(
        x:torch.Tensor, y:torch.Tensor, 
        nslice:Union[int, None]=None, band:Union[int, None]=None, 
        device:Union[torch.device, str]="cpu"
):
    """Compute sliced histogram distance between two tensors.
    either random sliced distance on tensors (if nslice is int) or distance
    over a chosen band (if nslice is not int and band is int).

    Args:
        x (torch.Tensor): first image
        y (torch.Tensor): second image
        nslice (Union[int, None], optional): _description_. Defaults to None.
         nslice (Union[int, None], optional): Number of random slice to
            perform.
            If ``int``, computes random sliced distance.
            Defaults to None.
        band (Union[int, None], optional): band index on which to compute the
            distance.
            If ``int``, computes distance on  chosen band.
            Defaults to None.

    Returns:
        _type_: _description_
    """
    if isinstance(nslice, int):
        fn = stochastic_slice(
            histogram_loss1D, nslice=nslice, device=device)
    elif isinstance(band, int):
        fn = band_slice(histogram_loss1D, band=band, device=device)

    return fn(x, y)

class HistogramLoss(SliceLoss):
    """Histogram Loss module.
    Compute sliced histogram distance between two images, using the L2
    distance between sorted histograms of the two images.

    Args:
        reduction (str, optional): reduction method to perform. 
            Should be 'none', 'mean' or 'sum'.
            Defaults to 'mean'.
        device (Union[torch.device, str], optional): device on which to place
            the tensors. 
            Defaults to "cpu".
    """
    def loss_fn(self, x:torch.Tensor, y:torch.Tensor):
        return histogram_loss1D(x, y)