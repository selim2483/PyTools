from typing import Union
import torch
import torch.nn.functional as F

from ..utils.checks import type_check
from ..utils.misc import unsqueeze_squeeze
from .slice import SliceLoss, sliced_distance

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
    x_sorted, x_indices = torch.sort(x[..., 0])
    y_sorted, y_indices = torch.sort(y[..., 0])

    stretched_y_proj = F.interpolate(
        input                  = y_sorted.unsqueeze(1), 
        size                   = y_indices.shape[-1],
        mode                   = 'nearest', 
        recompute_scale_factor = False
    ) # handles a generated image larger than the ground truth 
    diff = (stretched_y_proj[:, 0] - x_sorted) #[inv(indices)]

    return .5 * torch.mean(diff**p, dim=-1)

def sliced_histogram_loss(
        x:torch.Tensor, y:torch.Tensor, p:int=2,
        nslice:Union[int, None]=None, 
        device:Union[torch.device, str]="cpu"
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
        histogram_loss1D, x, y, p=p, nslice=nslice, device=device)

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