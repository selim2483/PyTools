from typing import Union
import torch

from .slice import sliced
from ..utils.checks import assert_dim, type_check

@type_check
def corr_length(im: torch.Tensor, device: Union[torch.device, str]="cpu") -> torch.Tensor:
    """Computes correlation length using second order polynomial
    approximation over a batch grey scale images.

    Args:
        im (torch.Tensor): batch of grey scale images.
        device (Union[torch.device, str], optional): device to place tensors on. Defaults to "cpu".

    Returns:
        torch.Tensor: correlation length of images.
    """
    assert_dim(min_ndim=3)
    im = im.to(device)
    x = ((im - im.mean([-2, -1], keepdim=True)) 
         / im.std([-2, -1], keepdim=True))
    
    c1 = (x[...,1:]*x[...,:-1]).mean([-2, -1])
    c2 = (x[...,2:]*x[...,:-2]).mean([-2, -1])
    D2x = c2 - 2 * c1 + 1

    c1 = (x[...,1:,:] * x[...,:-1,:]).mean([-2, -1])
    c2 = (x[...,2:,:] * x[...,:-2,:]).mean([-2, -1])
    D2y = c2 - 2 * c1 + 1

    out = torch.zeros(x.shape[0], device=device)
    out[(D2x + D2y) < 0] = torch.sqrt(-2 / (D2x + D2y))[(D2x + D2y) < 0]
    return out

def sliced_corr_length(
        im:     torch.Tensor, 
        nslice: int, 
        device: Union[torch.device, str]="cpu"
):
    return sliced(corr_length, im, nargs=1, nslice=nslice, device=device)