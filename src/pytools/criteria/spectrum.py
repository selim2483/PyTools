from typing import Union
import torch

from ..data import periodic_smooth_decomposition
from ..utils.checks import type_check
from .slice import SliceLoss, sliced_function

@type_check
def spectrum_ortho_loss1D(
        x:torch.Tensor, y:torch.Tensor, 
        p:int=2, remove_cross:bool=True
) -> torch.Tensor:
    """Computes the Lp distance in the image space between y and the set of
    images having the same spectrum (modulus of the 2D Fourier transform) as 
    x.

    Args:
        x (torch.Tensor): reference image
        y (torch.Tensor): synthetic image.
        p (int, optional): distance order.
        remove_cross (bool, optional): whether to remove the Fourier cross or
            not. 
            Defaults to True.
    Returns:
        torch.Tensor: spectral loss in image space
    """
    if remove_cross :
        x_fft, _ = periodic_smooth_decomposition(x, inverse_dft=False)
        x_fft = torch.fft.fftshift(x_fft, dim=(-2,-1)) 
        y_fft, _ = periodic_smooth_decomposition(y, inverse_dft=False)
        y_fft = torch.fft.fftshift(y_fft, dim=(-2,-1)) 
    else :
        x_fft = torch.fft.fft2(x)
        y_fft = torch.fft.fft2(y)

    # create an grey image with the phase from rec and the module from
    # the original image
    f_proj = y_fft / (y_fft.abs() + 1e-8) * x_fft.abs()
    proj = torch.fft.ifft2(f_proj).real.detach()
    return ((y - proj) ** p).mean(dim=(-1, -2))

def sliced_spectrum_ortho_loss(
        x            :torch.Tensor, 
        y            :torch.Tensor, 
        p            :int                      = 2, 
        remove_cross :bool                     = True,
        nslice       :Union[int, None]         = None, 
        band         :Union[int, None]         = None, 
        device       :Union[torch.device, str] = "cpu"
) -> torch.Tensor:
    fn = sliced_function(
        spectrum_ortho_loss1D, nslice=nslice, band=band, device=device)
    return fn(x, y, p=p, remove_cross=remove_cross)

class SpectralOrthoLoss(SliceLoss):
    """Orthogonal spectral loss module.
    Computes sliced spectral distance in the image space :
        - Projects images in a random direction in the color space.
        - Then, computes spectral Lp distance in the image space between a
        synthetic image and the set of images that presents the same Fourier's
        modulus as a reference image.

    Args:
        reduction (str, optional): reduction method to perform. 
            Should be 'none', 'mean' or 'sum'.
            Defaults to 'mean'.
        device (Union[torch.device, str], optional): device on which to place
            the tensors. 
            Defaults to "cpu".
    """
    def loss_fun(
            self, 
            x:torch.Tensor, y:torch.Tensor, 
            p:int=2, remove_cross:bool=True
    ) -> torch.Tensor:
        return spectrum_ortho_loss1D(x, y, p=p, remove_cross=remove_cross)

