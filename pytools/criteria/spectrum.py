from typing import Union
import torch

from ..data import periodic_smooth_decomposition
from ..utils.checks import assert_shape, type_check 
from ..utils.misc import unsqueeze_squeeze
from .slice import SliceLoss, sliced_distance

# ------------------------ Spectral orthogonal loss ------------------------ #

@type_check
def spectrum_ortho_loss1D(
        x: torch.Tensor, y: torch.Tensor, 
        p: int=2, remove_cross: bool=True, 
        device: Union[torch.device, str] = "cpu"
) -> torch.Tensor:
    r"""Computes the :math:`L_p` distance in the image space between a
    synthetic image :math:`y` and the set of images having the same spectrum
    (modulus of the 2D Fourier transform) as a reference image :math:`x` :

    .. math::
        \mathcal{L}_{spe} (x,y)
        =
        \left\|
        y - \mathcal{F}^{-1} 
        \left(
        \mathcal{F}(y) 
        \times 
        \left\|\frac{\mathcal{F}(x)}{\mathcal{F}(y)}\right\|
        \right)
        \right\|_2^2

    Periodic+smooth decomposition can be performed on images before the
    distance computation by the mean of the :attr:`remove_cross` argument.

    The :math:`L_p` space in which the distance is computed can be controled
    by the mean of the :attr:`p` argument.

    Args:
        x (torch.Tensor): reference image.
        y (torch.Tensor): synthetic image.
        p (int, optional): distance order.
        remove_cross (bool, optional): whether to remove the Fourier cross or
            not. 
            Defaults to True.

    Returns:
        torch.Tensor: spectral distance in image space
    """
    if remove_cross :
        x_fft = periodic_smooth_decomposition(
            x, inverse_dft=False, device=device)
        x_fft = torch.fft.fftshift(x_fft, dim=(-2,-1)) 
        y_fft = periodic_smooth_decomposition(
            y, inverse_dft=False, device=device)
        y_fft = torch.fft.fftshift(y_fft, dim=(-2,-1)) 
    else :
        x_fft = torch.fft.fft2(x.to(device))
        y_fft = torch.fft.fft2(y.to(device))

    # create an grey image with the phase from rec and the module from
    # the original image
    f_proj = y_fft / (y_fft.abs() + 1e-8) * x_fft.abs()
    proj = torch.fft.ifft2(f_proj).real.detach().to(device)
    return ((y - proj) ** p).mean(dim=(-1, -2))

def sliced_spectrum_ortho_loss(
        x            :torch.Tensor, 
        y            :torch.Tensor, 
        p            :int                      = 2, 
        remove_cross :bool                     = True,
        nslice       :Union[int, None]         = None, 
        device       :Union[torch.device, str] = "cpu",
        **kwargs
) -> torch.Tensor:
    """Computes sliced spectral distance in the image space :

    - Projects images in a random direction in the color space (if nslice is
      int), or on a chosen band (if nslice is not int and band is int).
    - Then, computes spectral Lp distance in the image space between a
      synthetic image and the set of images that presents the same Fourier's
      modulus as a reference image.

    Args:
        x (torch.Tensor): reference image.
        y (torch.Tensor): synthetic image.
        p (int, optional): distance order.
        remove_cross (bool, optional): whether to remove the Fourier cross or
            not. 
            Defaults to True.
        nslice (Union[int, None], optional): Number of random slice to
            perform.
            If ``int``, computes random sliced distance.
            Defaults to None.
        device (Union[torch.device, str], optional): device on which to place
            the tensors. 
            Defaults to "cpu".

    Returns:
        torch.Tensor: sliced spectral distance in image space.
    """
    return sliced_distance(
        spectrum_ortho_loss1D, 
        x, y, p=p, remove_cross=remove_cross, 
        nslice=nslice, device=device, 
        **kwargs
    )

class SpectralOrthoLoss(SliceLoss):
    """Orthogonal spectral loss module.

    Computes sliced spectral distance in the image space :
    
    - Projects images in a random direction in the color space.
    - Then, computes spectral Lp distance in the image space between a
      synthetic image and the set of images that presents the same Fourier's
      modulus as a reference image.

    Args:
        reduction (str, optional): reduction method to perform. 
            Should be none, mean or sum.
            Defaults to mean.
        device (Union[torch.device, str], optional): device on which to place
            the tensors. 
            Defaults to "cpu".
    """
    def loss_fun(
            self, 
            x:torch.Tensor, y:torch.Tensor, 
            p:int=2, remove_cross:bool=True
    ) -> torch.Tensor:
        return spectrum_ortho_loss1D(
            x, y, p=p, remove_cross=remove_cross, device=self.device)

# -------------------------- Basic spectral loss --------------------------- #

@unsqueeze_squeeze(ntensors=2)
def spectral_loss1D(x:torch.Tensor, y:torch.Tensor, p:int=2) -> torch.Tensor:
    """Computes Lp distance between the power spectrum densities (PSD) of two
    grey scale images.

    Args:
        x (torch.Tensor): first image.
        y (torch.Tensor): second image.
        p (int, optional): order of Lp distance. 
            Defaults to 2.

    Returns:
        torch.Tensor: PSD Lp distance.
    """
    b, h, w = x.shape
    assert_shape(y, (b, h, w))

    fpx = periodic_smooth_decomposition(x, inverse_dft=False)
    fpy = periodic_smooth_decomposition(y, inverse_dft=False)
    fpx, fpy = fpx / (h * w), fpy / (h * w)
    
    return torch.sqrt(torch.mean(
        (20 * torch.log(fpx.abs()) - 20 * torch.log(fpy.abs()))**p))

def sliced_spectral_loss(
        x            :torch.Tensor, 
        y            :torch.Tensor, 
        p            :int                      = 2, 
        nslice       :Union[int, None]         = None, 
        device       :Union[torch.device, str] = "cpu",
        **kwargs
) -> torch.Tensor:
    """Computes sliced PSD distance :

    - Projects images in a random direction in the color space (if nslice is
      int), or on a chosen band (if nslice is not int and band is int).
    - Then, computes spectral between the power spectrum densities (PSD) of
      the two so-obtained grey scale images.

    Args:
        x (torch.Tensor): reference image.
        y (torch.Tensor): synthetic image.
        p (int, optional): distance order.
        nslice (Union[int, None], optional): Number of random slice to
            perform.
            If ``int``, computes random sliced distance.
            Defaults to None.
        device (Union[torch.device, str], optional): device on which to place
            the tensors. 
            Defaults to "cpu".

    Returns:
        torch.Tensor: sliced PSD distance.
    """
    return sliced_distance(
        spectral_loss1D, x, y, p=p, nslice=nslice, device=device, **kwargs)

class SpectralLoss(SliceLoss):
    """PSD loss module.

    Computes sliced PSD distance :

    - Projects images in a random direction in the color space.
    - Then, computes Lp distance between the power spectrum densities (PSD) of
      the two so-obtained grey scale images.

    Args:
        reduction (str, optional): reduction method to perform. 
            Should be none, mean or sum.
            Defaults to mean.
        device (Union[torch.device, str], optional): device on which to place
            the tensors. 
            Defaults to "cpu".
    """
    def loss_fn(
            self, x:torch.Tensor, y:torch.Tensor, p:int=2) -> torch.Tensor:
        return spectral_loss1D(x, y, p=p)

# -------------------------- Radial spectral loss -------------------------- #

@unsqueeze_squeeze(ndim=4)
def radial_profile(
        img      :torch.Tensor, 
        bin_size :int                      = 1, 
        device   :Union[torch.device, str] = 'cpu'
) -> torch.Tensor:
    """Compute radial profile of the Fourier's transform modulus of a grey
    image.
    Usable for batches of images.

    Args:
        img (torch.Tensor): grey image
        bin_size (int, optional): size of bins used to compute histograms.
            Defaults to 1.
        device (Union[torch.device, str], optional): device on which to place
            the tensors. 
            Defaults to "cpu".

    Returns:
        torch.Tensor: Radial profile(s)
    """
    b, c, h, w = img.shape
    img = img.reshape(b * c, h, w)
    fp = periodic_smooth_decomposition(img, inverse_dft=False)
    
    fpshift = torch.fft.fftshift(fp, dim=(-2,-1)) / (h*w)
    mod = fpshift.abs().view(b * c, -1)**2

    xx, yy = torch.meshgrid(
        torch.arange(fpshift.shape[-2]), 
        torch.arange(fpshift.shape[-1]), 
        indexing='ij'
    )
    r = torch.sqrt((xx - h // 2)**2 + (yy - w // 2)**2)

    # make crowns over which compute the histogram, larger crowns (bin_size>1)
    # should get faster computations
    crowns = (r / bin_size).type(torch.int64).repeat(
        (b * c, 1)).view(b * c, -1).to(device)

    # compute histogram
    values_sum = torch.zeros(
        b * c, int(crowns.max() + 1), dtype=mod.dtype, device=device)
    values_sum = values_sum.scatter_add_(1, crowns, mod)
    crowns_sum = torch.zeros(
        b * c, int(crowns.max() + 1), dtype=mod.dtype, device=device)
    crowns_sum = crowns_sum.scatter_add_(
        1, crowns, torch.ones(mod.shape, dtype=mod.dtype, device=device))
    rad = values_sum / crowns_sum

    return rad.view(b, c, -1)

@unsqueeze_squeeze(ndim=3, ntensors=2)
def radial_spectral_loss1D(
        x:torch.Tensor, y:torch.Tensor, p:int=2) -> torch.Tensor:
    """Computes Lp distance between the azimuted power spectrum densities
    (PSD) of two grey scale images.

    Args:
        x (torch.Tensor): first image.
        y (torch.Tensor): second image.
        p (int, optional): order of Lp distance. 
            Defaults to 2.

    Returns:
        torch.Tensor: radial PSD Lp distance.
    """
    radx = radial_profile(x) 
    rady = radial_profile(y)   
    return torch.sqrt(torch.mean(
        (10 * torch.log(radx) - 10 * torch.log(rady))**p, dim=-1))

def sliced_radial_spectral_loss(
        x            :torch.Tensor, 
        y            :torch.Tensor, 
        p            :int                      = 2, 
        nslice       :Union[int, None]         = None, 
        device       :Union[torch.device, str] = "cpu",
        **kwargs
) -> torch.Tensor:
    """Computes sliced radial PSD distance :

    - Projects images in a random direction in the color space (if nslice is
      int), or on a chosen band (if nslice is not int and band is int).
    - Then, computes Lp distance between the azimuted power spectrum densities
      (PSD) of the two so-obtained grey scale images.

    Args:
        x (torch.Tensor): reference image.
        y (torch.Tensor): synthetic image.
        p (int, optional): distance order.
        nslice (Union[int, None], optional): Number of random slice to
            perform.
            If ``int``, computes random sliced distance.
            Defaults to None.
        device (Union[torch.device, str], optional): device on which to place
            the tensors. 
            Defaults to "cpu".

    Returns:
        torch.Tensor: sliced radial PSD distance.
    """
    return sliced_distance(
        radial_spectral_loss1D, 
        x, y, p=p, nslice=nslice, device=device,
        **kwargs
    )

class RadialSpectralLoss(SliceLoss):
    """Radial PSD loss module.

    Computes sliced PSD distance :

    - Projects images in a random direction in the color space.
    - Then, computes Lp distance between the azimuted power spectrum
      densities (PSD) of the two so-obtained grey scale images.

    Args:
        reduction (str, optional): reduction method to perform. 
            Should be none, mean or sum.
            Defaults to mean.
        device (Union[torch.device, str], optional): device on which to place
            the tensors. 
            Defaults to "cpu".
    """
    def loss_fn(
            self, x:torch.Tensor, y:torch.Tensor, p:int=2) -> torch.Tensor:
        return radial_spectral_loss1D(x, y, p=p)