from typing import Tuple, Union
import torch

from ..utils.misc import unsqueeze_squeeze

__all__ = ["periodic_smooth_decomposition"]

@unsqueeze_squeeze(ndim=3)
def periodic_smooth_decomposition(
        u:           torch.Tensor, 
        inverse_dft: bool                     = True, 
        smooth_comp: bool                     = False,
        device:      Union[torch.device, str] = "cpu"
) -> Tuple[torch.Tensor]:
    """Computes periodic + smooth decomposition from Moisan's paper
    (https://link.springer.com/article/10.1007/s10851-010-0227-1). When
    computing discrete Fourier transform, signals are assumed to be periodic.
    The periodic extension of the image then presents strong discontinuities
    since there are no reason for borders to be alike. This results in the
    presence of a cross in the Fourier spectrum of the image.
    This function decomposes the image in 2 composent : a periodic one and a
    smoothing one. 

    Args:
        u (torch.Tensor): image to process
        inverse_dft (bool, optional): whether to return the inverse Fourier
            transform of the image or not. 
            If True, the function returns the periodic component in the image 
            space.
            If False, the functuion returns the Periodic Fourier spectrum.
            Defaults to True.

    Returns:
        (torch.Tensor): image or Fourier spectrum periodic + smooth
            decomposition.
    """
    
    u = u.type(torch.complex128).to(device)
     
    arg = (2. * torch.tensor(torch.pi, device=device) 
           * torch.fft.fftfreq(u.shape[-2], 1., device=device))
    arg = arg.repeat(*u.shape[:-2], 1, 1).transpose(-2, -1)
    cos_h, sin_h = torch.cos(arg), torch.sin(arg)
    one_minus_exp_h = 1.0 - cos_h - 1j * sin_h

    arg = (2. * torch.tensor(torch.pi, device=device) 
           * torch.fft.fftfreq(u.shape[-1], 1., device=device))
    arg = arg.repeat(*u.shape[:-2], 1, 1)
    cos_w, sin_w = torch.cos(arg), torch.sin(arg)
    one_minus_exp_w = 1.0 - cos_w - 1j * sin_w

    w1 = u[..., -1] - u[..., 0]
    w1_dft = torch.fft.fft(w1).unsqueeze(-1)
    v_dft = w1_dft * one_minus_exp_w.to(device)

    w2 = u[..., -1, :] - u[..., 0, :]
    w2_dft = torch.fft.fft(w2).unsqueeze(-2)
    v_dft = v_dft + one_minus_exp_h.to(device) * w2_dft

    denom = 2.0 * (cos_h + cos_w - 2.0)
    denom[..., 0, 0] = 1.0

    s_dft = v_dft / denom
    s_dft[..., 0, 0] = 0.0

    if inverse_dft:
        s = torch.fft.ifft2(s_dft).real
        if smooth_comp:
            return u.real - s, s
        else:
            return u.real - s
    else:
        u_dft = torch.fft.fft2(u)
        if smooth_comp:
            return u_dft - s_dft, s_dft
        else:
            return u_dft - s_dft