from typing import Union

import torch
from torch.types import _size, _int

from ..utils.checks import type_check, assert_dim
from ..utils.misc import unsqueeze_squeeze

__all__ = [
    "normalize", "instance_norm2d", "adain_map", "compute_gram_matrix"
]

EPSILON = 1e-8

@type_check
def normalize(x:torch.Tensor, dim:Union[_size, _int, None]=-1):
    """Performs normalization across the wanted dimensions.

    Args:
        x (torch.Tensor): tensor to normalize.
        dim (Union[_size, _int, None], optional): dimensions in which
            normalization is performed. 
            Defaults to -1.

    Returns:
        (torch.Tensor): normalized tensor.
    """
    m, s = x.mean(dim=dim, keepdim=True), x.std(dim=dim, keepdim=True)
    return (x - m) / (s + EPSILON)

@type_check
def instance_norm2d(x:torch.Tensor):
    assert_dim(x, min_ndim=3)
    return normalize(x, dim=(-2,-1))

@type_check
def adain_map(
    x     :torch.Tensor, 
    mu    :torch.Tensor, 
    sigma :torch.Tensor, 
    width :_int=4
    ) -> torch.Tensor:
    """Performs the forward pass with mu and sigma being a 2D map instead of
    scalar.
    Each location is normalized by local statistics, and denormalized by `mu`
    and `sigma` evaluated at the given location.
    The local statistics are computed with a gaussian kernel of relative 
    `width` controlled by the variable of the same name
    """
    assert_dim(x, ndim=4)
    _, C, H0, W0 = x.shape
    # The pooling operation is used to make the computation of local mu and
    # sigma less expensive using the spacial smoothness of locally computed
    # statistics
    pool = torch.nn.AdaptiveAvgPool2d((min(H0, 128), min(W0, 128))) 
    x_pooled = pool(x) # x_pooled of maximum spacial size 128*128

    # Create the gaussian kernel for loacl stats computation
    B, C, H, W = x_pooled.shape
    rx, ry = H0 / H, W0 / W
    _width = width / min(rx,ry)
    kernel_size = [
        (max(int(2 * _width), 5) // 2) * 2 + 1, 
        (max(int(2 * _width), 5) // 2) * 2 + 1
    ] 
    _width = [_width, _width]
    kernel = 1
    mgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in kernel_size])
    
    for size, std, mgrid in zip(kernel_size, _width, mgrids):
        mean = (size - 1) / 2
        kernel *= torch.exp(-((mgrid - mean) / std) ** 2 / 2)
    kernel = kernel / torch.sum(kernel)
    kernel_1d = kernel.view(1, 1, *kernel.size())
    kernel = kernel_1d.repeat(C, *[1] * (kernel_1d.dim() - 1)).cuda()

    # create a weight map by convolution of a constant map with the 
    # gaussian kernel. It used to correctly compute the local statistics 
    # at the border of the image, accounting for zero padding
    ones = torch.ones(1,1,H,W).cuda()
    weight = torch.nn.functional.conv2d(
        ones,kernel_1d.cuda(),bias=None,padding='same')

    # define channel-wise gaussian convolution module conv
    conv = torch.nn.Conv2d(
        C, 
        C, 
        kernel_size, 
        groups=C, 
        bias=False, 
        stride=1, 
        padding=int((kernel_size[0] - 1) / 2), 
        padding_mode='zeros'
    )
    conv.weight.data = kernel
    conv.weight.requires_grad = False
    
    # pooling already performs local averaging, so it does not perturb the 
    # computation of the local mean
    local_mu = conv(x_pooled)/weight 

    # upsampling the local mean map to the original shape
    local_mu = torch.nn.functional.interpolate(
        local_mu, size=(H0,W0), mode='bilinear', align_corners=False) 
    
    # perform (x-local_mu)**2 at the high resolution, THEN pool and finally
    # smooth to get the local standard deviation.
    local_sigma = torch.sqrt(
        conv(pool(((x-local_mu)**2)) /weight) + 10 ** -8) 
    
    # upsampling the local std map to the original shape
    local_sigma = torch.nn.functional.interpolate(
        local_sigma, size=(H0,W0), mode='bilinear', align_corners=False) 

    # finally perform the local AdaIN operation using these maps of local_mu
    # and local_sigma to normalize, then denormalize with the given maps mu
    # and sigma.
    x_norm = (x - local_mu) / local_sigma
    return sigma * x_norm + mu

@type_check
@unsqueeze_squeeze()
def compute_gram_matrix(x:torch.Tensor, center_gram:bool=True) :
    """Computes gram matrix of a 3D or 4D tensor.

    Args:
        x (torch.Tensor): feature maps
        center_gram (bool, optional): Whether to center feature maps to
            computre Gram matrices or not. 
            Defaults to True.

    Returns:
        _type_: Gram matrix
    """
    F = x.flatten(start_dim=-2, end_dim=-1)
    if center_gram: 
        F = F - F.mean(dim=-1, keepdim=True)
    G = torch.bmm(F, F.transpose(-1, -2))
    return G.div_(F.shape[-1]) 