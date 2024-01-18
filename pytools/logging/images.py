from functools import partial
from math import ceil, sqrt
import os
from typing import Callable, Iterable, List, Optional, Tuple, Union
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision

from ..utils.checks import assert_dim, assert_shape
from ..utils.misc import unsqueeze_squeeze

@unsqueeze_squeeze()
def mean_project(x: torch.Tensor, nc_out: int) :
    nc_in = x.shape[1]
    kernel_size = ceil(nc_in / nc_out)
    if nc_in==1 :
        return x.repeat((1, 3, 1, 1))
    else :
        n, c, w, h = x.size()
        x = x.reshape(n, c, h * w).permute(0, 2, 1)
        pooled = F.avg_pool1d(x, kernel_size=kernel_size, ceil_mode=True)
        assert_shape(pooled, (None, None, nc_out))
        return pooled.permute(0, 2, 1).view(n, nc_out, w, h)

def reformat(mode:str, name:Optional[str]=None) -> str:
    if name is None :
        return mode
    else:
        return f"{name}_{mode}"
        
def to_rgb(
        img:torch.Tensor, 
        name:Optional[str]=None, 
        format:str="rgb"
) -> dict[str, torch.Tensor]:
    """Converts multipectral image to a bunch of rgb images.
    The chosen groups of spectral bands depend on the selected format.

    Args:
        img (torch.Tensor): Original image
        name (str, optional): Name to give to the images. 
            Defaults to "images".
        format (str, optional): format used to select the groups of band
            spectral.
            Defaults to "rgb".

    Returns:
        dict[str, torch.Tensor]: dict of RGB-like images (tensors).
    """
    assert_dim(img, ndim=3)
    
    if format=="sentinel2" :
        assert_shape(img, (11, None, None))
        return {
            reformat("rgb", name) : img[1:4],
            reformat("nir", name) : img[[4, 6, 7]],
            reformat("swir", name) : img[[8, 9, 10]],
            reformat("global", name) : mean_project(img, 3)
        }
    else :
        return {reformat(format, name) : img[:3]}
    
def scale_tensor(
        x:         torch.Tensor, 
        in_range:  Tuple[float, float], 
        out_range: Tuple[float, float]
) -> torch.Tensor :
    """Scales tensor from a given range ``in_range`` to the wanted one
    ``out_range``

    Args:
        x (torch.Tensor): tensor to scale.
        in_range (Tuple[float, float]): input range.
        out_range (Tuple[float, float]): output range.

    Returns:
        torch.Tensor: scaled tensor
    """
    a, b = in_range
    c, d = out_range
    return (torch.clamp(x, a, b) - (a + b)/2) * (d - c) / (b - a) + (c + d)/2

def norm_tensor(
        x:torch.Tensor, 
        out_range:Tuple[float, float]=(0., 1.)
) -> torch.Tensor :
    """Normalizes tensor using its extreme values.

    Args:
        x (torch.Tensor): tensor to normalize
        out_range (Tuple[float, float], optional): output range.
            Defaults to (0., 1.).

    Returns:
        torch.Tensor: normalized tensor.
    """
    return scale_tensor(x, in_range=(x.max(), x.min()), out_range=out_range)

def log_images(
        images_dict: dict[str, torch.Tensor], 
        logdir:      str, 
        name:        str,
        idx:         Optional[Union[int, str]] = None
):
    """Log images from a dictionnary of tensors.

    Args:
        images_dict (dict[str, torch.Tensor]): dictonnary containing the
            images.
        logdir (str): path or the directory where to log the images.
        name (str): name to give to the images.
        idx (Optional[Union[int, str]], optional): index of the images.
            Defaults to None.
    """
    for key, img in images_dict.items() :
        if idx is not None :
            img_name = f"{name}_{key}_{idx}.png"
        else :
            img_name = f"{name}_{key}.png"
        torchvision.utils.save_image(img, os.path.join(logdir, img_name))

def make_grid(
        x:          torch.Tensor, 
        *ys:        torch.Tensor,
        fn:         Callable[[torch.Tensor], torch.Tensor] = lambda img: img, 
        xrange:     Tuple[float, float]                  = (0, 255),
        yrange:     Tuple[float, float]                  = (-1., 1.),
        outrange:   Tuple[float, float]                  = (0., 1.),
        format:     str                                  = "rgb", 
        name:       str                                  = None,
        gt_right:   bool                                 = False,
        resolution: int                                  = 256
) -> dict[str, torch.Tensor]:  
    """Create a dictionnary of reconstruction grids given original and
    synthetic tensors ``x`` and ``ys``.

    Args:
        x (torch.Tensor): original tensors
        ys (torch.Tensor): synthetic tensors.
        fn (Callable[torch.Tensor, torch.Tensor]): additional operation to
            perform on images.
        xrange (Tuple[float, float], optional): original tensors range. 
            Defaults to (0, 255).
        yrange (Tuple[float, float], optional): synthetic tensors range.
            Defaults to (-1., 1.).
        outrange (Tuple[float, float], optional): desired output range.
            Defaults to (0., 1.).
        format (str, optional): format used to select the groups of band
            spectral.
            Defaults to "rgb".
        gt_right (bool, optional): whether to place the original images both
            right and left (``True``) or only left (``False``).
            Defaults to False.
        resolution (int, optional): image resolution.
            Defaults to 256.

    Returns:
        dict[str, torch.Tensor]: dict of RGB-like images (tensors)
    """
    x = fn(TF.center_crop(scale_tensor(x, xrange, outrange), resolution))
    _fn = lambda img: fn(TF.center_crop(
            scale_tensor(img, yrange, outrange), resolution))
    ys = map(_fn, ys)
    cat = F.interpolate(
        torch.cat((x, *ys, x) if gt_right else (x, *ys), dim=3), 
        scale_factor=2, 
        mode='nearest'
    )

    return to_rgb(
        torchvision.utils.make_grid(cat, nrow=1), name=name, format=format)

_iter_tensor = Union[Iterable[torch.Tensor], torch.Tensor]

def log_grid_plt(
        y1:       _iter_tensor,
        y2:       Optional[_iter_tensor] = None,
        label1:   str                    = "original",
        label2:   str                    = "reconstructed",
        savefig:  Optional[str]          = None
):
    """Makes a grid of plt plots of time series obtained by applying a
    function on images.

    Args:
        fn (Callable[[torch.Tensor], torch.Tensor]): Function to apply on
            images to obtain a time serie.
        savefig (Optional[str], optional): path where to save the figure.
            Defaults to None.
        nimg (int, optional): number of plots. 
            Defaults to 2.
    """
    x = torch.arange(y1[0].shape[-1]).reshape(-1, 1) + 1
    y2 = [None for _ in range(len(y1))] if y2 is None else y2

    nplots = ceil(sqrt(len(y1)))
    fig, axes = plt.subplots(
        nplots, nplots, figsize=(5 * nplots, 5 * nplots), dpi=100)
    for i, (_y1, _y2) in enumerate(zip(y1, y2)):
        axes[i].loglog(x.T[0], y1[i], color='b', label=f'{label1} {i}')
        if y2 is not None:
            axes[i].loglog(x.T[0], y2[i], color='r', label=f'{label2} {i}')
        axes[i].legend()

    if savefig is not None :
        fig.savefig(savefig)