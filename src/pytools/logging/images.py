from functools import partial
import os
from typing import Callable, List, Optional, Tuple, Union
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision

from ..nn.projectors import MeanProjector
from ..utils.checks import assert_dim, assert_shape


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
        
    if format=="sentinel" :
        assert_shape(img, (None, 11, None, None))
        return {
            reformat("rgb", name) : img[1:4],
            reformat("nir", name) : img[[4, 6, 7]],
            reformat("swir", name) : img[[8, 9, 10]],
            reformat("global", name) : MeanProjector(11, 3)(img)
        }
    else :
        return {name : img[:3]}
    
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
    x = TF.center_crop(scale_tensor(x, xrange, outrange), resolution)
    ys = map(
        lambda img: TF.center_crop(
            scale_tensor(x, yrange, outrange), resolution), ys)
    cat = F.interpolate(
        torch.cat((x, *ys, x) if gt_right else (x, *ys), dim=3), 
        scale_factor=2, 
        mode='nearest'
    )

    return to_rgb(torchvision.utils.make_grid(cat, nrow=1), format=format)

def make_grid_plt(
        *imgs:    torch.Tensor,
        fn:       Callable[[torch.Tensor], torch.Tensor],
        savefig:  Optional[str] = None,
        nplots:     int           = 2
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
    x = torch.arange(y[0].shape[-1]).reshape(-1, 1) + 1
    y = map(fn, imgs)

    fig, axes = plt.subplots(
        nplots, nplots, figsize=(5 * nplots, 5 * nplots), dpi=100)
    if nplots==1 :
        axes.loglog(x.T[0], y[0][0], color='b', label=f'original')
        axes.legend()
    else :
        for i,ax in enumerate(axes.flatten()):
            for j in range(len(y)):
                ax.loglog(
                    x.T[0], y[j][i], color='b', label=f'original {i}')
                ax.loglog(
                    x.T[0], y[j][i], 
                    color='r', label=f'reconstructed {i}'
                )
                ax.legend()

    if savefig is not None :
        fig.savefig(savefig)