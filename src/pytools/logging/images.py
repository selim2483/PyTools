import torch

from ..nn.projectors import MeanProjector
from ..utils.checks import assert_dim

def to_rgb(img:torch.Tensor, name:str="images", format:str="rgb"):
    """Converts multipectral image to a bunch of rgb images.
    The chosen groups of spectral bands depend on the selected format.

    Args:
        img (torch.Tensor): Original image
        name (str, optional): Name to give to the images. 
            Defaults to "images".
        format (str, optional): format used to select the groups of band spectral.
            Defaults to "rgb".

    Returns:
        _type_: _description_
    """
    assert_dim(img, ndim=3)
    if format=="sentinel" :
        return {
            f"{name}_rgb" : img[1:4],
            f"{name}_nir" : img[[4, 6, 7]],
            f"{name}_swir" : img[[8, 9, 10]],
            f"{name}_global" : MeanProjector(11, 3)(img)
        }
    else :
        return {name : img[:3]}