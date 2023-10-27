import torch

from ..nn.projectors import MeanProjector


def to_rgb(img:torch.Tensor, name:str="images", format:str="rgb"):
    if format=="sentinel" :
        return {
            f"{name}_rgb" : img[1:4],
            f"{name}_nir" : img[[4, 6, 7]],
            f"{name}_swir" : img[[8, 9, 10]],
            f"{name}_global" : MeanProjector(11, 3)(img)
        }
    else :
        return {name : img}