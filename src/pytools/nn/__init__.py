import torch
from . import functionnal
from .projectors import (
    RandomProjector, MeanProjector, SampleProjector, KDEProjector,
    PCAProjector, PCA, ConvProjector)

def f(x:torch.Tensor):
    """Test function

    Args:
        x (torch.Tensor): input

    Returns:
        torch.Tensor: output
    """
    return 2 * x