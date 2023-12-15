from functools import wraps
from typing import Callable, Tuple

import torch


def inversible_matrix(n:int):
    while True:
        A = torch.randn(n, n)
        if torch.det(A) != 0:
            return A
        
def autocov(mat:torch.Tensor, cholesky=False) -> torch.Tensor:
    mat = mat.flatten(start_dim=-2)
    mean = mat.mean(dim=-1, keepdim=True)
    mat = mat.sub(mean)
    cov = (mat @ mat.transpose(-1,-2)) / mat.shape[-1]
    if cholesky:
        return mean.squeeze(-1), torch.linalg.cholesky(cov)
    else:
        return mean.squeeze(-1), cov
    
def transform_color_statistics(
        img1: torch.Tensor, img2: torch.Tensor
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor], 
    Callable[[torch.Tensor], torch.Tensor]
]:
    """Defines transformations to match statistics of txo source and target
    images.

    Args:
        img1 (torch.Tensor): source image
        img2 (torch.Tensor): target image

    Returns:
        Tuple[
            Callable[[torch.Tensor], torch.Tensor], 
            Callable[[torch.Tensor], torch.Tensor]
        ]: transformations to match statistics in both ways.
    """
    mu1, cov1 = autocov(img1, cholesky=True)
    mu2, cov2 = autocov(img2, cholesky=True)
    t = (cov2 @ cov1.inverse()).transpose(-1,-2)
    t_inv = (cov1 @ cov2.inverse()).transpose(-1,-2)
    
    @color_operation
    def transform(x: torch.Tensor):
        return ((x - mu1).unsqueeze(-2) @ t).squeeze(-2) + mu2
    
    @color_operation
    def transform_inv(x: torch.Tensor):
        return ((x - mu2).unsqueeze(-2) @ t_inv).squeeze(-2) + mu1
    
    return transform, transform_inv
    
def color_operation(func: Callable[[torch.Tensor], torch.Tensor]):
    @wraps(func)
    def wrapper(x: torch.Tensor):
        return func(
            x.transpose(-3,-1).transpose(-2,0)
        ).transpose(-2,0).transpose(-3,-1)
    return wrapper