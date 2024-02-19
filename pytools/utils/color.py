from functools import wraps
import itertools
import math
from typing import Callable, Iterable, Tuple

import torch


def inversible_matrix(n:int):
    while True:
        A = torch.randn(n, n)
        if torch.det(A) != 0:
            return A
        
def autocov(
        mat: torch.Tensor | Iterable[torch.Tensor], cholesky: bool = False
    ) -> torch.Tensor:
    if isinstance(mat, torch.Tensor):
        mat = mat.flatten(start_dim=-2)
        mean = mat.mean(dim=-1, keepdim=True)
        mat = mat.sub(mean)
        cov = (mat @ mat.transpose(-1,-2)) / mat.shape[-1]
        if cholesky:
            return mean.squeeze(-1), torch.linalg.cholesky(cov)
        else:
            return mean.squeeze(-1), cov
    else:
        return mat
    
def transform_color_statistics(
        img1: torch.Tensor | Iterable[torch.Tensor],
        img2: torch.Tensor | Iterable[torch.Tensor]
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

def matrix_power(mat: torch.Tensor, p: float):
    d, v = torch.linalg.eigh(mat)
    return (v * torch.pow(d, p)) @ v.T

def transport_optimal_statistics(
        img1: torch.Tensor | Iterable[torch.Tensor],
        img2: torch.Tensor | Iterable[torch.Tensor]
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor], 
    Callable[[torch.Tensor], torch.Tensor]
]:
    """Defines optimal transport transformations to match statistics of two 
    source and target images.

    Args:
        img1 (torch.Tensor): source image
        img2 (torch.Tensor): target image

    Returns:
        Tuple[
            Callable[[torch.Tensor], torch.Tensor], 
            Callable[[torch.Tensor], torch.Tensor]
        ]: transformations to match statistics in both ways.
    """
    mu1, cov1 = autocov(img1)
    mu2, cov2 = autocov(img2)

    d1, v1 = torch.linalg.eigh(cov1)
    d2, v2 = torch.linalg.eigh(cov2)

    s1 = (v1 * torch.pow(d1, .5)) @ v1.T
    s2 = (v2 * torch.pow(d2, .5)) @ v2.T
    s1_inv = (v1 * torch.pow(d1, -.5)) @ v1.T
    s2_inv = (v2 * torch.pow(d2, -.5)) @ v2.T
    
    t = s1_inv @ matrix_power(s1 @ cov2 @ s1, .5) @ s1_inv
    t_inv = s2_inv @ matrix_power(s2 @ cov1 @ s2, .5) @ s2_inv

    @color_operation
    def transform(x: torch.Tensor):
        return ((x - mu1).unsqueeze(-2) @ t).squeeze(-2) + mu2
    
    @color_operation
    def transform_inv(x: torch.Tensor):
        return ((x - mu2).unsqueeze(-2) @ t_inv).squeeze(-2) + mu1
    
    return transform, transform_inv

def gaussian_barycenter(
        lambdas: torch.Tensor, sigmas: torch.Tensor, tol: float = 1e-20):
    S = torch.pow(torch.mean(torch.sqrt(sigmas), axis=0), 2)

    n = 0
    while True:
        n += 1
        d, v = torch.linalg.eigh(S)
        s = (v * torch.pow(d, .5)) @ v.T
        s_inv = (v * torch.pow(d, -.5)) @ v.T
        G = torch.zeros_like(s)
        for li, Si in zip(lambdas, sigmas):
            G += li * matrix_power(s @ Si @ s, .5)
        G = s_inv @ G @ G @ s_inv

        eps = float(torch.trace(S + G - 2 * matrix_power(s @ G @ s, .5)))
        # if n == 1:
        #     lvl = int(math.log(eps, 10)) + 1
        # if math.log(abs(eps), 10) < lvl:
        #     print(f"Iteration {n} : ", eps)
        #     lvl = min(lvl - 1, math.log(abs(eps), 10))

        if abs(eps) < tol:
            return G
        else:
            S = G

def multispectral_barycenter(
        img: torch.Tensor | Iterable[torch.Tensor], 
        tol: float = 1e-6, 
        ord: bool  = True
    ):
    mu, sigma = autocov(img)
    if ord == True:
        triplets = list(itertools.permutations(range(len(sigma)), 3))
    else:
        triplets = list(itertools.combinations(range(len(sigma)), 3))
    nt = len(triplets)
    lambdas = torch.tensor(1 / nt).repeat(nt)
    sigmas = torch.stack([
        sigma[*torch.meshgrid(
            torch.tensor(triplet), torch.tensor(triplet), indexing='ij')] 
        for triplet in triplets
    ])

    return (
        mu.mean(dim=-1).repeat(3), 
        gaussian_barycenter(lambdas, sigmas, tol=tol)
    )

def get_transforms(
        img1: torch.Tensor | Iterable[torch.Tensor],
        img2: torch.Tensor | Iterable[torch.Tensor],
        mode: str = "stats"
    ):

    if mode == "stats":
        return transform_color_statistics(img1, img2)
    elif mode == "optimal":
        return transport_optimal_statistics(img1, img2)
    
def color_operation(func: Callable[[torch.Tensor], torch.Tensor]):
    @wraps(func)
    def wrapper(x: torch.Tensor):
        return func(
            x.transpose(-3,-1).transpose(-2,0)
        ).transpose(-2,0).transpose(-3,-1)
    return wrapper

if __name__=="__main__":
    print("")
    print("Test multispectral_barycenter")
    print("=============================")
    img = torch.load("/scratchm/sollivie/data/sentinel2/gatys_512x512/S2A_MSIL2A_20230615T102031_N0509_R065_T30PZR_20230615T200553.SAFE.pt")
    _, sigma = autocov(img)
    print(multispectral_barycenter(sigma, ordered=False))
    print(multispectral_barycenter(sigma, ordered=True))