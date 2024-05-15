import torch

def matrix_power(mat: torch.Tensor, p: float):
    d, v = torch.linalg.eigh(mat)
    return (v * torch.pow(torch.max(torch.zeros_like(d), d), p)) @ v.T

def covariance_distance(cov1: torch.Tensor, cov2: torch.Tensor):
    s1 = matrix_power(cov1, .5)
    return torch.trace(cov1 + cov2 - 2 * matrix_power(s1 @ cov2 @ s1, .5))