from typing import Tuple
import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from .inception import InceptionV3

def compute_frechet_distance(
        mu1:np.ndarray, sigma1:np.ndarray, 
        mu2:np.ndarray, sigma2:np.ndarray, 
        eps:float=1e-6) :
    """Compute Fréchet Distance between two MVN distributions.

    Args:
        mu1 (np.ndarray): mean vector of the first distributions 
        sigma1 (np.ndarray): covariance matrix of the first distributions
        mu2 (np.ndarray): mean vector of the second distributions
        sigma2 (np.ndarray): covariance matrix of the second distributions
        eps (float, optional): perturbation to add to avoid singular matrix. 
            Defaults to 1e-6.

    Raises:
        ValueError: complex component in inverse matrix.

    Returns:
        _type_: Fréchet distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
            'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print('Imaginary component {}'.format(m))
            return None
        covmean = covmean.real

    return float(diff.dot(diff) 
                 + np.trace(sigma1) 
                 + np.trace(sigma2) 
                 - 2 * np.trace(covmean))

def compute_statistics(act:np.ndarray) -> Tuple[np.ndarray, np.ndarray] :
    """Computes first and second order statistics (mean and covariance matrix)
    over a batch of activations.

    Args:
        act (np.ndarray): activations array (B, H * W, C)

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean (B, C) and covariance matrix 
            (B, C, C)
    """
    mu = np.mean(act, axis=1)
    act = (act.transpose(1, 0, 2) - mu).transpose(1, 0, 2)
    sigma = act.transpose(0, 2, 1) @ act
    return mu, sigma

def compute_fid(act1:np.ndarray, act2:np.ndarray) :
    
    mu1, sigma1 = compute_statistics(act1)
    mu2, sigma2 = compute_statistics(act2)

    fids = []
    for i in range(mu1.shape[0]) :
        fid = compute_frechet_distance(mu1[i], sigma1[i], mu2[i], sigma2[i])
        if fid is not None :
            fids.append(fid)

    return fids

class SIFID :

    def __init__(
            self, inception_dims:int=64, device=None, **kwargs) :
        self.inception_dims = inception_dims
        self.device         = device
        self.model          = self.initialize_model()

    def initialize_model(self) :
        print("Initializing InceptionV3 model...")
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.inception_dims]
        return InceptionV3([block_idx]).to(self.device)
    
    def get_activations(self, x:torch.Tensor) :
        pred = self.model(x)[0]
        b, _, h, w = pred.shape
        return pred.cpu().data.numpy().transpose(0, 2, 3, 1).reshape(
            b, h * w, -1)
    
    def __call__(self, x:torch.Tensor, y:torch.Tensor) :
        return compute_fid(self.get_activations(x), self.get_activations(y))

    


    