import itertools
from math import ceil
import random
from typing import Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.types import _float

from ..utils.checks import assert_dim, assert_shape
from ..utils.misc import unsqueeze_squeeze

__all__ = [
    "RandomProjector", "SampleProjector", "KDEProjector", "PCAProjector",
    "PCA", "ConvProjector"
]

class StaticProjector(torch.nn.Module):
    """Base class containing basic methods for model saving and loading for
    projectors that do not use trainable parameters (e.g. sampling, PCA).
    """
    def state_dict(self) :
        return self.__dict__
    
    def load_state_dict(self, state_dict:dict) : 
        for k, v in state_dict.items() :
            self.__setattr__(k, v)


class RandomProjector(torch.nn.Module):
    """Implements a module that projects multispectral into 3 dimensionnal
    color space by selecting 3 random channels.
    For consistent training, the random channels need to be generated using
    the method provided for this purpose. 
    """
    def __init__(self, in_channels:int, all_arrangements:bool=False):
        """
        Args:
            in_channels (int): number of channels of images.
                all_arrangements (bool, optional): whether to test every
                possible arrangements in a deterministic order to avoid
                duplication. 
                This feature is usefull in the case of testing.
                If ``True``, :meth:`generate` changes the projection channels
                to the next arrangement.
                If ``False``, :meth:`generate` generates a new random
                arrangement.
                Defaults to ``False``.
        """
        super().__init__()
        self.in_channels = in_channels
        self.all_arrangements = all_arrangements
        if self.all_arrangements :
            self.triplets = itertools.permutations(
                range(self.in_channels), 3)
        self.generate()

    def generate(self) :
        """Generates a new arrangement, random if :attr:`arrangements` is
        ``True``, determinist if not.
        """
        if self.in_channels!=1 :
            if self.in_channels!=3 :
                if self.all_arrangements :
                    try : 
                        self.channels = next(self.triplets)
                    except StopIteration :
                        self.triplets = itertools.permutations(
                            range(self.in_channels), 3)
                        self.channels = next(self.triplets)
                else :
                    self.channels = random.sample(range(self.in_channels), 3)
            else :
                self.channels = [0,1,2]
        else:
            self.channels = [0]

    def forward(self, x:torch.Tensor) :
        return x[..., self.channels, :, :]
    
class MeanProjector(StaticProjector) :
    """_summary_

    Args:
        StaticProjector (_type_): _description_
    """
    def __init__(self, nc_in:int, nc_out:int) :
        super().__init__()
        self.nc_in = nc_in
        self.nc_out = nc_out
        self.kernel_size = ceil(nc_in / nc_out)

    @unsqueeze_squeeze
    def forward(self, x:torch.Tensor) :
        assert_shape(x, (None, self.nc_in, None, None))
        if self.nc_in==1 :
            y = x.repeat((1, 3, 1, 1))
        else :
            n, c, w, h = x.size()
            x = x.reshape(n, c, h * w).permute(0, 2, 1)
            pooled = F.avg_pool1d(
                x,
                kernel_size=self.kernel_size,
                ceil_mode=True
            )
            assert_shape(pooled, (None, None, self.nc_out))
            return pooled.permute(0, 2, 1).view(n, self.nc_out, w, h)
        
class SampleProjector(StaticProjector) :
    """Projects multispectral images into into a chosen sub-color space.
    """
    def __init__(self, bands:Iterable[int]) -> None:
        """
        Args:
            bands (Iterable[int]): Spectral bands constituting the wanted
                subcolor space.
        """
        super().__init__()
        self.bands = bands

    def forward(self, x:torch.Tensor) :
        return x[..., self.bands, :, :]

class KDEProjector(StaticProjector) :
    """Gives an intermediate solution between :class:`SampleProjector` and
    :class:`MeanProjector` : each channel of the projected output corresponds
    to a ponderated sum of the original channels using a gaussian kernel.
    """
    def __init__(
            self,
            mus    :Iterable[_float], 
            sigmas :Iterable[_float], 
            device :Union[torch.device, str]="cpu") -> None:
        """
        Args:
            mus (Iterable[_float]): Means of the gaussian kernels.
            sigmas (Iterable[_float]): standard deviations of the gaussian
                kernels
            device (Union[torch.device, str], optional): device to place
                tensors on. 
                Defaults to "cpu".
        """
        assert len(mus)==len(sigmas), "mus and sigmas arguments should have \
the same length."
        super().__init__()
        u      = torch.tensor(
            range(len(mus)), device=device).repeat(3,1).transpose(0,1)
        mus    = torch.tensor(mus, device=device)
        sigmas = torch.tensor(sigmas, device=device)
        
        self.v = torch.exp(- (u - mus / sigmas)**2 / 2)
        self.v = self.v / self.v.norm(p=1, dim=0)

    def forward(self, x:torch.Tensor) :
        return x.transpose(1,3).matmul(self.v).transpose(1,3)

class PCAProjector(StaticProjector):
    """Class implementing a PCA projection operator and its inversion
    counterpart.
    """
    def __init__(self, mu:torch.Tensor, lambdas:torch.Tensor, V:torch.Tensor):
        """
        Args:
            mu (torch.Tensor): Mean vector of the considered data.
            lambdas (torch.Tensor): Eigenvalues of the covariance matrix
                (ignoring a multiplication factor)
            V (torch.Tensor): principal directions
        """
        super().__init__()
        self.mu = mu
        self.lambdas = lambdas
        self.V = V
    
    def forward(self, x:torch.Tensor) :
        x = torch.matmul(
            x.transpose(1,3).sub(self.mu), 
            self.V)
        return x.div(self.lambdas).transpose(1,3)
    
    def back_project(self, x:torch.Tensor) :
        x = x.transpose(1,3).mul(self.lambdas)
        x = torch.matmul(x, self.V.transpose(0,1))
        return x.add(self.mu).transpose(1,3)

class PCA(torch.nn.Module) :
    """Implements PCA trainig and plotting processes based on multispectral
    images.
    """
    def __init__(self,  A:torch.Tensor, npixels:int, ncomp:int=3):
        """
        Args:
            A (torch.Tensor): batch of images on which PCA needs to be 
                performed.
            npixels (int): Number of pixels to use to perfrom PCA.
            ncomp (int, optional): Number of principal component to select.
                Defaults to 3.
        """
        assert_dim(A, 4)
        super().__init__()
        b, c, h, w = A.shape
        idx = random.sample(range(b * h * w), npixels)
        self.A = A.transpose(1, 3).reshape(-1, c)[idx, :]
        self.mu = self.A.mean(dim=0)
        self.nsamples, self.nfeatures = self.A.shape
        self.ncomp = ncomp

    @property
    def principal_components(self):
        return self.V[:, :self.ncomp]
    
    @property
    def eigenvalues(self):
        return self.S[:self.ncomp].sqrt() * self.norm_quantile

    def train_pca(self) :
        """Trains PCA on the selected data.
        Initializes 
        """
        _, self.S, self.V = torch.pca_lowrank(
            self.A, q=self.nfeatures, center=True)
        self.S = self.S ** 2
        self.S = self.S / torch.sum(self.S)
        self.V = self.V[:, :self.ncomp]

    def normalize(self, q=0.99) :
        """Finds a normalizing constant depending on the wanted quantile.

        Args:
            q (float, optional): Wanted quantile : by dividing by
                :attr:norm_quantile, The proportion of the data contained in
                the unit ball will be ``q``.
                Defaults to 0.99.
        """
        self.norm_quantile = torch.matmul(
            self.A - self.mu, 
            self.V).div(self.S[:self.ncomp].sqrt()).abs().quantile(q)
    
    def plot(self, path:str) :
        """Plots and save Variance percentages and eigenvectors/principal
        components.

        Args:
            path (str): where to save the figure.
        """
        fig = plt.figure(figsize=(20, 7),facecolor='w')

        # Eigen Values
        print("Eigen Values :")
        print(*list(self.S.cpu().numpy()))
        print(*list(self.S.cumsum(dim=0).cpu().numpy()))
        ax = plt.subplot(1, 2, 1)
        plt.bar(range(1, self.nfeatures + 1), self.S.cpu().numpy())
        ax.set_xlabel("Eigen value")
        ax.set_ylabel("Variance percentage")
        ax.set_title("PCA eigen values")

        # Eigen vectors
        ax = plt.subplot(1, 2, 2)
        for i in range(self.ncomp) :
            plt.plot(
                range(1, self.nfeatures + 1), 
                self.V[:, i].cpu().numpy(),
                label=f"u{i+1}")
        ax.set_xlabel("Band index")
        ax.set_title("PCA eigen vectors")
        ax.legend()

        plt.savefig(path, bbox_inches='tight')

    def plot_histograms(self, path:str, nbins:int) :
        """Plots histograms of the norm of pixels for each different
        normalization.

        Args:
            path (str): Where to save the figure
            nbins (int): Number of bins for histograms.
        """
        B = torch.matmul(self.A - self.mu, self.V)
        C = B.div(self.S[:3].sqrt())
        D = B.div(self.S[:3])

        self.quantiles = [
            data.abs().quantile(0.99) for data in [self.A, B, C, D]]
        print("0.99 quantiles :", self.quantiles)

        bins = np.linspace(-2, 2, nbins)

        plt.figure()
        
        for i in range(3) :
            ax = plt.subplot(3, 1, i+1)
            for data, label in zip(
                [self.A[:, 1:4], B, C, D], 
                ["rgb", "pca", "pca div sqrt", "pca div"]) :
                ax.hist(
                    data[:, i].numpy(), 
                    bins=bins, 
                    alpha=0.5, 
                    label=f"{label} {i+1}")
            ax.legend()
        plt.savefig(path)
    
class ConvProjector(torch.nn.Module) :
    __doc__ = """Embody a single or double layer CNN with or not
    non-linearities.
    Allows to use linear projection when no non linearities are used.

    Args:
        nlayers (int): Number of layers of the CNN
        nc_in (int): Number of input channels : number of spectral bands
            of input data.
        nc_out (int): Number of output channels : number of wanted
            spectral bands in output.
        activation (bool, optional): Whether to use non-linearities or not.
            If ``True``, LeakyReLU is used.
            Defaults to ``True``.
    """
    def __init__(
            self, 
            nlayers:int, 
            nc_in:int, 
            nc_out:int, 
            activation:bool=True) -> None:
        super().__init__()
        match nlayers :
            case 1 :
                if activation :
                    self.body = torch.nn.Sequential(
                        torch.nn.Conv2d(nc_in, nc_out, 1), 
                        torch.nn.LeakyReLU()
                    )
                else :
                    self.body = torch.nn.Conv2d(nc_in, nc_out, 1)
            case 2 :
                self.body = torch.nn.Sequential(
                    torch.nn.Conv2d(nc_in, nc_in, 1), torch.nn.LeakyReLU(),
                    torch.nn.Conv2d(nc_in, nc_out, 1), torch.nn.LeakyReLU()
                )

    def forward(self, x:torch.Tensor) :
        return self.body(x)