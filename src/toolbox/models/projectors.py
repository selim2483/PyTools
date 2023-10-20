import itertools
import random
from typing import Iterable, Union

import torch


class RandomProjector(torch.nn.Module):
    """Implements a module that projects multispectral into 3 dimensionnal
    color space by selecting 3 random channels.
    For consistent training, the random channels need to be generated using
    the method provided for this purpose. 
    """
    def __init__(self, in_channels:int, all_arrangements=False):
        """
        Args:
            in_channels (int): number of channels of images.
            all_arrangements (bool, optional): whether to test every possible
            arrangements in a deterministic order to avoid duplication. 
            This feature is usefull in the case of testing.
            If True, the `generate()` method changes the projection channels to the next arrangement.
            If False, the `generate()` method generates a new random arrangement.
            Defaults to False.
        """
        super().__init__()
        self.in_channels = in_channels
        self.all_arrangements = all_arrangements
        if self.all_arrangements :
            self.triplets = itertools.permutations(
                range(self.in_channels), 3)
        self.generate()

    def generate(self) :
        """Generates a new arrangement, random if self.arrangements==True,
        determinist if not.
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
        
class SampleProjector(torch.nn.Module) :

    def __init__(self, bands) -> None:
        super().__init__()
        self.bands = bands

    def forward(self, x:torch.Tensor) :
        return x[:, self.bands, :, :]
    
    def state_dict(self) :
        return self.__dict__
    
    def load_state_dict(self, state_dict:dict) : 
        for k, v in state_dict.items() :
            self.__setattr__(k, v)

class KDEProjector(torch.nn.Module) :

    def __init__(
            self,
            mus    :Iterable[int], 
            sigmas :Iterable[int], 
            nbands :int, 
            device :Union[torch.device, str]="cpu") -> None:
        super().__init__()
        u = torch.tensor(
            range(nbands), device=device).repeat(3,1).transpose(0,1)
        mus    = torch.tensor(mus, device=device)
        sigmas = torch.tensor(sigmas, device=device)
        
        self.v = torch.exp(- (u - mus / sigmas)**2 / 2)
        self.v = self.v / self.v.norm(p=1, dim=0)

    def forward(self, x:torch.Tensor) :
        return x.transpose(1,3).matmul(self.v).transpose(1,3)
    
    def state_dict(self) :
        return self.__dict__
    
    def load_state_dict(self, state_dict:dict) : 
        for k, v in state_dict.items() :
            self.__setattr__(k, v)

class PCA(torch.nn.Module) :

    def __init__(
            self, 
            data_options :DataOptions, 
            nimg         :int, 
            npixels      :int, 
            ncomp        :int, 
            device       :Union[torch.device, str] = "cpu") :
        super().__init__()
        data_options.batch_size = nimg
        try :
            loader = DataTAE(data_options).get_loader("train")
        except FileNotFoundError :
            loader = DataTAE(data_options).get_loader()
        A = next(iter(loader)).to(device)
        B, _, H, W = A.shape
        idx = random.sample(range(B * H * W), npixels)
        self.A = A.transpose(1, 3).reshape(-1, data_options.nbands)[idx, :]
        self.mean = self.A.mean(dim=0)
        self.nsamples, self.nfeatures = self.A.shape
        self.ncomp = ncomp

    def train_pca(self) :
        _, self.S, self.V = torch.pca_lowrank(
            self.A, q=self.nfeatures, center=True)
        self.S = self.S ** 2
        self.S = self.S / torch.sum(self.S)
        self.V = self.V[:, :self.ncomp]

    def normalize(self, q=0.99) :
        self.q = torch.matmul(
            self.A - self.mean, 
            self.V).div(self.S[:3].sqrt()).abs().quantile(q)
    
    def plot(self, path:str) :
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
        B = torch.matmul(self.A - self.mean, self.V)
        C = B.div(self.S[:3].sqrt())
        D = B.div(self.S[:3])

        self.quantiles = [data.abs().quantile(0.99) for data in [A, B, C, D]]
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

    def forward(self, x:torch.Tensor) :
        x = torch.matmul(
            x.transpose(1,3).sub(self.mean), 
            self.V)
        return x.div(self.S[:self.ncomp].sqrt() * self.q).transpose(1,3)
    
    def back_project(self, x:torch.Tensor) :
        x = x.transpose(1,3).mul(self.S[:self.ncomp].sqrt() * self.q)
        x = torch.matmul(x, self.V.transpose(0,1))
        return x.add(self.mean).transpose(1,3)
    
    def state_dict(self) :
        return self.__dict__
    
    def load_state_dict(self, state_dict:dict) : 
        for k, v in state_dict.items() :
            self.__setattr__(k, v)

    
class ConvProjector(torch.nn.Module) :

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