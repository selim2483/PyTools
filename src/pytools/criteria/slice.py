from functools import wraps
from typing import Union, Protocol, runtime_checkable

import torch

from ..utils.misc import unsqueeze_squeeze
from ..utils.checks import assert_shape, type_check
from .loss import Loss, reduce_loss


@runtime_checkable
class TensorCallable(Protocol):
    def __call__(
            self, 
            x:torch.Tensor, y:torch.Tensor, 
            *args, **kwargs
    ) -> torch.Tensor:
        ...

@type_check
def stochastic_slice(
    fn     :TensorCallable, 
    nslice :int, 
    device :Union[torch.device, str]="cpu"
):
    """Wraps a univariate function that compute a distance between two
    univariate tensors into a function that computes the corresponding random
    sliced distance.

    Args:
        fn (Callable[[torch.Tensor, torch.Tensor, ...], torch.Tensor]): 
            Univariate function
        nslice (int): Number of random slice to perform.
        device (Union[torch.device, str], optional): device on which to place
            the tensors. 
            Defaults to "cpu".

    Returns:
        Callable[[torch.Tensor, torch.Tensor, ...], torch.Tensor]: randomly
        sliced distance function
    """
    @wraps(fn)
    @unsqueeze_squeeze
    def sliced_fn(x:torch.Tensor, y:torch.Tensor, *args, **kwargs):
        assert_shape(y, x.shape)
        b, c, _, _ = x.shape
        res = torch.zeros(b).to(device)

        for _ in range(nslice):
            v = torch.randn(c, 1, dtype=x.dtype).to(device)
            v = v / v.norm(2)
            res += fn(
                torch.matmul(x.reshape(b, c, -1).transpose(1, 2), v), 
                torch.matmul(y.reshape(b, c, -1).transpose(1, 2), v),
                *args, **kwargs
            )

        return res
    
    return sliced_fn

@type_check
def band_slice(
    fn     :TensorCallable, 
    band   :int, 
    device :Union[torch.device, str]="cpu"
):
    """Wraps a univariate function that compute a distance between two
    univariate tensors into a function that computes the corresponding 
    distance between two multivariate tensors on a chosen band.

    Args:
        fn (Callable[[torch.Tensor, torch.Tensor, ...], torch.Tensor]):
            Univariate function
        band (int): band index on which to compute the distance.
        device (Union[torch.device, str], optional): device on which to place
            the tensors. 
            Defaults to "cpu".

    Returns:
        Callable[[torch.Tensor, torch.Tensor, ...], torch.Tensor]: band sliced
        distance function
    """
    @wraps(fn)
    @unsqueeze_squeeze
    def sliced_fn(x:torch.Tensor, y:torch.Tensor, *args, **kwargs):
        assert_shape(y, x.shape)
        b, c, _, _ = x.shape
        v = torch.tensor(
            [[0] if i!=band else [1] for i in range(x.shape[1])], 
            dtype=torch.float).to(device)
        v = v / v.norm(2)

        return fn(
            torch.matmul(x.reshape(b, c, -1).transpose(1, 2), v), 
            torch.matmul(y.reshape(b, c, -1).transpose(1, 2), v),
            *args, **kwargs
        )
    
    return sliced_fn

def sliced_function(
    fn     :TensorCallable, 
    nslice :Union[int, None]         = None, 
    band   :Union[int, None]         = None,
    device :Union[torch.device, str] = "cpu"
):
    if isinstance(nslice, int):
        return stochastic_slice(fn, nslice=nslice, device=device)
    elif isinstance(band, int):
        return band_slice(fn, band=band, device=device)
    else:
        raise ValueError(
            "Either nslice or band arguments should take an int value") 

class SliceLoss(Loss):
    """Basic sliced loss module. 
    Given a loss function that computes a distance between two univariate
    tensors, the module allows to compute the corresponding randomly sliced
    distance or the distance over a selected band.

    Args:
        reduction (str, optional): reduction method to perform. 
            Should be 'none', 'mean' or 'sum'.
            Defaults to 'mean'.
        device (Union[torch.device, str], optional): device on which to place
            the tensors. 
            Defaults to "cpu".
    """
    def forward(
            self, x:torch.Tensor, y:torch.Tensor, *args,
            nslice:Union[int, None]=None, band:Union[int, None]=None, **kwargs
        ):
        """Computes either random sliced distance on tensors (if nslice is
        int) or distance over a chosen band (if nslice is not int and band is
        int).

        Args:
            x (torch.Tensor): first image
            y (torch.Tensor): second image
            nslice (Union[int, None], optional): Number of random slice to
                perform.
                If ``int``, computes random sliced distance.
                Defaults to None.
            band (Union[int, None], optional): band index on which to compute
                the distance.
                If ``int``, computes distance on  chosen band.
                Defaults to None.

        Returns:
            torch.Tensor: Wanted distance.
        """
        fn = sliced_function(
            self.loss_fn, nslice=nslice, band=band, device=self.device)

        return reduce_loss(
            fn(x, y, *args, **kwargs), 
            reduction=self.reduction
        )

        

    
