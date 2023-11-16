from functools import wraps
from typing import Tuple, Union, Protocol, runtime_checkable

import torch

from ..utils.misc import unsqueeze_squeeze
from ..utils.checks import assert_shape, type_check
from .loss import Loss, reduce_loss


@runtime_checkable
class TensorCallable(Protocol):
    def __call__(self, x:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        ...

@type_check
def slice_tensors(args:Tuple[torch.Tensor], v:torch.Tensor):
    tensors = ()
    b, c, _, _ = args[0].shape
    for arg in args:
        assert_shape(arg, (b, c, None, None))
        tensors += (torch.matmul(arg.reshape(b, c, -1).transpose(1, 2), v),)  
        
    return tensors

@type_check
def sliced_function(
    fn:     TensorCallable, 
    nargs:  int,
    nslice: int, 
    device: Union[torch.device, str]="cpu"
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
    def sliced_fn(*args, **kwargs):
        assert isinstance(args[0], torch.Tensor), "input should be a tensor"
        b, c, _, _ = args[0].shape
        res = torch.zeros(b).to(device)

        if "device" in fn.__annotations__.keys():
            kwargs["device"] = device
            
        for _ in range(nslice):
            v = torch.randn(c, 1, dtype=args[0].dtype).to(device)
            v = v / v.norm(2)
            res += fn(
                *slice_tensors(args[:nargs], v), *args[nargs:], **kwargs)

        return res
    
    return sliced_fn

def sliced(
    fn:     TensorCallable, 
    *args,
    nargs:  int,
    nslice: Union[int, None]         = None, 
    device: Union[torch.device, str] = "cpu",
    **kwargs
) -> torch.Tensor:
    """Slices a function.

    Args:
        fn (TensorCallable): _description_
        nargs (int): _description_
        nslice (Union[int, None], optional): _description_. Defaults to None.
        device (Union[torch.device, str], optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """
    fn_sliced = sliced_function(
        fn, nargs=nargs, nslice=nslice, device=device)
    return fn_sliced(*args, **kwargs)

@type_check
def band_slice(
    fn         :TensorCallable, 
    nargs:int,
    band       :int, 
    device     :Union[torch.device, str]="cpu"
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
    @unsqueeze_squeeze(ntensors=nargs)
    def sliced_fn(*args, **kwargs):
        assert isinstance(args[0], torch.Tensor), "input should be a tensor"
        v = torch.tensor(
            [[0] if i!=band else [1] for i in range(args[0].shape[1])], 
            dtype=args[0].dtype).to(device)
        v = v / v.norm(2)
            
        if "device" in fn.__annotations__.keys():
            kwargs["device"] = device

        return fn(
            *slice_tensors(args[:nargs], v), 
            *args[nargs:], 
            **kwargs
        )
    
    return sliced_fn 

def sliced_distance(
    fn:     TensorCallable, 
    *args,
    nslice: Union[int, None]         = None, 
    device: Union[torch.device, str] = "cpu",
    **kwargs
) -> torch.Tensor:
    return sliced(fn, *args, nargs=2, nslice=nslice, device=device, **kwargs)
    
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
            nslice:Union[int, None]=None, **kwargs
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

        return reduce_loss(
            sliced_distance(
                self.loss_fn, 
                x, y, *args, 
                nslice=nslice, device=self.device, **kwargs
            ), 
            reduction=self.reduction
        )

        

    
