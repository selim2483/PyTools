from functools import wraps
from itertools import islice
import re
import sys
from typing import Any, Callable, Iterable, Optional, Tuple

import torch

_tensor_op = Callable[[torch.Tensor], torch.Tensor]

def error(msg: str):
    print('Error: ' + msg)
    sys.exit(1)

def parse_tuple(s: str) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the 
    attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

def unsqueeze_squeeze(ndim:int=4, ntensors:int=1):
    def decorator(func:Callable[..., torch.Tensor]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            assert isinstance(args[0], torch.Tensor), "Input in position \
should be a tensors"
            if args[0].ndim == ndim - 1:
                res = func(
                    *map(lambda x: x.unsqueeze(0), args[:ntensors]), 
                    *args[ntensors:], 
                    **kwargs
                )
                if isinstance(res, torch.Tensor):
                    return res.squeeze(0)
                elif isinstance(res, Iterable):
                    return map(lambda x: x.squeeze(0), res)
                else:
                    raise ValueError(f"Function should return either a Tensor \
or an Iterable of tensors, given was {type(res)}.")
            elif args[0].ndim >= ndim:
                return func(*args, **kwargs)
            else:
                raise ValueError(f"Input tensors should \
have {ndim} dimension, one of the given tensors has {args[0].ndim}")
            
        return wrapper
    return decorator

def get_device() -> torch.device : 
    """Check the availability of the GPU.

    Returns:
        (torch.device): device to use : cuda (GPU) or cpu.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device name : ", torch.cuda.get_device_name())
        print("Device name : ", torch.cuda.get_device_properties(device))
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("Device name : CPU")

    print("")

    return device

def slice_tensors(*args:torch.Tensor, start:int, stop:int) :
    return [arg[start:stop] for arg in args]

def tensor2list(x:torch.Tensor) :
    assert type(x)==torch.Tensor, "x should be a tensor"
    if x.ndim==0:
        x = x.unsqueeze(0)
    return list(x.detach().cpu().numpy())

def batch_iter(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def map_dict(fn: Callable[[str, Any], Any], _dict: dict):
    res_dict = {}
    for key, value in _dict.items():
        res_dict[key] = fn(key, value)
    return res_dict