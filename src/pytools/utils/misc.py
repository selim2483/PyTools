from functools import wraps
import re
import sys
from typing import Any, Callable, Optional, Tuple

import torch


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
            for i, x in enumerate(args[:ntensors]):
                assert isinstance(x, torch.Tensor), "Input in position {i} \
should be a tensor"
                if x.ndim == ndim - 1:
                    return func(x.unsqueeze(0), *args, **kwargs).squeeze(0)
                elif x.ndim == ndim:
                    return func(x, *args, **kwargs)
                else:
                    raise ValueError(f"Input tensor in position {i} should \
have {ndim} dimension, given has {x.ndim}")
            
        return wrapper
    return decorator