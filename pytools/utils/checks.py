import contextlib
from functools import wraps
from typing import get_type_hints, Any, Iterable, Union
import warnings

import torch
from torch.types import _int

__all__ = ["type_check"]

def check_arg(_name:str, _value:Any, _type:type) :
    if hasattr(_type, "__origin__"):
        if _type.__origin__ == Union:
            if not any([isinstance(_value, arg_type) 
                        for arg_type in _type.__args__]):
                raise TypeError(
                    f"Argument '{_name}' must be of type " 
                    + " or ".join([f"'{arg_type.__name__}'" 
                                   for arg_type in _type.__args__])
                    + ", not '{type(_value).__name__}'"
                )
        else:
            return check_arg(_name, _value, _type.__origin__)
    elif not isinstance(_value, _type) :
        raise TypeError(f"Argument '{_name}' must be of type '{_type.__name__}', \
not '{type(_value).__name__}'")

def type_check(func) :
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the type hints for the decorated function
        hints = get_type_hints(func)

        # Check argument types
        for arg_name, arg_type in hints.items():
            if arg_name in kwargs:
                check_arg(arg_name, kwargs[arg_name], arg_type)
            else:
                arg_index = list(hints.keys()).index(arg_name)
                if arg_index < len(args):
                    check_arg(arg_name, args[arg_index], arg_type)

        # Call the decorated function
        return func(*args, **kwargs)

    return wrapper

@contextlib.contextmanager
def suppress_tracer_warnings():
    flt = ('ignore', None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)

def assert_shape(tensor:torch.Tensor, ref_shape:Iterable):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim},\
 expected {len(ref_shape)}')
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            # as_tensor results are registered as constants
            with suppress_tracer_warnings(): 
                torch._assert(
                    torch.equal(torch.as_tensor(size), ref_size), 
                    f'Wrong size for dimension {idx}'
                )
        elif isinstance(size, torch.Tensor):
            # as_tensor results are registered as constants
            with suppress_tracer_warnings(): 
                torch._assert(
                    torch.equal(size, torch.as_tensor(ref_size)), 
                    f'Wrong size for dimension {idx}: expected {ref_size}'
                )
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got \
{size}, expected {ref_size}')
        
def assert_dim(
        tensor:torch.Tensor, 
        ndim:Union[_int, None]=None, 
        min_ndim:Union[_int, None]=None, 
        max_ndim:Union[_int, None]=None
    ):
    if ndim is not None and tensor.ndim != ndim :
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim},\
 expected {ndim}')
    elif min_ndim is not None and tensor.ndim < min_ndim:
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim},\
 expected minimum {min_ndim}')
    elif max_ndim is not None and tensor.ndim > max_ndim:
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim},\
 expected maximum {max_ndim}')

        
