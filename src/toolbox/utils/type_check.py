from functools import wraps
from typing import get_type_hints, Any

__all__ = ["type_check"]

def check_arg(_name:str, _value:Any, _type:type) :
    if not isinstance(_value, _type) :
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