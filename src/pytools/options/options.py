from dataclasses import dataclass
from typing import Union

__all__ = [
    "Options", "DataOptions", "LoggingOptions", "OptimizerOptions", 
    "TrainingOptions"
]

class Options :
    """Options base class. 
    
    Options object are meant to be dataclasses that carry every options
    relative to a unique subdomain (e.g. losses, data, optimizers). 
    This class behaves like a dict but enables the use of code completion via
    Pylance, comparison operations. Default values can also be provided.
    2 methods are added on top of a basic dataclasses :
        - A custom __repr__ method.
        - A classmethod allowing one to construct the 
    """

    def __repr__(self) :
        short_repr = self._short_repr()
        if len(short_repr)<79:
            return short_repr
        else:
            return self._long_repr()

    def _short_repr(self):
        _repr = f"{self.__class__.__name__}("
        for key, value in self.__dict__.items() :
            _repr += f"{key}={value}, "
        _repr = _repr[:-2] + ")"
        return _repr
    
    def _long_repr(self):
        _repr = f"{self.__class__.__name__}(\n"
        key_len = max([len(key) for key in self.__dict__.keys()])
        for key, value in self.__dict__.items() :
            spaced_key = key + (key_len - len(key))*" "
            _repr += f"\t{spaced_key} = {value},\n"
        _repr = _repr[:-2] + ")"
        return _repr
    
    @classmethod
    def from_attribute_dict(cls, attr_dict:dict):
        new_attr_dict = {}
        for key in cls.__annotations__:
            if key in attr_dict :
                new_attr_dict[key] = attr_dict[key]
            elif key in cls.__dict__:
                new_attr_dict[key] = cls.__dict__[key]
            else:
                raise KeyError(f"Provided 'attr_dict' argument should have a \
    '{key}' as the {cls.__class__.__name__} dataclass does not provide \
    default value")
        return cls(**new_attr_dict)
    

@dataclass(repr=False)
class DataOptions(Options) :
    path        :str
    max_size    :Union[int, None] = None
    resolution  :int              = 256
    use_labels  :bool             = False
    random_seed :int              = 0
    augment     :bool             = True
    nchannels    :int             = 3
        
@dataclass(repr=False)
class LoggingOptions(Options) :
    toplogdir             :str
    logger                :str  = "tensorboard"
    progress_key          :str  = "step"
    image_interval_local  :int  = 10000
    image_interval_logger :int  = 1000
    checkpoint_interval   :int  = 10000
    save_training_data    :bool = True
    max_queue_train       :int  = 50
    nimg                  :str  = "nimg"

@dataclass(repr=False)
class OptimizerOptions(Options) :
    lr              :float = 0.001
    beta1           :float = 0.0
    beta2           :float = 0.99
    from_checkpoint :bool  = False
        
@dataclass(repr=False)
class TrainingOptions(Options) :
    max_steps :Union[int, None] = None
    max_kimg  :Union[int, None] = None