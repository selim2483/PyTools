from dataclasses import dataclass, field, is_dataclass

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
    Only __repr__ is customized from basic dataclasses.
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

@dataclass(repr=False)
class DataOptions(Options) :
    path        :str
    max_size    :int
    resolution  :int  = 256
    use_labels  :bool = False
    random_seed :int  = 0
    augment     :bool = True
        
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
    lr_factor_map   :float = 0.01
    beta1           :float = 0.0
    beta2           :float = 0.99
    from_checkpoint :bool  = False

    @property
    def lr_map(self):
        return self.lr * self.lr_factor_map
        
@dataclass(repr=False)
class TrainingOptions(Options) :
    max_steps           :str  = "max_steps"
    max_kimg            :str  = "max_kimg"
    starting_resolution :int  = 8
    batch_sizes         :dict = field(default_factory={
        4: 128, 
        8: 128, 
        16: 128, 
        32: 128, 
        64: 64, 
        128: 32, 
        256: 16, 
        512: 8, 
        1024: 4
    })
    prog_fadin_kimg     :int = 600
    prog_stab_kimg      :int = 600