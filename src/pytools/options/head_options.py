import os
from typing import Union

import torch
import yaml

from .options import Options, DataOptions, LoggingOptions, TrainingOptions

__all__ = ["HeadOptions", "CoachOptions"]

class HeadOptions :

    def __init__(self, config:Union[str, dict, None]=None) -> None:

        if isinstance(config, str) :
            with open(config, "r") as ymlfile:
                self._cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)
        elif isinstance(config, dict) : 
            self._cfg = config
        elif config is None : 
            self._cfg = {}
        else :
            raise ValueError("config should be either a path, a dict or None")
        
        self.checkpoint_path = self._cfg.get("checkpoint")
        self.get_options()

    @property
    def cfg(self) :
        _cfg = {}
        for key, value in self.__dict__.items() :
            if isinstance(value, Options) :
                _cfg[key[:-7].lower()] = vars(value)
        return _cfg
        
    def dump_configfile(self, logdir:str) :
        path = os.path.join(logdir, "config.yaml")
        with open(path, "w") as ymlfile :
            yaml.dump(self.cfg, ymlfile)

    def get_options(self) :
        if self.checkpoint_path is not None :
            self.get_model_options_from_checkpoint()
        else :
            self.get_model_options_from_config()

        self.get_general_options_from_config()

    def get_model_options_from_checkpoint(self) :
        raise NotImplementedError
    
    def get_model_options_from_config(self) :
        raise NotImplementedError
    
    def get_general_options_from_config(self) :
        raise NotImplementedError
    
class CoachOptions(HeadOptions) :
 
    def get_model_options_from_checkpoint(self):
        self.checkpoint_dict:dict = torch.load(self.checkpoint_path)
    
    def get_general_options_from_config(self) :
        self.data_options           = DataOptions(**self._cfg.get("data")) 
        self.logging_options        = LoggingOptions(
            **self._cfg.get("logging")) 
        self.training_options       = TrainingOptions(
            **self._cfg.get("training")) 