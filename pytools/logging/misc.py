from functools import wraps
from math import sqrt
from typing import Any, Callable, Iterable, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from ..utils.checks import type_check

@type_check
def num_parameters(model:torch.nn.Module) :
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@type_check
def concatenate_loss_dict(agg_loss_dict:Iterable[dict], list_output:bool=False) :
    vals = {}
    for output in agg_loss_dict:
        for key in output:
            if list_output :
                vals[key] = vals.setdefault(key, []) + output[key]
            else :
                vals[key] = vals.setdefault(key, []) + [output[key]]
    return vals

@type_check
def mean_loss_dict(agg_loss_dict:Iterable[dict], list_output:bool=False) :
    mean_vals = concatenate_loss_dict(agg_loss_dict, list_output=list_output)
    for key in mean_vals:
        if len(mean_vals[key]) > 0:
            mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
        else:
            print('{} has no value'.format(key))
            mean_vals[key] = 0
    return mean_vals

@type_check
def std_loss_dict(agg_loss_dict:dict, list_output:bool=False) :
    mean_vals = mean_loss_dict(agg_loss_dict, list_output=list_output)
    std_vals = concatenate_loss_dict(agg_loss_dict, list_output=list_output)                
    for key in std_vals:
        if len(std_vals[key]) > 0:
            dev_vals = [(elt - mean_vals[key])**2 for elt in std_vals[key]]
            std_vals[key] = sqrt(sum(dev_vals) / len(std_vals[key]))
        else:
            print('{} has no value'.format(key))
            std_vals[key] = 0
    return std_vals

def console_print(text: str) :
    def console_print_decorateur(func) :
        @wraps(func)
        def decorateur(self, *args, **kwargs) :
            with self.console.status(f"[bold green]{text}...") as status :
                res = func(self, *args, **kwargs)
                self.console.log(f"[green]Finish {text}[/green]")
            print("")
            return res
        return decorateur
    return console_print_decorateur

def log_metrics(
        logger:       SummaryWriter, 
        metrics_dict: dict, 
        step:         int, 
        dirname:      Optional[str]
):
    for key, value in metrics_dict.items():
        logger.add_scalar(f"{dirname}/{key}" if dirname else key, value, step)

def log_images(
        logger:     SummaryWriter, 
        image_dict: dict, 
        step:       int, 
        dirname:    Optional[str]
):
    for key, value in image_dict.items():
        logger.add_image(f"{dirname}/{key}" if dirname else key, value, step)
