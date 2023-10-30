from functools import wraps
from typing import Iterable

import torch

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
def aggregate_loss_dict(agg_loss_dict:Iterable[dict], list_output:bool=False) :
    mean_vals = concatenate_loss_dict(agg_loss_dict, list_output=list_output)
    for key in mean_vals:
        if len(mean_vals[key]) > 0:
            mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
        else:
            print('{} has no value'.format(key))
            mean_vals[key] = 0
    return mean_vals

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