import torch

from ..utils.checks import type_check

@type_check
def num_parameters(model:torch.nn.Module) :
    return sum(p.numel() for p in model.parameters() if p.requires_grad)