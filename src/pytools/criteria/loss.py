from typing import Union
import torch
from torch import nn 

from ..utils.checks import type_check


@type_check
def reduce_loss(loss:torch.Tensor, reduction='mean') :
    """Perform reducing operation over the computed loss.

    Args:
        loss (torch.Tensor): given loss. Shape can be (), (B) or (B,C).
        reduction (str, optional): reduction to perform. 
            Defaults to 'mean'.

    Returns:
        torch.Tensor: reduced loss.
    """
    if loss.ndim==0 or reduction=='none' :
        return loss
    elif reduction=='mean' :
        return loss.mean(dim=0)
    elif reduction=='sum' :
        return loss.sum(dim=0)

class Loss(nn.Module):
    """Base class for losses.

    Args:
        reduction (str, optional): reduction method to perform. 
            Should be 'none', 'mean' or 'sum'.
            Defaults to 'mean'.
        device (Union[torch.device, str], optional): device on which to place
            the tensors. 
            Defaults to "cpu".
    """
    def __init__(
            self, 
            reduction :str                      = 'mean', 
            device    :Union[torch.device, str] = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.reduction = reduction
    
    def loss_fn(self, *args, **kwargs):
        """Loss function to use.
        Should be overrided in child class.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    def forward(self, *args, **kwargs):
        return reduce_loss(
            self.loss_fn(*args, **kwargs), reduction=self.reduction)