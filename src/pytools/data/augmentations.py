import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from ..utils.checks import type_check

@type_check
def augment(
        real:torch.Tensor, 
        scale:torch.Tensor, 
        theta:torch.Tensor, 
        resolution:int
    ) :
    if real.dim()==3:
        real = real.unsqueeze(0)
    aff = torch.stack(
        (torch.stack(
            (torch.cos(theta) * scale, -torch.sin(theta) * scale, 0 * scale),
            dim=1),
        torch.stack(
            (torch.sin(theta) * scale, torch.cos(theta) * scale, 0 * scale),
            dim=1)
        ),
        dim=1)
    
    grid = F.affine_grid(aff, real.size(), align_corners=False)
    t_real = F.grid_sample(real, grid, align_corners=False)
    t_real = TF.center_crop(t_real, resolution)
    return t_real