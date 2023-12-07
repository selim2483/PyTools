import torch


def image_gradient(im:torch.Tensor):
    dx, dy = im.diff(dim=-2)[...,:-1], im.diff(dim=-1)[...,:-1,:]
    mag = torch.sqrt(dx**2 + dy**2)
    return dx, dy, mag