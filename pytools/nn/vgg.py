from typing import Union

import torch
import torchvision

from ..options import VGGOptions
from ..utils.misc import unsqueeze_squeeze


MEAN = (0.485, 0.456, 0.406)

@unsqueeze_squeeze()
def prep(x:torch.Tensor):
    if x.shape[1]==3 :
        mean = torch.as_tensor(MEAN).view(-1, 1, 1).to(x.device)
    else :
        mean = torch.as_tensor([0.5]*x.shape[1]).view(-1, 1, 1).to(x.device)
    x = x / 2 + .5
    return x.sub_(mean).mul_(255)

def initialize_vgg(
    vgg_options: VGGOptions, 
    nbands:      int                      = 3, 
    device:      Union[torch.device, str] = "cpu"
):
    """Instantiate a VGG19 network and load pretrained wheigts.
    The model uses a hook to retrieve the wanted feature maps, the latter
    are accessible after a call to ``vgg`` in ``outputs``.
    """
    print(vgg_options)
    if vgg_options.mode == "rgb":
        vgg = torchvision.models.vgg19().features.to(device)
    else:
        vgg = VGGMS(nbands, vgg_options.batchnorm)
    
    if vgg_options.mode != "raw":
        pretrained_dict = torch.load(vgg_options.path)
        for param, item in zip(vgg.parameters(), pretrained_dict.keys()):
            if vgg_options.mode == "rgb" or "conv1_1" not in item:
                param.data = pretrained_dict[item].type(
                        torch.FloatTensor).to(device)
                if vgg_options.mode in ["rgb", "freeze"]:
                    param.requires_grad = False 

    outputs = {}
    def save_output(name):
        def hook(module, module_in, module_out):
            outputs[name] = module_out
        return hook
    
    for layer in vgg_options.layers:
        handle = vgg[layer].register_forward_hook(save_output(layer))

    def get_features(x:torch.Tensor):
        vgg(prep(x.to(device)))
        return [outputs[key] for key in vgg_options.layers]

    return vgg, outputs, get_features

class VGGBlock(torch.nn.Module) :

    def __init__(
            self,
            inchannels:int, 
            outchannels:int, 
            nlayers:int, 
            batchnorm:bool=False):
        super().__init__()

        # Body
        modules = [
            torch.nn.Conv2d(
                inchannels, outchannels, kernel_size=3, stride=1, padding=1)
        ]
        if batchnorm :
            modules += [torch.nn.BatchNorm2d(outchannels)]
        modules += [torch.nn.ReLU()]
        for i in range(nlayers - 1) :
            modules += [
                torch.nn.Conv2d(
                    outchannels, 
                    outchannels, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1)]
            if batchnorm :
                modules += [torch.nn.BatchNorm2d(outchannels)]
            modules += [torch.nn.ReLU(inplace=True)]

        self.body = torch.nn.Sequential(*modules)

        # Pool
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x:torch.Tensor) :
        x = self.body(x)
        x = self.pool(x)
        return x
    
    def __getitem__(self, index) :
        if index == len(self) - 1 :
            return self.pool
        else :
            return self.body[index]
    
    def __len__(self) :
        return len(self.body) + 1 


class VGGMS(torch.nn.Module) :

    def __init__(self, nbands: int, batchnorm: bool = False):
        super().__init__()

        self.blocks = torch.nn.Sequential(
            VGGBlock(nbands, 64, 2, batchnorm=batchnorm),
            VGGBlock(64, 128, 2, batchnorm=batchnorm),
            VGGBlock(128, 256, 4, batchnorm=batchnorm),
            VGGBlock(256, 512, 4, batchnorm=batchnorm),
            VGGBlock(512, 512, 4, batchnorm=batchnorm)
        )

    def forward(self, x:torch.Tensor) :
        return self.blocks(x)
    
    def __getitem__(self, index) :
        block_index = 0
        count = 0
        while block_index < len(self.blocks._modules) :
            if index - count < len(self.blocks[block_index]) :
                return self.blocks[block_index][index - count]
            else :
                count += len(self.blocks[block_index])
                block_index += 1

    def __len__(self) :
        return sum([len(block) for block in self.blocks._modules.values()])
