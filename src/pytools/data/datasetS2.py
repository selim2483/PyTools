import os
from typing import Union
import numpy as np

import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from .mask_crop import get_mask_crop


def separate_paths(image_path:str) :
    for f in os.listdir(image_path) :
        if "MASK" in f :
            mask = f
        else :
            img = f

    return mask, img

def get_band_indexs(mode:str, nbands:int) :
    if mode=="grey" :
        return [2]
    elif mode=="rgb" :
        return [2, 3, 4]
    elif mode=="multispectral" :
        return list(range(1, nbands + 1))

class DatasetS2Determinist(Dataset) :

    def __init__(
            self, 
            root:str, 
            nbands:int=11, 
            height:int=256, 
            width:int=256, 
            mode="multispectral") :
        
        self.root = root
        self.img_names = os.listdir(self.root)
        self.nbands = nbands
        self.height, self.width = height, width
        self.band_indexs = get_band_indexs(mode, nbands)

    def __getitem__(self, index) :
        path = os.path.join(self.root, self.img_names[index])
        img = np.empty(
            (len(self.band_indexs), self.height, self.width), dtype="uint8")

        file = rasterio.open(
            path, 
            driver = "Gtiff", 
            dtype  = "uint8",
            count  = 11,
            width  = self.width,
            height = self.height
        )

        for i, band_index in enumerate(self.band_indexs) :
            img[i,:,:] = file.read(band_index)

        return F.convert_image_dtype(torch.tensor(img), torch.float) * 2 - 1
    
    def __len__(self) :
        return len(self.img_names)
    
class DatasetS2Random(Dataset) :

    def __init__(
            self, 
            root:str, 
            nbands:int=11, 
            height:int=256, 
            width:int=256, 
            mode:int="multispectral",
            s:int=0.8) :
        
        self.root = root
        self.img_names = os.listdir(self.root)
        self.nbands = nbands
        self.height, self.width = height, width
        self.band_indexs = get_band_indexs(mode, nbands)
        self.s = s

    def __getitem__(self, index):
        path = os.path.join(self.root, self.img_names[index])

        mask_path, img_path = separate_paths(path)
        mask = torch.load(os.path.join(path, mask_path), map_location='cpu')
        x, y = get_mask_crop(mask, self.width, self.height, self.s)
        
        img = torch.load(
            os.path.join(path, img_path), 
            map_location='cpu')[:, x:x + self.width, y:y + self.height]

        return img
    
    def __len__(self):
        return len(self.img_names)