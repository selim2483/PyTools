import os
from pathlib import Path
from typing import Union

import PIL.Image


__all__ = ["file_ext", "is_image_ext", "generate_unique_name", 
    "generate_unique_path"]

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore

def generate_unique_name(root:str, name:str, ext:str) :
    i = 0
    while(True):
        unique_name = name + f"_{i:03}"
        if not os.path.exists(os.path.join(root, unique_name + "." + ext)) :
            return unique_name
        i += 1

def generate_unique_path(path:str) :
    root, tail = os.path.split(path)
    name, ext = tail.split(".")
    return os.path.join(
        root, generate_unique_name(root, name, ext) + "." + ext)
