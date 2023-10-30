import os
from pathlib import Path
from typing import Union

import PIL.Image


__all__ = ["file_ext", "is_image_ext", "generate_unique_name", 
    "generate_unique_path"]

def file_ext(name: Union[str, Path]) -> str:
    """Checks the extensions of a given path.

    Args:
        name (Union[str, Path]): Path to check.

    Returns:
        str: extension.
    """
    return str(name).split('.')[-1]

def generate_unique_name(root:str, name:str, ext:Union[str, None]):
    """Generates a unique name for a file/directory.

    Args:
        root (str): where to place the new file/directory.
        name (str): Wanted name.
        ext (Union[str, None]): extesnion.

    Returns:
        _type_: Unique name.
    """
    if ext is None:
        ext=""
    else:
        ext = "." + ext
    i = 0
    while(True):
        unique_name = name + f"_{i:03}" + ext
        if not os.path.exists(os.path.join(root, unique_name)):
            return unique_name
        i += 1

def generate_unique_path(path:str) :
    """Generates a unique file/directory path by adding numerotation to the
    wanted path.

    Args:
        path (str): Wanted path.

    Returns:
        _type_: Unique path
    """
    root, tail = os.path.split(path)
    try :
        name, ext = tail.split(".")
    except ValueError:
        name, ext = tail.split(".")[0], None
    return os.path.join(
        root, generate_unique_name(root, name, ext))

def is_image_ext(fname: Union[str, Path]) -> bool:
    """Checks if the path corresponds to an image path.

    Args:
        fname (Union[str, Path]): supposed image path.

    Returns:
        bool: Whether it is an image path or not.
    """
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore
