import torch

__all__ = ["convert_corner", "get_mask_crop"]

def convert_corner(x:int, y:int, width:int, height:int, new_corner="ul") :
    """Converts upper left corner coordinates into new wanted corner
    coordinates for a given mask.

    Args:
        x (int): upper left corner first coordinate.
        y (int): upper left corner second coordinate.
        width (int): 
        height (int): 
        new_corner (str): new wanted corner. 
            Must be :
                - ``ul`` : upper left, 
                - ``ur`` : upper right, 
                - ``bl`` : bottom left,
                - ``br`` : bottom right.
            default to 'ul'.
    """
    if new_corner=='ul' :
        return x, y
    elif new_corner=='ur' :
        return x + width, y
    elif new_corner=='bl' :
        return x, y + height
    elif new_corner=='br' :
        return x + width, y + height
    else :
        raise ValueError(
            "'new_corner' argument shoul be either 'ul', 'ur', 'bl', or 'br'")

def get_mask_crop(
        mask       :torch.Tensor, 
        width      :int, 
        height     :int, 
        s          :float, 
        new_corner :str="ul", 
        **kwargs   :dict
    ):
    """Generate random crop satisfying a given treshold coordinates given an
    image and a mask. 
    The function returns the coordinates of the upper left corner of the crop. 
    If there are no crop satisfying the treshold in the image, the treshold
    value is replaced with a quantile that can be specified with "q" keyarg.
    The coordinates of another crop "x0" and "y0"  could be specified in
    kwargs if the crop needs to overlap the other crop.

    Args:
        mask (torch.Tensor): mask tensor
        width (int): desired width of the crop.
        height (int): desired height of the crop.
        s (float): 

    Returns:
        _type_: _description_
    """
    imgw, imgh = mask.shape
    minx = max(0, kwargs.get("x0", 0) - width)
    maxx = min(imgw - width, kwargs.get("x0", imgw - 2 * width) + width)
    miny = max(0, kwargs.get("y0", 0) - height)
    maxy = min(imgh - height, kwargs.get("y0", imgh - 2 * height) + height)
    indices = (mask[minx:maxx, miny:maxy] >= s).nonzero()
    
    if len(indices)==0 :
        quant = mask[minx:maxx, miny:maxy].quantile(kwargs.get("q", 0.95))
        indices = (mask[minx:maxx, miny:maxy] >= quant).nonzero()
    idx = indices[torch.randint(len(indices), (1,))]

    return convert_corner(
        x          = int(idx[0,0] + minx), 
        y          = int(idx[0,1] + miny),
        width      = width, 
        height     = height, 
        new_corner = new_corner
    )