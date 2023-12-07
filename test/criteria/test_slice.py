import torch

from pytools.criteria import (
    slice_tensors, sliced_function, sliced_distance, sliced)

def test_slice_tensors():
    v = torch.randn(3)
    v = v / v.norm(2)

    # 4D tensors
    a = torch.randn((8,3,10,10))
    b = torch.randn((8,3,10,10))

    a_s, b_s = slice_tensors((a, b), v)
    print(a_s.shape, b_s.shape)
    a_s = slice_tensors(a, v)
    print(a_s.shape)

    # 3D tensors
    a = torch.randn((3,10,10))
    b = torch.randn((3,10,10))

    a_s, b_s = slice_tensors((a, b), v)
    print(a_s.shape, b_s.shape)
    a_s = slice_tensors(a, v)
    print(a_s.shape)

test_slice_tensors()
if __name__=="__main__":
    test_slice_tensors()


