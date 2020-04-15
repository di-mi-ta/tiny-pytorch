import torch 
from core.autograd import Value 

def test_sanity_check():
    # AutoDiff in tiny-pytorch
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x 
    y.backward()
    xtnpt, ytnpt = x, y

    # AutoDiff in official Pytorch 
    x = torch.Tensor([-4.0])
    x.requires_grad = True 
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y 

    # forward pass went well 
    assert ytnpt.data == ypt.data.item()
    # backward pass went well 
    assert xtnpt.data == xpt.data.item()
