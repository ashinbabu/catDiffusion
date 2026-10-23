import torch
from catdiffusion.model import UNet
def test_unet_shapes_and_backward():
    m=UNet(base_channels=4)
    for size in (32,64):
        x=torch.randn(2,3,size,size,requires_grad=True); t=torch.tensor([0,5])
        y=m(x,t); assert y.shape==x.shape and torch.isfinite(y).all()
        y.square().mean().backward()
