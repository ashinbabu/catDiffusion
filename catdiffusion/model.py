"""A compact timestep-conditioned U-Net for 32 or 64 pixel RGB images."""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int): super().__init__(); self.dim=dim
    def forward(self,t: Tensor)->Tensor:
        half=self.dim//2; freq=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/max(half-1,1))
        x=t.float()[:,None]*freq[None,:]; return torch.cat([x.sin(),x.cos()],1)

class ResBlock(nn.Module):
    def __init__(self,channels:int,emb_dim:int):
        super().__init__(); self.n1=nn.GroupNorm(min(8,channels),channels); self.c1=nn.Conv2d(channels,channels,3,padding=1)
        self.e=nn.Linear(emb_dim,channels); self.n2=nn.GroupNorm(min(8,channels),channels); self.c2=nn.Conv2d(channels,channels,3,padding=1)
    def forward(self,x,e):
        h=self.c1(F.silu(self.n1(x))); h=h+self.e(F.silu(e))[:,:,None,None]; return x+self.c2(F.silu(self.n2(h)))

class UNet(nn.Module):
    """Encoder, bottleneck, and decoder predicting RGB noise."""
    def __init__(self,in_channels=3,out_channels=3,base_channels=16,time_dim=None,image_size=None):
        super().__init__(); time_dim=time_dim or base_channels*4; c1=base_channels; c2=base_channels*2
        self.time=nn.Sequential(SinusoidalEmbedding(time_dim),nn.Linear(time_dim,time_dim),nn.SiLU(),nn.Linear(time_dim,time_dim))
        self.inc=nn.Conv2d(in_channels,c1,3,padding=1); self.down=nn.Conv2d(c1,c2,4,2,1)
        self.e1=ResBlock(c1,time_dim); self.e2=ResBlock(c2,time_dim); self.mid=ResBlock(c2,time_dim)
        self.up=nn.ConvTranspose2d(c2,c1,4,2,1); self.dec=ResBlock(c1,time_dim); self.out=nn.Conv2d(c1,out_channels,3,padding=1)
    def forward(self,x:Tensor,t:Tensor)->Tensor:
        if x.ndim!=4 or t.ndim!=1 or x.shape[0]!=t.shape[0]: raise ValueError("expected x=(B,C,H,W), t=(B,)")
        e=self.time(t); h1=self.e1(self.inc(x),e); h2=self.e2(self.down(h1),e); h=self.mid(h2,e)
        return self.out(self.dec(self.up(h)+h1,e))
