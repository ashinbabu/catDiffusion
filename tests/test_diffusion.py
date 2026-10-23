import torch
from catdiffusion.diffusion import DDPM

def test_schedule_and_q_sample():
    d=DDPM(100)
    assert torch.all((d.betas>0)&(d.betas<1))
    assert torch.all(d.alpha_bars[1:] <= d.alpha_bars[:-1])
    assert float(d.alpha_bars[-1]) < 1e-3
    x=torch.ones(2,3,4,4); n=torch.full_like(x,.25); y,_=d.q_sample(x,torch.tensor([0,99]),n)
    assert torch.allclose(y[0],d.sqrt_alpha_bars[0]*x[0]+d.sqrt_one_minus_alpha_bars[0]*n[0])

class Zero(torch.nn.Module):
    def forward(self,x,t): return torch.zeros_like(x)

def test_sample_shape_and_seed():
    d=DDPM(4); m=Zero()
    a=d.sample(m,(2,3,8,8),"cpu",seed=3); b=d.sample(m,(2,3,8,8),"cpu",seed=3)
    assert a.shape==(2,3,8,8) and torch.isfinite(a).all() and torch.equal(a,b)
