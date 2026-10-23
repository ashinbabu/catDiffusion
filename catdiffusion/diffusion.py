"""Forward and reverse DDPM processes.

Python timestep ``i`` is zero based and represents mathematical timestep
``t=i+1``. The cosine schedule is recommended for short demonstrations.
"""
from __future__ import annotations
import math
from typing import Iterable, Optional
import torch
from torch import Tensor, nn

def cosine_betas(steps: int, s: float = 0.008) -> Tensor:
    if steps < 2: raise ValueError("steps must be at least 2")
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    abar = torch.cos(((x / steps + s) / (1 + s)) * math.pi / 2) ** 2
    return (1 - abar[1:] / abar[:-1]).clamp(1e-5, 0.999).float()

def linear_betas(steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> Tensor:
    if steps < 1 or not (0 < beta_start < beta_end < 1): raise ValueError("invalid linear schedule")
    return torch.linspace(beta_start, beta_end, steps)

class DDPM(nn.Module):
    """DDPM coefficients, q(x_t|x_0), and educational ancestral sampling."""
    def __init__(self, steps: int = 100, schedule: str = "cosine"):
        super().__init__()
        if schedule not in {"cosine", "linear"}: raise ValueError("schedule must be cosine or linear")
        betas = cosine_betas(steps) if schedule == "cosine" else linear_betas(steps)
        alphas = 1 - betas; abar = torch.cumprod(alphas, 0)
        abar_prev = torch.cat([torch.ones(1), abar[:-1]])
        posterior = (betas * (1 - abar_prev) / (1 - abar)).clamp_min(1e-20)
        self.steps, self.schedule = steps, schedule
        for name, value in {"betas": betas, "alphas": alphas, "alpha_bars": abar,
            "sqrt_alpha_bars": torch.sqrt(abar), "sqrt_one_minus_alpha_bars": torch.sqrt(1-abar),
            "sqrt_recip_alphas": torch.rsqrt(alphas), "posterior_variance": posterior}.items():
            self.register_buffer(name, value)

    @staticmethod
    def _extract(values: Tensor, timesteps: Tensor, shape: tuple[int, ...]) -> Tensor:
        if timesteps.ndim != 1 or timesteps.shape[0] != shape[0]: raise ValueError("timesteps must have shape (batch,)")
        return values.to(timesteps.device).gather(0, timesteps.long()).reshape((shape[0],) + (1,) * (len(shape)-1))

    def q_sample(self, x0: Tensor, timesteps: Tensor, noise: Optional[Tensor] = None):
        if x0.ndim != 4: raise ValueError("x0 must have shape (B,C,H,W)")
        t = timesteps.to(device=x0.device, dtype=torch.long)
        if torch.any((t < 0) | (t >= self.steps)): raise ValueError("timestep out of bounds")
        noise = torch.randn_like(x0) if noise is None else noise.to(x0)
        if noise.shape != x0.shape: raise ValueError("noise and x0 must have identical shapes")
        return (self._extract(self.sqrt_alpha_bars,t,x0.shape)*x0 +
                self._extract(self.sqrt_one_minus_alpha_bars,t,x0.shape)*noise, noise)

    @torch.no_grad()
    def p_sample(self, model: nn.Module, xt: Tensor, t: Tensor) -> Tensor:
        t=t.to(device=xt.device,dtype=torch.long)
        if xt.ndim != 4 or t.ndim != 1 or t.shape[0] != xt.shape[0] or torch.any((t<0)|(t>=self.steps)):
            raise ValueError("invalid sample inputs")
        eps=model(xt,t); beta=self._extract(self.betas,t,xt.shape)
        mean=self._extract(self.sqrt_recip_alphas,t,xt.shape)*(xt-beta*eps/self._extract(self.sqrt_one_minus_alpha_bars,t,xt.shape))
        var = self._extract(self.posterior_variance, t, xt.shape)
        if torch.all(t == 0):
            return mean
        noise = torch.randn_like(xt)
        mask = (t != 0).reshape(-1, 1, 1, 1)
        return mean + mask * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple[int,int,int,int], device, seed: Optional[int]=None,
               capture_steps: Optional[Iterable[int]]=None):
        if len(shape)!=4: raise ValueError("shape must be (B,C,H,W)")
        device=torch.device(device); g=torch.Generator(device=device)
        if seed is not None: g.manual_seed(seed)
        x=torch.randn(shape,device=device,generator=g); capture=set(capture_steps or ()); states={}
        training=model.training; model.eval()
        for i in reversed(range(self.steps)):
            t=torch.full((shape[0],),i,device=device,dtype=torch.long); eps=model(x,t)
            beta=self._extract(self.betas,t,x.shape); denom=self._extract(self.sqrt_one_minus_alpha_bars,t,x.shape)
            mean=self._extract(self.sqrt_recip_alphas,t,x.shape)*(x-beta*eps/denom)
            x=mean if i==0 else mean+torch.sqrt(self._extract(self.posterior_variance,t,x.shape))*torch.randn(shape,device=device,generator=g)
            if i in capture: states[i]=x.detach().cpu()
        if training: model.train()
        return (x,states) if capture_steps is not None else x

# Backwards-compatible descriptive name used in the educational documentation.
DiffusionSchedule = DDPM
