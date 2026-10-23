import torch
from catdiffusion.diffusion import DDPM
from catdiffusion.model import UNet
def test_training_and_sampling_smoke():
    torch.manual_seed(0); m=UNet(base_channels=4); d=DDPM(3); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    x=torch.randn(2,3,8,8); t=torch.randint(3,(2,)); xt,n=d.q_sample(x,t)
    loss=torch.nn.functional.mse_loss(m(xt,t),n); loss.backward(); opt.step()
    out=d.sample(m,(1,3,8,8),"cpu",seed=1); assert out.shape==(1,3,8,8)
