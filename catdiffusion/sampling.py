"""Checkpoint reconstruction and sample image output."""
from pathlib import Path
import torch
from .diffusion import DDPM
from .model import UNet
from .training import load_checkpoint
from .utils import save_grid
def load_model(path,device="cpu"):
    payload=load_checkpoint(path,device); model=UNet(**payload["model_config"]).to(device); diffusion=DDPM(**payload["diffusion_config"]).to(device)
    model.load_state_dict(payload["model"]); model.eval(); return model,diffusion,payload
def generate(path,count=16,device="cpu",seed=None,output="outputs/generated_cats.png"):
    device=torch.device(device); model,diffusion,payload=load_model(path,device); size=payload["model_config"].get("image_size",32)
    samples=diffusion.sample(model,(count,3,size,size),device,seed=seed); save_grid(samples,output); return samples
