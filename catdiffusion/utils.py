"""Seeding, image grids, configuration, and checkpoint helpers."""
from pathlib import Path
import csv, random
import numpy as np
import torch
from torchvision.utils import save_image
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def save_grid(images,path,nrow=4):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); save_image((images.detach().cpu().clamp(-1,1)+1)/2,path,nrow=nrow)
def save_loss_csv(rows,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f: w=csv.writer(f); w.writerow(["step","loss"]); w.writerows(rows)
def resolve_device(value):
    if value=="auto": value="cuda" if torch.cuda.is_available() else "cpu"
    if value=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    return torch.device(value)
