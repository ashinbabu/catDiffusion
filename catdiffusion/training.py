"""Checkpointable DDPM training loop."""
from pathlib import Path
import torch
from torch.nn import functional as F
from tqdm import tqdm
from .utils import save_loss_csv
def train(model,diffusion,loader,optimizer,device,epochs=1,max_steps=None,checkpoint_path=None,model_config=None,diffusion_config=None,output_dir=None):
    model.to(device); diffusion.to(device); rows=[]; step=0
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(loader,desc=f"epoch {epoch+1}/{epochs}"):
            x0=(batch[0] if isinstance(batch,(tuple,list)) else batch).to(device); t=torch.randint(diffusion.steps,(x0.shape[0],),device=device)
            xt,noise=diffusion.q_sample(x0,t); loss=F.mse_loss(model(xt,t),noise)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); step+=1; rows.append((step,float(loss.detach())))
            if max_steps and step>=max_steps: break
        if max_steps and step>=max_steps: break
    if output_dir: save_loss_csv(rows,Path(output_dir)/"loss.csv")
    if checkpoint_path: save_checkpoint(checkpoint_path,model,optimizer,epoch,step,model_config,diffusion_config)
    return rows
def train_step(model,diffusion,x0,optimizer,device):
    """Run one optimizer update; useful for notebooks and smoke tests."""
    model.train(); x0=x0.to(device); t=torch.randint(diffusion.steps,(x0.shape[0],),device=device)
    xt,noise=diffusion.q_sample(x0,t); loss=F.mse_loss(model(xt,t),noise)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return float(loss.detach())
def save_checkpoint(path,model,optimizer=None,epoch=0,step=0,model_config=None,diffusion_config=None):
    data={"model":model.state_dict(),"model_config":model_config or {},"diffusion_config":diffusion_config or {},
          "normalization":"x in [-1,1] from torchvision Normalize(0.5, 0.5)",
          "epoch":epoch,"step":step}
    if optimizer is not None: data["optimizer"]=optimizer.state_dict()
    Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(data,path)
def load_checkpoint(path,device="cpu"):
    try: return torch.load(path,map_location=device,weights_only=True)
    except TypeError: return torch.load(path,map_location=device)
