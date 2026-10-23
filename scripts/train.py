#!/usr/bin/env python3
import argparse,yaml,torch
from torch.utils.data import DataLoader
from catdiffusion.dataset import CatOnlyDataset
from catdiffusion.diffusion import DDPM
from catdiffusion.model import UNet
from catdiffusion.training import train,load_checkpoint
from catdiffusion.utils import resolve_device,seed_everything
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/cpu.yaml"); p.add_argument("--data-root",default="data")
    p.add_argument("--device",default="auto"); p.add_argument("--epochs",type=int); p.add_argument("--batch-size",type=int)
    p.add_argument("--max-images",type=int); p.add_argument("--max-steps",type=int); p.add_argument("--image-size",type=int)
    p.add_argument("--seed",type=int); p.add_argument("--output-dir",default="outputs"); p.add_argument("--checkpoint",default="checkpoints/model.pt")
    p.add_argument("--resume",help="resume from a compatible checkpoint"); p.add_argument("--dry-run",action="store_true"); a=p.parse_args()
    with open(a.config) as f: c=yaml.safe_load(f)
    for k,v in {"epochs":a.epochs,"batch_size":a.batch_size,"max_images":a.max_images,"max_steps":a.max_steps,"image_size":a.image_size,"seed":a.seed}.items():
        if v is not None: c[k]=v
    if a.dry_run: c["max_steps"]=1
    seed_everything(c["seed"]); device=resolve_device(a.device); ds=CatOnlyDataset(a.data_root,image_size=c["image_size"],max_images=c.get("max_images"))
    loader=DataLoader(ds,batch_size=c["batch_size"],shuffle=True,num_workers=c.get("num_workers",0))
    mc={"in_channels":3,"out_channels":3,"base_channels":c["base_channels"],"image_size":c["image_size"]}; dc={"steps":c["diffusion_steps"],"schedule":c["schedule"]}
    model=UNet(**mc); d=DDPM(**dc); opt=torch.optim.AdamW(model.parameters(),lr=c["learning_rate"])
    if a.resume:
        payload=load_checkpoint(a.resume,device)
        if payload.get("model_config") != mc or payload.get("diffusion_config") != dc:
            raise ValueError("resume checkpoint configuration does not match the requested model/schedule")
        model.load_state_dict(payload["model"])
        if "optimizer" in payload: opt.load_state_dict(payload["optimizer"])
    train(model,d,loader,opt,device,epochs=c["epochs"],max_steps=c.get("max_steps"),checkpoint_path=a.checkpoint,model_config=mc,diffusion_config=dc,output_dir=a.output_dir)
if __name__=="__main__": main()
