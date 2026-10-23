#!/usr/bin/env python3
import argparse,torch
from catdiffusion.diffusion import DDPM
from catdiffusion.utils import save_grid
def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",default="outputs/forward_diffusion.png"); p.add_argument("--steps",type=int,default=100); p.add_argument("--config"); p.add_argument("--device",default="cpu"); a=p.parse_args()
    d=DDPM(a.steps); x=torch.zeros(1,3,32,32); ts=torch.linspace(0,a.steps-1,6).long(); save_grid(torch.cat([d.q_sample(x,torch.tensor([int(t)]))[0] for t in ts]),a.output,nrow=6)
if __name__=="__main__": main()
