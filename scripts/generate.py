#!/usr/bin/env python3
import argparse
from pathlib import Path
from catdiffusion.sampling import generate
from catdiffusion.utils import resolve_device
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--count",type=int,default=16); p.add_argument("--device",default="auto"); p.add_argument("--seed",type=int); p.add_argument("--output",default="outputs/generated_cats.png"); p.add_argument("--output-dir")
    a=p.parse_args()
    if a.output_dir: a.output=str(Path(a.output_dir)/Path(a.output).name)
    if not Path(a.checkpoint).exists(): p.error(f"checkpoint not found: {a.checkpoint}")
    x=generate(a.checkpoint,a.count,resolve_device(a.device),a.seed,a.output); print(f"saved {a.output}: {tuple(x.shape)}")
if __name__=="__main__": main()
