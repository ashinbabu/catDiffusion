"""Oxford-IIIT Pet cat-only dataset and reproducible image transforms."""
from __future__ import annotations
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import Dataset

CAT_BREEDS=frozenset({"Abyssinian","Bengal","Birman","Bombay","British_Shorthair","Egyptian_Mau",
                      "Maine_Coon","Persian","Ragdoll","Russian_Blue","Siamese","Sphynx"})

class CatOnlyDataset(Dataset):
    """Wrap Oxford-IIIT Pet and retain only its explicitly named cat breeds."""
    def __init__(self,root: str|Path,split="trainval",image_size=32,download=True,augment=True,max_images=None):
        if split not in {"trainval","test"}: raise ValueError("split must be trainval or test")
        ops=[transforms.Resize(image_size),transforms.CenterCrop(image_size)]
        if augment and split=="trainval": ops.append(transforms.RandomHorizontalFlip())
        ops += [transforms.ToTensor(),transforms.Normalize((.5,)*3,(.5,)*3)]
        try: base=datasets.OxfordIIITPet(root=str(root),split=split,target_types="category",transform=transforms.Compose(ops),download=download)
        except Exception as exc: raise RuntimeError("Could not download Oxford-IIIT Pet; check network and permissions.") from exc
        classes=list(getattr(base,"classes",[]))
        if not classes: raise RuntimeError("torchvision dataset did not expose breed metadata")
        cats={i for i,name in enumerate(classes) if name in CAT_BREEDS}
        if not cats: raise RuntimeError("No cat breeds found in torchvision metadata")
        labels=getattr(base,"_labels",None)
        if labels is None: raise RuntimeError("Unsupported torchvision OxfordIIITPet: category labels unavailable")
        self.base=base; self.indices=[i for i,target in enumerate(labels) if target in cats]
        if max_images is not None: self.indices=self.indices[:max(0,max_images)]
    def __len__(self): return len(self.indices)
    def __getitem__(self,index): return self.base[self.indices[index]][0]

def make_transform(image_size:int,augment=False):
    ops=[transforms.Resize(image_size),transforms.CenterCrop(image_size)]
    if augment: ops.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(ops+[transforms.ToTensor(),transforms.Normalize((.5,)*3,(.5,)*3)])
