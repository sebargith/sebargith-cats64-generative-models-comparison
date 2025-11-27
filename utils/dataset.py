# utils/dataset.py
from pathlib import Path
from typing import List

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CatDataset64(Dataset):
    def __init__(self, list_file: str, train: bool = True):
        self.paths: List[Path] = []
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.paths.append(Path(line))

        # Transformaciones
        t = []
        if train:
            t.append(transforms.RandomHorizontalFlip(p=0.5))
        t.extend([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # Normalizamos a [-1, 1] (GANs con tanh)
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
        self.transform = transforms.Compose(t)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img
