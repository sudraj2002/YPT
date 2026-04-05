import json
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from collections import Counter, defaultdict
import torch.distributed as dist
from typing import *
import torch

def get_rank_safe():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class JsonlDataset_paired_image_test(Dataset):
    def __init__(self, jsonl_path, size = (480, 832), verbose=False):
        self.verbose = verbose
        self.verbose_first_time = True

        self.H, self.W = size

        # Transforms
        self.tform = T.Compose([
            T.Resize((self.H, self.W), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])  # Return 0 to 1

        # Load and filter items
        self.items = json.load(open(jsonl_path))
        self.items = self.filter_paired(self.items)
        if self.verbose_first_time:
            print(f"[JsonlDataset] Loaded {len(self.items)} items after filtering from {jsonl_path}")

        # Build sample weights by dataset (for sampler)
        self.dataset_counts = Counter([item["dataset"] for item in self.items])

        self.train = False

        if self.verbose_first_time:
            print(f"[JsonlDataset] Dataset sample counts: {dict(self.dataset_counts)}")


    def filter_paired(self, items):
        filtered = []
        for line in items:
            filtered.append(line)
        if self.verbose:
            print(f"[JsonlDataset] After filtering: {len(filtered)} items")
        self.dataset_name = line['dataset']

        return filtered

    def _load_clip(self, path) -> torch.Tensor:
        # Image dataset, but for consistency with other functions we just treat it as one frame of a video
        frames = []
        img = Image.open(path).convert("RGB")
        frames.append(self.tform(img))
        return torch.stack(frames, dim=0).permute(1,0,2,3).contiguous()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        rec = self.items[i]
        hazy_paths  = rec["image_path"]
        clean_paths = rec["target_path"]

        deg = self._load_clip(hazy_paths)  # [C,T,H,W]
        tgt = self._load_clip(clean_paths)  # [C,T,H,W]

        if self.verbose and self.verbose_first_time:
            print(f"[JsonlDataset] First item:\n  - image: {hazy_paths}\n")
            self.verbose_first_time = False

        return {
            "deg": deg, "tgt": tgt,
            "video_id": i,
            "dataset": rec.get("dataset","UNK"),
            "degradation": rec.get("degradation",""),
            "paths": hazy_paths
        }

