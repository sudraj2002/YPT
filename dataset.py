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

class JsonlDataset_paired_test(Dataset):
    def __init__(self, jsonl_path, size = (480, 832), T_clip: int = 33, temporal_stride: int = 1,
                 verbose=False):
        print(f"Dataset test. Size: {size}")
        self.verbose = verbose
        self.verbose_first_time = True

        self.T_clip = T_clip
        self.H, self.W = size

        # Transforms
        self.tform = T.Compose([
            T.Resize((self.H, self.W), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])  # Return 0 to 1

        # Load and filter items
        self.items = json.load(open(jsonl_path))
        import random
        random.Random(42).shuffle(self.items)
        self.items = self.filter_paired(self.items)

        if self.verbose_first_time:
            print(f"[JsonlDataset] Loaded {len(self.items)} items after filtering from {jsonl_path}")

        # Build sample weights by dataset (for sampler)
        self.dataset_counts = Counter([item["dataset"] for item in self.items])
        self.temporal_stride = temporal_stride

        if self.verbose_first_time:
            print(f"[JsonlDataset] Dataset sample counts: {dict(self.dataset_counts)}")


    def filter_paired(self, items):
        filtered = []
        for line in items:
            dataset = line['dataset']
            if 'target_path' in line:
                if len(line['target_path']) != len(line['image_path']):
                    if len(line['target_path']) == 0:
                        line['target_path'] = line['image_path']
                    else:
                        print(len(line['target_path']), len(line['image_path']))
                        print(f"Skipping {line} as input GT lengths dont match")
                        continue

                filtered.append(line)
        self.dataset_name = dataset

        if self.verbose:
            print(f"[JsonlDataset] After filtering: {len(filtered)} items")
        return filtered

    def _pick_start(self, n_eff: int) -> int:
        # n_eff = usable length given stride
        if self.train:
            return random.randint(0, max(0, n_eff - self.T_clip))
        else:
            return max(0, (n_eff - self.T_clip)//2)

    def _sample_indices(self, n: int) -> List[int]:
        # 33 frames testing
        return list(range(min(n, self.T_clip)))

    def _load_clip(self, paths: List[str], idxs: List[int]) -> torch.Tensor:
        frames = []
        for k in idxs:
            path = paths[k]
            img = Image.open(path).convert("RGB")
            frames.append(self.tform(img))
        return torch.stack(frames, dim=0).permute(1,0,2,3).contiguous()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        rec = self.items[i]
        hazy_paths  = rec["image_path"]
        clean_paths = rec["target_path"]
        n = min(len(hazy_paths), len(clean_paths))

        idxs = self._sample_indices(n)

        deg = self._load_clip(hazy_paths,  idxs)  # [C,T,H,W]
        tgt = self._load_clip(clean_paths, idxs)  # [C,T,H,W]

        if self.verbose and self.verbose_first_time:
            print(f"[JsonlDataset] First item:\n  - image: {hazy_paths}\n")
            self.verbose_first_time = False

        return {
            "deg": deg, "tgt": tgt,
            "video_id": i,
            "paths": hazy_paths,
            "dataset": rec.get("dataset","UNK"),
            "degradation": rec.get("degradation",""),
            "meta": {"idxs": idxs}
        }