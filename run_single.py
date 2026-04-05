import os
import argparse

from contextlib import nullcontext

import torch
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from logger import redirect_prints_tee
from model_prompts import WANUNetDirect
from utils import (
    encode_frames,
    preview_decode_from_latent,
    save_video_frames,
    bridge_sample,
    EBRCustomBridge,
    save_images,
    save_video_mp4
)

import numpy as np
import random

import torch.nn.functional as F
from tqdm import tqdm

from pathlib import Path
import cv2
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def resize_chw(img: torch.Tensor, size_hw):
    # img: C,H,W
    if img.shape[-2:] == size_hw:
        return img
    img = img.unsqueeze(0)
    img = F.interpolate(img, size=size_hw, mode="bicubic", align_corners=False)
    return img.squeeze(0)


def load_image_rgb(path):
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img)   # C,H,W in [0,1]


def load_video_from_frame_folder(folder, max_frames=33):
    folder = Path(folder)
    frame_paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])
    if len(frame_paths) == 0:
        raise ValueError(f"No image frames found in folder: {folder}")

    frames = [load_image_rgb(str(p)) for p in frame_paths][:max_frames]  # list of C,H,W
    video = torch.stack(frames, dim=1)  # C,T,H,W
    return video, [str(p) for p in frame_paths]


def load_video_from_file(video_path, max_frames=33):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 12.0  # Default

    frames = []
    frame_names = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transforms.ToTensor()(frame)  # C,H,W
        frames.append(frame)
        frame_names.append(f"{Path(video_path).stem}_{idx:05d}.jpg")
        idx += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from video: {video_path}")

    video = torch.stack(frames, dim=1)  # C,T,H,W
    return video[:, :max_frames], frame_names[:max_frames], fps


class SingleInputDataset(Dataset):
    def __init__(self, input_path, input_type, max_frames=33):
        self.input_path = input_path
        self.input_type = input_type
        self.max_frames = max_frames

        if input_type == "image":
            self.size = (1024, 1024)   # H,W
        elif input_type == "video":
            self.size = (480, 832)     # H,W
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.input_type == "image":
            img = load_image_rgb(self.input_path)
            img = resize_chw(img, self.size)

            return {
                "deg": img,                     # C,H,W
                "paths": [self.input_path],
                "dataset": "single_image",
            }

        else:
            in_path = Path(self.input_path)

            if in_path.is_dir():
                vid, names = load_video_from_frame_folder(in_path, max_frames=self.max_frames)
                fps = 12.0  # Default
            elif in_path.suffix.lower() in VID_EXTS:
                vid, names, fps = load_video_from_file(in_path, max_frames=self.max_frames)
            else:
                raise ValueError(
                    f"For args.input_type video, input must be a folder of frames or a video file. Got: {in_path}"
                )

            # vid: C,T,H,W -> resize framewise
            vid = vid.permute(1, 0, 2, 3)  # T,C,H,W
            vid = torch.stack([resize_chw(frame, self.size) for frame in vid], dim=0)
            vid = vid.permute(1, 0, 2, 3).contiguous()  # C,T,H,W

            return {
                "deg": vid,
                "paths": names,
                "dataset": "single_video",
                "fps": fps
            }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, required=True, help="Huggingface model ID")
    p.add_argument("--backend", type=str, default="wan", choices=["wan", "flux"], help="Backbone")
    p.add_argument("--output_dir", type=str, default="./results", help="Result saving directory")
    p.add_argument("--resume_path", type=str, default=None, help="CKPT path")
    p.add_argument("--mixed_paths", type=str, nargs="+", default=None,
                   help="More than one path for mixed degradations")
    p.add_argument("--bridge_sampler_steps", type=int, default=40)
    p.add_argument("--t0", type=float, default=0.3, help="Max timestep for bridge")
    p.add_argument("--pow", type=float, default=1.0, help="Timestep skew for bridge")
    p.add_argument("--t_min", type=float, default=0.0001)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--single_input", type=str, default=None, help="Path to single file/folder")
    p.add_argument("--input_type", type=str, choices=["image", "video"], default=None,
                   help="Specify whether single input is an image or video")

    args = p.parse_args()
    args.output_dir = f"{args.output_dir}/{args.backend}"

    if args.backend == 'wan':
        deep_prompt_len = 226
    elif args.backend == 'flux':
        deep_prompt_len = 512
    else:
        raise ValueError("Invalid backbone")

    dataset_test = SingleInputDataset(
        input_path=args.single_input,
        input_type=args.input_type,
    )

    args.output_dir = f"{args.output_dir}/single/"

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    redirect_prints_tee(output_dir=args.output_dir, rank=0)  # Logs

    device = "cuda"
    mp_dtype = torch.float32
    if args.mixed_precision == "fp16":
        mp_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        mp_dtype = torch.bfloat16

    model = WANUNetDirect(
        model_id=args.model_id,
        backend=args.backend,
        dtype=mp_dtype,
        device=str(device),
        deep_prompt_len=deep_prompt_len,
    ).to(device)

    vae = model.vae
    scheduler = model.scheduler

    test_loader_img = DataLoader(dataset_test, batch_size=1, num_workers=4, shuffle=False)

    if args.mixed_paths is None:
        mixed_inference = False
        deg_ids = None
        saved = torch.load(args.resume_path, map_location="cpu", weights_only=False)
        saved_backend = saved.get("backend")
        if saved_backend is not None and saved_backend != model.backend_name:
            print(f"[WARN] checkpoint backend={saved_backend} but current backend={model.backend_name}")

        if "deep_prompt_bank" in saved:
            model.deep_prompt_bank.load_state_dict(saved["deep_prompt_bank"], strict=True)
            alpha_vals = saved.get("alphas", [])
            if not alpha_vals:
                raise ValueError(f"Alphas not found in checkpoint: {alpha_vals}")
            for wrap, alpha in zip(model._deep_prompt_wrappers, alpha_vals):
                wrap.alpha.data.copy_(alpha.to(wrap.alpha.device, dtype=wrap.alpha.dtype))
        else:
            raise ValueError(f"Invalid checkpoint save format")

    else:
        mixed_inference = True
        deg_ids = list(range(len(args.mixed_paths)))
        model.load_learned_prompts_from_paths(args.mixed_paths)
        print(f"Using mixed inference")

    print(
        f"TESTING; Model name: {args.backend}, Output directory: {args.output_dir}, checkpoint: {args.resume_path}")

    bridge = EBRCustomBridge(T0=args.t0, sigma_max=1.0, sigma_min=args.t_min, power=args.pow)

    # Keep VAE in fp32 for stability
    vae.to(device=device, dtype=torch.float32)

    def autocast_ctx():
        if args.mixed_precision == "no":
            return nullcontext()
        # "fp16" or "bf16"
        dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    print("\nTesting:")
    test(
        args=args,
        bridge=bridge,
        vae=vae,
        model=model,
        scheduler=scheduler,
        test_loader_img=test_loader_img,
        device=device,
        autocast_ctx=autocast_ctx,
        bridge_steps=args.bridge_sampler_steps,
        mixed_inference=mixed_inference,
        deg_ids=deg_ids,
        print_all=False,  # Print each sample metrics
    )

    print("Testing done.")


@torch.no_grad()
def test(args, bridge, vae, model, scheduler, test_loader_img, device="cuda",
         autocast_ctx=None, print_all=False, bridge_steps=None,
         mixed_inference=False, deg_ids=None,):
    autocast_ctx = autocast_ctx or (lambda: nullcontext())
    model.eval()
    vae.eval()
    model.set_prompt_enabled(True)

    global_step = 0
    num_test = len(test_loader_img)
    print(f"Testing {num_test} samples")

    pbar = tqdm(enumerate(test_loader_img), total=min(len(test_loader_img), 100), desc="Testing")

    for i, batch in pbar:
        if args.backend == 'flux':
            scheduler.set_timesteps(1000, mu=None, sigmas=None)
        else:
            scheduler.set_timesteps(1000)

        degraded = batch["deg"]
        if args.backend != "wan":
            if degraded.ndim == 5:
                # Select just first frame, input shape is B C T H W
                degraded = degraded.transpose(1, 2).contiguous()
                degraded = degraded[:, 0, :, :, :]

        degraded = degraded.to(device, non_blocking=True)
        degraded_in = degraded * 2.0 - 1.0

        with torch.no_grad():
            latents = encode_frames(vae, degraded_in)

            x = bridge_sample(bridge=bridge, model=model, scheduler=scheduler, autocast_ctx=autocast_ctx,
                              x_T=latents, steps=bridge_steps, clip_denoised=False, t0=args.t0, t_min=args.t_min,
                              mixed_inference=mixed_inference, deg_ids=deg_ids)

            vid = preview_decode_from_latent(vae, scheduler, x)

            paths = batch["paths"]
            if args.input_type == "video":
                fps = batch["fps"][0]
            if args.backend == 'wan':
                paths = [path[0] for path in paths]
                names = [paths[i].split('/')[-1].replace('.png', '.jpg') for i in range(len(paths))]
                save_video_frames(vid, folder=f"{args.output_dir}/pred", names=names)

                mp4_path = os.path.join(f"{args.output_dir}/pred", 'result.mp4')
                save_video_mp4(vid, mp4_path, fps=fps)
            else:
                paths = [path[0] for path in paths]
                names = [(paths[i].split('/')[-1].replace('.png', '.jpg')) for i in
                         range(len(paths))]
                save_images(vid, folder=f"{args.output_dir}/pred", name=names[0])

        global_step += 1

    print("Single inference completed.")


if __name__ == "__main__":
    main()
