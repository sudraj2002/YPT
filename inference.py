import argparse

from contextlib import nullcontext

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from dataset import JsonlDataset_paired_test
from dataset_images import JsonlDataset_paired_image_test
from logger import redirect_prints_tee
from model_prompts import WANUNetDirect
from utils import (
    encode_frames,
    preview_decode_from_latent,
    save_video_frames,
    bridge_sample,
    EBRCustomBridge,
    save_images
)

import numpy as np
import random
import pyiqa
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from metric_utils.val_utils import compute_psnr_ssim, AverageMeter

import torch.nn.functional as F
from tqdm import tqdm


def resize_video_for_metrics(x: torch.Tensor, size: int) -> torch.Tensor:
    # Reshape to (size x size) for fair evaluation
    # Case 0: [B, C, H, W]
    if x.ndim == 4:
        if x.shape[-2] == size and x.shape[-1] == size:
            return x
        return F.interpolate(x, size=(size, size), mode="bicubic")

    if x.ndim != 5:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)}")

    # Case 1: [B, T, C, H, W]
    if x.shape[2] in (1, 3) and x.shape[1] != 1 and x.shape[1] != 3:
        B, T, C, H, W = x.shape
        if H == size and W == size:
            return x
        xt = x.reshape(B * T, C, H, W)
        yt = F.interpolate(xt, size=(size, size), mode="bicubic")
        return yt.reshape(B, T, C, size, size)

    # Case 2: [B, C, T, H, W]
    B, C, T, H, W = x.shape
    if H == size and W == size:
        return x
    xt = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [B*T, C, H, W]
    yt = F.interpolate(xt, size=(size, size), mode="bicubic")
    y = yt.reshape(B, T, C, size, size).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
    return y


def _to_bcthw(x: torch.Tensor) -> torch.Tensor:
    """
    Make tensor BxCxTxHxW.
    Accepts: BCHW, BCTHW, BTCHW.
    """
    if x.ndim == 4:
        # BCHW -> BCTHW with T=1 (FLUX)
        return x.unsqueeze(2)
    if x.ndim == 5:
        # Could be BCTHW or BTCHW
        if x.shape[1] in (1, 3):
            return x

        if x.shape[2] in (1, 3):
            return x.permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(f"Unsupported shape for video/image tensor: {tuple(x.shape)}")


def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)

def _safe_mean(vals):
    return vals.avg


class MetricComputer:
    def __init__(self, device: torch.device):
        self.device = device

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()

        # pyiqa: CLIPIQA+, MUSIQ, DISTS
        self.iqa = {}
        for name in ["clipiqa+", "musiq", "dists"]:
            try:
                self.iqa[name] = pyiqa.create_metric(name, device=device).eval()
            except Exception as e:
                print(f"[WARN] Could not create pyiqa metric '{name}': {e}")
                self.iqa[name] = None

    @torch.no_grad()
    def iqa_nr(self, name: str, pred: torch.Tensor) -> float:
        """
        No-reference IQA on BCHW in [0,1]
        """
        m = self.iqa.get(name, None)
        if m is None:
            return float("nan")
        v = m(pred)
        if isinstance(v, torch.Tensor):
            return float(v.mean().item())
        return float(v)

    @torch.no_grad()
    def iqa_fr(self, name: str, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """
        Full-reference IQA on BCHW in [0,1]
        """
        m = self.iqa.get(name, None)
        if m is None:
            return float("nan")
        v = m(pred, gt)
        if isinstance(v, torch.Tensor):
            return float(v.mean().item())
        return float(v)

    @torch.no_grad()
    def compute_video(self, pred_bcthw: torch.Tensor, gt_bcthw: torch.Tensor):
        """
        pred/gt: BCTHW in [0,1]. Average metrics over T.
        """
        pred_bcthw = _clamp01(_to_bcthw(pred_bcthw)).to(self.device, dtype=torch.float32)
        gt_bcthw = _clamp01(_to_bcthw(gt_bcthw)).to(self.device, dtype=torch.float32)

        B, C, T, H, W = pred_bcthw.shape
        psnr_list = AverageMeter()
        ssim_list = AverageMeter()
        lpips_list = AverageMeter()
        dists_list = AverageMeter()
        clipiqa_list = AverageMeter()
        musiq_list = AverageMeter()

        for t in range(T):
            pred = pred_bcthw[:, :, t]
            gt = gt_bcthw[:, :, t]

            restored_lpips = 2 * pred - 1
            clean_lpips = 2 * gt - 1

            temp_psnr, temp_ssim, N = compute_psnr_ssim(pred, gt)
            psnr_list.update(temp_psnr, N)
            ssim_list.update(temp_ssim, N)
            lpips_list.update(self.lpips(restored_lpips, clean_lpips), N)
            dists_list.update(self.iqa_fr("dists", pred, gt), N)

            clipiqa_list.update(self.iqa_nr("clipiqa+", pred), N)
            musiq_list.update(self.iqa_nr("musiq", pred), N)

        out = {
            "psnr": _safe_mean(psnr_list),
            "ssim": _safe_mean(ssim_list),
            "lpips": _safe_mean(lpips_list),
            "dists": _safe_mean(dists_list),
            "clipiqa_plus": _safe_mean(clipiqa_list),
            "musiq": _safe_mean(musiq_list),
        }
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, required=True, help="Huggingface model ID")
    p.add_argument("--backend", type=str, default="wan", choices=["wan", "flux"], help="Backbone")
    p.add_argument("--output_dir", type=str, default="./results", help="Result saving directory")
    p.add_argument("--json_path", type=str, default="JSON path for metadata")
    p.add_argument("--resume_path", type=str, default=None, help="CKPT path")
    p.add_argument("--mixed_paths", type=str, nargs="+", default=None,
                   help="More than one path for mixed degradations")
    p.add_argument("--bridge_sampler_steps", type=int, default=40)
    p.add_argument("--t0", type=float, default=0.3, help="Max timestep for bridge")
    p.add_argument("--pow", type=float, default=1.0, help="Timestep skew for bridge")
    p.add_argument("--t_min", type=float, default=0.0001)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=123)

    args = p.parse_args()
    args.output_dir = f"{args.output_dir}/{args.backend}"
    json_path = args.json_path

    if args.backend == 'wan':
        deep_prompt_len = 226
        im_size = (480, 832) # Video size for WAN inference
        dataset_test = JsonlDataset_paired_test(json_path, size=im_size)
    elif args.backend == 'flux':
        deep_prompt_len = 512
        im_size = (1024, 1024)  # Image size for FLUX inference
        dataset_test = JsonlDataset_paired_image_test(json_path, size=im_size)
    else:
        raise ValueError("Invalid backbone")

    # Add dataset name
    args.output_dir = f"{args.output_dir}/{dataset_test.dataset_name}"

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
    print(f"Testing {num_test} samples from {test_loader_img.dataset.dataset_name}")

    total_loss = 0.0

    metric_comp = MetricComputer(device=torch.device(device))

    agg = {
        "psnr": AverageMeter(),
        "ssim": AverageMeter(),
        "lpips": AverageMeter(),
        "dists": AverageMeter(),
        "clipiqa_plus": AverageMeter(),
        "musiq": AverageMeter(),
        "mse_latent": AverageMeter(),
    }

    # Inference takes long time so limit large datasets, change as needed
    max_samples = 2001
    pbar = tqdm(enumerate(test_loader_img), total=min(len(test_loader_img), max_samples), desc="Testing")

    for i, batch in pbar:
    # for i, batch in enumerate(test_loader_img):
        if args.backend == 'flux':
            scheduler.set_timesteps(1000, mu=None, sigmas=None)
        else:
            scheduler.set_timesteps(1000)

        degraded, target = batch["deg"], batch["tgt"]
        img_path = batch["paths"][0]
        if 'SICE' in img_path:
            deg_name = img_path.split('/')[-1]
            if int(deg_name.split('.')[0]) > 3:
                continue
        if args.backend != "wan":
            if degraded.ndim == 5:
                # Select just first frame, input shape is B C T H W
                degraded = degraded.transpose(1, 2).contiguous()
                target = target.transpose(1, 2).contiguous()
                degraded = degraded[:, 0, :, :, :]
                target = target[:, 0, :, :, :]

        target = target.to(device, non_blocking=True)
        degraded = degraded.to(device, non_blocking=True)

        target_in = target * 2.0 - 1.0
        degraded_in = degraded * 2.0 - 1.0

        with torch.no_grad():
            latents = encode_frames(vae, degraded_in)
            latents_tgt = encode_frames(vae, target_in)

            x = bridge_sample(bridge=bridge, model=model, scheduler=scheduler, autocast_ctx=autocast_ctx,
                              x_T=latents, steps=bridge_steps, clip_denoised=False, t0=args.t0, t_min=args.t_min,
                              mixed_inference=mixed_inference, deg_ids=deg_ids)

            loss = F.mse_loss(x.float(), latents_tgt.float(), reduction="mean")
            total_loss += loss.item()
            if print_all:
                print(f"Test: sample {i + 1} | loss {loss.item():.6f}")

            vid = preview_decode_from_latent(vae, scheduler, x)
            vid_gt = target_in

            vid_metric = (vid + 1.0) * 0.5
            vid_gt_metric = (vid_gt + 1.0) * 0.5

            vid_metric = resize_video_for_metrics(vid_metric, size=512)
            vid_gt_metric = resize_video_for_metrics(vid_gt_metric, size=512)

            # Compute metrics on decoded outputs (expects RGB in [0,1])
            # Make sure shapes are BCTHW (helpers handle BCHW too)
            metrics = metric_comp.compute_video(pred_bcthw=vid_metric, gt_bcthw=vid_gt_metric)
            metrics["mse_latent"] = float(loss.item())

            for k in agg.keys():
                v = metrics.get(k, float("nan"))
                agg[k].update(v, 1)

            if print_all:
                # Print
                print(
                    " | ".join([
                        f"PSNR {metrics['psnr']:.3f}",
                        f"SSIM {metrics['ssim']:.4f}",
                        f"LPIPS {metrics['lpips']:.4f}",
                        f"DISTS {metrics['dists']:.4f}",
                        f"CLIPIQA+ {metrics['clipiqa_plus']:.3f}",
                        f"MUSIQ {metrics['musiq']:.3f}",
                    ])
                )

            paths = batch["paths"]
            if args.backend == 'wan':
                paths = [path[0] for path in paths]
                subdir = paths[0].split('/')[-2] if batch['dataset'][0] != 'GoPro' else paths[0].split('/')[-3]
                names = [paths[i].split('/')[-1].replace('.png', '.jpg') for i in range(len(paths))]
                save_video_frames(vid, folder=f"{args.output_dir}/{subdir}/pred", names=names)
            else:
                if batch['dataset'][0] != 'GoPro' and batch['dataset'][0] != 'SICE':
                    names = [paths[i].split('/')[-1].replace('.png', '.jpg') for i in range(len(paths))]
                elif batch['dataset'][0] == 'SICE':
                    subdir = paths[0].split('/')[-2]
                    names = [(subdir + '_' + paths[i].split('/')[-1]).replace('.png', '.jpg') for i in
                             range(len(paths))]
                else:
                    subdir = paths[0].split('/')[-3]
                    names = [(subdir + '_' + paths[i].split('/')[-1].replace('.png', '.jpg')) for i in
                             range(len(paths))]
                save_images(vid, folder=f"{args.output_dir}/pred", name=names[0])

        global_step += 1

    print("\n=== Test Summary (mean over samples) ===")
    for k in ["psnr", "ssim", "lpips", "dists", "clipiqa_plus", "musiq", "mse_latent"]:
        m = agg[k].avg
        print(f"{k:16s}: {m:.4f}")

    print()


if __name__ == "__main__":
    main()
