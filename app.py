import os
import cv2
import uuid
import random
from pathlib import Path
from contextlib import nullcontext
from typing import Optional, Tuple, Dict, Any, List

import gradio as gr
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import gc
import av
from fractions import Fraction

# =========================
# Environment
# =========================
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Set before launch if desired:
# export HF_HOME=/data/
# export CUDA_VISIBLE_DEVICES=1

# =========================
# IMPORTS FROM YOUR PROJECT
# =========================
from run_single import (
    WANUNetDirect,
    EBRCustomBridge,
    encode_frames,
    bridge_sample,
    preview_decode_from_latent,
)

# =========================
# Config
# =========================
CKPT_DIR = Path("./ckpts")
DEGRADATIONS = ["blur", "haze", "lowlight", "rain", "snow"]

MODEL_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
APP_TMP_ROOT = Path("./gradio_tmp")
APP_TMP_ROOT.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# =========================
# Utility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_mp_dtype(mixed_precision: str):
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def make_autocast_ctx(mixed_precision: str):
    def autocast_ctx():
        if mixed_precision == "no":
            return nullcontext()
        dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    return autocast_ctx


def get_model_id_from_backend(backend: str) -> str:
    if backend == "wan":
        return "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    elif backend == "flux":
        return "black-forest-labs/FLUX.1-dev"
    raise ValueError(f"Unsupported backend: {backend}")


def get_deep_prompt_len(backend: str) -> int:
    if backend == "wan":
        return 226
    if backend == "flux":
        return 512
    raise ValueError(f"Invalid backend: {backend}")


def clear_model_cache():
    global MODEL_CACHE

    for _, state in MODEL_CACHE.items():
        for k in ["model", "vae", "bridge", "scheduler"]:
            obj = state.get(k, None)
            if obj is None:
                continue
            try:
                if hasattr(obj, "cpu"):
                    obj.cpu()
            except Exception:
                pass

    MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_input_type_from_backend(backend: str) -> str:
    if backend == "wan":
        return "video"
    elif backend == "flux":
        return "image"
    raise ValueError(f"Unsupported backend: {backend}")


def get_target_size(input_type: str):
    if input_type == "image":
        return (1024, 1024)  # H,W
    if input_type == "video":
        return (480, 832)  # H,W
    raise ValueError(f"Invalid input_type: {input_type}")


def resize_chw(img: torch.Tensor, size_hw):
    if tuple(img.shape[-2:]) == tuple(size_hw):
        return img
    img = img.unsqueeze(0)
    img = F.interpolate(img, size=size_hw, mode="bicubic", align_corners=False)
    return img.squeeze(0)


def ensure_video_bcthw(video: torch.Tensor) -> torch.Tensor:
    if video.ndim == 5:
        return video
    if video.ndim == 4:
        if video.shape[0] == 3:
            return video.unsqueeze(0)  # [1,3,T,H,W]
        if video.shape[1] == 3:
            return video.unsqueeze(2)  # [B,3,1,H,W]
    if video.ndim == 3:
        return video.unsqueeze(0).unsqueeze(2)  # [1,3,1,H,W]
    raise ValueError(f"Unsupported video tensor shape: {tuple(video.shape)}")


def video_tensor_to_uint8(video: torch.Tensor):
    video = ensure_video_bcthw(video)
    video = video[0]  # [3,T,H,W]
    video = video.clamp(-1, 1)
    video = (video + 1.0) * 0.5
    video = (video * 255.0).byte()
    video = video.permute(1, 2, 3, 0).cpu().numpy()  # [T,H,W,3]
    return video


def save_image_from_tensor(video_or_img: torch.Tensor, out_path: str):
    frames = video_tensor_to_uint8(video_or_img)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(frames[0]).save(out_path)


def save_video_mp4(video: torch.Tensor, out_path: str, fps: float = 24.0, codec: str = "mpeg4"):
    """
    video: tensor in [-1,1], shape [3,T,H,W], [B,3,T,H,W], [B,3,H,W], or [3,H,W]
    saves mp4 using PyAV
    codec:
      - "mpeg4" is more likely to work broadly
      - "h264" is better for browser playback if available in your PyAV/FFmpeg libs
    """
    frames = video_tensor_to_uint8(video)  # [T,H,W,3]
    T, H, W, _ = frames.shape

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with av.open(out_path, mode="w") as container:
        stream = container.add_stream(codec, rate=Fraction(str(fps)))
        stream.width = W
        stream.height = H
        stream.pix_fmt = "yuv420p"

        for frame in frames:
            av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def load_image_rgb(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img)  # C,H,W in [0,1]


def load_video_from_file(video_path: str, max_frames: int = 33):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 12.0

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transforms.ToTensor()(frame)
        frames.append(frame)
        idx += 1
        if idx >= max_frames:
            break

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from video: {video_path}")

    video = torch.stack(frames, dim=1)  # [C,T,H,W]
    return video, fps


def load_and_preprocess(single_input: str, input_type: str, max_frames: int = 33):
    if input_type == "image":
        img = load_image_rgb(single_input)
        img = resize_chw(img, get_target_size("image"))
        return {
            "deg": img.unsqueeze(0),  # [B,C,H,W]
            "fps": None,
        }

    in_path = Path(single_input)
    if in_path.suffix.lower() not in VID_EXTS:
        raise ValueError("For WAN/video input, upload a video file such as .mp4/.mov/.avi")

    vid, fps = load_video_from_file(str(in_path), max_frames=max_frames)  # [C,T,H,W]
    vid = vid.permute(1, 0, 2, 3)  # [T,C,H,W]
    vid = torch.stack([resize_chw(frame, get_target_size("video")) for frame in vid], dim=0)
    vid = vid.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W]

    return {
        "deg": vid.unsqueeze(0),  # [B,C,T,H,W]
        "fps": fps,
    }


def get_ckpt_path(backend: str, degradation: str) -> Path:
    ckpt = CKPT_DIR / f"{backend}_{degradation}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def resolve_ckpt_selection(backend: str, degradations: List[str]):
    if not degradations or len(degradations) == 0:
        raise ValueError("Please select at least one degradation.")

    paths = [str(get_ckpt_path(backend, d)) for d in degradations]
    if len(paths) == 1:
        return paths[0], None
    return None, paths


# =========================
# Model loading
# =========================
def load_pipeline(
        model_id: str,
        backend: str,
        mixed_precision: str,
        resume_path: Optional[str],
        t0: float,
        t_min: float,
        bridge_sampler_steps: int,
        mixed_paths: Optional[List[str]] = None,
):
    cache_key = (
        model_id,
        backend,
        mixed_precision,
        os.path.abspath(resume_path) if resume_path else None,
        tuple(sorted(mixed_paths)) if mixed_paths is not None else None,
    )
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    # keep only one pipeline on GPU at a time
    clear_model_cache()

    device = "cuda"
    mp_dtype = get_mp_dtype(mixed_precision)
    deep_prompt_len = get_deep_prompt_len(backend)

    model = WANUNetDirect(
        model_id=model_id,
        backend=backend,
        dtype=mp_dtype,
        device=str(device),
        deep_prompt_len=deep_prompt_len,
    ).to(device)

    vae = model.vae
    scheduler = model.scheduler

    if mixed_paths is None:
        mixed_inference = False
        deg_ids = None

        if not resume_path:
            raise ValueError("resume_path is required when mixed_paths is not provided.")

        print(f"[single ckpt] loading checkpoint: {resume_path}")

        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
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
            raise ValueError("Invalid checkpoint save format")

        saved_backend = saved.get("backend")
        if saved_backend is not None and saved_backend != model.backend_name:
            raise ValueError(
                f"Checkpoint/backend mismatch: checkpoint says {saved_backend}, current model is {model.backend_name}"
            )
    else:
        mixed_inference = True
        deg_ids = list(range(len(mixed_paths)))
        model.load_learned_prompts_from_paths(mixed_paths)
        print(f"[mixed ckpt] loading checkpoints: {mixed_paths}")
        for p in mixed_paths:
            fname = Path(p).name
            if not fname.startswith(f"{backend}_"):
                raise ValueError(f"Mixed checkpoint {p} does not match backend={backend}")

    bridge = EBRCustomBridge(T0=t0, sigma_max=1.0, sigma_min=t_min, power=1.0)

    vae.to(device=device, dtype=torch.float32)

    state = {
        "model": model,
        "vae": vae,
        "scheduler": scheduler,
        "bridge": bridge,
        "device": device,
        "autocast_ctx": make_autocast_ctx(mixed_precision),
        "mixed_inference": mixed_inference,
        "deg_ids": deg_ids,
        "bridge_sampler_steps": bridge_sampler_steps,
    }

    MODEL_CACHE[cache_key] = state
    return state


# =========================
# Inference
# =========================
@torch.no_grad()
def run_inference(
        uploaded_file: str,
        backend: str,
        mixed_precision: str,
        degradations: List[str],
        t0: float,
        t_min: float,
        bridge_sampler_steps: int,
        seed: int,
        max_frames: int,
):
    if uploaded_file is None:
        raise gr.Error("Please upload an input file.")

    input_type = get_input_type_from_backend(backend)
    model_id = get_model_id_from_backend(backend)

    set_seed(seed)

    resume_path, mixed_paths = resolve_ckpt_selection(backend, degradations)

    print(f"[request] backend={backend}")
    print(f"[request] input_type={input_type}")
    print(f"[request] degradations={degradations}")
    print(f"[request] resume_path={resume_path}")
    print(f"[request] mixed_paths={mixed_paths}")

    out_img = None
    out_vid = None
    msg = None

    try:
        state = load_pipeline(
            model_id=model_id,
            backend=backend,
            mixed_precision=mixed_precision,
            resume_path=resume_path,
            t0=t0,
            t_min=t_min,
            bridge_sampler_steps=bridge_sampler_steps,
            mixed_paths=mixed_paths,
        )

        model = state["model"]
        vae = state["vae"]
        scheduler = state["scheduler"]
        bridge = state["bridge"]
        device = state["device"]
        autocast_ctx = state["autocast_ctx"]
        mixed_inference = state["mixed_inference"]
        deg_ids = state["deg_ids"]

        model.eval()
        vae.eval()
        model.set_prompt_enabled(True)

        sample = load_and_preprocess(uploaded_file, input_type=input_type, max_frames=max_frames)
        degraded = sample["deg"].to(device, non_blocking=True)
        fps = sample["fps"]

        if backend == "flux":
            scheduler.set_timesteps(1000, mu=None, sigmas=None)
        else:
            scheduler.set_timesteps(1000)

        degraded_in = degraded * 2.0 - 1.0

        latents = encode_frames(vae, degraded_in)

        x = bridge_sample(
            bridge=bridge,
            model=model,
            scheduler=scheduler,
            autocast_ctx=autocast_ctx,
            x_T=latents,
            steps=bridge_sampler_steps,
            clip_denoised=False,
            t0=t0,
            t_min=t_min,
            mixed_inference=mixed_inference,
            deg_ids=deg_ids,
        )

        vid = preview_decode_from_latent(vae, scheduler, x)

        job_dir = APP_TMP_ROOT / str(uuid.uuid4())
        job_dir.mkdir(parents=True, exist_ok=True)

        if input_type == "image":
            out_path = job_dir / "result.jpg"
            save_image_from_tensor(vid, str(out_path))
            mode_msg = f"single={degradations[0]}" if len(degradations) == 1 else f"mixed={','.join(degradations)}"
            out_img, out_vid, msg = str(out_path), None, f"Done. backend={backend}, {mode_msg}"
        else:
            out_path = job_dir / "result.mp4"
            save_video_mp4(vid, str(out_path), fps=float(fps if fps is not None else 12.0), codec="h264")
            mode_msg = f"single={degradations[0]}" if len(degradations) == 1 else f"mixed={','.join(degradations)}"
            out_img, out_vid, msg = None, str(out_path), f"Done. backend={backend}, {mode_msg}"

        return out_img, out_vid, msg

    finally:
        for var_name in [
            "sample", "degraded", "degraded_in", "latents", "x", "vid",
            "fps", "state", "model", "vae", "scheduler", "bridge",
            "autocast_ctx", "deg_ids"
        ]:
            if var_name in locals():
                del locals()[var_name]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def infer_wrapper(
        uploaded_file,
        backend,
        mixed_precision,
        degradations,
        t0,
        t_min,
        bridge_sampler_steps,
        seed,
        max_frames,
):
    try:
        out_img, out_vid, msg = run_inference(
            uploaded_file=uploaded_file,
            backend=backend,
            mixed_precision=mixed_precision,
            degradations=degradations,
            t0=float(t0),
            t_min=float(t_min),
            bridge_sampler_steps=int(bridge_sampler_steps),
            seed=int(seed),
            max_frames=int(max_frames),
        )
        return out_img, out_vid, msg
    except Exception as e:
        return None, None, f"Error: {str(e)}"


# =========================
# UI helpers
# =========================
def ui_info_for_backend(backend: str):
    input_type = get_input_type_from_backend(backend)
    model_id = get_model_id_from_backend(backend)
    label = f"Select degradation(s) for {backend}"
    return (
        input_type,
        model_id,
        gr.update(choices=DEGRADATIONS, value=[], label=label),
        None,  # out_image
        None,  # out_video
        f"Switched to {backend}. Please reselect degradation(s) and upload a matching input."
    )


# =========================
# Gradio UI
# =========================
with gr.Blocks(title="Restore Demo") as demo:
    gr.Markdown(
        """
# Restore Demo

- `wan` uses `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` and expects **video**
- `flux` uses `black-forest-labs/FLUX.1-dev` and expects **image**

Select one degradation for single-checkpoint inference, or multiple degradations for mixed inference.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            uploaded_file = gr.File(label="Upload input file", type="filepath")

            backend = gr.Dropdown(
                label="backend",
                choices=["wan", "flux"],
                value="wan",
            )

            input_type = gr.Textbox(
                label="input_type",
                value="video",
                interactive=False,
            )

            model_id_view = gr.Textbox(
                label="model_id",
                value="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                interactive=False,
            )

            degradations = gr.CheckboxGroup(
                label="Select degradation(s) for wan",
                choices=DEGRADATIONS,
                value=[],
            )

            mixed_precision = gr.Dropdown(
                label="mixed_precision",
                choices=["no", "fp16", "bf16"],
                value="fp16",
            )

            t0 = gr.Number(label="t0", value=0.4)
            t_min = gr.Number(label="t_min", value=0.0001)

            bridge_sampler_steps = gr.Slider(
                label="bridge_sampler_steps",
                minimum=1,
                maximum=100,
                step=1,
                value=20,
            )

            seed = gr.Number(label="seed", value=123, precision=0)

            max_frames = gr.Slider(
                label="max_frames (video only)",
                minimum=1,
                maximum=120,
                step=1,
                value=33,
            )

            run_btn = gr.Button("Run Inference", variant="primary")

        with gr.Column(scale=1):
            out_image = gr.Image(label="Output image", type="filepath")
            out_video = gr.Video(label="Output video")
            status = gr.Textbox(label="Status")

    backend.change(
        fn=ui_info_for_backend,
        inputs=[backend],
        outputs=[input_type, model_id_view, degradations, out_image, out_video, status],
    )

    run_btn.click(
        fn=infer_wrapper,
        inputs=[
            uploaded_file,
            backend,
            mixed_precision,
            degradations,
            t0,
            t_min,
            bridge_sampler_steps,
            seed,
            max_frames,
        ],
        outputs=[out_image, out_video, status],
    )

demo.queue(max_size=20)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )