import os
from typing import Dict, Optional, Tuple

import torch
from PIL import Image
import torch.distributed as dist
import numpy as np
import cv2


# -----------------------------------------------------------------------------
# VAE helpers
# -----------------------------------------------------------------------------

def _vae_has_wan_stats(vae) -> bool:
    cfg = getattr(vae, "config", None)
    return cfg is not None and hasattr(cfg, "latents_mean") and hasattr(cfg, "latents_std")


def _vae_scaling_factor(vae) -> float:
    cfg = getattr(vae, "config", None)
    scale = getattr(cfg, "scaling_factor", None) if cfg is not None else None
    return float(scale) if scale is not None else 1.0


# FLUX VAE is different
def _vae_has_flux_affine(vae) -> bool:
    cfg = getattr(vae, "config", None)
    return cfg is not None and hasattr(cfg, "shift_factor") and getattr(cfg, "shift_factor", None) is not None


def _get_mean_std(vae, like: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(vae.config.latents_mean, device=like.device, dtype=like.dtype)
    std = torch.tensor(vae.config.latents_std, device=like.device, dtype=like.dtype)
    mean = mean.view(1, -1, 1, 1, 1)
    std = std.view(1, -1, 1, 1, 1)
    return mean, std


def _flatten_bt(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    if x.ndim == 4:
        return x, x.shape[0], 1
    if x.ndim != 5:
        raise ValueError(f"Expected 4D or 5D tensor, got ndim={x.ndim}")
    b, c, t, h, w = x.shape
    flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    return flat, b, t


def _unflatten_bt(x: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
    if frames == 1:
        return x
    bt, c, h, w = x.shape
    if bt != batch * frames:
        raise ValueError(f"Unexpected flattened batch: got {bt}, expected {batch * frames}")
    return x.view(batch, frames, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def _expand_indices_for_frames(indices: torch.Tensor, frames: int) -> torch.Tensor:
    return indices.repeat_interleave(frames, dim=0) if frames > 1 else indices


# -----------------------------------------------------------------------------
# Bridge helpers
# -----------------------------------------------------------------------------

def get_rank_safe():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _nearest_index_1d(values_1d: torch.Tensor, targets_1d: torch.Tensor) -> torch.Tensor:
    """
    values_1d: [S] (scheduler.sigmas)
    targets_1d: [B]
    returns: [B] long indices
    """
    d = (values_1d.view(1, -1) - targets_1d.view(-1, 1)).abs()
    return d.argmin(dim=1)


def bridge_sample(
        bridge,
        model,
        scheduler,
        autocast_ctx,
        x_T,
        steps=40,
        clip_denoised=True,
        t0=0.3,
        t_min=0.0001,
        mixed_inference=False,
        deg_ids=None
):

    def denoiser(x_t, sigma):
        noisy_ctx = bridge.make_model_input(x_t=x_t, scheduler=scheduler, sigmas=sigma)
        t = noisy_ctx["t"]
        sigma = noisy_ctx["sigma"]

        noisy_latents_model = x_t

        with autocast_ctx():
            if not mixed_inference:
                noise_pred = model(x_in=noisy_latents_model, t_idx=t)
                denoised = bridge.get_model_output(noise_pred=noise_pred, y=noisy_latents_model,
                                                   sigma=sigma)
            else:
                assert deg_ids and len(deg_ids) > 1, f"{deg_ids} is invalid"
                noise_pred_by_deg = {}
                for did in deg_ids:
                    model.set_active_learned_prompt(did)
                    noise_pred = model(x_in=noisy_latents_model, t_idx=t)

                    noise_pred_by_deg[did] = noise_pred  # keep tensor on GPU

                noise_preds = torch.stack([noise_pred_by_deg[d] for d in deg_ids], dim=0)
                noise_pred_mix = noise_preds.mean(dim=0)
                denoised = bridge.get_model_output(noise_pred=noise_pred_mix, y=noisy_latents_model,
                                                   sigma=sigma)

        if clip_denoised:
            denoised = denoised.clamp(-1, 1)

        return denoised

    x_0 = ebr_custom_multistep_sample(bridge=bridge, denoiser=denoiser, Y_latents=x_T,
                                             steps=steps, T0=t0, t_min=t_min, scheduler=scheduler)

    return x_0


def _append_dims(v: torch.Tensor, target_ndim: int) -> torch.Tensor:
    while v.ndim < target_ndim:
        v = v.view(-1, *([1] * (target_ndim - 1)))
    return v

class EBRCustomBridge:
    """
    My implementation for EBR.
    """

    def __init__(self, T0: float = 1.0, sigma_min: float = 1e-4, sigma_max: float = None, power: float = 1.0,):
        self.T0 = float(T0)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(self.T0 if sigma_max is None else sigma_max)
        self.power = power
        if self.power != 1.0:
            if get_rank_safe() == 0:
                print(f"Power sampling enabled")  # Skewed timesteps

    def sample_t_real_uniform(self, batch_size, device, dtype=torch.float32):
        """
        power < 1  -> more mass near 1 -> more samples near T0 (higher t)
        power = 1  -> uniform
        power > 1  -> more mass near 0 -> more samples near sigma_min (lower t)
        """
        u = torch.rand(batch_size, device=device, dtype=dtype)
        u = u.pow(self.power)
        return self.sigma_min + (self.T0 - self.sigma_min) * u

    def _coeffs(self, t: torch.Tensor):
        """
        Returns:
          alpha = (1 - t), beta = t
          lam = t/T0
          A(t), B(t), C(t) (Eq. 10)
          plus "a_t" and "b_t" as the mean-path coefficients on X0 and Y inside X_t
        """
        t = t.to(torch.float32)
        T0 = torch.tensor(self.T0, device=t.device, dtype=torch.float32)

        lam = (t / T0).clamp(0.0, 1.0)
        alpha = (1.0 - t).clamp(0.0, 1.0)
        beta = t.clamp(0.0, 1.0)


        A = alpha * lam  # (1 - t) * (t/T0)
        B = beta  # t
        C = 1.0 - A  # 1 - (1 - t) * (t/T0)

        # X_t mean part = alpha*((1-lam)X0 + lam*Y) for which coeffs reqd
        a_mean = alpha * (1.0 - lam)  # coefficient of X0 in X_t (mean)
        b_mean = alpha * lam  # coefficient of Y  in X_t (mean)
        std_xt = beta  # since X_t includes + beta * X_T and X_T ~ N(0,I)

        return alpha, beta, lam, A, B, C, a_mean, b_mean, std_xt

    def A_B_C(self, t: torch.Tensor):
        """
        coeffs:
          A(t) = (1 - t) * (t/T0)
          B(t) = t
          C(t) = 1 - A(t)
        """
        t = t.to(torch.float32)
        T0 = torch.tensor(self.T0, device=t.device, dtype=torch.float32)
        lam = (t / T0).clamp(0.0, 1.0)
        alpha = (1.0 - t).clamp(0.0, 1.0)
        A = alpha * lam
        B = t.clamp(0.0, 1.0)
        C = 1.0 - A
        return A, B, C

    def get_bridge_input(self, scheduler, x_start, xT, sigmas, noise):
        X0 = x_start
        Y = xT
        t = torch.as_tensor(sigmas, device=X0.device, dtype=torch.float32)

        alpha, beta, lam, A, B, C, a_mean, b_mean, std_xt = self._coeffs(t)

        # build X_t per:  X_t = (1-t)*X̃_t + t*X_T
        X_T = noise  # paper samples X_T ~ N(0,I)
        X_tilde = _append_dims(1.0 - lam, X0.ndim) * X0 + _append_dims(lam, X0.ndim) * Y
        X_t = _append_dims(alpha, X0.ndim) * X_tilde + _append_dims(beta, X0.ndim) * X_T

        # map t to your scheduler index
        sigmas_grid = scheduler.sigmas.to(device=X0.device, dtype=torch.float32)
        # sigma_model is same as t here
        sigma_model = t
        k = _nearest_index_1d(sigmas_grid, sigma_model).clamp(max=len(scheduler.timesteps) - 1)

        timesteps = scheduler.timesteps.to(device=X0.device)
        t_idx = timesteps.index_select(0, k)

        return {
            "noisy_latents": X_t,
            "sigma": sigma_model,
            "t": t_idx,
            "k": k,
            "A_t": A, "B_t": B, "C_t": C,
        }

    def make_model_input(self, x_t: torch.Tensor, sigmas: torch.Tensor, scheduler):
        t = torch.as_tensor(sigmas, device=x_t.device, dtype=torch.float32)

        sigmas_grid = scheduler.sigmas.to(device=x_t.device, dtype=torch.float32)
        sigma_model = t
        k = _nearest_index_1d(sigmas_grid, sigma_model).clamp(max=len(scheduler.timesteps) - 1)

        timesteps = scheduler.timesteps.to(device=x_t.device)
        t_idx = timesteps.index_select(0, k)

        return {
            "noisy_latents": x_t,
            "sigma": sigma_model,
            "t": t_idx,
        }

    def get_model_output(self, noise_pred, y, sigma,):
        """
          Unlike calculating paper x0_hat, we can directly estimate x0_hat using flow matching eqns.
        """

        sigma = _append_dims(sigma, y.ndim)

        return y - sigma * noise_pred


@torch.no_grad()
def ebr_custom_multistep_sample(
        bridge: EBRCustomBridge,
        denoiser,  # denoiser(x_t, sigma=t, xT=Y) -> x0_hat
        scheduler,
        Y_latents,  # degraded latents
        steps: int = 40,
        T0: float = None,
        t_min: float = 1e-4,
        seed: int = 42,
):
    device = Y_latents.device
    dtype = Y_latents.dtype
    Bsz = Y_latents.shape[0]
    ones = Y_latents.new_ones([Bsz])

    if T0 is None:
        T0 = float(getattr(bridge, "T0", 0.3))

    # time grid from T0 -> t_min
    ts = torch.linspace(T0, t_min, steps, device=device, dtype=torch.float32)

    # init: X_{T0} = (1-T0)*Y + T0*X_T0, with X_T0 ~ N(0,I)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    XT0 = torch.randn(Y_latents.shape, device=device, dtype=dtype, generator=g)
    x = (1.0 - T0) * Y_latents + T0 * XT0  # Construct at max t=t0

    for i in range(len(ts) - 1):
        t = ts[i] * ones  # [B]
        t_next = ts[i + 1] * ones  # [B]

        # denoise at current t -> x0_hat
        x0_hat = denoiser(x, t)  # denoiser ignores xT arg if it wants; fine

        # coefficients at current t
        A, Bc, C = bridge.A_B_C(t)  # [B]
        coef_X0 = (C - Bc)  # = (1-t)*(1 - t/T0)

        # estimate terminal noise X_T from forward equation:
        # x = coef_X0*X0 + A*Y + B*X_T  =>  X_T_hat = (x - coef_X0*x0_hat - A*Y)/B
        Bc_safe = torch.clamp(Bc, min=max(t_min, 1e-6))
        coef_X0_ = _append_dims(coef_X0, x.ndim)
        A_ = _append_dims(A, x.ndim)
        B_ = _append_dims(Bc_safe, x.ndim)

        XT_hat = (x - coef_X0_ * x0_hat - A_ * Y_latents) / B_

        # advance to next time using bridge.get_bridge_input
        # x_start := X0_hat, xT := Y, noise := X_T_hat, sigmas := t_next
        ctx_next = bridge.get_bridge_input(
            scheduler=scheduler,
            x_start=x0_hat,
            xT=Y_latents,
            sigmas=t_next,
            noise=XT_hat,
        )
        x = ctx_next["noisy_latents"]

    # final denoise at smallest t
    t = ts[-1] * ones
    x0_hat = denoiser(x, t)
    return x0_hat


# ----------------- VAE encode/decode -----------------
@torch.no_grad()
def encode_frames(vae, video: torch.Tensor) -> torch.Tensor:
    vae.eval()
    video = video.to(torch.float32)

    if video.ndim == 5 and _vae_has_wan_stats(vae):
        # WAN-style standardized latents
        raw = vae.encode(video).latent_dist.sample()
        mean, std = _get_mean_std(vae, raw)
        return (raw - mean) / std

    # FLUX VAE
    if _vae_has_flux_affine(vae):
        flat, batch, frames = _flatten_bt(video)
        z_raw = vae.encode(flat).latent_dist.sample()
        scale = float(vae.config.scaling_factor)
        shift = float(vae.config.shift_factor)
        z_std = (z_raw - shift) * scale
        return _unflatten_bt(z_std, batch=batch, frames=frames)

    flat, batch, frames = _flatten_bt(video)
    latents = vae.encode(flat).latent_dist.sample()
    latents = latents * _vae_scaling_factor(vae)
    return _unflatten_bt(latents, batch=batch, frames=frames)


@torch.no_grad()
def decode_frames(vae, latents: torch.Tensor) -> torch.Tensor:
    vae.eval()
    latents = latents.to(torch.float32)

    if latents.ndim == 5 and _vae_has_wan_stats(vae):
        mean, std = _get_mean_std(vae, latents)
        raw = latents * std + mean
        video = vae.decode(raw, return_dict=False)[0]
        return video.clamp(-1, 1)

    # FLUX VAE
    if _vae_has_flux_affine(vae):
        flat, batch, frames = _flatten_bt(latents)
        scale = float(vae.config.scaling_factor)
        shift = float(vae.config.shift_factor)
        z_raw = flat / scale + shift
        img = vae.decode(z_raw, return_dict=False)[0]
        return _unflatten_bt(img.clamp(-1, 1), batch=batch, frames=frames)

    flat, batch, frames = _flatten_bt(latents)
    raw = flat / _vae_scaling_factor(vae)
    video = vae.decode(raw, return_dict=False)[0]
    video = video.clamp(-1, 1)
    return _unflatten_bt(video, batch=batch, frames=frames)


@torch.no_grad()
def decode_x0_std_to_video(vae, x0_std: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    video = decode_frames(vae, x0_std)
    return video.clamp(-1, 1) if clamp else video


@torch.no_grad()
def preview_decode_from_latent(vae, scheduler, latent: torch.Tensor):
    del scheduler
    vid = decode_x0_std_to_video(vae, latent, clamp=True)
    return vid

# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def _ensure_video_bcthw(video: torch.Tensor) -> torch.Tensor:
    if video.ndim == 5:
        return video
    if video.ndim == 4:
        if video.shape[0] == 3:
            return video.unsqueeze(0)
        return video.unsqueeze(2)
    if video.ndim == 3:
        return video.unsqueeze(0).unsqueeze(2)
    raise ValueError(f"Unsupported video ndim={video.ndim}")


def video_tensor_to_uint8(video: torch.Tensor):
    """
    video: [3,T,H,W], [B,3,T,H,W], [B,3,H,W], or [3,H,W] in [-1,1]
    returns: uint8 numpy array [T,H,W,3]
    """
    video = _ensure_video_bcthw(video)
    video = video[0]

    video = video.clamp(-1, 1)
    video = (video + 1) * 0.5
    video = (video * 255).byte()
    video = video.permute(1, 2, 3, 0).cpu().numpy()
    return video


def save_video_frames(video: torch.Tensor, folder: str, names: list):
    os.makedirs(folder, exist_ok=True)
    frames = video_tensor_to_uint8(video)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(folder, names[i]))


def save_video_frames_prefix(video: torch.Tensor, folder: str, prefix: str = "frame", num_save=1):
    os.makedirs(folder, exist_ok=True)
    frames = video_tensor_to_uint8(video)
    for i, frame in enumerate(frames):
        if i > num_save:
            break
        Image.fromarray(frame).save(os.path.join(folder, f"{prefix}_{i:04d}.jpg"))

def save_video_mp4(video: torch.Tensor, out_path: str, fps: float = 12.0):
    frames = video_tensor_to_uint8(video)   # [T,H,W,3]
    T, H, W, _ = frames.shape

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (W, H))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

    for frame in frames:
        # cv2 expects BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()


def save_images(image: torch.Tensor, folder: str, name: str):
    os.makedirs(folder, exist_ok=True)
    frames = video_tensor_to_uint8(image)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(folder, name))
        assert i < 1, "Only one frame needs to be there for image"