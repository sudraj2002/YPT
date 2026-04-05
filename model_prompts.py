import html
import inspect
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers import FlowMatchEulerDiscreteScheduler

try:
    from diffusers import FluxPipeline
except Exception:  # pragma: no cover - optional dependency at runtime
    FluxPipeline = None

try:
    from diffusers import WanPipeline
except Exception:  # pragma: no cover - optional dependency at runtime
    WanPipeline = None

from prompt_helpers import DeepPromptBank, CrossAttnAddPrompts, CrossAttnAddPromptsFlux

try:
    import ftfy
except Exception:
    class _FtfyStub:
        @staticmethod
        def fix_text(text: str) -> str:
            return text
    ftfy = _FtfyStub()


# -----------------------------------------------------------------------------
# Prompt cleaning helpers
# -----------------------------------------------------------------------------

def basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prompt_clean(text: str) -> str:
    return whitespace_clean(basic_clean(text))

# -----------------------------------------------------------------------------
# Cross-attention discovery helpers
# -----------------------------------------------------------------------------

def _has_arg(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except Exception:
        return False


def _infer_cross_attn_dim(attn_module: nn.Module) -> int:
    # diffusers Attention modules usually expose to_k with in_features=cross_attn_dim
    to_k = getattr(attn_module, "to_k", None)
    if to_k is None:
        raise RuntimeError("Cross-attn module has no to_k; cannot infer context dim")
    in_features = getattr(to_k, "in_features", None)
    if in_features is not None:
        return int(in_features)
    weight = getattr(to_k, "weight", None)
    if weight is not None:
        return int(weight.shape[1])
    raise RuntimeError("Unable to infer cross-attention dimension from to_k")


def _collect_cross_attn_targets(root: nn.Module, flux: bool = False) -> List[Tuple[nn.Module, str, nn.Module]]:
    """
    Find parent modules that have a cross-attn attribute (attn2/cross_attn/etc.).
    Returns (parent, attr_name, attn_module) tuples.
    """
    if not flux:
        candidates = ("attn2", "cross_attn", "cross_attention", "attn_cross", "crossattn")
    else:
        candidates = ("attn2", "cross_attn", "cross_attention", "attn_cross", "crossattn", "attn")
    targets: List[Tuple[nn.Module, str, nn.Module]] = []
    seen = set()
    for parent in root.modules():
        for name in candidates:
            if not hasattr(parent, name):
                continue
            attn = getattr(parent, name)
            if not isinstance(attn, nn.Module):
                continue
            key = (id(parent), name)
            if key in seen:
                continue
            seen.add(key)
            targets.append((parent, name, attn))
            break
    return targets


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _expand_timesteps(t: torch.Tensor, frames: int) -> torch.Tensor:
    if frames == 1:
        return t
    return t.repeat_interleave(frames, dim=0)

def _prepare_2d_input(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """
    For 2D models, flatten [B,C,T,H,W] -> [B*T,C,H,W].
    Returns flattened tensor plus (B, T).
    """
    if x.ndim == 4:
        return x, x.shape[0], 1
    if x.ndim != 5:
        raise ValueError(f"Unsupported latent ndim={x.ndim}; expected 4D or 5D")
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), b, t


def _restore_2d_output(x: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
    if frames == 1:
        return x
    bt, c, h, w = x.shape
    if bt != batch * frames:
        raise ValueError(f"Unexpected flattened batch: got {bt}, expected {batch * frames}")
    return x.view(batch, frames, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def _get_latent_channels(vae: nn.Module, fallback: int = 4) -> int:
    cfg = getattr(vae, "config", None)
    print(cfg)
    if cfg is not None:
        for key in ("latent_channels", "z_channels", "z_dim"):
            val = getattr(cfg, key, None)
            if val is not None:
                return int(val)
    return int(fallback)


def _resolve_backend_name(name: str) -> str:
    n = name.lower()
    if n in {"wan", "wan2", "wan14b"}:
        return "wan"
    if n in {"flux", "flux-dev", "flux_dev"}:
        return "flux"
    raise ValueError(f"Unsupported backend '{name}'. Expected one of: wan, sdxl, flux")


# -----------------------------------------------------------------------------
# Backend (Universal)
# -----------------------------------------------------------------------------

@dataclass
class PromptBatch:
    encoder_hidden_states: torch.Tensor
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None


class DiffusionBackend(nn.Module):
    backend_name: str = "base"
    is_video_native: bool = False

    def __init__(self, model_id: str, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        self.pipeline = None
        self.unet = None
        self.vae = None
        self.scheduler = None

    def encode_prompt_uncond(self, batch_size: int, image_size: Optional[Tuple[int, int]] = None) -> PromptBatch:
        raise NotImplementedError

    def forward_unet(self, x_in: torch.Tensor, t_idx: torch.Tensor, prompt_batch: PromptBatch) -> torch.Tensor:
        raise NotImplementedError

    def cross_attn_targets(self, flux) -> List[Tuple[nn.Module, str, nn.Module]]:
        return _collect_cross_attn_targets(self.unet, flux)


class WanBackend(DiffusionBackend):
    backend_name = "wan"
    is_video_native = True

    def __init__(
            self,
            model_id: str,
            dtype: torch.dtype,
            device: torch.device,
            max_sequence_length: int = 226,
            cache_prompt=True,  # Dummy unused for WAN alone, because already it does so
    ):
        if WanPipeline is None:
            raise ImportError("WanPipeline is not available in this environment")
        super().__init__(model_id=model_id, dtype=dtype, device=device)
        self.max_sequence_length = max_sequence_length

        pipe = WanPipeline.from_pretrained(model_id, torch_dtype=dtype)
        vae_scale_factor = int(getattr(pipe, "vae_scale_factor", 8))
        self.pipeline = SimpleNamespace(vae_scale_factor=vae_scale_factor)
        self.unet = pipe.transformer.cpu()
        self.unet.eval()

        # WAN uses flow prediction; keep prior scheduler config
        self.scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=1.0,
        )

        self.vae = pipe.vae.to(device)
        self.vae.eval()

        # text encoder path (T5)
        self.tokenizer = pipe.tokenizer
        self.text_enc = pipe.text_encoder.to(device)
        self.text_enc.eval()

        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.text_enc.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            # Cache the null-text embedding to free up memory
            self.prompt_embeds_uncond_1 = self._get_t5_prompt_embeds(
                prompt="",
                max_sequence_length=self.max_sequence_length,
                device=device,
                dtype=dtype,
            )

        # Drop heavy pipeline modules once components are extracted
        self.cache_prompt = cache_prompt
        del pipe
        del self.text_enc
        torch.cuda.empty_cache()


    def _get_t5_prompt_embeds(
            self,
            prompt: Union[str, Sequence[str]],
            num_videos_per_prompt: int = 1,
            max_sequence_length: Optional[int] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        # Mirrors diffusers
        device = device or self.device
        dtype = dtype or self.dtype
        max_sequence_length = max_sequence_length or self.max_sequence_length

        prompt_list = [prompt] if isinstance(prompt, str) else list(prompt)
        prompt_list = [prompt_clean(u) for u in prompt_list]
        batch_size = len(prompt_list)

        text_inputs = self.tokenizer(
            prompt_list,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        mask = text_inputs.attention_mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_enc(input_ids, mask).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        return prompt_embeds

    def encode_prompt_uncond(self, batch_size: int, image_size: Optional[Tuple[int, int]] = None) -> PromptBatch:
        del image_size
        enc = self.prompt_embeds_uncond_1.expand(batch_size, -1, -1)
        return PromptBatch(encoder_hidden_states=enc)

    def forward_unet(self, x_in: torch.Tensor, t_idx: torch.Tensor, prompt_batch: PromptBatch) -> torch.Tensor:
        return self.unet(
            hidden_states=x_in,
            timestep=t_idx,
            encoder_hidden_states=prompt_batch.encoder_hidden_states.to(x_in.device, dtype=x_in.dtype),
            return_dict=False,
        )[0]

    def cross_attn_targets(self, flux) -> List[Tuple[nn.Module, str, nn.Module]]:
        # Flux is False for WAN
        return _collect_cross_attn_targets(self.unet, flux=flux)


class FluxBackend(DiffusionBackend):
    backend_name = "flux"
    is_video_native = False

    def __init__(self, model_id: str, dtype: torch.dtype, device: torch.device, cache_prompt: bool = True,):
        if FluxPipeline is None:
            raise ImportError("FluxPipeline is not available in this environment")
        super().__init__(model_id=model_id, dtype=dtype, device=device)
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipeline = pipe
        self.unet = pipe.transformer.to(device)
        self.unet.eval()
        self.vae = pipe.vae.to(device)
        self.vae.eval()
        self.scheduler = pipe.scheduler
        self.cache_prompt = cache_prompt

        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)

        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)

        for enc in (getattr(pipe, "text_encoder", None), getattr(pipe, "text_encoder_2", None)):
            # 2 encoders for FLUX
            if enc is not None:
                if not cache_prompt:
                    enc.to(device)
                enc.eval()
                for p in enc.parameters():
                    p.requires_grad_(False)

        self._forward_sig = inspect.signature(self.unet.forward)

    @staticmethod
    def calculate_shift(
            image_seq_len,
            base_seq_len: int = 256,
            max_seq_len: int = 4096,
            base_shift: float = 0.5,
            max_shift: float = 1.15,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    @staticmethod
    def get_mu_sigma(scheduler, image_size, num_inference_steps):
        if image_size == (512, 512):
            image_seq_len = 1024
        elif image_size == (768, 768):
            image_seq_len = 2304
        elif image_size == (1024, 1024):
            image_seq_len = 4096
        else:
            raise ValueError(f"Invalid image size of {image_size}")

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
            sigmas = None
        mu = FluxBackend.calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )

        return mu, sigmas

    def encode_prompt_uncond(self, batch_size: int, image_size: Optional[Tuple[int, int]] = None) -> PromptBatch:
        del image_size
        prompt = ["" for _ in range(batch_size)]
        encoded = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=512
        )
        if not isinstance(encoded, (tuple, list)):
            raise RuntimeError("Unexpected encode_prompt output for FLUX backend")

        prompt_embeds = encoded[0]
        pooled = encoded[1] if len(encoded) > 1 else None
        text_ids = encoded[2] if len(encoded) > 2 else None

        added: Dict[str, torch.Tensor] = {}
        if pooled is not None:
            # diffusers Flux transformer expects pooled_projections
            added["pooled_projections"] = pooled
        if text_ids is not None:
            added["txt_ids"] = text_ids

        return PromptBatch(encoder_hidden_states=prompt_embeds, added_cond_kwargs=added or None)

    @staticmethod
    def _prepare_latent_image_ids(height, width, device, dtype):
        # height/width here are LATENT PACK grid sizes (H//8, W//8)
        latent_image_ids = torch.zeros(height // 2, width // 2, 3, device=device, dtype=dtype)
        latent_image_ids[..., 1] += torch.arange(height // 2, device=device, dtype=dtype)[:, None]
        latent_image_ids[..., 2] += torch.arange(width // 2, device=device, dtype=dtype)[None, :]
        return latent_image_ids.reshape(-1, 3)

    @staticmethod
    def _pack_latents(latents, batch_size, height, width):
        """
        latents: [B, C, H, W] where C=16 for Flux latent channels
        returns: [B, (H//2)*(W//2), C*4] => token dim
        """
        B, C, H, W = latents.shape
        latents = latents.view(batch_size, C, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), C * 4)
        return latents

    @staticmethod
    def _unpack_latents(tokens_3d: torch.Tensor, h: int, w: int):
        # tokens_3d: [B, N, L] -> [B, 16, h, w]
        B, N, D = tokens_3d.shape
        assert D == 64
        assert (h % 2 == 0) and (w % 2 == 0)
        x = tokens_3d.view(B, h // 2, w // 2, 16, 2, 2)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, 16, h, w)
        return x

    def forward_unet(self, x_in: torch.Tensor, t_idx: torch.Tensor, prompt_batch: PromptBatch) -> torch.Tensor:
        kwargs: Dict[str, Any] = {}

        # Latent packing
        B = x_in.shape[0]
        _, _, height_lat, width_lat = x_in.shape

        if x_in.ndim == 4:
            x_in = self._pack_latents(x_in, B, height_lat, width_lat)  # [B, N, 64]

        if "hidden_states" in self._forward_sig.parameters:
            kwargs["hidden_states"] = x_in
        """elif "sample" in self._forward_sig.parameters:
            kwargs["sample"] = x_in
        else:
            kwargs[next(iter(self._forward_sig.parameters))] = x_in"""

        if "img_ids" in self._forward_sig.parameters:
            kwargs["img_ids"] = self._prepare_latent_image_ids(height_lat, width_lat, device=x_in.device,
                                                               dtype=x_in.dtype)

        # For flux u need to scale the timestep by 1000 (expects float)
        kwargs["timestep"] = t_idx.to(x_in.device, dtype=torch.float32) / 1000.0
        kwargs["encoder_hidden_states"] = prompt_batch.encoder_hidden_states.to(x_in.device, dtype=x_in.dtype)

        # guidance: if transformer has guidance embeddings
        if "guidance" in self._forward_sig.parameters:
            g = torch.ones((B,), device=x_in.device, dtype=torch.float32)
            kwargs["guidance"] = g

        if prompt_batch.added_cond_kwargs:
            for k, v in prompt_batch.added_cond_kwargs.items():
                if k in self._forward_sig.parameters:
                    kwargs[k] = v.to(x_in.device, dtype=x_in.dtype) if torch.is_tensor(v) else v

        out = self.unet(return_dict=False, **kwargs)[0]

        # Unpack latents
        out = self._unpack_latents(out, height_lat, width_lat)

        return out


def build_backend(
        backend: str,
        model_id: str,
        dtype: torch.dtype,
        device: torch.device,
        max_sequence_length: int,
        cache_prompt: bool,
) -> DiffusionBackend:
    name = _resolve_backend_name(backend)
    if name == "wan":
        return WanBackend(
            model_id=model_id,
            dtype=dtype,
            device=device,
            max_sequence_length=max_sequence_length,
            cache_prompt=cache_prompt
        )

    if name == "flux":
        return FluxBackend(model_id=model_id, dtype=dtype, device=device, cache_prompt=cache_prompt)
    raise AssertionError("Unreachable backend resolution")


# -----------------------------------------------------------------------------
# Model-agnostic deep-prompt wrapper
# -----------------------------------------------------------------------------

class PromptedDiffusionModel(nn.Module):
    def __init__(
            self,
            model_id: str,
            backend: str = "wan",
            dtype: torch.dtype = torch.float16,
            device: Union[str, torch.device] = "cuda",
            max_sequence_length: int = 226,
            enable_deep_prompts: bool = True,
            deep_prompt_len: int = 226,
            deep_prompt_init_std: float = 0.02,
            deep_prompt_alpha_init: float = 0.0,
            cache_prompt: bool = True,
    ):
        super().__init__()
        device = torch.device(device)
        self.dtype = dtype
        self.device = device
        self.backend_name = _resolve_backend_name(backend)

        self.backend = build_backend(
            backend=self.backend_name,
            model_id=model_id,
            dtype=dtype,
            device=device,
            max_sequence_length=max_sequence_length,
            cache_prompt=cache_prompt
        )

        self.unet = self.backend.unet
        self.vae = self.backend.vae
        self.scheduler = self.backend.scheduler

        self.enable_deep_prompts = enable_deep_prompts
        self.deep_prompt_len = deep_prompt_len
        self.deep_prompt_init_std = deep_prompt_init_std
        self.deep_prompt_alpha_init = deep_prompt_alpha_init

        self.deep_prompt_bank: Optional[DeepPromptBank] = None
        self._deep_prompt_wrappers: List[CrossAttnAddPrompts] = []

        if self.enable_deep_prompts:
            if self.backend_name == 'flux':
                self._init_deep_prompts_flux()
            else:
                self._init_deep_prompts()

    # -------------------- Deep prompts --------------------
    def _init_deep_prompts(self) -> None:
        if self.backend_name == 'flux':
            flux_ = True
        else:
            flux_ = False
        targets = self.backend.cross_attn_targets(flux_)

        if not targets:
            raise RuntimeError(f"[{self.backend_name}] found 0 cross-attn modules to patch")

        dims = [_infer_cross_attn_dim(attn) for _, _, attn in targets]
        ctx_dim = dims[0]
        if any(d != ctx_dim for d in dims):
            uniq = sorted(set(dims))
            raise RuntimeError(f"[{self.backend_name}] inconsistent cross-attn dims: {uniq}")

        router_in_dim = _get_latent_channels(self.vae, fallback=ctx_dim)
        print(
            f"[DeepPrompts:{self.backend_name}] patching {len(targets)} cross-attn modules | ctx_dim={ctx_dim} | router_in={router_in_dim}")

        self.deep_prompt_bank = DeepPromptBank(
            num_blocks=len(targets),
            prompt_len=self.deep_prompt_len,
            hidden_dim=ctx_dim,
            init_std=self.deep_prompt_init_std,
        ).to(self.device)

        for p in self.deep_prompt_bank.parameters():
            p.requires_grad_(True)

        for idx, (parent, name, attn) in enumerate(targets):
            if isinstance(attn, CrossAttnAddPrompts):
                raise RuntimeError("Cross-attn already wrapped; refusing to double-patch")
            wrapper = CrossAttnAddPrompts(attn, self.deep_prompt_bank, block_idx=idx,
                                          alpha_init=self.deep_prompt_alpha_init)
            setattr(parent, name, wrapper)
            self._deep_prompt_wrappers.append(wrapper)

        print(f"[DeepPrompts:{self.backend_name}] patched {len(self._deep_prompt_wrappers)} modules")

    # Flux
    def _init_deep_prompts_flux(self) -> None:
        if self.backend_name == 'flux':
            flux_ = True
        else:
            flux_ = False
        targets = self.backend.cross_attn_targets(flux_)

        if not targets:
            raise RuntimeError(f"[{self.backend_name}] found 0 cross-attn modules to patch")

        dims = [_infer_cross_attn_dim(attn) for _, _, attn in targets]
        ctx_dim = dims[0]
        if any(d != ctx_dim for d in dims):
            uniq = sorted(set(dims))
            raise RuntimeError(f"[{self.backend_name}] inconsistent cross-attn dims: {uniq}")

        router_in_dim = _get_latent_channels(self.vae, fallback=ctx_dim)
        print(
            f"[DeepPrompts:{self.backend_name}] patching {len(targets)} cross-attn modules | ctx_dim={ctx_dim} | router_in={router_in_dim}")

        self.deep_prompt_bank = DeepPromptBank(
            num_blocks=len(targets),
            prompt_len=self.deep_prompt_len,
            hidden_dim=ctx_dim,
            init_std=self.deep_prompt_init_std,
        ).to(self.device)

        for p in self.deep_prompt_bank.parameters():
            p.requires_grad_(True)

        for idx, (parent, name, attn) in enumerate(targets):
            if isinstance(attn, CrossAttnAddPromptsFlux):
                raise RuntimeError("Cross-attn already wrapped; refusing to double-patch")
            wrapper = CrossAttnAddPromptsFlux(attn, self.deep_prompt_bank, block_idx=idx,
                                              alpha_init=self.deep_prompt_alpha_init, flux=flux_)
            setattr(parent, name, wrapper)
            self._deep_prompt_wrappers.append(wrapper)

        print(f"[DeepPrompts:{self.backend_name}] patched {len(self._deep_prompt_wrappers)} modules")

    def set_prompt_enabled(self, enabled: bool) -> None:
        for w in self._deep_prompt_wrappers:
            w.enable_prompts = enabled

    def _init_learned_prompt_store(self):
        # Mixed prompts
        self._learned_prompt_payloads = {}  # deg_id -> payload
        self._active_deg_id = None

    def load_learned_prompts_from_paths(self, paths, map_location="cpu"):
        """
        Load N prompt checkpoints for mixed degradations.
        """
        self._init_learned_prompt_store()

        deg_ids = list(range(len(paths)))

        for deg_id, path in zip(deg_ids, paths):
            payload = torch.load(path, map_location=map_location)

            if "deep_prompt_bank" in payload:
                bank_sd = payload["deep_prompt_bank"]
                alphas = payload.get("alphas", [])
            else:
                raise ValueError(f"Invalid checkpoint format: {path}")

            self._learned_prompt_payloads[int(deg_id)] = {
                "path": path,
                "deep_prompt_bank": bank_sd,
                "alphas": alphas,
                "backend": payload.get("backend", None),
            }

        print(f"[mixed prompts] loaded {len(self._learned_prompt_payloads)} profiles: {sorted(self._learned_prompt_payloads.keys())}")

    @torch.no_grad()
    def set_active_learned_prompt(self, deg_id: int, strict=True):
        deg_id = int(deg_id)
        if getattr(self, "_active_deg_id", None) == deg_id:
            return

        if not hasattr(self, "_learned_prompt_payloads") or deg_id not in self._learned_prompt_payloads:
            raise RuntimeError(f"deg_id={deg_id} not loaded. Call load_learned_prompts_from_paths().")

        payload = self._learned_prompt_payloads[deg_id]

        # sanity check
        if payload.get("backend") is not None and str(payload["backend"]) != str(self.backend_name):
            raise RuntimeError(f"Prompt ckpt backend={payload['backend']} but model backend={self.backend_name}")

        if self.deep_prompt_bank is None:
            raise RuntimeError("deep_prompt_bank is None (enable_deep_prompts is off?)")

        miss, unexp = self.deep_prompt_bank.load_state_dict(payload["deep_prompt_bank"], strict=strict)
        if strict and (miss or unexp):
            raise RuntimeError(f"[deg_id={deg_id}] missing={len(miss)} unexpected={len(unexp)}")

        alphas = payload.get("alphas", None)
        if alphas is not None and len(alphas) > 0:
            if len(alphas) != len(self._deep_prompt_wrappers):
                raise RuntimeError(
                    f"Alpha count mismatch: ckpt has {len(alphas)}, model has {len(self._deep_prompt_wrappers)}")
            for w, a in zip(self._deep_prompt_wrappers, alphas):
                w.alpha.data.copy_(a.to(device=w.alpha.device, dtype=w.alpha.dtype))

        self._active_deg_id = deg_id

    # -------------------- Forward --------------------
    def _infer_image_size(self, x_in: torch.Tensor) -> Tuple[int, int]:
        if x_in.ndim == 5:
            _, _, _, h, w = x_in.shape
        elif x_in.ndim == 4:
            _, _, h, w = x_in.shape
        else:
            raise ValueError(f"Unsupported input ndim={x_in.ndim}; expected 4D or 5D")

        # attempt to recover pixel size from latent size
        scale = int(getattr(self.backend.pipeline, "vae_scale_factor", 8))
        return int(h * scale), int(w * scale)

    def forward(
            self,
            x_in: torch.Tensor,
            t_idx: torch.Tensor,
    ) -> torch.Tensor:
        batch = x_in.shape[0]

        image_size = self._infer_image_size(x_in)

        if self.backend.is_video_native:
            # WAN
            prompt_batch = self.backend.encode_prompt_uncond(batch_size=batch, image_size=image_size)
            out = self.backend.forward_unet(x_in, t_idx, prompt_batch)
            return out.float()

        # 2D backends: flatten time dimension if present
        x_model, batch_size, frames = _prepare_2d_input(x_in)
        t_model = _expand_timesteps(t_idx, frames)

        if not self.backend.cache_prompt:
            # Text encoder is very heavy, might OOM
            prompt_batch = self.backend.encode_prompt_uncond(batch_size=x_model.shape[0], image_size=image_size)
        else:
            # Cache null-text once
            if not hasattr(self, "prompt_batch"):
                self.backend.unet.cpu()
                pipe = getattr(self.backend, "pipeline", None)

                assert pipe is not None, "Pipeline missing"
                te1 = getattr(pipe, "text_encoder", None)
                te2 = getattr(pipe, "text_encoder_2", None)
                if te1 is not None:
                    te1.to(self.device)
                if te2 is not None:
                    te2.to(self.device)
                self.prompt_batch = self.backend.encode_prompt_uncond(batch_size=x_model.shape[0],
                                                                      image_size=image_size)

                if te1 is not None:
                    te1.cpu()
                if te2 is not None:
                    te2.cpu()
                torch.cuda.empty_cache()
                self.backend.unet.to(self.device)

        if not self.backend.cache_prompt:
            out_model = self.backend.forward_unet(x_model, t_model, prompt_batch)
        else:
            out_model = self.backend.forward_unet(x_model, t_model, self.prompt_batch)

        out = _restore_2d_output(out_model, batch=batch_size, frames=frames)
        return out.float()


# Backward-compatible name used by existing scripts
WANUNetDirect = PromptedDiffusionModel
