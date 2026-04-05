import inspect
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Dict, List


class DeepPromptBank(nn.Module):
    def __init__(self, num_blocks, prompt_len, hidden_dim,
                 film_hidden_dim=256, init_std=0.02,):
        super().__init__()
        # Shared prompt
        self.prompts = nn.Parameter(torch.randn(1, prompt_len, hidden_dim) * init_std)


    def get(self, batch_size, device):
        # Prompts return in fp32
        p = self.prompts[0].to(device=device, dtype=torch.float32)  # [K,D]
        p = p.unsqueeze(0).expand(batch_size, -1, -1)  # [B,K,D]

        return p


def _has_arg(fn, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
        return name in sig.parameters
    except Exception:
        return False


class CrossAttnAddPrompts(nn.Module):
    """
    Wrap a cross-attn module and add an extra prompt-only cross-attn residual
    All attn weights are frozen; only prompt tokens + alpha train.
    For WAN
    """

    def __init__(self, attn_module: nn.Module, prompt_bank: DeepPromptBank, block_idx: int,
                 alpha_init: float = 0.0):
        super().__init__()
        self.attn = attn_module
        self.prompt_bank = prompt_bank
        self.block_idx = block_idx
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        # We call the module's original forward
        self._old_forward = attn_module.forward
        self.enable_prompts = True  # <-- global default

        # detect common kw names
        self._kw_encoder = "encoder_hidden_states" if _has_arg(self._old_forward, "encoder_hidden_states") else None
        self._kw_context = "context" if _has_arg(self._old_forward, "context") else None

        if self._kw_encoder is None and self._kw_context is None:
            raise RuntimeError("Cross-attn forward has neither encoder_hidden_states nor context")

    def forward(self, hidden_states, *args, **kwargs):
        # get text context
        text_ctx = None
        if self._kw_encoder is not None:
            text_ctx = kwargs.get("encoder_hidden_states", None)
        if text_ctx is None and self._kw_context is not None:
            text_ctx = kwargs.get("context", None)

        # If no context provided, fall back to original behavior
        if text_ctx is None:
            return self._old_forward(hidden_states, *args, **kwargs)

        B = hidden_states.shape[0]
        device = hidden_states.device
        dtype = text_ctx.dtype

        B, Ltxt, D = text_ctx.shape

        # call original cross-attn with text
        out_text = self._old_forward(hidden_states, *args, **kwargs)  # encoder_hidden_states of previous block was inp

        prompt_ctx = self.prompt_bank.get(B, device=device)  # [B,K,D]

        # HARD OFF
        if not self.enable_prompts:
            return out_text

        # print(f"Prompts enabled")
        kwargs2 = dict(kwargs)
        if self._kw_encoder is not None:
            emb_fp32 = kwargs["encoder_hidden_states"].to(dtype=torch.float32) + prompt_ctx
            kwargs2["encoder_hidden_states"] = emb_fp32.to(dtype=dtype)
        else:
            kwargs2["context"] = prompt_ctx.to(dtype=dtype)

        # Output for prompt influenced/prompt-based encoder_hidden_states
        out_prompt = self._old_forward(hidden_states, *args, **kwargs2)

        a = self.alpha.to(dtype=torch.float32, device=out_text.device)
        out_text = out_text.to(dtype=torch.float32)
        out_prompt = out_prompt.to(dtype=torch.float32)

        out = out_text + a * out_prompt

        out = out.to(dtype=dtype)
        return out


class CrossAttnAddPromptsFlux(nn.Module):
    """
    Wrap a cross-attn module and add an extra prompt-only cross-attn by gating input to cross-attn/attn:
    All attn weights are frozen; only prompt tokens + alpha train.
    For FLUX
    """

    def __init__(self, attn_module: nn.Module, prompt_bank: DeepPromptBank, block_idx: int,
                 alpha_init: float = 0.0, flux: bool = False):
        super().__init__()
        self.attn = attn_module
        self.prompt_bank = prompt_bank
        self.block_idx = block_idx
        assert flux is True
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        # We call the module's original forward
        self._old_forward = attn_module.forward
        self.enable_prompts = True  # <-- global default

        # detect common kw names
        self._kw_encoder = "encoder_hidden_states" if _has_arg(self._old_forward, "encoder_hidden_states") else None
        # If self._kw_encoder is None, it means single flux transformer block
        self._kw_context = "context" if _has_arg(self._old_forward, "context") else None

        if self._kw_encoder is None and self._kw_context is None:
            raise RuntimeError("Cross-attn forward has neither encoder_hidden_states nor context")

    def forward(self, hidden_states, *args, **kwargs):
        if not self.enable_prompts:
            # call original cross-attn with text
            return self._old_forward(hidden_states, *args, **kwargs)  # encoder_hidden_states of previous block was inp

        # get text context
        # For flux do single pass with perturbed encoder_hidden_states/hidden_states[:, :Ltxt]
        text_ctx = None
        if self._kw_encoder is not None:
            text_ctx = kwargs.get("encoder_hidden_states", None)
        if text_ctx is None and self._kw_context is not None:
            text_ctx = kwargs.get("context", None)

        # If no context provided, single stream otherwise joint
        if text_ctx is not None:
            # Joint
            B = hidden_states.shape[0]
            device = hidden_states.device
            dtype = text_ctx.dtype

            B, Ltxt, D = text_ctx.shape  # B 512 3072 for FLUX

            prompt_ctx = self.prompt_bank.get(B, device=device)  # [B,K,D]
            B, Ltxt, D = kwargs["encoder_hidden_states"].shape
            if prompt_ctx.shape[1] != Ltxt:
                prompt_full = prompt_ctx.new_zeros(B, Ltxt, D)
                K = min(Ltxt, prompt_ctx.shape[1])
                prompt_full[:, :K] = prompt_ctx[:, :K]
                prompt_ctx = prompt_full

            kwargs2 = dict(kwargs)
            a = self.alpha.to(dtype=torch.float32, device=device)
            res = a * prompt_ctx.to(torch.float32)

            # Directly gate input
            kwargs2["encoder_hidden_states"] = kwargs["encoder_hidden_states"] + res.to(dtype=dtype)

            # Output for prompt influenced/prompt-based encoder_hidden_states
            out_prompt = self._old_forward(hidden_states, *args, **kwargs2)

            return out_prompt

        else:
            # Single
            B = hidden_states.shape[0]
            D = hidden_states.shape[-1]
            device = hidden_states.device
            dtype = hidden_states.dtype
            Ltxt = 512  # Max sequence length

            prompt_ctx = self.prompt_bank.get(B, device=device)  # [B,K,D]
            if prompt_ctx.shape[1] != Ltxt:
                prompt_full = prompt_ctx.new_zeros(B, Ltxt, D)
                K = min(Ltxt, prompt_ctx.shape[1])
                prompt_full[:, :K] = prompt_ctx[:, :K]
                prompt_ctx = prompt_full

            kwargs2 = dict(kwargs)
            a = self.alpha.to(dtype=torch.float32, device=device)
            res = a * prompt_ctx.to(torch.float32)

            hidden2 = hidden_states.clone()
            hidden2[:, :Ltxt] = hidden2[:, :Ltxt] + res.to(dtype=dtype)

            # Output for prompt influenced/prompt-based encoder_hidden_states
            out_prompt = self._old_forward(hidden2, *args, **kwargs2)

            return out_prompt