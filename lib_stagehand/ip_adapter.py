"""Anima IP-Adapter: LuciferTC/Anima-IP-Adapter, "ip_adapter-Character_Reference-10".

A port of the inference path in github.com/LuciferTC9527/ComfyUI-Anima_IP-Adapter, cut
down to what the published checkpoint actually contains (no token compressor, null
tokens, key norm or shared projection):

    tokens = SigLIP2(reference).last_hidden_state                 [1, 1024, 768]
    per DiT block i, after the whole block:
        q    = cross_attn.q_norm(cross_attn.q_proj(x_cross))      the block's own query
        out += adaln_ip[i](emb) * sdpa(q, ip_k_proj[i](tokens), ip_v_proj[i](tokens))

Two deliberate differences from that node:
- The checkpoint's LoRA covers self_attn, cross_attn AND mlp of every block. The node's
  regex only matches cross_attn, silently dropping two thirds of what was trained.
  All of it is loaded here (lora_patch_source).
- Strength scales the injected output, so 0 is exactly no effect. The node scales the
  SigLIP tokens instead, which never reaches zero through the projections' biases.

Kept free of Forge imports so test_core.py can exercise it.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

SIGLIP_SIZE = 512


class _IPBlock(nn.Module):
    def __init__(self, embed_dim: int, inner_dim: int):
        super().__init__()
        self.ip_k_proj = nn.Linear(embed_dim, inner_dim)
        self.ip_v_proj = nn.Linear(embed_dim, inner_dim)
        self.adaln_ip = nn.Sequential(nn.SiLU(), nn.Linear(inner_dim, inner_dim))


class AnimaIPAdapter(nn.Module):
    """Per-block K/V projections and gates. Key layout matches the checkpoint exactly."""

    def __init__(self, num_blocks: int, embed_dim: int, inner_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList(_IPBlock(embed_dim, inner_dim) for _ in range(num_blocks))


def load_adapter(path: str) -> tuple[AnimaIPAdapter, dict[str, torch.Tensor], dict[str, str]]:
    """Returns (adapter, lora state dict with 'lora.base_model.model.' stripped, metadata)."""
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        metadata = f.metadata() or {}
        state = {k: f.get_tensor(k) for k in f.keys()}

    prefix = "lora.base_model.model."
    lora = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    own = {k: v for k, v in state.items() if not k.startswith("lora.")}

    num_blocks = 1 + max(int(k.split(".")[1]) for k in own)
    inner_dim, embed_dim = own["blocks.0.ip_k_proj.weight"].shape
    adapter = AnimaIPAdapter(num_blocks, embed_dim, inner_dim)
    # strict: a checkpoint variant with the node's optional parts must fail loudly here,
    # not run with those parts silently missing.
    adapter.load_state_dict(own)
    adapter.eval().requires_grad_(False)
    return adapter, lora, metadata


def lora_patch_source(lora: dict[str, torch.Tensor], alpha: float | None, prefix: str = "diffusion_model.") -> tuple[dict, dict]:
    """Shape the adapter's LoRA for Forge's comfy load_lora(lora, to_load).

    Keys arrive as 'blocks.N.<module>.lora_A.weight'; load_lora reads the diffusers
    'lora_A'/'lora_B' names directly, so only the target map and alpha are added.
    No alpha means scale 1, which is also what alpha == rank gives.
    """
    lora = dict(lora)
    to_load = {}
    for key in list(lora):
        if key.endswith(".lora_A.weight"):
            module = key[: -len(".lora_A.weight")]
            to_load[module] = f"{prefix}{module}.weight"
            if alpha is not None:
                lora[f"{module}.alpha"] = torch.tensor(float(alpha))
    return lora, to_load


def has_image(value) -> bool:
    """A reference card's value holds an image: an array (UI), a path or base64 (API).
    Not `value == ""`: on an array that compares every pixel and raises."""
    return value is not None and not (isinstance(value, str) and not value)


def flatten(image) -> np.ndarray:
    """Any PIL image or array -> RGB array, with transparency flattened onto white.

    Character art often has no background, and a plain RGB conversion turns the
    transparent area black -- which the reference then faithfully transfers.
    """
    pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype(np.uint8))
    if pil.mode.startswith("I;16"):  # 16-bit greyscale: convert("RGB") clips it to near-white
        pil = Image.fromarray((np.asarray(pil, dtype=np.float32) / 257).astype(np.uint8))
    if pil.mode in ("RGBA", "LA", "PA") or "transparency" in pil.info:
        pil = Image.alpha_composite(Image.new("RGBA", pil.size, "white"), pil.convert("RGBA"))
    return np.asarray(pil.convert("RGB"))


def letterbox(image: np.ndarray, size: int = SIGLIP_SIZE) -> np.ndarray:
    """Fit inside a size x size square on black, as the node's Apply does."""
    pil = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    w, h = pil.size
    ratio = size / max(w, h)
    new_w, new_h = max(1, round(w * ratio)), max(1, round(h * ratio))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(pil.resize((new_w, new_h), Image.BILINEAR), ((size - new_w) // 2, (size - new_h) // 2))
    return np.array(canvas)  # writable copy: torch.from_numpy warns on read-only arrays


def image_key(image: np.ndarray) -> str:
    image = np.ascontiguousarray(image)
    return hashlib.sha1(image.tobytes() + str(image.shape).encode()).hexdigest()


@torch.no_grad()
def encode_reference(encoder, image: np.ndarray, device) -> torch.Tensor:
    """SigLIP2 last_hidden_state for one reference: [1, 1024, 768] float32."""
    pixels = torch.from_numpy(letterbox(image)).float().div(127.5).sub(1.0).permute(2, 0, 1).unsqueeze(0)
    return encoder(pixel_values=pixels.to(device), interpolate_pos_encoding=True).last_hidden_state.float()


class IPReference:
    """Strength for the positive and negative passes. Every block, every step: the A/B
    tests found the early blocks contribute nothing and the step window little."""

    def __init__(self, tokens: torch.Tensor, cond_weight: float, uncond_weight: float, target: int | None = None):
        self.tokens = tokens
        self.cond_weight = float(cond_weight)
        self.uncond_weight = float(uncond_weight)
        # a character's number: only over her region (Character Prompts'); None = everywhere
        self.target = target


def region_mask(regions, number: int, length: int, device, dtype) -> torch.Tensor | None:
    """[length]: how much of every token is this character's -- Character Prompts' region
    weights, the same blend her prompt gets. None when she isn't drawn in this pass (no
    regions, or no such character)."""
    numbers = list(getattr(regions, "numbers", None) or [])
    if number not in numbers:
        return None
    weights = regions.weights(length, device, dtype)
    return None if weights is None else weights[numbers.index(number) + 1]


class IPSession:
    """Computes each block's IP attention from its cross-attention input, then adds it
    (gated) to the block's output. Lives in transformer_options for one sampling pass."""

    def __init__(self, adapter: AnimaIPAdapter, references: list[IPReference]):
        self.adapter = adapter
        self.references = references
        self.pending = {}
        # K/V depend only on the reference and the block, never on the step: project once
        self.kv = {}

    def _row_weights(self, batch, transformer_options):
        cond = list(transformer_options.get("cond_indices") or [])
        uncond = list(transformer_options.get("uncond_indices") or [])
        if len(cond) + len(uncond) != batch:
            return []  # unexpected batch layout: leave the pass untouched rather than guess

        out = []
        for n, ref in enumerate(self.references):
            weights = [0.0] * batch
            for row in cond:
                weights[row] = ref.cond_weight
            for row in uncond:
                weights[row] = ref.uncond_weight
            if any(weights):
                out.append((n, ref, weights))
        return out

    def _project(self, n, ref, index, heads, dim, like):
        key = (n, index, like.device, like.dtype)
        if key not in self.kv:
            block = self.adapter.blocks[index]
            tokens = ref.tokens.to(device=like.device, dtype=like.dtype)
            k = _linear(block.ip_k_proj, tokens).view(1, -1, heads, dim).transpose(1, 2)
            v = _linear(block.ip_v_proj, tokens).view(1, -1, heads, dim).transpose(1, 2)
            self.kv[key] = (k, v)
        return self.kv[key]

    def capture(self, index: int, q: torch.Tensor, transformer_options: dict, regions=None) -> None:
        """Called with a block's cross-attention query, [B, L, heads, dim] after q_norm --
        the same q the node computes from the cross-attention input. regions: Character
        Prompts' session, for the references that belong to one character."""
        batch, length, heads, dim = q.shape
        rows = self._row_weights(batch, transformer_options)
        if not rows:
            return
        query = q.transpose(1, 2)
        total = None
        for n, ref, weights in rows:
            mask = None
            if ref.target is not None:
                mask = region_mask(regions, ref.target, length, q.device, q.dtype)
                if mask is None:
                    continue
            k, v = self._project(n, ref, index, heads, dim, q)
            out = F.scaled_dot_product_attention(query, k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1))
            out = out.transpose(1, 2).reshape(batch, length, heads * dim)
            out = out * torch.tensor(weights, device=out.device, dtype=out.dtype).view(batch, 1, 1)
            if mask is not None:
                out = out * mask.to(out.dtype)[None, :, None]
            total = out if total is None else total + out
        if total is not None:
            self.pending[index] = total

    def apply(self, index: int, out: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Called with a block's output [B, T, H, W, D] and its timestep embedding [B, T', D]
        (T' is 1 when Forge's Anima reference frames make T > 1: broadcast, as the block does)."""
        ip = self.pending.pop(index, None)
        if ip is None:
            return out
        batch, frames, height, width, dim = out.shape
        gate = _linear(self.adapter.blocks[index].adaln_ip[1], F.silu(emb.to(ip.dtype)))
        return out + (gate[:, :, None, None, :] * ip.view(batch, frames, height, width, dim)).to(out.dtype)

    def clear(self) -> None:
        """Drop the cached K/V (~224 MB per reference) once the generation is done."""
        self.kv.clear()
        self.pending.clear()


def _linear(layer, x):
    """layer(x) with its weights brought to x: under VRAM pressure Forge can leave part of
    the adapter on the CPU, and plain nn.Linear has no manual cast."""
    bias = None if layer.bias is None else layer.bias.to(x)
    return F.linear(x, layer.weight.to(x), bias)
