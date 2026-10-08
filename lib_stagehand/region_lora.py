"""Per-character LoRAs: a LoRA typed in a character's card changes only her.

Forge merges a LoRA into the model's weights, so it reaches every pixel. A card's LoRA stays
out of the weights instead: its low-rank pair (down, up) is added after each layer it patches,
on the fly, one of two ways (Settings > Stagehand, "Character LoRAs"):

- Masked: one model call. The LoRA's change to every image token is scaled by that
  character's region weight -- the same weights her prompt gets. On the cross-attention's
  text side (k/v projections) it applies in full to her own text, to no one else's.
- Separate pass: one extra model call per character with a LoRA, that LoRA on everywhere;
  her call's prediction replaces the plain one over her region (latent space).

"Whole image" (the old way) leaves the tags to Forge. A LoRA's text-encoder half reaches her
card's text only: character_prompts encodes it with a patched copy of the text encoder.

Kept free of Forge imports at module level so test_core.py can exercise it; load() imports
Forge's LoRA code when it runs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

KEY = "stagehand_region_lora"
MODES = ["Whole image", "Masked", "Separate pass"]
DEFAULT = "Masked"  # won the 2026-10-08 blind test (NOTES.md, "Character LoRAs")
current = None  # the session of the model call in progress (anima_hooks' Block wrapper sets it)


class LoraSession:
    """loras: {owner: {module: [(down [r, in], up [out, r], scale)]}}. regions: the pass's
    RegionSession (its grid, weights, numbers and images). owners: {owner: (character number,
    the batch's image indices that have this LoRA set, or None for all)} -- a wildcard can roll
    a different LoRA per image. Without it, every owner is a character number, on every image."""

    def __init__(self, loras, regions, mode, owners=None):
        self.loras = loras
        self.regions = regions
        self.mode = mode
        self.owners = owners or {o: (o, None) for o in loras}
        self.pass_owner = None  # Separate pass: whose LoRA this model call carries
        self.context_owner = None  # Masked: whose text the cross-attention is projecting
        self._cast = {}
        self._rows = {}

    def owner(self, n, image):
        """Whose LoRA character n has in image `image` of the batch, or None."""
        return next((o for o, (m, images) in self.owners.items() if m == n and (images is None or image in images)), None)

    def rows(self, o, like):
        """1 on the model call's rows (image b % batch) that have owner o's LoRA set, shaped to
        broadcast over `like`; None when every row has it."""
        images = self.owners[o][1]
        if images is None:
            return None
        count = len(self.regions.images)
        key = (o, like.shape[0], like.ndim, like.device, like.dtype)
        if key not in self._rows:
            mask = torch.tensor([float(b % count in images) for b in range(like.shape[0])], device=like.device, dtype=like.dtype)
            self._rows[key] = mask.reshape(-1, *[1] * (like.ndim - 1))
        return self._rows[key]

    def key(self):
        """What a cached k/v projection depends on, besides its text."""
        return self.pass_owner, self.context_owner

    def _delta(self, n, module, x):
        total = None
        for down, up, scale in self.loras.get(n, {}).get(module, ()):
            k = (id(down), x.device, x.dtype)
            if k not in self._cast:
                self._cast[k] = (down.to(x), up.to(x))
            d, u = self._cast[k]
            y = F.linear(F.linear(x, d), u) * scale
            total = y if total is None else total + y
        return total

    def adjust(self, module, x, y):
        """A hooked Linear's output y (of input x), with the characters' LoRAs added."""
        if self.mode == "Separate pass" or getattr(module, "_rl_context", False):
            n = self.pass_owner if self.mode == "Separate pass" else self.context_owner
            d = self._delta(n, module, x) if n is not None else None
            return y if d is None else y + d
        weights = self.regions.weights(y.shape[1:-1].numel(), y.device, y.dtype)
        if weights is None:
            return y
        for o in self.loras:
            d = self._delta(o, module, x)
            if d is None:
                continue
            d = d * weights[self.regions.numbers.index(self.owners[o][0]) + 1].reshape(1, *y.shape[1:-1], 1)
            rows = self.rows(o, y)
            y = y + (d if rows is None else d * rows)
        return y

    def latent_mask(self, n, like):
        """Character n's region weights at the latent's size, shaped to broadcast over it."""
        frames, h, w = self.regions.grid
        weights = self.regions.weights(frames * h * w, like.device, like.dtype)
        mask = F.interpolate(weights[self.regions.numbers.index(n) + 1].reshape(1, frames, h, w), size=like.shape[-2:], mode="nearest")
        return mask.reshape(1, 1, frames, *like.shape[-2:]) if like.ndim == 5 else mask.reshape(1, 1, *like.shape[-2:])

    def wrapper(self, previous=None):
        """Separate pass: Forge's model_function_wrapper, chained onto any earlier one."""

        def run(apply_model, args):
            def call():
                if previous is not None:
                    return previous(apply_model, args)
                return apply_model(args["input"], args["timestep"], **args["c"])

            base = out = call()
            for o in self.loras:
                self.pass_owner = o
                try:
                    other = call()
                finally:
                    self.pass_owner = None
                mask = self.latent_mask(self.owners[o][0], base)
                rows = self.rows(o, base)
                out = out + (mask if rows is None else mask * rows) * (other - base)
            return out

        return run


def hook_linears(block) -> None:
    """Every Linear of a DiT block's attention and MLP adds the current session's LoRAs.
    The cross-attention's k/v project text, not image tokens: marked _rl_context."""
    for part in (block.self_attn, block.cross_attn, block.mlp):
        for name, module in part.named_modules():
            if not isinstance(module, torch.nn.Linear) or getattr(module, "_rl_hooked", False):
                continue
            module._rl_context = part is block.cross_attn and name in ("k_proj", "v_proj")
            original = module.forward

            def forward(x, *args, _original=original, _module=module, **kwargs):
                y = _original(x, *args, **kwargs)
                return y if current is None else current.adjust(_module, x, y)

            module.forward = forward
            module._rl_hooked = True


# ---------------------------------------------------------------------------- loading
def parse_tags(tags) -> list:
    """['<lora:name:0.7>', ...] -> [(name, te, unet)], Forge's own reading of the numbers."""
    from modules import extra_networks

    out = []
    for params in extra_networks.parse_prompt(" ".join(tags))[1].get("lora", []):
        if not params.positional:
            continue
        te = (0.0 if "@" in params.positional[1] else float(params.positional[1])) if len(params.positional) > 1 else 1.0
        te = float(params.named.get("te", te))
        unet = float(params.positional[2]) if len(params.positional) > 2 else te
        out.append((params.positional[0], te, float(params.named.get("unet", unet))))
    return out


_files = {}


def _state_dict(name):
    import os

    import networks
    from backend.state_dict import state_dict_prefix_replace

    on_disk = networks.available_network_aliases.get(name) or networks.available_networks.get(name)
    if on_disk is None:
        networks.list_available_networks()
        on_disk = networks.available_network_aliases.get(name) or networks.available_networks.get(name)
    if on_disk is None:
        return None
    key = (on_disk.filename, os.path.getmtime(on_disk.filename))
    if key not in _files:
        sd = networks.load_lora_state_dict(on_disk.filename)
        if any(k.startswith("lora_unet__") for k in sd):
            sd = state_dict_prefix_replace(sd, {"lora_unet__": "lora_unet_"})
        _files.clear()  # ponytail: keeps one batch's files; an LRU if switching sets gets slow
        _files[key] = sd
    return _files[key]


def load(unet, clip, tags):
    """One character's LoRA tags -> ({module: [(down, up, scale)]}, a text encoder patched
    with their text-encoder halves or None, [names not found or not low-rank])."""
    import networks
    from backend.patcher.lora import load_lora, model_lora_keys_clip, model_lora_keys_unet

    diffusion_model = unet.model.diffusion_model
    unet_keys, clip_keys = model_lora_keys_unet(unet.model), model_lora_keys_clip(clip.cond_stage_model)
    pairs, patched, problems = {}, None, []
    for name, te, strength in parse_tags(tags):
        sd = _state_dict(name)
        if sd is None:
            problems.append(f"{name} (not found)")
            continue
        sd = dict(sd)
        networks.process_anima(sd, len(diffusion_model.blocks))  # renames the LLM adapter's keys, remaps block counts
        patches, rest = load_lora(sd, unet_keys)
        skipped = 0
        for key, adapter in patches.items():
            w = getattr(adapter, "weights", None)
            if (not isinstance(key, str) or not key.startswith("diffusion_model.blocks.") or getattr(adapter, "name", "") != "lora"
                    or w is None or w[3] is not None or w[4] is not None):
                skipped += 1
                continue
            up, down, alpha = w[0].flatten(1), w[1].flatten(1), w[2]
            scale = strength * (alpha / down.shape[0] if alpha is not None else 1.0)
            module = diffusion_model.get_submodule(key[len("diffusion_model."): -len(".weight")])
            pairs.setdefault(module, []).append((down, up, scale))
        if skipped:
            problems.append(f"{name} ({skipped} layers not plain LoRA or outside the blocks, left out)")
        te_patches, _ = load_lora(rest, clip_keys)
        if te_patches and te:
            patched = (patched or clip).clone()
            patched.add_patches(filename=name, patches=te_patches, strength_patch=te)
    return pairs, patched, problems
