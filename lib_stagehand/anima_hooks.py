"""The one place Anima's attention gets wrapped, for both features.

backend/nn/anima.py dispatches no attention patches, so SelfCrossAttention.forward and
Block.forward are wrapped at import time instead of editing the fork. Both features work
on the cross-attention: Character Prompts changes how it's computed (regional), and
Precise Reference reads its query (the IP-Adapter attends with the block's own q). One
wrapper runs q/k/v once and hands the same q to both -- recomputing it per feature costs a
projection per block per step and advances ControlLLLite's per-call counter on q_proj.

Sessions travel in transformer_options; a cross-attention without one runs the original.
This module is imported once (sys.modules), so a Reload UI can't stack the wrappers.
The Block wrapper also makes a pass's per-character LoRAs (region_lora) current while the
block runs: its Linears read them from there.
"""

from backend.nn import anima as anima_nn

from lib_stagehand import region_lora

REGIONS_KEY = "nai_character_regions"
IP_KEY = "precise_reference_ip"


def install() -> None:
    original_forward = anima_nn.SelfCrossAttention.forward
    if not getattr(original_forward, "_nai_hooked", False):

        def forward(self, x, context=None, rope_emb=None, transformer_options={}):
            options = transformer_options or {}
            index = getattr(self, "_pr_ip_index", None)
            regions = options.get(REGIONS_KEY)
            ip = options.get(IP_KEY) if index is not None else None
            if self.is_SelfAttn or context is None or (regions is None and ip is None):
                return original_forward(self, x, context, rope_emb, transformer_options)

            # the body of SelfCrossAttention.forward, with q kept for both features
            q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
            if regions is not None:
                out = regions.attend(self, q, k, v, context, options)
            else:
                out = self.compute_attention(q, k, v, transformer_options=options)
            if ip is not None:
                ip.capture(index, q, options, regions)
            return out

        forward._nai_hooked = True
        anima_nn.SelfCrossAttention.forward = forward

    original_block = anima_nn.Block.forward
    if not getattr(original_block, "_nai_hooked", False):

        def block_forward(self, x, emb, *args, **kwargs):
            region_lora.current = (kwargs.get("transformer_options") or {}).get(region_lora.KEY)
            try:
                out = original_block(self, x, emb, *args, **kwargs)
            finally:
                region_lora.current = None
            index = getattr(self, "_pr_ip_index", None)
            if index is not None:
                session = (kwargs.get("transformer_options") or {}).get(IP_KEY)
                if session is not None:
                    out = session.apply(index, out, emb)
            return out

        block_forward._nai_hooked = True
        anima_nn.Block.forward = block_forward


def tag_blocks(diffusion_model) -> list:
    """Anima's DiT blocks, each (and its cross-attention) tagged with its index."""
    blocks = [m for m in diffusion_model.modules() if isinstance(m, anima_nn.Block)] if diffusion_model is not None else []
    for i, block in enumerate(blocks):
        block._pr_ip_index = i
        block.cross_attn._pr_ip_index = i
        region_lora.hook_linears(block)
    return blocks
