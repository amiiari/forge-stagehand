"""Process-wide state for Precise Reference: the adapter, its ModelPatcher, the SigLIP2
encoder and the encoded-reference cache.

It lives in a lib module (imported once, cached in sys.modules) rather than in the script,
which Forge re-executes on every Reload UI -- that left the old adapter (~0.5 GB), encoder
and patcher alive next to fresh copies.
"""

import os

import torch

from modules.paths import models_path

from backend import memory_management
from backend.patcher.base import ModelPatcher

from lib_precise_reference.ip_adapter import encode_reference, image_key, load_adapter

ADAPTER_DIR = os.path.join(models_path, "precise_reference")
ADAPTER_FILE = "ip_adapter-Character_Reference-10.safetensors"
SIGLIP_DIR = "siglip2-base-patch16-512"

_adapter = None
_encoder = None
_patcher = None
_tokens = {}


def adapter():
    """(AnimaIPAdapter, lora state dict, metadata), loaded on first use."""
    global _adapter
    if _adapter is None:
        path = os.path.join(ADAPTER_DIR, ADAPTER_FILE)
        if not os.path.isfile(path) or not os.path.isdir(os.path.join(ADAPTER_DIR, SIGLIP_DIR)):
            raise FileNotFoundError(f"Precise Reference needs {ADAPTER_FILE} and {SIGLIP_DIR}/ in {ADAPTER_DIR} -- see the extension's README")
        _adapter = load_adapter(path)
    return _adapter


def patcher(unet, module):
    """One ModelPatcher for the process: Forge tracks loaded models by patcher identity, so
    a fresh one per generation (add_extra_torch_module_during_sampling) re-uploaded the
    adapter every time."""
    global _patcher
    # Must match the activations, i.e. the computation dtype. (add_extra_torch_module's
    # own cast reads diffusion_model.dtype, which Anima's module doesn't have.)
    module.to(unet.model.computation_dtype)
    if _patcher is None or _patcher.load_device != unet.load_device:
        _patcher = ModelPatcher(model=module, load_device=unet.load_device, offload_device=unet.offload_device)
    return _patcher


def reference_tokens(image) -> torch.Tensor:
    """SigLIP2 tokens of a flattened RGB array, cached per image (16 most recent)."""
    global _encoder
    key = image_key(image)
    if key not in _tokens:
        if _encoder is None:
            from transformers import SiglipVisionModel

            _encoder = SiglipVisionModel.from_pretrained(os.path.join(ADAPTER_DIR, SIGLIP_DIR)).eval()
        device = memory_management.get_torch_device()
        _encoder.to(device)
        try:
            tokens = encode_reference(_encoder, image, device).cpu()
        finally:
            _encoder.to("cpu")
        if len(_tokens) >= 16:
            _tokens.pop(next(iter(_tokens)))
        _tokens[key] = tokens
    return _tokens[key]
