"""Transparent backgrounds. Anima has no alpha channel of its own (LayerDiffuse is SDXL/SD1.5
only), so a finished image has its background cut out by ToonOut: BiRefNet fine-tuned on
anime characters (joelseytre/toonout, MIT). Its weights and BiRefNet's model code
(ZhengPeng7/BiRefNet, pinned) live in models/toonout.

An edit keeps the input's own pixels -- colour and alpha -- outside the inpaint mask, so only
what was repainted is cut again. Forge flattens a transparent input onto grey
(img2img_background_color) before sampling, which also gives the cut a plain background.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SIZE = 1024  # BiRefNet's training resolution
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
WEIGHTS = "birefnet_finetuned_toonout.pth"

# ponytail: one model, loaded on first use and kept on the GPU (~0.5 GB in fp16), outside
# Forge's memory management; callers hold Forge's queue_lock, so it never runs beside sampling
_model = None


def _load(folder: Path):
    global _model
    if _model is None:
        package = types.ModuleType("stagehand_toonout")  # birefnet.py imports its config relatively
        package.__path__ = [str(folder)]
        sys.modules[package.__name__] = package
        code = importlib.import_module("stagehand_toonout.birefnet")
        model = code.BiRefNet(config=code.BiRefNetConfig(bb_pretrained=False))
        state = torch.load(folder / WEIGHTS, map_location="cpu", weights_only=True)
        model.load_state_dict({k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in state.items()})
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = model.eval().to(device, torch.float16 if device == "cuda" else torch.float32)
    return _model


def rgba(image: Image.Image) -> Image.Image:
    """Greyscale+alpha and palette images with transparency as RGBA: Forge's flatten only puts
    RGBA on grey, the rest would lose their alpha to whatever colour is stored under it."""
    if image.mode in ("LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        return image.convert("RGBA")
    return image


def has_alpha(image: Image.Image | None) -> bool:
    return image is not None and image.mode == "RGBA" and image.getextrema()[-1][0] < 255


@torch.inference_mode()
def matte(image: Image.Image, folder: Path) -> Image.Image:
    """The foreground's alpha, at the image's size."""
    model = _load(folder)
    param = next(model.parameters())
    x = np.asarray(image.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC), np.float32) / 255
    x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)[None]).to(param.device, param.dtype)
    alpha = model(x)[-1].sigmoid()[0, 0].float().cpu().numpy()
    return Image.fromarray((alpha * 255).round().astype(np.uint8)).resize(image.size, Image.BILINEAR)


def cut_out(image: Image.Image, cut: Image.Image, source: Image.Image | None = None, mask: Image.Image | None = None) -> Image.Image:
    """`image` with `cut` as its alpha -- and where `mask` (white = repainted) didn't repaint,
    `source`'s own pixels (RGBA, the output's size), so an edge isn't re-blended with grey."""
    out = image.convert("RGBA")
    out.putalpha(cut)
    if source is not None and mask is not None:
        out = Image.composite(out, source, mask.convert("L").resize(out.size, Image.BILINEAR))
    out.info = dict(image.info)
    return out


if __name__ == "__main__":
    # self-check, no model needed
    grey = Image.new("RGB", (4, 4), (128, 128, 128))  # what Forge sampled on
    source = Image.new("RGBA", (4, 4), (0, 0, 0, 128))  # a half-transparent black edge
    mask = Image.new("L", (4, 4), 0)
    mask.paste(255, (0, 0, 2, 4))  # left half repainted
    out = cut_out(grey, Image.new("L", (4, 4), 0), source, mask)
    assert out.getpixel((0, 0)) == (128, 128, 128, 0), out.getpixel((0, 0))  # repainted: cut
    assert out.getpixel((3, 0)) == (0, 0, 0, 128), out.getpixel((3, 0))  # kept: no grey fringe
    assert cut_out(grey, Image.new("L", (4, 4), 9)).getpixel((3, 3)) == (128, 128, 128, 9)
    assert not has_alpha(Image.new("RGB", (2, 2))) and not has_alpha(Image.new("RGBA", (2, 2), (0, 0, 0, 255)))
    assert has_alpha(Image.new("RGBA", (2, 2))) and has_alpha(rgba(Image.new("LA", (2, 2))))
    assert rgba(Image.new("L", (2, 2))).mode == "L"
    print("transparent: ok")
