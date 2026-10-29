"""Transparent backgrounds. Anima has no alpha channel of its own (LayerDiffuse is SDXL/SD1.5
only), so a finished image has its background cut out by ToonOut: BiRefNet fine-tuned on
anime characters (joelseytre/toonout, MIT); ViTMatte then works out the edge (hair, fur,
lace) and the edge's colours are cleaned of the old background (cutout()). The weights,
BiRefNet's model code (ZhengPeng7/BiRefNet, pinned) and ViTMatte live in models/toonout.

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


def _input_size(image: Image.Image, keep_shape: bool) -> tuple[int, int]:
    """BiRefNet's input: its 1024x1024, or about as many pixels in the image's own shape."""
    if not keep_shape:
        return SIZE, SIZE
    scale = (SIZE * SIZE / (image.width * image.height)) ** 0.5
    return max(32, round(image.width * scale / 32) * 32), max(32, round(image.height * scale / 32) * 32)


@torch.inference_mode()
def matte(image: Image.Image, folder: Path, keep_shape=False) -> Image.Image:
    """The foreground's alpha, at the image's size."""
    model = _load(folder)
    param = next(model.parameters())
    x = np.asarray(image.convert("RGB").resize(_input_size(image, keep_shape), Image.BICUBIC), np.float32) / 255
    x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)[None]).to(param.device, param.dtype)
    alpha = model(x)[-1].sigmoid()[0, 0].float().cpu().numpy()
    return Image.fromarray((alpha * 255).round().astype(np.uint8)).resize(image.size, Image.BILINEAR)


_vitmatte = None


def _load_vitmatte(folder: Path):
    global _vitmatte
    if _vitmatte is None:
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = VitMatteForImageMatting.from_pretrained(folder / "vitmatte").eval().to(device)
        _vitmatte = model, VitMatteImageProcessor.from_pretrained(folder / "vitmatte")
    return _vitmatte


def trimap(alpha: Image.Image) -> Image.Image:
    """Sure foreground (255), sure background (0) and the band between (128) to work out,
    from a cut: its soft edge, widened by a few pixels each way."""
    import cv2

    a = np.asarray(alpha.convert("L"))
    k = max(3, round(max(a.shape) / 200)) | 1  # ~5 px at 1024, 13 at 2560
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.erode((a >= 242).astype(np.uint8), kernel)
    bg = cv2.erode((a <= 13).astype(np.uint8), kernel)
    return Image.fromarray(np.where(fg == 1, 255, np.where(bg == 1, 0, 128)).astype(np.uint8))


@torch.inference_mode()
def refine(image: Image.Image, alpha: Image.Image, folder: Path) -> Image.Image:
    """The cut's edge worked out by ViTMatte (hustvl/vitmatte-base-composition-1k): real
    partial transparency for hair and soft edges, where a cut only guesses."""
    model, processor = _load_vitmatte(folder)
    tri = trimap(alpha)
    inputs = processor(images=image.convert("RGB"), trimaps=tri, return_tensors="pt").to(model.device)
    out = model(**inputs).alphas[0, 0, : image.height, : image.width].float().cpu().numpy()  # the processor pads
    t = np.asarray(tri)
    out = np.where(t == 255, 1.0, np.where(t == 0, 0.0, out))  # only the band is worked out
    return Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8))


def decontaminate(image: Image.Image, alpha: Image.Image) -> Image.Image:
    """The foreground's own colour where it's partly transparent, without the old background
    blended in (no light or grey fringe): Forte & Pitie's blur-fusion estimate (2021)."""
    import cv2

    i = np.asarray(image.convert("RGB"), np.float32) / 255
    a = (np.asarray(alpha.convert("L"), np.float32) / 255)[:, :, None]

    def fuse(f, b, r):
        blurred_a = cv2.blur(a, (r, r))[:, :, None]
        blurred_f = cv2.blur(f * a, (r, r)) / (blurred_a + 1e-5)
        blurred_b = cv2.blur(b * (1 - a), (r, r)) / ((1 - blurred_a) + 1e-5)
        return np.clip(blurred_f + a * (i - a * blurred_f - (1 - a) * blurred_b), 0, 1), blurred_b

    f, b = fuse(i, i, 90)
    f, _ = fuse(f, b, 6)
    return Image.fromarray((f * 255).round().astype(np.uint8))


def cutout(image: Image.Image, folder: Path) -> tuple[Image.Image, Image.Image]:
    """The cut (colour, alpha): ToonOut's mask, its edge worked out by ViTMatte, the edge's
    colours cleaned. Won the user's blind round (NOTES.md, "Transparent")."""
    alpha = refine(image, matte(image, folder), folder)
    return decontaminate(image, alpha), alpha


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
