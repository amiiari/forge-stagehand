"""Transparent: images come out with their background cut away (lib_stagehand/transparent.py).

The "Transparent" pill in the Stagehand header turns it on. img2img does it by itself for an
input that already has transparency (a cut-out, or a Krita layer), and so does txt2img's
hires fix of a transparent result, so an edit stays transparent instead of coming back on a
flat background.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import gradio as gr

from modules import images, script_callbacks, scripts, shared
from modules.paths_internal import models_path

from lib_stagehand import transparent

FOLDER = Path(models_path) / "toonout"
LABEL = "Transparent background"
MAX_PIXELS = 4096 * 4096  # the matte endpoint's biggest image


def _off_on_paste(infotext, params):
    """Pasting an opaque image's PNG info switches it off again."""
    if "Steps" in params:
        params.setdefault(LABEL, "False")


script_callbacks.on_infotext_pasted(_off_on_paste)


def _api(_demo, app):
    """POST /stagehand/v1/matte {"image": base64} -> {"alpha": base64 PNG, "image": base64 PNG}:
    the cut on its own -- its alpha, and the image as RGBA with the edge colours cleaned --
    for images made elsewhere (Krita's ComfyUI generations). Behind --api and --api-auth like
    Forge's own routes, and in Forge's queue: it waits for a generation that's running."""
    from secrets import compare_digest

    from fastapi import Body, Depends, HTTPException
    from fastapi.security import HTTPBasic, HTTPBasicCredentials

    from modules.api.api import decode_base64_to_image
    from modules.call_queue import queue_lock

    if not shared.cmd_opts.api:
        return
    users = dict(a.split(":", 1) for a in (shared.cmd_opts.api_auth or "").split(",") if a)

    def auth(credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
        if credentials.username in users and compare_digest(credentials.password, users[credentials.username]):
            return True
        raise HTTPException(401, "Incorrect username or password", headers={"WWW-Authenticate": "Basic"})

    def matte(body: dict = Body(...)):
        if not isinstance(body.get("image"), str):
            raise HTTPException(422, '"image" (base64) is required')
        image = decode_base64_to_image(body["image"])
        if image.width * image.height > MAX_PIXELS:
            raise HTTPException(413, f"at most {MAX_PIXELS} pixels")
        with queue_lock:
            rgb, alpha = transparent.cutout(image, FOLDER)

        def png(im):  # PNG whatever Forge saves images as: the matte must be exact
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()

        return {"alpha": png(alpha), "image": png(transparent.cut_out(rgb, alpha))}

    app.add_api_route("/stagehand/v1/matte", matte, methods=["POST"], dependencies=[Depends(auth)] if users else [])


script_callbacks.on_app_started(_api)


class Transparent(scripts.Script):
    create_group = False

    def title(self):
        return "Transparent"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "i2i" if is_img2img else "t2i"
        # shown as a pill in the Stagehand header (javascript/stagehand.js)
        on = gr.Checkbox(value=False, label="Transparent", elem_id=f"nai_{tab}_alpha_on", elem_classes=["nai-hidden"])
        self.infotext_fields = [(on, LABEL)]
        self.paste_field_names = [LABEL]
        return [on]

    def process(self, p, on=False, *args):
        # before p.init(), which flattens the init images onto grey
        if getattr(p, "init_images", None):
            p.init_images = [transparent.rgba(i) for i in p.init_images]
        auto = any(transparent.has_alpha(i) for i in getattr(p, "init_images", None) or [])
        # txt2img's hires fix of a gallery image: Forge reads it as RGB, 4 channels would crash it
        first = getattr(p, "firstpass_image", None)
        if first is not None and transparent.has_alpha(transparent.rgba(first)):
            p.firstpass_image = images.flatten(transparent.rgba(first), shared.opts.img2img_background_color)
            auto = True
        # one p serves a whole img2img batch folder: set it per image, both ways
        if on or auto:
            p.extra_generation_params[LABEL] = True
        else:
            p.extra_generation_params.pop(LABEL, None)

    def postprocess_image_after_composite(self, p, pp, on=False, *args):
        if getattr(p, "_ad_inner", False) or not p.extra_generation_params.get(LABEL):
            return
        rgb, cut = transparent.cutout(pp.image, FOLDER)
        rgb.info = dict(pp.image.info)
        source = self._source(p, pp.index)
        mask = getattr(p, "mask_for_overlay", None) if getattr(p, "image_mask", None) is not None else None
        if not transparent.has_alpha(source) or mask is None:
            source = None
        elif source.size != pp.image.size:  # as Forge resized it ("Latent upscale" resizes plainly)
            mode = 0 if p.resize_mode == 3 else p.resize_mode
            source = images.resize_image(mode, source, *pp.image.size, force_RGBA=True)
        pp.image = transparent.cut_out(rgb, cut, source, mask)

    @staticmethod
    def _source(p, index):
        init = getattr(p, "init_images", None) or []
        return init[index % len(init)] if init else None
