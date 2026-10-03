"""Precise Reference — NovelAI-style reference images for Anima, in Forge Neo.

Runs the trained Anima IP-Adapter (LuciferTC/Anima-IP-Adapter, "Character_Reference"):
SigLIP2 encodes the reference and every DiT block gets an extra cross-attention onto it.
lib_stagehand/ip_adapter.py holds the port; NOTES.md has the A/B results that
picked it over the training-free reference-attention method this extension started with.

Anima's attention is wrapped once, for this and Character Prompts, in
lib_stagehand/anima_hooks.py; the adapter and encoder live in adapter_runtime.py.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from modules import script_callbacks, scripts, shared
from modules.paths_internal import data_path

from backend.args import dynamic_args
from backend.patcher.lora import load_lora

from lib_stagehand import adapter_runtime, anima_hooks
from lib_stagehand.ip_adapter import IPReference, IPSession, flatten, has_image, image_key, lora_patch_source

anima_hooks.install()

# NovelAI's three types. The adapter has one mode, and the A/B grids found no setting that
# separates a character from its art style, so a type only picks the starting Strength.
TYPE_STRENGTH = {"Character": 1.0, "Style": 0.5, "Character & Style": 1.0}
REFERENCE_TYPES = tuple(TYPE_STRENGTH)
# For people who've never used NovelAI: collapsed under the panel's title until asked for.
HELP = """<details class="nai-help"><summary>How to use</summary><div>
<ol>
<li><b>+ Add reference</b> adds a card: an image of a character to keep (face, hair, outfit), an art style to copy, or
a composition to follow. Up to 4 cards, all blended together -- e.g. the same character from several angles. Your
prompt still sets the scene. For solo images or the whole image: two different characters' references blend into
one character (NovelAI's do the same).</li>
<li><b>Type</b> only sets a starting Strength: 1.0 for Character and Character &amp; Style, 0.5 for Style.</li>
<li><b>Strength:</b> how much of the reference goes in. 0 turns the card off; below 0 pushes away from it.</li>
<li><b>Fidelity:</b> how hard the reference is to override with your prompt. Lower it if the prompt
(pose, outfit) isn't being followed.</li>
<li><b>also in ADetailer:</b> keeps the reference when ADetailer repaints faces, so the likeness holds better.
Off by default.</li>
</ol>
<p>Clean images on plain backgrounds work best: a busy or dark background gets copied into the picture.
Anima only.</p>
</div></details>"""

# Cards are pre-built and revealed by "+": Gradio can't add components at runtime.
MAX_REFS = 4
CARD_FIELDS = 4  # image, type, strength, fidelity
INFOTEXT = ("image", "type", "strength", "fidelity")


# The browser never says where a dropped file came from (and Gradio hands over the pixels, not
# the file), so each reference is kept as a copy here, and that path goes into the image's PNG
# info: pasting the image -- or a batch tab re-running it -- loads the reference back.
REFERENCE_DIR = os.path.join(data_path, "outputs", "stagehand references")


def _as_array(image):
    """UI images arrive as PIL, API callers send base64 or a path; "" is no image."""
    if not has_image(image):
        return None
    if isinstance(image, str):
        if os.path.isfile(image):
            with Image.open(image) as im:
                return flatten(im)
        from modules.api.api import decode_base64_to_image

        image = decode_base64_to_image(image)
    return flatten(image)


def _keep(array) -> str:
    """Path of the kept copy of a reference (as used: transparency already flattened). Named
    by content, so the same reference is kept once however often it's used."""
    path = os.path.join(REFERENCE_DIR, f"reference {image_key(array)[:12]}.png")
    if not os.path.exists(path):
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        # a full disk mid-save must not leave a broken file that exists() then reuses forever
        tmp = path + ".tmp"
        Image.fromarray(array).save(tmp, format="PNG")
        os.replace(tmp, path)
    return path


def _pasted_image(params, n):
    """Paste: the kept copy back into card n -- or an empty card, when the image had none."""
    if "Steps" not in params:
        return None
    path = params.get(f"PR {n} image")
    if path and os.path.isfile(path):
        return gr.update(value=path)
    if path:
        print(f"[Precise Reference] reference {n} isn't at {path} anymore -- card left empty")
    return gr.update(value=None)


def _enabled() -> bool:
    """Settings > Stagehand > Precise Reference. Off, show() keeps Forge from building the
    section or ever running the script, so its XYZ axes and ADetailer hook (set up in ui())
    never appear either."""
    return bool(getattr(shared.opts, "stagehand_precise_reference", True))


def _defaults_on_paste(infotext, params):
    """Pasting restores the references exactly, including having none."""
    if "Steps" not in params or not _enabled():
        return
    params.setdefault("PR in ADetailer", "False")
    for i in range(MAX_REFS):
        params.setdefault(f"PR {i + 1} type", "Character")
        params.setdefault(f"PR {i + 1} strength", 1.0)
        params.setdefault(f"PR {i + 1} fidelity", 1.0)


script_callbacks.on_infotext_pasted(_defaults_on_paste)


_xyz_registered = False


def _register_xyz() -> None:
    global _xyz_registered
    if _xyz_registered:
        return

    module = None
    for data in scripts.scripts_data:
        if data.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid"):
            module = data.module
            break
    if module is None:
        return

    def apply(field):
        def setter(p, x, xs):
            overrides = getattr(p, "_pr_xyz", None)
            if overrides is None:
                overrides = {}
                setattr(p, "_pr_xyz", overrides)
            overrides[field] = float(x)

        return setter

    # Keyed by card position, since a card's type is chosen in the UI.
    for i in range(MAX_REFS):
        module.axis_options.extend(
            [
                module.AxisOption(f"[Precise Ref] Ref {i + 1} strength", float, apply(f"{i}:strength")),
                module.AxisOption(f"[Precise Ref] Ref {i + 1} fidelity", float, apply(f"{i}:fidelity")),
            ]
        )

    _xyz_registered = True


def _hook_adetailer() -> None:
    """ADetailer's face pass runs only the scripts named in its "Script names" setting. When
    a generation ticks "also in ADetailer", this script joins that pass's filtered list for
    that generation only -- the setting itself is never touched."""
    for data in scripts.scripts_data:
        cls = data.script_class
        if Path(data.path).stem != "!adetailer" or not hasattr(cls, "script_filter"):
            continue
        if getattr(cls.script_filter, "_pr_hooked", False):
            return
        original = cls.script_filter

        def script_filter(self, p, *args, **kwargs):
            runner, script_args = original(self, p, *args, **kwargs)
            if getattr(p, "_pr_in_adetailer", False):
                ours = [s for s in p.scripts.alwayson_scripts if Path(s.filename).stem == "precise_reference"]
                runner.alwayson_scripts = runner.alwayson_scripts + [s for s in ours if s not in runner.alwayson_scripts]
                runner._pr_xyz = getattr(p, "_pr_xyz", None)  # an XYZ cell's strength, not the UI's
            return runner, script_args

        script_filter._pr_hooked = True
        cls.script_filter = script_filter
        return


def _add_card(shown):
    shown = list(shown)
    if False in shown:
        shown[shown.index(False)] = True
    return [shown] + [gr.update(visible=v) for v in shown]


def _remover(i):
    def remove(shown):
        shown = list(shown)
        shown[i] = False
        # back to a fresh card, so the next "+" doesn't reopen the old settings
        return shown, gr.update(visible=False), None, "Character", 1.0, 1.0

    return remove


def _revealer(i):
    """A pasted reference lands in a hidden card; show the card when that happens."""

    def reveal(image, shown):
        if not has_image(image) or shown[i]:
            return gr.update(), gr.update()
        shown = list(shown)
        shown[i] = True
        return shown, gr.update(visible=True)

    return reveal


def _type_strength(kind):
    return gr.update(value=TYPE_STRENGTH.get(kind, 1.0))


class PreciseReference(scripts.Script):
    # javascript/stagehand.js moves the panel under the prompt boxes; a script
    # group would be left behind in the scripts column as an empty frame.
    create_group = False

    def title(self):
        return "Precise Reference"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if _enabled() else False

    def ui(self, is_img2img):
        for setup in (_register_xyz, _hook_adetailer):
            try:
                setup()
            except Exception as e:  # without the panel, every generation would fail on its args
                print(f"[Precise Reference] {setup.__name__} failed, carrying on without it: {e!r}")
        tab = "i2i" if is_img2img else "t2i"
        components = []
        infotext = []

        with gr.Column(elem_id=f"nai_{tab}_pr", elem_classes=["nai-panel"]):
            # The feature's on/off: shown as a pill in the Stagehand header (javascript/stagehand.js).
            on = gr.Checkbox(value=True, label="Precise Reference", elem_id=f"nai_{tab}_pr_on", elem_classes=["nai-hidden"])
            with gr.Row(elem_classes=["nai-head"]):
                gr.HTML('<div class="nai-section">References</div>', elem_classes=["nai-title-cell"])
                in_adetailer = gr.Checkbox(value=False, label="also in ADetailer", elem_id=f"nai_{tab}_pr_adetailer", elem_classes=["nai-auto"], scale=0, min_width=150)
                gr.HTML('<span class="nai-hint">clean images on plain backgrounds work best</span>', elem_classes=["nai-title-cell"])
                gr.HTML("", elem_classes=["nai-spacer"])  # pushes the add button to the right
                add = gr.Button("+ Add reference", elem_classes=["nai-add"], min_width=40, scale=0)
            shown = gr.State([False] * MAX_REFS)
            cards = []
            buttons = [add]

            for i in range(MAX_REFS):
                with gr.Group(visible=False, elem_classes=["nai-card"]) as card:
                    with gr.Row(equal_height=False):
                        image = gr.Image(
                            show_label=False,
                            # Not "filepath": Forge's patched save_pil_to_file rejects the
                            # `name` Gradio passes it, and every upload failed.
                            type="pil",
                            image_mode="RGBA",  # keep alpha so it can be flattened onto white
                            height=170,
                            sources=["upload", "clipboard"],
                            elem_id=f"nai_{tab}_pr{i + 1}_image",
                            elem_classes=["nai-ref-image"],
                            scale=1,
                            min_width=140,
                        )
                        with gr.Column(scale=3, min_width=220):
                            with gr.Row(elem_classes=["nai-card-head"]):
                                kind = gr.Dropdown(
                                    list(REFERENCE_TYPES),
                                    value="Character",
                                    show_label=False,
                                    container=False,
                                    elem_id=f"nai_{tab}_pr{i + 1}_type",
                                )
                                remove = gr.Button("🗑", elem_classes=["nai-icon", "nai-remove-ref"], min_width=36, scale=0)
                            strength = gr.Slider(label="Strength", minimum=-1.0, maximum=2.0, step=0.01, value=1.0)
                            fidelity = gr.Slider(label="Fidelity", minimum=0.0, maximum=1.0, step=0.01, value=1.0)
                cards.append(card)
                buttons.append(remove)

                # .input, not .change: pasting infotext sets the type too, and .change would
                # then overwrite the pasted strength with the type's default.
                kind.input(fn=_type_strength, inputs=[kind], outputs=[strength], show_progress=False)
                remove.click(fn=_remover(i), inputs=[shown], outputs=[shown, card, image, kind, strength, fidelity], show_progress=False)
                image.change(fn=_revealer(i), inputs=[image, shown], outputs=[shown, card], show_progress=False)

                fields = [image, kind, strength, fidelity]
                components += fields
                infotext += [
                    (image, lambda params, n=i + 1: _pasted_image(params, n)),
                    (kind, f"PR {i + 1} type"),
                    (strength, f"PR {i + 1} strength"),
                    (fidelity, f"PR {i + 1} fidelity"),
                ]

            add.click(fn=_add_card, inputs=[shown], outputs=[shown] + cards, show_progress=False)
            gr.HTML(HELP)
            components.append(in_adetailer)
            for component in components + buttons:
                # ui-config.json keys come from a component's LABEL, so every card would
                # collapse onto one entry -- and a saved entry overrides the code.
                component.do_not_save_to_config = True

        infotext.append((in_adetailer, "PR in ADetailer"))
        # pasting an image made with references switches the feature on
        infotext.append((on, lambda params: True if any(params.get(f"PR {n} image") for n in range(1, MAX_REFS + 1)) else None))
        self.infotext_fields = infotext
        self.paste_field_names = [label for _, label in infotext if isinstance(label, str)]
        # [4 x (image, type, strength, fidelity), also in ADetailer, on/off]: added fields go last
        return components + [on]

    def args_from_infotext(self, params):
        """This script's args for re-running an image from its PNG info (the batch hires-fix
        tab calls this): its references come back from the kept copies. None if it had none."""
        args, found = [], False
        for i in range(MAX_REFS):
            path = params.get(f"PR {i + 1} image")
            path = path if path and os.path.isfile(path) else None
            found = found or path is not None
            try:
                strength, fidelity = float(params.get(f"PR {i + 1} strength", 1.0)), float(params.get(f"PR {i + 1} fidelity", 1.0))
            except (TypeError, ValueError):
                strength, fidelity = 1.0, 1.0
            args += [path, params.get(f"PR {i + 1} type", "Character"), strength, fidelity]
        return args + [str(params.get("PR in ADetailer", "")) == "True", True] if found else None

    def process_before_every_sampling(self, p, *args, **kwargs):
        # the previous pass's reference K/V (~224 MB each on the GPU) is done with
        for old in getattr(p, "_pr_sessions", []):
            old.clear()
        p._pr_sessions = []
        on = args[MAX_REFS * CARD_FIELDS + 1] if len(args) > MAX_REFS * CARD_FIELDS + 1 else True
        cards = [args[i * CARD_FIELDS : (i + 1) * CARD_FIELDS] for i in range(MAX_REFS)]
        if on is False or not any(has_image(card[0]) for card in cards):
            return
        # ADetailer hands over every script when its "only selected scripts" is off:
        # "also in ADetailer" unticked must still keep the reference out of the face pass
        if getattr(p, "_ad_inner", False) and not (len(args) > MAX_REFS * CARD_FIELDS and args[MAX_REFS * CARD_FIELDS]):
            return
        overrides = getattr(p, "_pr_xyz", None) or getattr(getattr(p, "scripts", None), "_pr_xyz", None) or {}
        # A copied p (XYZ cells) shares extra_generation_params: start from a clean slate.
        p.extra_generation_params.pop("PR in ADetailer", None)
        for i in range(MAX_REFS):
            for key in INFOTEXT:
                p.extra_generation_params.pop(f"PR {i + 1} {key}", None)

        unet = p.sd_model.forge_objects.unet
        blocks = anima_hooks.tag_blocks(getattr(unet.model, "diffusion_model", None))
        if not blocks:
            print("[Precise Reference] the adapter is trained for Anima only; this checkpoint isn't Anima -- skipping.")
            return
        try:
            adapter, lora, metadata = adapter_runtime.adapter()
        except FileNotFoundError as e:  # one line and a note in the image info, not a traceback per image
            print(f"[Precise Reference] {e}")
            p.comment(f"Precise Reference skipped: {e}")
            return
        if len(blocks) != len(adapter.blocks):
            # e.g. the 40-block anima29B: same width, so it would run until block 28 indexed
            # past the adapter and killed the generation mid-sampling
            print(f"[Precise Reference] the adapter has weights for {len(adapter.blocks)} blocks; this checkpoint has {len(blocks)} -- skipping.")
            return

        # Encode every card before recording any: one that fails must not leave the
        # infotext claiming references that were never applied.
        references, used = [], []
        for i, (source, kind, strength, fidelity) in enumerate(cards):
            image = _as_array(source)
            strength = float(overrides.get(f"{i}:strength", strength))
            fidelity = min(max(float(overrides.get(f"{i}:fidelity", fidelity)), 0.0), 1.0)
            if image is None or strength == 0.0:  # negative is meaningful (push away); zero is not
                continue
            # Fidelity is how much CFG amplifies the reference: at 1 only the positive pass
            # sees it (hardest to override with the prompt), at 0 both passes do.
            references.append(IPReference(adapter_runtime.reference_tokens(image), strength, strength * (1.0 - fidelity)))
            try:
                path = _keep(image)
            except OSError as e:  # the reference still applies; only its PNG info path is lost
                print(f"[Precise Reference] couldn't keep a copy of reference {i + 1}: {e!r}")
                path = None
            used.append((i, kind, strength, fidelity, path))
        if not references:
            return

        unet = unet.clone()
        # The checkpoint's LoRA covers self_attn, cross_attn and mlp; all of it is loaded.
        # (Without it the adapter copies the reference's pose -- see NOTES.md.)
        alpha = metadata.get("lora_alpha")
        source, to_load = lora_patch_source(lora, float(alpha) if alpha else None)
        patches, _ = load_lora(source, to_load)
        base_uuid = unet.patches_uuid
        loaded = unet.add_patches(filename=adapter_runtime.ADAPTER_FILE, patches=patches, strength_patch=1.0, online_mode=dynamic_args.online_lora)
        if len(loaded) != len(patches):
            print(f"[Precise Reference] adapter LoRA matched only {len(loaded)} of {len(patches)} keys")
        # add_patches mints a fresh uuid, so Forge re-patched the whole model every
        # generation (~1.6 s). The same base weights + the same adapter LoRA are the same
        # patch set, so derive the id from exactly those and let Forge keep them patched.
        unet.patches_uuid = uuid.uuid5(base_uuid, adapter_runtime.ADAPTER_FILE)

        session = IPSession(adapter, references)
        unet.add_extra_model_patcher_during_sampling(adapter_runtime.patcher(unet, adapter))
        unet.set_transformer_option(anima_hooks.IP_KEY, session)
        p.sd_model.forge_objects.unet = unet
        p._pr_sessions = [session]
        # Off by default. On: ADetailer's face pass gets the reference too (better likeness
        # in testing, and ~3 s/image faster since the model isn't re-patched between passes).
        p._pr_in_adetailer = len(args) > MAX_REFS * CARD_FIELDS and bool(args[MAX_REFS * CARD_FIELDS])

        if p._pr_in_adetailer:
            p.extra_generation_params["PR in ADetailer"] = True
        for i, kind, strength, fidelity, path in used:
            if path:
                p.extra_generation_params[f"PR {i + 1} image"] = path
            p.extra_generation_params[f"PR {i + 1} type"] = kind
            p.extra_generation_params[f"PR {i + 1} strength"] = round(strength, 3)
            p.extra_generation_params[f"PR {i + 1} fidelity"] = round(fidelity, 3)
        print("[Precise Reference] " + ", ".join(f"{k} {s:+.2f} fid {f:.2f}" for _, k, s, f, _ in used))

    def postprocess_batch(self, p, *args, **kwargs):
        # each batch's sampling is over (ADetailer, which runs after this, makes its own)
        self.postprocess(p, None)

    def postprocess(self, p, processed, *args):
        # the session stays reachable from forge_objects until the next generation resets it
        for session in getattr(p, "_pr_sessions", []):
            session.clear()
