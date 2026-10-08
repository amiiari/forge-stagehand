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
import json
import os
import re
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
# separates a character from its art style: a type changed nothing but the starting Strength,
# so the UI no longer shows it. Kept as an arg and in the PNG info for API callers.
REFERENCE_TYPES = ("Character", "Style", "Character & Style")
# For people who've never used NovelAI: collapsed under the panel's title until asked for.
HELP = """<details class="nai-help"><summary>How to use</summary><div>
<ol>
<li><b>+ Add reference</b> here adds a reference for the whole image: an image of a character to keep (face, hair,
outfit) or an art style to copy -- not a pose: it carries looks, not composition. Up to 4 references in all, e.g. the
same character from several angles. Your prompt still sets the scene.</li>
<li><b>A character's own references</b> go in the <b>Reference</b> tab of her Character Prompts card: they only go
into her part of the image (and her face in ADetailer), so two characters can each keep their own looks. References
here, for the whole image, blend: two different characters become one (NovelAI's do the same).</li>
<li><b>Strength:</b> how much of the reference goes in: about 1 for a character, about 0.5 for an art style (at 1 it
copies the whole artwork). 0 turns the card off; below 0 pushes away from it.</li>
<li><b>Fidelity:</b> how hard the reference is to override with your prompt. 0.6 by default (it beat 1.0 in a blind
test); lower it if the prompt (pose, outfit) isn't being followed, raise it for a closer copy.</li>
<li><b>Hires fix / ADetailer:</b> also use this card in that pass. Off: the card only shapes the first pass, which
looked best in testing. In ADetailer, a character's card goes only to her own face. Both off by default.</li>
</ol>
<p>Clean images on plain backgrounds work best: a busy or dark background gets copied into the picture.
Anima only.</p>
</div></details>"""

# Cards are pre-built and revealed by "+": Gradio can't add components at runtime.
MAX_REFS = 4
CARD_FIELDS = 4  # image, type, strength, fidelity
# per card, added after [cards..., ADetailer (the old panel-wide flag), on]: for, in Hires, in ADetailer
EXTRA_FIELDS = 3
TARGETS = ["Whole image"] + [f"Character {n}" for n in range(1, 7)]  # Character Prompts' 6 cards
INFOTEXT = ("image", "type", "strength", "fidelity", "for", "hires", "ADetailer")


def _target(value):
    """'Character 2' (or 2, from an API caller) -> 2; anything else is the whole image."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    match = re.fullmatch(r"(?:Character )?(\d+)", str(value).strip())
    return int(match[1]) if match else None


def _cards(args):
    """Per card: (image, type, strength, fidelity, target, in Hires, in ADetailer). Forge's
    API pads an older caller's shorter list with the UI defaults (whole image, in neither
    pass); its panel-wide ADetailer flag, when set, still puts every card in ADetailer."""
    base = MAX_REFS * CARD_FIELDS
    legacy = len(args) > base and bool(args[base])
    extra = list(args[base + 2 :])
    out = []
    for i in range(MAX_REFS):
        more = extra[i * EXTRA_FIELDS : (i + 1) * EXTRA_FIELDS]
        target, hires, adetailer = more if len(more) == EXTRA_FIELDS else (None, False, False)
        out.append(tuple(args[i * CARD_FIELDS : (i + 1) * CARD_FIELDS]) + (_target(target), bool(hires), bool(adetailer) or legacy))
    return out


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


def _fill_defaults(params):
    """What a card the PNG info doesn't spell out had. Images from before the per-card
    options: every reference was in the hires pass, and in ADetailer's when "PR in
    ADetailer" was set."""
    legacy = str(params.get("PR in ADetailer", "False"))
    for n in range(1, MAX_REFS + 1):
        old = f"PR {n} image" in params and f"PR {n} hires" not in params
        params.setdefault(f"PR {n} type", "Character")
        params.setdefault(f"PR {n} strength", 1.0)
        params.setdefault(f"PR {n} fidelity", 1.0 if f"PR {n} image" in params else EMPTY[3])  # a fresh card: today's default
        params.setdefault(f"PR {n} for", TARGETS[0])
        params.setdefault(f"PR {n} hires", "True" if old else "False")
        params.setdefault(f"PR {n} ADetailer", legacy if old else "False")
    return params


def _defaults_on_paste(infotext, params):
    """Pasting restores the references exactly, including having none."""
    if "Steps" not in params or not _enabled():
        return
    _fill_defaults(params)


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
    a generation has a card ticked for ADetailer, this script joins that pass's filtered list
    for that generation only -- the setting itself is never touched."""
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
    """The panel's +: a whole-image card (a target an undo left on a hidden card is reset)."""
    shown = list(shown)
    targets = [gr.update()] * MAX_REFS
    if False in shown:
        i = shown.index(False)
        shown[i] = True
        targets[i] = TARGETS[0]
    return [shown] + [gr.update(visible=v) for v in shown] + targets


EMPTY = (None, "Character", 1.0, 0.6, TARGETS[0], False, False)  # a fresh card's fields


def _act(text, shown, *values):
    """What a character card does to her references (stagehand.js writes it, as JSON):
    add n -- a new card for character n; drop n -- remove hers (her card was deleted);
    copy from to -- her references again for the duplicate, as many as fit; swap a b -- the
    cards were reordered; targets [...] -- an undo puts them back.
    values: every card's (image, type, strength, fidelity, for, hires, ADetailer)."""
    shown = list(shown)
    size = len(EMPTY)
    cards = [list(values[i * size : (i + 1) * size]) for i in range(MAX_REFS)]
    try:
        action = json.loads(text)
        op = action["op"]
    except (ValueError, KeyError, TypeError):
        op = None
    target = lambda n: f"Character {int(n)}"  # noqa: E731
    owner = lambda card: _target(card[4])  # noqa: E731
    free = [i for i in range(MAX_REFS) if not shown[i]]
    if op == "add" and free:
        i = free[0]
        shown[i], cards[i] = True, list(EMPTY[:4]) + [target(action["n"])] + list(EMPTY[5:])
    elif op == "add":
        gr.Warning(f"All {MAX_REFS} references are in use. Remove one first.")
    elif op == "drop":
        for i in range(MAX_REFS):
            if shown[i] and owner(cards[i]) == int(action["n"]):
                shown[i], cards[i] = False, list(EMPTY)
    elif op == "copy":
        hers = [i for i in range(MAX_REFS) if shown[i] and owner(cards[i]) == int(action["from"])]
        for i, j in zip(hers, free):
            shown[j], cards[j] = True, cards[i][:4] + [target(action["to"])] + cards[i][5:]
        if len(hers) > len(free):
            gr.Warning(f"Copied {len(free)} of her {len(hers)} references: {MAX_REFS} at most.")
    elif op == "swap":
        a, b = int(action["a"]), int(action["b"])
        for card in cards:
            card[4] = target(b) if owner(card) == a else target(a) if owner(card) == b else card[4]
    elif op == "targets":
        for i, value in enumerate(list(action["values"])[:MAX_REFS]):
            if shown[i] and isinstance(value, str):
                cards[i][4] = value
    return [shown] + [gr.update(visible=v) for v in shown] + [v for card in cards for v in card]


def _remover(i):
    def remove(shown):
        shown = list(shown)
        shown[i] = False
        # back to a fresh card, so the next "+" doesn't reopen the old settings
        return (shown, gr.update(visible=False)) + EMPTY

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


class PreciseReference(scripts.Script):
    # javascript/stagehand.js moves the panel under the prompt boxes; a script
    # group would be left behind in the scripts column as an empty frame.
    create_group = False
    cards = MAX_REFS  # forge link reads the arg layout from this

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
                # the old panel-wide "also in ADetailer": kept as an arg for API callers, now per card
                in_adetailer = gr.Checkbox(value=False, visible=False)
                gr.HTML('<span class="nai-hint">clean images on plain backgrounds work best</span>', elem_classes=["nai-title-cell"])
                gr.HTML("", elem_classes=["nai-spacer"])  # pushes the add button to the right
                add = gr.Button("+ Add reference", elem_classes=["nai-add"], min_width=40, scale=0)
            shown = gr.State([False] * MAX_REFS)
            # a character card's add / delete / duplicate / reorder, from stagehand.js (_act)
            action = gr.Textbox(value="", elem_id=f"nai_{tab}_pr_action", elem_classes=["nai-hidden"], show_label=False, container=False)
            cards = []
            buttons = [add]
            extras = []
            every = []  # each card's fields in _act's order

            for i in range(MAX_REFS):
                # stagehand.js moves a character's card into her Character Prompts card's Reference tab
                with gr.Group(visible=False, elem_id=f"nai_{tab}_pr{i + 1}", elem_classes=["nai-card", "nai-ref-card"]) as card:
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
                            with gr.Row(elem_classes=["nai-card-head", "nai-ref-head"]):
                                kind = gr.Dropdown(list(REFERENCE_TYPES), value="Character", show_label=False, container=False,
                                                   elem_id=f"nai_{tab}_pr{i + 1}_type", elem_classes=["nai-hidden"])
                                # "Whole image" or "Character N": stagehand.js shows it in her card's Reference tab
                                target = gr.Textbox(value=TARGETS[0], show_label=False, container=False,
                                                    elem_id=f"nai_{tab}_pr{i + 1}_for", elem_classes=["nai-hidden", "nai-ref-for"])
                                gr.HTML("", elem_classes=["nai-spacer"])
                                remove = gr.Button("🗑", elem_classes=["nai-icon", "nai-remove-ref"], min_width=36, scale=0)
                            strength = gr.Slider(label="Strength", minimum=-1.0, maximum=2.0, step=0.01, value=1.0)
                            fidelity = gr.Slider(label="Fidelity", minimum=0.0, maximum=1.0, step=0.01, value=0.6)
                            with gr.Row(elem_classes=["nai-ref-passes"]):
                                hires = gr.Checkbox(value=False, label="Hires fix", elem_id=f"nai_{tab}_pr{i + 1}_hires", elem_classes=["nai-auto", "nai-ref-hires"], scale=0, min_width=110)
                                adetailer = gr.Checkbox(value=False, label="ADetailer", elem_id=f"nai_{tab}_pr{i + 1}_adetailer", elem_classes=["nai-auto", "nai-ref-adetailer"], scale=0, min_width=110)
                cards.append(card)
                buttons.append(remove)

                remove.click(fn=_remover(i), inputs=[shown], outputs=[shown, card, image, kind, strength, fidelity, target, hires, adetailer], show_progress=False)
                image.change(fn=_revealer(i), inputs=[image, shown], outputs=[shown, card], show_progress=False)

                fields = [image, kind, strength, fidelity]
                components += fields
                extras += [target, hires, adetailer]
                every += [image, kind, strength, fidelity, target, hires, adetailer]
                infotext += [
                    (image, lambda params, n=i + 1: _pasted_image(params, n)),
                    (kind, f"PR {i + 1} type"),
                    (strength, f"PR {i + 1} strength"),
                    (fidelity, f"PR {i + 1} fidelity"),
                    (target, f"PR {i + 1} for"),
                    (hires, f"PR {i + 1} hires"),
                    (adetailer, f"PR {i + 1} ADetailer"),
                ]

            add.click(fn=_add_card, inputs=[shown], outputs=[shown] + cards + extras[::EXTRA_FIELDS], show_progress=False)
            action.input(fn=_act, inputs=[action, shown] + every, outputs=[shown] + cards + every, show_progress=False)
            gr.HTML(HELP)
            components.append(in_adetailer)
            for component in components + extras + buttons + [action]:
                # ui-config.json keys come from a component's LABEL, so every card would
                # collapse onto one entry -- and a saved entry overrides the code.
                component.do_not_save_to_config = True

        # pasting an image made with references switches the feature on
        infotext.append((on, lambda params: True if any(params.get(f"PR {n} image") for n in range(1, MAX_REFS + 1)) else None))
        self.infotext_fields = infotext
        self.paste_field_names = [label for _, label in infotext if isinstance(label, str)]
        # [4 x (image, type, strength, fidelity), ADetailer (old, panel-wide), on/off,
        #  4 x (for, in Hires, in ADetailer)]: added fields go last
        return components + [on] + extras

    def args_from_infotext(self, params):
        """This script's args for re-running an image from its PNG info (the batch hires-fix
        tab calls this): its references come back from the kept copies. None if it had none."""
        params = _fill_defaults(dict(params))
        args, extras, found = [], [], False
        for n in range(1, MAX_REFS + 1):
            path = params.get(f"PR {n} image")
            path = path if path and os.path.isfile(path) else None
            found = found or path is not None
            try:
                strength, fidelity = float(params[f"PR {n} strength"]), float(params[f"PR {n} fidelity"])
            except (TypeError, ValueError):
                strength, fidelity = 1.0, 1.0
            args += [path, params[f"PR {n} type"], strength, fidelity]
            extras += [params[f"PR {n} for"], str(params[f"PR {n} hires"]) == "True", str(params[f"PR {n} ADetailer"]) == "True"]
        return args + [False, True] + extras if found else None

    def process_before_every_sampling(self, p, *args, **kwargs):
        # the previous pass's reference K/V (~224 MB each on the GPU) is done with
        for old in getattr(p, "_pr_sessions", []):
            old.clear()
        p._pr_sessions = []
        on = args[MAX_REFS * CARD_FIELDS + 1] if len(args) > MAX_REFS * CARD_FIELDS + 1 else True
        cards = _cards(args)
        if on is False or not any(has_image(card[0]) for card in cards):
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

        # Which pass this is: the first, the hires fix's (Batch Hires-Fix runs only that one),
        # or one of ADetailer's faces -- which hands over every script when its "only selected
        # scripts" is off, so a card not ticked for it must be kept out here.
        hr = bool(getattr(p, "is_hr_pass", False))
        inner = bool(getattr(p, "_ad_inner", False))
        face = getattr(p, "_nai_character", None)  # whose face, from Character Prompts' matching
        numbers = (getattr(p, "_nai_chars", None) or {}).get("numbers") or []

        # Encode every card before recording any: one that fails must not leave the
        # infotext claiming references that were never applied. Every card in use is
        # recorded, also on a pass it sits out, so the image keeps all of them.
        references, used, applied = [], [], []
        for i, (source, kind, strength, fidelity, target, in_hires, in_adetailer) in enumerate(cards):
            image = _as_array(source)
            strength = float(overrides.get(f"{i}:strength", strength))
            fidelity = min(max(float(overrides.get(f"{i}:fidelity", fidelity)), 0.0), 1.0)
            if image is None or strength == 0.0:  # negative is meaningful (push away); zero is not
                continue
            try:
                path = _keep(image)
            except OSError as e:  # the reference still applies; only its PNG info path is lost
                print(f"[Precise Reference] couldn't keep a copy of reference {i + 1}: {e!r}")
                path = None
            used.append((i, kind, strength, fidelity, path, target, in_hires, in_adetailer))
            if (hr and not in_hires) or (inner and not in_adetailer):
                continue
            if inner and target is not None and target != face:
                continue  # another character's face, or one no character owns
            if not inner and target is not None and target not in numbers:
                if not hr:
                    note = f"reference {i + 1} is for Character {target}, who isn't in this image -- skipped"
                    print(f"[Precise Reference] {note}")
                    p.comment(f"Precise Reference: {note}")
                continue
            # Fidelity is how much CFG amplifies the reference: at 1 only the positive pass
            # sees it (hardest to override with the prompt), at 0 both passes do. A face
            # crop is all one character's, so there it goes everywhere.
            tokens = adapter_runtime.reference_tokens(image)
            references.append(IPReference(tokens, strength, strength * (1.0 - fidelity), None if inner else target))
            applied.append((kind, strength, fidelity, target))
        if not used:
            return

        if not inner:
            # ADetailer's face pass runs only its selected scripts: _hook_adetailer adds this one.
            # (A card in it: better likeness in testing, and ~3 s/image faster since the model
            # isn't re-patched between passes.)
            p._pr_in_adetailer = any(card[-1] for card in used)
        for i, kind, strength, fidelity, path, target, in_hires, in_adetailer in used:
            if path:
                p.extra_generation_params[f"PR {i + 1} image"] = path
            p.extra_generation_params[f"PR {i + 1} type"] = kind
            p.extra_generation_params[f"PR {i + 1} strength"] = round(strength, 3)
            p.extra_generation_params[f"PR {i + 1} fidelity"] = round(fidelity, 3)
            if target is not None:
                p.extra_generation_params[f"PR {i + 1} for"] = f"Character {target}"
            p.extra_generation_params[f"PR {i + 1} hires"] = in_hires
            p.extra_generation_params[f"PR {i + 1} ADetailer"] = in_adetailer
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
        where = "ADetailer" if inner else "hires" if hr else "first pass"
        print(f"[Precise Reference] {where}: " + ", ".join(
            f"{k} {s:+.2f} fid {f:.2f}" + (f" -> Character {t}" if t is not None else "") for k, s, f, t in applied))

    def postprocess_batch(self, p, *args, **kwargs):
        # each batch's sampling is over (ADetailer, which runs after this, makes its own)
        self.postprocess(p, None)

    def postprocess(self, p, processed, *args):
        # the session stays reachable from forge_objects until the next generation resets it
        for session in getattr(p, "_pr_sessions", []):
            session.clear()
