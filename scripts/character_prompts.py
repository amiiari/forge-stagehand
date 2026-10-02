"""Character Prompts — NovelAI-style multi-character prompting for Anima, in Forge Neo.

A box per character (prompt + Undesired Content), placed by "AI's Choice" (equal columns)
or by boxes dragged over the output image. lib_precise_reference/characters.py has the
regional attention and the prompt plumbing; this file is the UI and the Forge wiring:

  before_process        merge the boxes into the prompt with markers, so Set Queue words
                        and Dynamic Prompts wildcards roll inside them too
  before_process_batch  split them back out; the stored prompts become the base alone, so
                        ADetailer and the infotext never see a marker
  process_batch         add the interaction-tag sentences to what gets encoded
  every sampling pass   encode each character, install the regional session
"""

from __future__ import annotations

import functools
import inspect
import re
from pathlib import Path

import gradio as gr

from modules import processing, prompt_parser, script_callbacks, scripts, sd_samplers
from modules.processing_scripts.comments import strip_comments

from lib_precise_reference import anima_hooks
from lib_precise_reference.characters import (
    RegionSession,
    auto_boxes,
    center,
    default_cells,
    format_cell,
    has_marks,
    match_faces,
    merge,
    parse_cell,
    placement,
    read,
    real_length,
    region_weights,
    show,
    split,
    translate_actions,
)

anima_hooks.install()

MAX_CHARS = 6  # NovelAI V4.5's limit; regional prompting gets unreliable well before V5's 22
CARD_FIELDS = 6  # enabled, name, prompt, uc, box, face -- a card's components in the UI
ARG_FIELDS = 5  # the args carry each card's first five, then every card's face at the end
PROMPT = 2  # index of the prompt within a card's fields
FACE = 5  # ...and of its ADetailer face pick
NTH = ("1st", "2nd", "3rd", "4th", "5th", "6th")
FACES = ["Face: auto"] + [f"Face: {nth} from left" for nth in NTH]
# With AI's Choice off, how the characters are placed by hand: boxes dragged over the output,
# or NovelAI V4.5's 5x5 grid (a cell per character = its center, with a soft area around it).
MANUAL = ("Boxes", "Grid")
# For people who've never used NovelAI: collapsed under the panel's title until asked for.
HELP = """<details class="nai-help"><summary>How to use</summary><div>
<ol>
<li><b>Main prompt:</b> the scene, the style, and how many people (<code>2girls</code>, <code>1boy, 1girl</code>).
Don't describe the characters there.</li>
<li><b>+</b> adds a character. Describe only that character in its box: hair, eyes, outfit, expression.
Whatever that character must not have goes under <b>Undesired Content</b>. A card's <b>On</b> pill switches that
character off without deleting it; the <b>Character Prompts</b> pill in the Stagehand header does that for all of
them (it works with Stagehand closed).</li>
<li><b>AI's Choice</b> on: the characters stand left to right in card order (&uarr; &darr; to reorder).
Off: place them yourself, over the output image. <b>Boxes</b>: drag a box by its name tab, resize it by its corner
dot. <b>Grid</b>: NovelAI's 5&times;5 grid; drag a character's dot to a cell, which marks its center. Grid points blend
into each other more softly, so characters touching or overlapping tend to look more natural. Both scale with the
image, so they keep working at any size.</li>
<li><b>Interactions:</b> <code>source#hug</code> in the box of the one doing it, <code>target#hug</code> in the
box of the one it's done to, <code>mutual#kiss</code> in both for a shared action. If it comes out the wrong way
round on every seed, swap the two cards (&uarr; &darr;): some poses have a side the model likes to put the doer on.</li>
<li><b>Face</b> (ADetailer): leave it on <i>auto</i>, and each face gets repainted with its own character's prompt.
Pick "Nth from left" only if a face got the wrong character.</li>
</ol>
<p>LoRAs typed in a box apply to the whole image. Wildcards and Set Queue words work inside boxes.
Up to 6 characters; 2&ndash;3 work best. Anima only. Saved images list the characters in their prompt, under the
main prompt; pasting one (or a batch ADetailer / hires-fix run) brings them back.</p>
</div></details>"""
# Extra-network tags (<lora:...>) apply to the whole model, so a box's tags move to the base,
# where Forge activates them -- "any LoRA, wherever you type it, applies to the whole image".
_NETWORK = re.compile(r"<[^<>:]+:[^<>]+>")


def _parse_box(text):
    try:
        x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in str(text).replace(",", " ").split())
    except ValueError:
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)) if x1 != x0 and y1 != y0 else None


def _format_box(box):
    return " ".join(f"{v:.3f}" for v in box)


def _parse_place(text):
    """A card's position: a grid cell ('C3') as a point, or a box, or None."""
    return parse_cell(text) or _parse_box(text)


def _format_place(shape):
    return format_cell(shape) if len(shape) == 2 else _format_box(shape)


def _characters(args):
    """[(number, name, prompt, uc, box, face)] for every enabled card with something to say --
    a box that is only a comment would otherwise still take an AI's Choice column. face is
    the hand-picked ADetailer detection (0 = 1st from the left), or None."""
    out = []
    faces = list(args[1 + MAX_CHARS * ARG_FIELDS :]) + [FACES[0]] * MAX_CHARS
    for i in range(MAX_CHARS):
        enabled, name, prompt, uc, box = args[1 + i * ARG_FIELDS : 1 + (i + 1) * ARG_FIELDS]
        if enabled and strip_comments(prompt or "").strip():
            face = FACES.index(faces[i]) - 1 if faces[i] in FACES[1:] else None
            out.append((i + 1, (name or "").strip(), prompt, uc or "", _parse_place(box), face))
    return out


def _tail(args):
    """(on, manual): the args after the Face picks -- the feature's on/off pill (on unless the
    caller says otherwise) and the hand placement style."""
    rest = list(args[1 + MAX_CHARS * ARG_FIELDS + MAX_CHARS :])
    on = rest[0] if rest and rest[0] is not None else True
    manual = rest[1] if len(rest) > 1 and rest[1] in MANUAL else MANUAL[0]
    return bool(on), manual


def _places(characters, auto, manual):
    """{number: box or point} for the characters, by AI's Choice or by hand."""
    numbers = [c[0] for c in characters]
    if auto:
        return dict(zip(numbers, auto_boxes(len(numbers))))
    if manual == "Grid":
        cells = dict(zip(numbers, default_cells(len(numbers))))
        return {c[0]: center(c[4]) if c[4] else parse_cell(cells[c[0]]) for c in characters}
    columns = dict(zip(numbers, auto_boxes(len(numbers))))
    return {c[0]: c[4] or columns[c[0]] for c in characters}


def _face_picks(args):
    """{card number: picked detection (0 = 1st from left)} for every card not on auto."""
    faces = list(args[1 + MAX_CHARS * ARG_FIELDS :])
    return {i + 1: FACES.index(f) - 1 for i, f in enumerate(faces[:MAX_CHARS]) if f in FACES[1:]}


def _lift_networks(base: str, parts: dict) -> tuple[str, dict]:
    tags = [t for text in parts.values() for t in _NETWORK.findall(text)]
    parts = {n: _NETWORK.sub("", text).strip(" ,") for n, text in parts.items()}
    return (f"{base}, {' '.join(tags)}" if tags else base), parts


def _schedule_steps(p, hr):
    """(steps, hires_steps) exactly as Forge builds the main prompt's schedules
    (processing.py setup_conds / calculate_hr_conds), so editing in a box switches together
    with the main prompt -- including samplers that take several model calls per step."""
    if not hr:
        return getattr(p, "firstpass_steps", p.steps), None
    config = sd_samplers.find_sampler_config(getattr(p, "hr_sampler_name", None) or p.sampler_name)
    steps = getattr(p, "hr_second_pass_steps", 0) or p.steps
    return getattr(p, "firstpass_steps", p.steps), (config.total_steps(steps) if config else steps)


def _encode(p, texts, steps, hires_steps):
    """Prompt-editing schedules of each text: [(end_at_step, tokens [n, 1024])], real tokens only.
    is_negative_prompt keeps Anima's get_learned_conditioning off its reference-latent state."""
    texts = list(dict.fromkeys(t for t in texts if t))
    if not texts:
        return {}
    conditioning = prompt_parser.SdConditioning(texts, is_negative_prompt=True, width=p.width, height=p.height)
    schedules = prompt_parser.get_learned_conditioning(p.sd_model, conditioning, steps, hires_steps)
    out = {}
    for text, schedule in zip(texts, schedules):
        entries = []
        for entry in schedule:
            tokens = entry.cond.reshape(-1, entry.cond.shape[-1])
            entries.append((entry.end_at_step, tokens[: real_length(tokens)].detach()))
        out[text] = entries
    return out


# ---------------------------------------------------------------------------- UI callbacks
# `shown` is a gr.State, written back when a request ends; concurrent requests (a paste
# fills several cards at once) can leave it stale. So a card also counts as taken whenever
# its prompt has text, and the no-op path never writes the state back.
def _taken(shown, values, i):
    return shown[i] or bool((values[i * CARD_FIELDS + PROMPT] or "").strip())


def _card_values(enabled=True, name="", prompt="", uc="", box="", face=FACES[0]):
    return [enabled, name, prompt, uc, box, face]


def _no_change(extra=0):
    return [gr.update()] * (MAX_CHARS * CARD_FIELDS + MAX_CHARS + extra)


def _add(shown, *values):
    shown = list(shown)
    free = [i for i in range(MAX_CHARS) if not _taken(shown, values, i)]
    if not free:
        return [gr.update()] + _no_change()
    i = free[0]
    shown[i] = True
    updates = _no_change()
    updates[i * CARD_FIELDS : (i + 1) * CARD_FIELDS] = _card_values()
    updates[MAX_CHARS * CARD_FIELDS + i] = gr.update(visible=True)
    return [shown] + updates


def _remove(i):
    def remove(shown):
        shown = list(shown)
        shown[i] = False
        return [shown, gr.update(visible=False)] + _card_values()

    return remove


def _duplicate(i):
    def duplicate(shown, *values):
        shown = list(shown)
        free = [j for j in range(MAX_CHARS) if not _taken(shown, values, j)]
        if not free:
            return [gr.update()] + _no_change()
        j = free[0]
        shown[j] = True
        updates = _no_change()
        updates[j * CARD_FIELDS : (j + 1) * CARD_FIELDS] = list(values[i * CARD_FIELDS : (i + 1) * CARD_FIELDS])
        updates[j * CARD_FIELDS + FACE] = FACES[0]  # one detection per character: not the original's
        updates[MAX_CHARS * CARD_FIELDS + j] = gr.update(visible=True)
        return [shown] + updates

    return duplicate


def _swap(i, j):
    def swap(shown, *values):
        if not 0 <= j < MAX_CHARS:
            return [gr.update()] + _no_change()
        shown = list(shown)
        visible = {k: _taken(shown, values, k) for k in (i, j)}
        updates = _no_change()
        updates[i * CARD_FIELDS : (i + 1) * CARD_FIELDS] = list(values[j * CARD_FIELDS : (j + 1) * CARD_FIELDS])
        updates[j * CARD_FIELDS : (j + 1) * CARD_FIELDS] = list(values[i * CARD_FIELDS : (i + 1) * CARD_FIELDS])
        shown[i], shown[j] = visible[j], visible[i]
        updates[MAX_CHARS * CARD_FIELDS + i] = gr.update(visible=shown[i])
        updates[MAX_CHARS * CARD_FIELDS + j] = gr.update(visible=shown[j])
        return [shown] + updates

    return swap


def _revealer(i):
    """Pasting infotext fills a hidden card's prompt; show the card when that happens."""

    def reveal(prompt, shown):
        if not (prompt or "").strip() or shown[i]:
            return gr.update(), gr.update()  # don't write a possibly stale state back
        shown = list(shown)
        shown[i] = True
        return shown, gr.update(visible=True)

    return reveal


def _clear_on_paste(infotext, params):
    """Pasting a generation restores its characters exactly -- including having none --
    instead of leaving the previous image's boxes to be generated again. The characters are
    lines in the prompt section (see characters.show); older images had them as parameters."""
    if "Steps" not in params:
        return
    base, chars = read(params.get("Prompt", ""))
    if chars:
        params["Prompt"] = base
        for n, c in chars.items():
            params[f"Char {n} prompt"], params[f"Char {n} name"], params[f"Char {n} position"] = c["text"], c["name"], c["box"]
    negative, ucs = read(params.get("Negative prompt", ""))
    if ucs:
        params["Negative prompt"] = negative
        for n, c in ucs.items():
            params[f"Char {n} UC"] = c["text"]
    # Forge's infotext keys can't contain an apostrophe, so "Char AI's Choice" never parses.
    # Positions are written exactly when it was off; grid cells mean Grid.
    positions = [params.get(f"Char {n} position") for n in range(1, MAX_CHARS + 1)]
    params.setdefault("Char AI's Choice", str(not any(positions)))
    params.setdefault("Char placement", "Grid" if any(parse_cell(v) for v in positions if v) else "Boxes")
    for n in range(1, MAX_CHARS + 1):
        for key in (f"Char {n} prompt", f"Char {n} UC", f"Char {n} name", f"Char {n} position"):
            params.setdefault(key, "")
        params.setdefault(f"Char {n} ADetailer face", FACES[0])


script_callbacks.on_infotext_pasted(_clear_on_paste)


def _with_characters(text, a):
    """create_infotext's text with the image's characters added to its prompt section.
    `a` is create_infotext's bound arguments; the image index is worked out as it does."""
    p = a["p"]
    info = getattr(p, "_nai_chars", None)
    if not info or not info["images"]:
        return text
    if a.get("use_main_prompt"):
        g, prompt, negative = 0, p.main_prompt, p.main_negative_prompt
    else:
        index = a.get("index")
        if index is None:
            index = g = a.get("position_in_batch", 0) + a.get("iteration", 0) * p.batch_size
        else:  # Forge passes either the job's lists (with p.all_seeds) or the batch's
            g = index if a["all_seeds"] is p.all_seeds else p.iteration * p.batch_size + index
        negatives = a.get("all_negative_prompts")
        prompt, negative = a["all_prompts"][index], (negatives if negatives is not None else p.all_negative_prompts)[index]
    image = info["images"].get(g)
    body = f"{prompt}{chr(10) + 'Negative prompt: ' + negative if negative else ''}".lstrip()
    if image is None or not text.startswith(body):
        return text
    prompts = image.get("prompt_raw") or image["prompt"]  # with Save Raw Comments, the comments too
    ucs = image.get("uc_raw") or image["uc"]
    names = info.get("names", {})
    positive = show(prompt, [(n, names.get(n, ""), None if info["auto"] else _format_place(info["boxes"][n]), prompts.get(n, "")) for n in info["numbers"]])
    negative = show(negative or "", [(n, "", None, ucs.get(n, "")) for n in info["numbers"]])
    params = text[len(body):].lstrip(chr(10))  # with an empty prompt and negative, strip() ate the newline
    return f"{positive}{chr(10) + 'Negative prompt: ' + negative if negative else ''}{chr(10)}{params}".strip()


def _install_infotext(module=processing):
    """Every image's PNG info goes through processing.create_infotext (ADetailer keeps its own
    reference to it too), so that's where the characters are added."""
    original = module.create_infotext
    if getattr(original, "_nai_hooked", False):
        return
    signature = inspect.signature(original)

    @functools.wraps(original)
    def create_infotext(*args, **kwargs):
        text = original(*args, **kwargs)
        try:
            return _with_characters(text, signature.bind(*args, **kwargs).arguments)
        except Exception as e:  # never cost an image its PNG info
            print(f"[Character Prompts] couldn't add the characters to the PNG info: {e!r}")
            return text

    create_infotext._nai_hooked = True
    module.create_infotext = create_infotext


_install_infotext()


# ---------------------------------------------------------------------------- ADetailer
# ADetailer inpaints each detection with "[PROMPT]" = the image's prompt, which here is the
# base alone, so every face would lose its character's traits. Two wrappers on its script
# class fix that per detection: pred_preprocessing (one pass's sorted masks) matches them to
# characters, and i2i_prompts_replace (just before each is inpainted) makes [PROMPT] the base
# plus that character's prompt, and the negative the base's plus its Undesired Content.
_faces = {"pass": None}


def _uses_prompt(text):
    """ADetailer reads an empty [SEP] part as [PROMPT] too."""
    return any(not part.strip() or "[PROMPT]" in part for part in re.split(r"\[SEP\]", text or ""))


def _face_pass(script, p, args, masks):
    info = getattr(p, "_nai_chars", None)
    if not info or getattr(p, "_ad_inner", False) or not masks:
        return None
    if not (_uses_prompt(args.ad_prompt) or _uses_prompt(args.ad_negative_prompt)):
        return None
    i = p.iteration * p.batch_size + getattr(p, "batch_index", 0)
    image = info["images"].get(i)
    if image is None:
        return None
    numbers = info["numbers"]
    face = {"masks": masks, "owners": {}, "script": script, "p": p, "args": args, "i": i, "image": image, "numbers": numbers, "steps": info["steps"]}
    # No character per detection when regional prompting didn't run ("ran": Anima only; the
    # characters were ignored) or with one merged mask over every face -- but every detection
    # still gets the finished prompt (the batch tab leaves that to Stagehand for these images).
    if not info.get("ran") or getattr(args, "ad_mask_merge_invert", "None") != "None":
        return face
    width, height = masks[0].size
    # the regions exactly as sampling had them: Anima's token grid (8x VAE, 2x2 patches)
    h, w = -(-(height // 8) // 2), -(-(width // 8) // 2)
    weights = region_weights([info["boxes"][n] for n in numbers], h, w)[1:].reshape(len(numbers), h, w)
    found = [(j, m.getbbox()) for j, m in enumerate(masks)]
    found = [(j, box) for j, box in found if box]
    # hands come several per character: each goes to the region it's in, no picks
    hands = "hand" in str(getattr(args, "ad_model", "")).lower()
    picked = match_faces([box for _, box in found], (width, height), weights, None if hands else info.get("faces"), one_to_one=not hands)
    owners = {j: r for (j, _), r in zip(found, picked) if r is not None}
    if owners:
        rank = {j: k for k, (j, _) in enumerate(sorted(found, key=lambda f: f[1][0] + f[1][2]))}
        nth = lambda k: NTH[k] if k < len(NTH) else f"{k + 1}th"  # noqa: E731
        pairs = ", ".join(f"{nth(rank[j])} from left -> character {numbers[r]}" for j, r in sorted(owners.items(), key=lambda o: rank[o[0]]))
        print(f"[Character Prompts] ADetailer: {pairs}")
    face["owners"] = owners
    return face


def _join(*texts):
    return ", ".join(t.strip(" ,") for t in texts if t and t.strip(" ,"))


def _final(text, steps):
    """The prompt in the state the image finished in. ADetailer re-samples a face from step 0,
    so [from:to:7] would paint `from` again for its first steps. Alternation [a|b] and LoRA
    tags stay as they are (same rules as batch-adetailer's final_prompt)."""
    if not text or "[" not in text:
        return text
    import lark

    class Final(lark.Transformer):
        def scheduled(self, args):
            before, after, _, when, _ = args
            n = int(float(str(when)) * steps) if "." in str(when) else int(float(str(when)))
            return (before or "") if n >= steps else (after or "")

        def alternate(self, args):
            return "[" + "|".join(a or "" for a in args) + "]"

        def __default__(self, data, children, meta):
            return "".join(str(c) for c in children if c is not None)

    nets = _NETWORK.findall(text)
    body = _NETWORK.sub("", text)
    try:
        final = Final().transform(prompt_parser.schedule_parser.parse(body))
    except Exception:
        return text
    if final == body:
        return text
    return ", ".join([final.strip(" ,")] + nets) if nets else final


def _face_prompts(face, i2i, j):
    """Every detection of a character image gets the finished prompt; one that belongs to a
    character gets that character's prompt and Undesired Content added."""
    r = face["owners"].get(j)
    p, i, image, steps = face["p"], face["i"], face["image"], face["steps"]
    n = face["numbers"][r] if r is not None else None
    text = (image.get("encode_prompt") or image["prompt"]).get(n, "") if n else ""
    uc = image["uc"].get(n, "") if n else ""
    saved = p.all_prompts[i], p.all_negative_prompts[i]
    try:
        p.all_prompts[i] = _final(_join(saved[0], text), steps)
        p.all_negative_prompts[i] = _final(_join(saved[1], uc), steps)
        prompts, negatives = face["script"].get_prompt(p, face["args"])  # ADetailer's own [PROMPT]/[SEP] handling
    finally:
        p.all_prompts[i], p.all_negative_prompts[i] = saved
    i2i.prompt = prompts[min(j, len(prompts) - 1)]
    i2i.negative_prompt = negatives[min(j, len(negatives) - 1)]


def _hook_adetailer_faces():
    for data in scripts.scripts_data:
        cls = data.script_class
        if Path(data.path).stem != "!adetailer" or not all(hasattr(cls, a) for a in ("pred_preprocessing", "i2i_prompts_replace", "get_prompt")):
            continue
        if getattr(cls.pred_preprocessing, "_nai_hooked", False):
            return
        _install_infotext(data.module)  # its own from-import of create_infotext ("-ad-before" images)
        preprocess, replace = cls.pred_preprocessing, cls.i2i_prompts_replace

        # A failure here must not cost the image its ADetailer pass: ADetailer re-raises, and
        # Forge then skips its whole postprocess. Falls back to ADetailer's own prompts.
        def pred_preprocessing(self, p, pred, args, *rest, **kwargs):
            masks = preprocess(self, p, pred, args, *rest, **kwargs)
            try:
                _faces["pass"] = _face_pass(self, p, args, masks)
            except Exception as e:
                _faces["pass"] = None
                print(f"[Character Prompts] ADetailer: per-character faces skipped: {e!r}")
            return masks

        def i2i_prompts_replace(i2i, prompts, negative_prompts, j, *rest, **kwargs):
            replace(i2i, prompts, negative_prompts, j, *rest, **kwargs)
            face = _faces["pass"]
            # only inside the pass that matched these very masks
            if face and j < len(face["masks"]) and getattr(i2i, "image_mask", None) is face["masks"][j]:
                try:
                    _face_prompts(face, i2i, j)
                except Exception as e:
                    print(f"[Character Prompts] ADetailer: per-character face skipped: {e!r}")

        pred_preprocessing._nai_hooked = True
        cls.pred_preprocessing = pred_preprocessing
        cls.i2i_prompts_replace = staticmethod(i2i_prompts_replace)
        return


class CharacterPrompts(scripts.Script):
    # javascript/stagehand.js moves the panel under the prompt boxes; a script
    # group would be left behind in the scripts column as an empty frame.
    create_group = False

    def title(self):
        return "Character Prompts"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        _hook_adetailer_faces()
        tab = "i2i" if is_img2img else "t2i"
        fields, cards, infotext = [], [], []
        ups, downs, copies, removes = [], [], [], []

        with gr.Column(elem_id=f"nai_{tab}_chars", elem_classes=["nai-panel"]):
            # The feature's on/off: shown as a pill in the Stagehand header (javascript/stagehand.js).
            on = gr.Checkbox(value=True, label="Character Prompts", elem_id=f"nai_{tab}_chars_on", elem_classes=["nai-hidden"])
            with gr.Row(elem_classes=["nai-head"]):
                gr.HTML('<div class="nai-section">Characters</div>')
                auto = gr.Checkbox(value=True, label="AI's Choice", elem_id=f"nai_{tab}_chars_auto", elem_classes=["nai-auto"], scale=0, min_width=120)
                manual = gr.Radio(list(MANUAL), value=MANUAL[0], show_label=False, container=False, elem_id=f"nai_{tab}_chars_manual", elem_classes=["nai-manual"], scale=0, min_width=160)
                gr.HTML("")  # spacer: pushes + to the right
                add = gr.Button("+", elem_classes=["nai-add"], min_width=40, scale=0)
            shown = gr.State([False] * MAX_CHARS)

            for i in range(MAX_CHARS):
                with gr.Group(visible=False, elem_id=f"nai_{tab}_char{i + 1}", elem_classes=["nai-card", "nai-char-card", f"nai-char-{i + 1}"]) as card:
                    with gr.Row(elem_classes=["nai-card-head"]):
                        enabled = gr.Checkbox(value=True, label="On", container=False, scale=0, min_width=60, elem_classes=["nai-char-on"])
                        name = gr.Textbox(value="", show_label=False, container=False, placeholder=f"Character {i + 1}", max_lines=1, elem_classes=["nai-char-name"])
                        # which of ADetailer's detections gets this character's prompt; auto = by position
                        face = gr.Dropdown(FACES, value=FACES[0], show_label=False, container=False, scale=0, min_width=170, elem_classes=["nai-char-face"])
                        up = gr.Button("↑", elem_classes=["nai-icon"], min_width=30, scale=0)
                        down = gr.Button("↓", elem_classes=["nai-icon"], min_width=30, scale=0)
                        copy = gr.Button("⧉", elem_classes=["nai-icon"], min_width=30, scale=0)
                        remove = gr.Button("🗑", elem_classes=["nai-icon"], min_width=30, scale=0)
                    with gr.Tabs(elem_classes=["nai-char-tabs"]):
                        with gr.Tab("Prompt"):
                            prompt = gr.Textbox(value="", show_label=False, lines=2, placeholder="girl, purple hair, ...", elem_id=f"nai_{tab}_char{i + 1}_prompt")
                        with gr.Tab("Undesired Content"):
                            uc = gr.Textbox(value="", show_label=False, lines=2, elem_id=f"nai_{tab}_char{i + 1}_uc")
                    # Rendered but hidden by CSS: the position overlay writes here.
                    box = gr.Textbox(value="", elem_id=f"nai_{tab}_char{i + 1}_box", elem_classes=["nai-hidden"], show_label=False, container=False)
                cards.append(card)
                ups.append(up)
                downs.append(down)
                copies.append(copy)
                removes.append(remove)
                # value="" matters: Forge's paste converts with type(component.value), and a
                # Textbox without one is None -- every pasted character was silently dropped
                card_fields = [enabled, name, prompt, uc, box, face]
                for component in card_fields:
                    # ui-config.json keys come from a component's LABEL; every card would share one
                    component.do_not_save_to_config = True
                fields += card_fields
                infotext += [
                    (name, f"Char {i + 1} name"),
                    (prompt, f"Char {i + 1} prompt"),
                    (uc, f"Char {i + 1} UC"),
                    (box, f"Char {i + 1} position"),
                    (face, f"Char {i + 1} ADetailer face"),
                    # a pasted character is switched on, or it would be pasted and never drawn
                    (enabled, lambda params, n=i + 1: True if params.get(f"Char {n} prompt") else None),
                ]
                prompt.change(_revealer(i), [prompt, shown], [shown, card], show_progress="hidden")
                remove.click(_remove(i), [shown], [shown, card] + card_fields, show_progress="hidden")

            everything = fields + cards
            for i in range(MAX_CHARS):
                ups[i].click(_swap(i, i - 1), [shown] + fields, [shown] + everything, show_progress="hidden")
                downs[i].click(_swap(i, i + 1), [shown] + fields, [shown] + everything, show_progress="hidden")
                copies[i].click(_duplicate(i), [shown] + fields, [shown] + everything, show_progress="hidden")

            add.click(_add, [shown] + fields, [shown] + everything, show_progress="hidden")
            gr.HTML(HELP)

        for component in [auto, add] + ups + downs + copies + removes:
            component.do_not_save_to_config = True  # ui-config keys collide by label
        # "True"/"False" rather than a callable key, so Send to img2img carries it too
        infotext.append((auto, "Char AI's Choice"))
        infotext.append((manual, "Char placement"))
        # pasting an image with characters switches the feature on
        infotext.append((on, lambda params: True if any(params.get(f"Char {n} prompt") for n in range(1, MAX_CHARS + 1)) else None))
        self.infotext_fields = infotext
        self.paste_field_names = [key for _, key in infotext if isinstance(key, str)]
        # Added fields go last, so API callers written for the older layouts keep working:
        # [AI's Choice] + 6 x [on, name, prompt, uc, position] + 6 faces + [on/off, placement]
        per_card = [fields[i * CARD_FIELDS : (i + 1) * CARD_FIELDS] for i in range(MAX_CHARS)]
        return [auto] + [c for card in per_card for c in card[:ARG_FIELDS]] + [card[ARG_FIELDS] for card in per_card] + [on, manual]

    # ------------------------------------------------------------------ processing
    def args_from_infotext(self, params):
        """Args for re-running an image from its PNG info (the batch tabs call this). The
        characters themselves ride in the prompt; only the Face picks live in parameters."""
        faces = [params.get(f"Char {n} ADetailer face", FACES[0]) for n in range(1, MAX_CHARS + 1)]
        if all(f == FACES[0] for f in faces):
            return None
        return [True] + [x for _ in range(MAX_CHARS) for x in (True, "", "", "", "")] + faces + [True, MANUAL[0]]

    def before_process(self, p, *args):
        if getattr(p, "_ad_inner", False):
            return
        # Prompt Matrix and friends hand over a list per image; leave those alone.
        if not isinstance(p.prompt, str) or not isinstance(p.negative_prompt, str):
            return
        on, manual = _tail(args)
        # Off: the cards are left out. A prompt that carries characters (below) still has them:
        # they're that image's characters, whatever the panel says.
        characters = _characters(args) if on else []
        auto = bool(args[0])

        # A prompt copied from an image's PNG info (an API caller, the batch tabs, a paste the
        # panel hasn't split): its character lines are the characters, as the image had them.
        base, shown_chars = read(p.prompt)
        if shown_chars:
            negative, shown_ucs = read(p.negative_prompt)
            p.prompt = merge(base, {n: c["text"] for n, c in shown_chars.items()})
            p.negative_prompt = merge(negative, {n: c["text"] for n, c in shown_ucs.items() if n in shown_chars})
            known = {c[0]: c for c in characters}
            picks = _face_picks(args)
            characters = [(n, c["name"] or known.get(n, (n, ""))[1], c["text"], "", _parse_place(c["box"]),
                           picks.get(n)) for n, c in sorted(shown_chars.items())]
            places = [_parse_place(c["box"]) for c in shown_chars.values()]
            auto = not any(places)
            manual = "Grid" if any(pl and len(pl) == 2 for pl in places) else "Boxes"
        elif has_marks(p.prompt):
            # Already merged (a restored template the panel hasn't split yet, or an API
            # caller): the marks are the characters; boxes only add names and positions.
            marked = sorted(set(split(p.prompt)[1]) | set(split(p.negative_prompt)[1]))
            known = {c[0]: c for c in characters}
            characters = [known.get(n, (n, "", "", "", None, None)) for n in marked]
        elif characters:
            p.prompt = merge(p.prompt, {c[0]: c[2] for c in characters})
            p.negative_prompt = merge(p.negative_prompt, {c[0]: c[3] for c in characters})
            # A separate hires prompt is NOT merged: Dynamic Prompts and Set Queue never roll
            # it, so it would carry raw wildcards. The hires pass reuses the first pass's rolls.
        if not characters:
            return

        numbers = [c[0] for c in characters]
        boxes = _places(characters, auto, manual)
        faces = {r: c[5] for r, c in enumerate(characters) if c[5] is not None}

        names = {c[0]: c[1] for c in characters if c[1]}
        # the prompts, names and positions go into the PNG info's prompt section (_with_characters)
        p._nai_chars = {"numbers": numbers, "boxes": boxes, "auto": auto, "manual": manual, "faces": faces, "names": names, "steps": p.steps, "images": {}}
        for n, _, _, _, _, face in characters:
            if face is not None:
                p.extra_generation_params[f"Char {n} ADetailer face"] = FACES[face + 1]

    def before_process_batch(self, p, *args, batch_number=0, **kwargs):
        info = getattr(p, "_nai_chars", None)
        if not info:
            return
        start = batch_number * p.batch_size
        hr_prompts = getattr(p, "all_hr_prompts", None)
        hr_negatives = getattr(p, "all_hr_negative_prompts", None)
        for k in range(len(p.prompts)):
            g = start + k
            base, parts = _lift_networks(*split(p.prompts[k]))
            negative, ucs = split(p.negative_prompts[k])
            # "prompt"/"uc" are what was typed (the infotext); "encode*" what the model gets
            image = {"prompt": parts, "uc": ucs, "hr_prompt": parts, "hr_uc": ucs}
            p.prompts[k], p.negative_prompts[k] = base, negative
            p.all_prompts[g], p.all_negative_prompts[g] = base, negative
            if hr_prompts and g < len(hr_prompts) and has_marks(hr_prompts[g]):
                hr_base, image["hr_prompt"] = _lift_networks(*split(hr_prompts[g]))
                hr_prompts[g] = hr_base
            if hr_negatives and g < len(hr_negatives) and has_marks(hr_negatives[g]):
                hr_negatives[g], image["hr_uc"] = split(hr_negatives[g])
            # Forge's "Save Raw Comments" keeps an unstripped copy for the infotext; lifted
            # the same way, or get_hr_prompt sees base != base+<lora> and writes a Hires prompt
            raw = getattr(p, "_all_prompts_c", None)
            if raw and g < len(raw) and has_marks(raw[g]):
                raw[g], image["prompt_raw"] = _lift_networks(*split(raw[g]))
            raw = getattr(p, "_all_negative_prompts_c", None)
            if raw and g < len(raw) and has_marks(raw[g]):
                raw[g], image["uc_raw"] = split(raw[g])
            info["images"][g] = image
        # grids and the job-wide infotext read main_prompt
        for attr in ("main_prompt", "main_negative_prompt"):
            text = getattr(p, attr, None)
            if has_marks(text):
                setattr(p, attr, _lift_networks(*split(text))[0] if attr == "main_prompt" else split(text)[0])

    def process_batch(self, p, *args, batch_number=0, **kwargs):
        """Interaction tags: the plain action into each character, the who-does-what sentence
        into what gets encoded (not into the stored prompt -- it's rebuilt on paste). Custom
        boxes also say where each character is ("a boy on the left")."""
        info = getattr(p, "_nai_chars", None)
        if not info:
            return
        boxes = [info["boxes"][n] for n in info["numbers"]]
        start = batch_number * p.batch_size
        for k in range(len(p.prompts)):
            image = info["images"].get(start + k)
            if image is None:
                continue
            for kind, prompts in (("prompt", p.prompts), ("hr_prompt", getattr(p, "hr_prompts", None))):
                texts = [image[kind].get(n, "") for n in info["numbers"]]
                cleaned, phrases = translate_actions(texts, boxes)
                if not info["auto"]:
                    phrases = placement(texts, boxes) + phrases
                image["encode_" + kind] = {n: t for n, t in zip(info["numbers"], cleaned) if t}
                if phrases and prompts is not None and k < len(prompts):
                    prompts[k] = f"{prompts[k]}, {', '.join(phrases)}"

    def process_before_every_sampling(self, p, *args, **kwargs):
        info = getattr(p, "_nai_chars", None)
        if not info or getattr(p, "_ad_inner", False):
            return
        unet = p.sd_model.forge_objects.unet
        diffusion_model = getattr(unet.model, "diffusion_model", None)
        if not anima_hooks.tag_blocks(diffusion_model):
            print("[Character Prompts] regional prompting is implemented for Anima only -- characters ignored.")
            return

        hr = getattr(p, "is_hr_pass", False)
        start = p.iteration * p.batch_size
        # len(p.seeds), not batch_size: Dynamic Prompts' combinatorial mode can leave the
        # last batch short
        images = [info["images"].get(start + k) for k in range(len(p.seeds))]
        if any(image is None for image in images):
            print("[Character Prompts] characters missing for this batch -- skipped.")
            return
        kind, uc_kind = ("encode_hr_prompt", "hr_uc") if hr else ("encode_prompt", "uc")
        texts = [image[key].get(n) for image in images for key in (kind, uc_kind) for n in info["numbers"]]
        encoded = _encode(p, texts, *_schedule_steps(p, hr))

        regions = [[(encoded.get(image[kind].get(n)), encoded.get(image[uc_kind].get(n))) for n in info["numbers"]] for image in images]
        denoiser = p.sampler.model_wrap_cfg
        session = RegionSession(
            [info["boxes"][n] for n in info["numbers"]],
            regions,
            step=lambda: denoiser.step,
            patch=getattr(diffusion_model, "patch_spatial", 2),
        )
        unet = unet.clone()
        unet.set_transformer_option(anima_hooks.REGIONS_KEY, session)
        unet.add_conditioning_modifier(session.modifier)
        p.sd_model.forge_objects.unet = unet
        info["ran"] = True
        layout = "AI's Choice" if info["auto"] else info["manual"].lower()
        print(f"[Character Prompts] {len(info['numbers'])} characters, {layout}{' (hires)' if hr else ''}")

    def postprocess(self, p, processed, *args):
        # not from inside ADetailer's own pass, which runs every script when "Apply only
        # selected scripts" is off: that would drop the matches for the faces still to come
        if not getattr(p, "_ad_inner", False):
            _faces["pass"] = None
