"""Character Prompts — NovelAI-style multi-character prompting for Anima, in Forge Neo.

A box per character (prompt + Undesired Content), in equal columns by default, or placed by
hand: boxes dragged on the card's canvas or the output image, or on NovelAI's 5x5 grid.
lib_stagehand/characters.py has the regional attention and the prompt plumbing; this file is
the UI and the Forge wiring:

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
import json
import math
import os
import re
from pathlib import Path

import gradio as gr

from modules import processing, prompt_parser, script_callbacks, scripts, sd_samplers, shared
from modules.paths_internal import data_path
from modules.processing_scripts.comments import strip_comments

from lib_stagehand import anima_hooks, region_lora
from lib_stagehand.characters import (
    SHARE,
    crop_places,
    RegionSession,
    auto_boxes,
    cat_schedules,
    center,
    default_cells,
    format_cell,
    has_marks,
    match_faces,
    merge,
    parse_cell,
    placement,
    read,
    shapes,
    real_length,
    region_weights,
    show,
    split,
    translate_actions,
)

anima_hooks.install()

MAX_CHARS = 6  # NovelAI V4.5's limit; regional prompting gets unreliable well before V5's 22
CARD_FIELDS = 7  # enabled, name, prompt, uc, box, face, share -- a card's components in the UI
ARG_FIELDS = 5  # the args carry each card's first five, then every card's face, then every share
PROMPT = 2  # index of the prompt within a card's fields
FACE = 5  # ...and of its ADetailer face pick
NTH = ("1st", "2nd", "3rd", "4th", "5th", "6th")
FACES = ["Face: auto"] + [f"Face: {nth} from left" for nth in NTH]
# How a character dragged into place is placed: boxes, or NovelAI V4.5's 5x5 grid (a cell per
# character = its center; nearest cell wins). One never dragged stands in its default column.
MANUAL = ("Boxes", "Grid")
# For people who've never used NovelAI: collapsed under the panel's title until asked for.
HELP = """<details class="nai-help"><summary>How to use</summary><div>
<ol>
<li><b>Main prompt:</b> the scene, the style, and how many people (<code>2girls</code>, <code>1boy, 1girl</code>).
Don't describe the characters there. The count is yours to keep right: switching a card off doesn't change it.</li>
<li><b>+ Add character</b> adds a card. Start its box with <code>girl</code> or <code>boy</code> (NovelAI's habit: it
also makes the position words say "a girl on the left" instead of "a character on the left"), then describe only
that character: hair, eyes, outfit, expression. Whatever that character must not have goes under <b>Undesired
Content</b> (a dot on the tab when it has something). A card's <b>On</b> / <b>Off</b> pill switches that character off
without deleting it; the <b>Character Prompts</b> pill in the Stagehand header does that for all of them (it works
with Stagehand closed).</li>
<li><b>Presets:</b> name a card, then &#128190; saves it (prompt and Undesired Content, line breaks kept). To use one,
type in a card's name box and pick it from the list: the card's text is replaced (Ctrl+Z undoes it), its place and
references stay. &#8943; on a card has Delete preset (and the ADetailer face); &#10697; duplicates a card.</li>
<li><b>Positions:</b> by default the characters stand left to right in card order (&uarr; &darr; to reorder).
Drag one to place it yourself -- on the small canvas beside its card, or over the output image. <b>Boxes</b>: drag
a box anywhere on it, resize it by any edge or corner. <b>Grid</b>: NovelAI's 5&times;5 grid; drag a character's
dot to a cell, which marks its center, and each character gets the part of the image nearest its dot. Good for
layouts columns can't do: one above the other (bunk beds), diagonal. Put a character's cell where its
<i>head</i> will be. Both scale with the image. <b>Reset boxes</b> puts everyone back in the default columns;
<b>Switch</b> swaps two characters' places; <b>+</b> on a card's small map adds another place for that character
(&times; on a place, or click it and press Delete, removes it); Ctrl+Z / Ctrl+Y undo and redo (outside a text box).</li>
<li><b>Interactions:</b> <code>source#hug</code> in the box of the one doing it, <code>target#hug</code> in the
box of the one it's done to, <code>mutual#kiss</code> in both for a shared action. If it comes out the wrong way
round on every seed, swap the two cards (&uarr; &darr;): some poses have a side the model likes to put the doer on.
The model also tends to give the passive role (carried, lying down) to the softer-looking outfit, which no card
order fixes: re-roll.</li>
<li><b>Reference</b> tab: images of this character to copy her look from (Precise Reference). Used only in her
part of the image, so each character keeps her own look; the References help explains Strength and Fidelity.</li>
<li><b>Overlap %</b> (beside the name): where two characters' places overlap, they split it in proportion (70 vs 30
gives 70/30); 50 each by default. <b>Face</b> (ADetailer, in &#8943;): leave it on <i>auto</i>, and each face gets
repainted with its own character's prompt. Pick "Nth from left" only if a face got the wrong character.</li>
</ol>
<p>LoRAs typed in a box apply only to that character (Settings &gt; Stagehand &gt; Character LoRAs can make them
apply to the whole image). Wildcards and Set Queue words work inside boxes.
Up to 6 characters; 2&ndash;3 work best. Anima only. Saved images list the characters in their prompt, under the
main prompt; pasting one (or a batch ADetailer / hires-fix run) brings them back.</p>
</div></details>"""
# Extra-network tags (<lora:...>) apply to the whole model, so a box's tags move to the base,
# where Forge activates them -- "any LoRA, wherever you type it, applies to the whole image".
# Unless Character LoRAs (Settings) is Masked or Separate pass: then they stay in the box and
# region_lora applies them to that character only.
_NETWORK = re.compile(r"<[^<>:]+:[^<>]+>")


def _parse_box(text):
    try:
        values = [float(v) for v in str(text).replace(",", " ").split()]
        x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in values)
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)) if x1 != x0 and y1 != y0 else None


def _format_box(box):
    return " ".join(f"{v:.3f}" for v in box)


def _parse_place(text):
    """A card's position: a grid cell ('C3') as a point, or a box -- or several, " + " between
    them, as a tuple of those -- or None."""
    found = [s for s in (parse_cell(part) or _parse_box(part) for part in str(text or "").split("+")) if s]
    return None if not found else found[0] if len(found) == 1 else tuple(found)


def _format_place(place):
    return " + ".join(format_cell(s) if len(s) == 2 else _format_box(s) for s in shapes(place))


def _of_kind(place, size):
    """The place's shapes of one kind (2: grid points, 4: boxes), as a place, or None."""
    own = [s for s in shapes(place) if s and len(s) == size] if place else []
    return None if not own else own[0] if len(own) == 1 else tuple(own)


def _characters(args):
    """[(number, name, prompt, uc, box, face, share)] for every enabled card with something to
    say -- a box that is only a comment would otherwise still take a column. face
    is the hand-picked ADetailer detection (0 = 1st from the left), or None; share the card's
    claim on overlaps, in percent."""
    out = []
    faces = list(args[1 + MAX_CHARS * ARG_FIELDS :]) + [FACES[0]] * MAX_CHARS
    shares = _shares(args)
    for i in range(MAX_CHARS):
        enabled, name, prompt, uc, box = args[1 + i * ARG_FIELDS : 1 + (i + 1) * ARG_FIELDS]
        if enabled and strip_comments(prompt or "").strip():
            face = FACES.index(faces[i]) - 1 if faces[i] in FACES[1:] else None
            out.append((i + 1, (name or "").strip(), prompt, uc or "", _parse_place(box), face, shares[i]))
    return out


def _shares(args):
    """Every card's overlap share, after (on, manual) -- SHARE for callers that send none."""
    rest = list(args[1 + MAX_CHARS * ARG_FIELDS + MAX_CHARS + 2 :])[:MAX_CHARS]
    out = []
    for v in rest + [SHARE] * (MAX_CHARS - len(rest)):
        try:
            out.append(min(max(int(round(float(v))), 0), 100))
        except (TypeError, ValueError):
            out.append(SHARE)
    return out


def _tail(args):
    """(on, manual): the args after the Face picks -- the feature's on/off pill (on unless the
    caller says otherwise) and the hand placement style."""
    rest = list(args[1 + MAX_CHARS * ARG_FIELDS + MAX_CHARS :])
    on = rest[0] if rest and rest[0] is not None else True
    manual = rest[1] if len(rest) > 1 and rest[1] in MANUAL else MANUAL[0]
    return bool(on), manual


def _auto(auto, characters, manual):
    """AI's Choice -- the default columns, no position words -- unless someone was placed by
    hand (a position of the mode's kind). The UI's switch is gone and always on; an API caller
    can still turn it off to get the columns with position words."""
    size = 2 if manual == "Grid" else 4
    return bool(auto) and not any(_of_kind(c[4], size) for c in characters)


def _places(characters, auto, manual):
    """{number: box or point} for the characters, by AI's Choice or by hand."""
    numbers = [c[0] for c in characters]
    if auto:
        return dict(zip(numbers, auto_boxes(len(numbers))))
    # Strictly the mode's own kind of position: a leftover of the other kind (a card that was
    # off while Boxes <-> Grid was switched) gets the default, as the editor draws it.
    if manual == "Grid":
        cells = dict(zip(numbers, default_cells(len(numbers))))
        return {c[0]: _of_kind(c[4], 2) or parse_cell(cells[c[0]]) for c in characters}
    columns = dict(zip(numbers, auto_boxes(len(numbers))))
    return {c[0]: _of_kind(c[4], 4) or columns[c[0]] for c in characters}


def _in_crop(p, places):
    """Inpaint "Only masked" samples just the crop around the mask (p.paste_to, in the whole
    image's pixels). Positions are fractions of the whole image, so they're moved into the
    crop's frame: inpainting one face keeps it that character's, instead of splitting the crop
    into columns. Anything else comes back unchanged."""
    crop = getattr(p, "paste_to", None) if getattr(p, "inpaint_full_res", False) else None
    return crop_places(places, crop, getattr(getattr(p, "mask_for_overlay", None), "size", None))


def _face_picks(args):
    """{card number: picked detection (0 = 1st from left)} for every card not on auto."""
    faces = list(args[1 + MAX_CHARS * ARG_FIELDS :])
    return {i + 1: FACES.index(f) - 1 for i, f in enumerate(faces[:MAX_CHARS]) if f in FACES[1:]}


def _lift_networks(base: str, parts: dict) -> tuple[str, dict]:
    tags = [t for text in parts.values() for t in _NETWORK.findall(text)]
    parts = {n: _NETWORK.sub("", text).strip(" ,") for n, text in parts.items()}
    return (f"{base}, {' '.join(tags)}" if tags else base), parts


def _keep_networks(base: str, parts: dict) -> tuple[str, dict]:
    return base, parts


def _lora_mode():
    mode = getattr(shared.opts, "stagehand_cp_lora", region_lora.DEFAULT)
    return mode if mode in region_lora.MODES else region_lora.DEFAULT


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


def _card_mode():
    """How cards are read (Settings > Stagehand), or {} for the old way (separate, full strength).
    "global": the main prompt Forge encodes carries every card too, so whatever isn't a
    region -- outside the boxes, under a region's partial strength -- is the one-prompt image."""
    opts = shared.opts
    x = {"one_pass": bool(getattr(opts, "stagehand_cp_one_pass", True)),
         "strength": float(getattr(opts, "stagehand_cp_region_strength", 0.65))}
    x["global"] = x["strength"] < 1
    x["whole"] = x["one_pass"] or x["global"]
    if not x["whole"]:
        return {}
    x["label"] = "; ".join(label for on, label in (
        (x["one_pass"], "one pass"), (x["global"], f"strength {x['strength']:g}")) if on)
    return x


def _whole_regions(p, images, numbers, hr, x, steps):
    """Each region's entire context, positive and negative: its main prompt and card in one
    pass, or each encoded alone and joined (the old way's context, rebuilt because the main
    prompt Forge encoded may now carry every card)."""
    kind, uc_kind = ("hr_prompt", "hr_uc") if hr else ("prompt", "uc")
    pairs = [[((image["main_" + kind], image["encode_" + kind].get(n)), (image["main_" + uc_kind], image[uc_kind].get(n)))
              for n in numbers] for image in images]
    flat = [pair for row in pairs for both in row for pair in both]
    if x["one_pass"]:
        encoded = _encode(p, [_join(a, b) for a, b in flat], *steps)
        return [[tuple(encoded.get(_join(a, b)) for a, b in both) for both in row] for row in pairs]
    encoded = _encode(p, [t for pair in flat for t in pair], *steps)
    return [[tuple(cat_schedules(encoded.get(a), encoded.get(b)) for a, b in both) for both in row] for row in pairs]


# ---------------------------------------------------------------------------- UI callbacks
# `shown` is a gr.State, written back when a request ends; concurrent requests (a paste
# fills several cards at once) can leave it stale. So a card also counts as taken whenever
# its prompt has text, and the no-op path never writes the state back.
def _taken(shown, values, i):
    return shown[i] or bool((values[i * CARD_FIELDS + PROMPT] or "").strip())


def _card_values(enabled=True, name="", prompt="", uc="", box="", face=FACES[0], share=SHARE):
    return [enabled, name, prompt, uc, box, face, share]


def _no_change(extra=0):
    return [gr.update()] * (MAX_CHARS * CARD_FIELDS + MAX_CHARS + extra)


# ---------------------------------------------------------------------------- presets
# A character saved by name: its prompt and Undesired Content exactly as typed, newlines
# included. In Forge's folder, not the extension's, so an update or reinstall keeps them.
PRESETS = os.path.join(data_path, "stagehand character presets.json")


def _presets(strict=False):
    """The saved presets. strict (before writing): an existing file that can't be read raises
    instead of reading as empty, so a save can't wipe it."""
    try:
        with open(PRESETS, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        if strict:
            raise
        return {}
    if not isinstance(data, dict):
        if strict:
            raise ValueError("not a JSON object")
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _presets_for_writing():
    try:
        return _presets(strict=True)
    except (OSError, ValueError) as e:
        gr.Warning(f"Couldn't read {os.path.basename(PRESETS)} ({e}). Fix or move it; nothing was changed.")
        return None


def _write_presets(presets):
    # whole file to a temp first: a crash mid-write must not cost every saved character
    tmp = PRESETS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(presets.items(), key=lambda kv: kv[0].lower())), f, ensure_ascii=False, indent=2)
    os.replace(tmp, PRESETS)


def _presets_json(presets=None):
    """Every preset, for the cards' name-box search (stagehand.js reads it from a hidden box)."""
    return json.dumps(presets if presets is not None else _presets(), ensure_ascii=False)


def _save_preset(name, prompt, uc):
    name = (name or "").strip()
    if not name:
        gr.Warning("Give the card a name first: the preset is saved under it.")
        return gr.update()
    if not (prompt or "").strip():
        gr.Warning("This card has no prompt to save.")
        return gr.update()
    presets = _presets_for_writing()
    if presets is None:
        return gr.update()
    replaced = name in presets
    presets[name] = {"prompt": prompt, "uc": uc or ""}
    _write_presets(presets)
    gr.Info(f'{"Updated" if replaced else "Saved"} the preset "{name}".')
    return _presets_json(presets)


def _delete_preset(name):
    if not name:  # nothing picked, or the confirmation was cancelled: leave the list as it is
        return gr.update()
    presets = _presets_for_writing()
    if presets is None:
        return gr.update()
    if presets.pop(name, None) is not None:
        _write_presets(presets)
        gr.Info(f'Deleted the preset "{name}".')
    return _presets_json(presets)


def _add(shown, *values):
    shown = list(shown)
    free = [i for i in range(MAX_CHARS) if not _taken(shown, values, i)]
    if not free:
        gr.Warning(f"All {MAX_CHARS} character cards are in use. Delete one first.")
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


def _restore(text):
    """Undo / redo (stagehand.js): every card -- shown or not, and its fields -- and Boxes /
    Grid as the browser recorded them, as JSON. Through here because a card's visibility
    lives in the server's `shown` state, which the browser can't set."""
    try:
        state = json.loads(text)
        cards = list(state["cards"])
    except (ValueError, KeyError, TypeError):
        return [gr.update()] + _no_change() + [gr.update()]
    shown, values, visible = [], [], []
    for i in range(MAX_CHARS):
        c = cards[i] if i < len(cards) and isinstance(cards[i], dict) else {}
        try:
            share = min(max(int(round(float(c.get("share", SHARE)))), 0), 100)
        except (TypeError, ValueError):
            share = SHARE
        face = c.get("face") if c.get("face") in FACES else FACES[0]
        shown.append(bool(c.get("visible")))
        values += _card_values(bool(c.get("enabled", True)), str(c.get("name", "")), str(c.get("prompt", "")),
                               str(c.get("uc", "")), str(c.get("box", "")), face, share)
        visible.append(gr.update(visible=shown[-1]))
    manual = state.get("manual") if state.get("manual") in MANUAL else gr.update()
    return [shown] + values + visible + [manual]


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
            params[f"Char {n} share"] = c["share"]
    negative, ucs = read(params.get("Negative prompt", ""))
    if ucs:
        params["Negative prompt"] = negative
        for n, c in ucs.items():
            params[f"Char {n} UC"] = c["text"]
    # Positions are written exactly when someone was placed by hand; grid cells mean Grid.
    positions = [params.get(f"Char {n} position") for n in range(1, MAX_CHARS + 1)]
    if any(positions):  # without positions the image doesn't say, so the switch is left alone
        params.setdefault("Char placement", "Grid" if any(parse_cell(v) for v in positions if v) else "Boxes")
    # pasted characters are switched on, and so is the feature when the image had any
    if any(params.get(f"Char {n} prompt") for n in range(1, MAX_CHARS + 1)):
        params.setdefault("Char feature", "True")
    for n in range(1, MAX_CHARS + 1):
        if params.get(f"Char {n} prompt"):
            params.setdefault(f"Char {n} on", "True")
    for n in range(1, MAX_CHARS + 1):
        for key in (f"Char {n} prompt", f"Char {n} UC", f"Char {n} name", f"Char {n} position"):
            params.setdefault(key, "")
        params.setdefault(f"Char {n} ADetailer face", FACES[0])
        params.setdefault(f"Char {n} share", SHARE)


script_callbacks.on_infotext_pasted(_clear_on_paste)


def _settings():
    # registered here because Character Prompts always loads; precise_reference.py reads it
    shared.opts.add_option("stagehand_precise_reference", shared.OptionInfo(
        True, "Precise Reference", gr.Checkbox, section=("stagehand", "Stagehand"),
    ).info("off removes it completely: its section and pill under the prompt, its paste handling, "
           "its XYZ Plot axes and its ADetailer hook").needs_reload_ui())
    # How cards are read (NOTES.md, "The card look"). Read on their own and glued on at full
    # strength, cards gave images a look of their own; the defaults won blind tests against
    # that. Off + 1.0 is the old behaviour. Images record what they used ("Char cards").
    for key, default, label, component, extra in (
        ("stagehand_cp_one_pass", True, "Character Prompts: read each card together with the main prompt",
         gr.Checkbox, None),
        ("stagehand_cp_region_strength", 0.65, "Character Prompts: card strength (each character's area follows her card this much; "
         "the rest is one prompt with every card)", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.05}),
    ):
        shared.opts.add_option(key, shared.OptionInfo(default, label, component, extra, section=("stagehand", "Stagehand")))
    shared.opts.add_option("stagehand_cp_lora", shared.OptionInfo(
        region_lora.DEFAULT, "Character Prompts: LoRAs typed in a card", gr.Radio, {"choices": region_lora.MODES},
        section=("stagehand", "Stagehand"),
    ).info("Whole image: like anywhere in Forge, every character gets them; Masked: only her area; "
           "Separate pass: only her area, one extra model run per character with a LoRA"))


script_callbacks.on_ui_settings(_settings)


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
    shares = info.get("shares", {})
    positive = show(prompt, [(n, names.get(n, ""), None if info["auto"] else _format_place(info["boxes"][n]), prompts.get(n, ""), shares.get(n))
                             for n in info["numbers"]])
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
    i = p.iteration * p.batch_size + getattr(p, "batch_index", 0)
    image = info["images"].get(i)
    if image is None:
        return None
    numbers = info["numbers"]
    # owners are matched either way: Precise Reference's per-character references follow them
    prompts = _uses_prompt(args.ad_prompt) or _uses_prompt(args.ad_negative_prompt)
    face = {"masks": masks, "owners": {}, "script": script, "p": p, "args": args, "i": i, "image": image, "numbers": numbers,
            "steps": info["steps"], "prompts": prompts}
    # No character per detection when regional prompting didn't run ("ran": Anima only; the
    # characters were ignored) or with one merged mask over every face -- but every detection
    # still gets the finished prompt (the batch tab leaves that to Stagehand for these images).
    if not info.get("ran") or getattr(args, "ad_mask_merge_invert", "None") != "None":
        return face
    width, height = masks[0].size
    # the regions exactly as sampling had them: Anima's token grid (8x VAE, 2x2 patches)
    h, w = -(-(height // 8) // 2), -(-(width // 8) // 2)
    weights = region_weights([info["boxes"][n] for n in numbers], h, w, shares=[info.get("shares", {}).get(n, SHARE) for n in numbers])
    weights = weights[1:].reshape(len(numbers), h, w)
    found = [(j, m.getbbox()) for j, m in enumerate(masks)]
    found = [(j, box) for j, box in found if box]
    # hands and eyes come several per character (or one, an eye covered): each goes to the
    # region it's in, no picks
    model = str(getattr(args, "ad_model", "")).lower()
    many = "hand" in model or "eye" in model
    picked = match_faces([box for _, box in found], (width, height), weights, None if many else info.get("faces"), one_to_one=not many)
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
    # her own LoRAs (Character LoRAs: Masked / Separate pass): on her face, Forge applies them whole
    text = _join(text, *image.get("loras_prompt", {}).get(n, [])) if n else text
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
        if Path(data.path).stem != "!adetailer":
            continue
        if not all(hasattr(cls, a) for a in ("pred_preprocessing", "i2i_prompts_replace", "get_prompt")):
            # an ADetailer update renamed what's wrapped below; without this, faces would
            # quietly get the main prompt (ui() runs per tab, so say it once)
            if not getattr(_hook_adetailer_faces, "warned", False):
                _hook_adetailer_faces.warned = True
                print("[Character Prompts] WARNING: this ADetailer version doesn't have the functions "
                      "per-character faces hook into. ADetailer will give every face the main prompt.")
            return
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
            # whose face this is: Precise Reference gives it only that character's references
            i2i._nai_character = None
            # only inside the pass that matched these very masks
            if face and j < len(face["masks"]) and getattr(i2i, "image_mask", None) is face["masks"][j]:
                r = face["owners"].get(j)
                i2i._nai_character = face["numbers"][r] if r is not None else None
                if not face["prompts"]:
                    return
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
                gr.HTML('<div class="nai-section">Characters</div>', elem_classes=["nai-title-cell"])
                # The old "AI's Choice" switch: an API arg still, always on in the UI. A character
                # without a position stands in its default column either way (before_process).
                auto = gr.Checkbox(value=True, label="AI's Choice", elem_id=f"nai_{tab}_chars_auto", elem_classes=["nai-hidden"])
                manual = gr.Radio(list(MANUAL), value=MANUAL[0], show_label=False, container=False, elem_id=f"nai_{tab}_chars_manual", elem_classes=["nai-manual"], scale=0, min_width=160)
                # Reset, Switch and the canvas toggles: stagehand.js builds them in here
                gr.HTML("", elem_id=f"nai_{tab}_chars_tools", elem_classes=["nai-title-cell", "nai-tools"])
                gr.HTML("", elem_classes=["nai-spacer"])  # pushes the add button to the right
                add = gr.Button("+ Add character", elem_classes=["nai-add"], min_width=40, scale=0)
            shown = gr.State([False] * MAX_CHARS)
            # every preset as JSON, for the name boxes' search (stagehand.js); read on page load
            # and whenever a name box gets focus, so presets saved elsewhere show up
            presets = gr.Textbox(value=_presets_json, elem_id=f"nai_{tab}_chars_presets", elem_classes=["nai-hidden"], show_label=False, container=False)
            references = getattr(shared.opts, "stagehand_precise_reference", True)
            saves, deletes = [], []

            for i in range(MAX_CHARS):
                with gr.Group(visible=False, elem_id=f"nai_{tab}_char{i + 1}", elem_classes=["nai-card", "nai-char-card", f"nai-char-{i + 1}"]) as card:
                    with gr.Row(elem_classes=["nai-card-head"]):
                        enabled = gr.Checkbox(value=True, label="On", container=False, scale=0, min_width=60, elem_classes=["nai-char-on"])
                        # also the preset search: stagehand.js lists the presets under it, and picking one fills the card
                        name = gr.Textbox(value="", show_label=False, container=False, placeholder=f"Character {i + 1} (type to find a preset)",
                                          max_lines=1, min_width=80, elem_classes=["nai-char-name"])
                        # only matters where places overlap
                        share = gr.Number(value=SHARE, minimum=0, maximum=100, step=5, label="Overlap %", scale=0, min_width=110,
                                          elem_id=f"nai_{tab}_char{i + 1}_share", elem_classes=["nai-share"])
                        up = gr.Button("↑", elem_classes=["nai-icon", "nai-up"], min_width=30, scale=0)
                        down = gr.Button("↓", elem_classes=["nai-icon", "nai-down"], min_width=30, scale=0)
                        copy = gr.Button("⧉", elem_classes=["nai-icon", "nai-copy"], min_width=30, scale=0)
                        save = gr.Button("💾", elem_classes=["nai-icon", "nai-save-preset"], min_width=30, scale=0)
                        # opens the row below (stagehand.js; no server round trip)
                        more = gr.Button("⋯", elem_classes=["nai-icon", "nai-more-btn"], min_width=30, scale=0)
                        remove = gr.Button("🗑", elem_classes=["nai-icon", "nai-remove"], min_width=30, scale=0)
                    with gr.Row(elem_classes=["nai-more"]):
                        # which of ADetailer's detections gets this character's prompt; auto = by position
                        face = gr.Dropdown(FACES, value=FACES[0], show_label=False, container=False, scale=0, min_width=150, elem_classes=["nai-char-face"])
                        delete = gr.Button("🗑 Delete preset", elem_classes=["nai-small", "nai-delete-preset"], min_width=40, scale=0)
                    with gr.Tabs(elem_classes=["nai-char-tabs"]):
                        with gr.Tab("Prompt"):
                            prompt = gr.Textbox(value="", show_label=False, lines=2, placeholder="girl, purple hair, ...", elem_id=f"nai_{tab}_char{i + 1}_prompt")
                        with gr.Tab("Undesired Content"):
                            uc = gr.Textbox(value="", show_label=False, lines=2, elem_id=f"nai_{tab}_char{i + 1}_uc")
                        if references:
                            with gr.Tab("Reference", elem_classes=["nai-ref-tab"]):
                                # Precise Reference's cards for this character land here (stagehand.js)
                                gr.HTML('<div class="nai-hint nai-ref-tab-hint">Images of this character: her look is copied into her part '
                                        'of the image only. Clean images on plain backgrounds work best.</div>'
                                        f'<div class="nai-ref-slot" data-n="{i + 1}"></div>'
                                        f'<button type="button" class="nai-ref-add" data-n="{i + 1}">+ Add reference</button>')
                    # Rendered but hidden by CSS: the position overlay writes here.
                    box = gr.Textbox(value="", elem_id=f"nai_{tab}_char{i + 1}_box", elem_classes=["nai-hidden"], show_label=False, container=False)
                cards.append(card)
                ups.append(up)
                downs.append(down)
                copies.append(copy)
                removes.append(remove)
                # value="" matters: Forge's paste converts with type(component.value), and a
                # Textbox without one is None -- every pasted character was silently dropped
                card_fields = [enabled, name, prompt, uc, box, face, share]
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
                    (share, f"Char {i + 1} share"),
                    # a pasted character is switched on, or it would be pasted and never drawn
                    # string keys, so Send to img2img carries on/off too (_clear_on_paste fills them)
                    (enabled, f"Char {i + 1} on"),
                ]
                prompt.change(_revealer(i), [prompt, shown], [shown, card], show_progress="hidden")
                remove.click(_remove(i), [shown], [shown, card] + card_fields, show_progress="hidden")
                save.click(_save_preset, [name, prompt, uc], [presets], show_progress="hidden")
                # asks first; a cancel hands the backend "" and nothing is deleted
                delete.click(_delete_preset, [name], [presets], show_progress="hidden",
                             _js="(name) => [name && confirm(`Delete the character preset \"${name}\"?`) ? name : '']")
                name.focus(lambda: _presets_json(), None, [presets], show_progress="hidden")
                saves.append(save)
                deletes += [delete, more]

            everything = fields + cards
            for i in range(MAX_CHARS):
                ups[i].click(_swap(i, i - 1), [shown] + fields, [shown] + everything, show_progress="hidden")
                downs[i].click(_swap(i, i + 1), [shown] + fields, [shown] + everything, show_progress="hidden")
                copies[i].click(_duplicate(i), [shown] + fields, [shown] + everything, show_progress="hidden")

            add.click(_add, [shown] + fields, [shown] + everything, show_progress="hidden")
            # Undo / redo: stagehand.js writes the state to go back to here (_restore)
            restore = gr.Textbox(value="", elem_id=f"nai_{tab}_chars_restore", elem_classes=["nai-hidden"], show_label=False, container=False)
            restore.input(_restore, [restore], [shown] + everything + [manual], show_progress="hidden")
            gr.HTML(HELP)

        for component in [auto, add, presets, restore] + ups + downs + copies + removes + saves + deletes:
            component.do_not_save_to_config = True  # ui-config keys collide by label
        # "True"/"False" rather than a callable key, so Send to img2img carries it too
        infotext.append((manual, "Char placement"))
        # pasting an image with characters switches the feature on
        infotext.append((on, "Char feature"))
        self.infotext_fields = infotext
        self.paste_field_names = [key for _, key in infotext if isinstance(key, str)]
        # Added fields go last, so API callers written for the older layouts keep working:
        # [AI's Choice] + 6 x [on, name, prompt, uc, position] + 6 faces + [on/off, placement] + 6 shares
        per_card = [fields[i * CARD_FIELDS : (i + 1) * CARD_FIELDS] for i in range(MAX_CHARS)]
        return ([auto] + [c for card in per_card for c in card[:ARG_FIELDS]] + [card[FACE] for card in per_card] + [on, manual]
                + [card[FACE + 1] for card in per_card])

    # ------------------------------------------------------------------ processing
    def args_from_infotext(self, params):
        """Args for re-running an image from its PNG info (the batch tabs call this). The
        characters themselves ride in the prompt; only the Face picks live in parameters."""
        faces = [params.get(f"Char {n} ADetailer face", FACES[0]) for n in range(1, MAX_CHARS + 1)]
        if all(f == FACES[0] for f in faces):
            return None
        return [True] + [x for _ in range(MAX_CHARS) for x in (True, "", "", "", "")] + faces + [True, MANUAL[0]]

    def prompt_lines(self, prompt, negative, *args):
        """(prompt, negative) with this job's cards written in as the PNG info's character
        lines -- the form an API caller sends, which before_process reads back as cards on the
        Forge that runs the job (forge link's slots forward their jobs that way). Unchanged
        when there are no cards, or when the prompt already carries its own lines or marks
        (before_process would ignore the cards then too)."""
        if not isinstance(prompt, str) or not isinstance(negative, str) or read(prompt)[1] or has_marks(prompt):
            return prompt, negative
        on, manual = _tail(args)
        characters = _characters(args) if on else []
        if not characters:
            return prompt, negative
        auto = _auto(args[0], characters, manual)
        places = None if auto else _places(characters, auto, manual)
        positive = show(prompt, [(n, name, None if auto else _format_place(places[n]), text, share)
                                 for n, name, text, _uc, _box, _face, share in characters])
        negative = show(negative, [(n, "", None, uc) for n, _name, _text, uc, _box, _face, _share in characters])
        return positive, negative

    def before_process(self, p, *args):
        if getattr(p, "_ad_inner", False):
            return
        p._nai_chars = None
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
                           picks.get(n), c["share"]) for n, c in sorted(shown_chars.items())]
            places = [_parse_place(c["box"]) for c in shown_chars.values()]
            auto = not any(places)
            manual = "Grid" if any(pl and len(shapes(pl)[0]) == 2 for pl in places) else "Boxes"
        elif has_marks(p.prompt):
            # Already merged (a restored template the panel hasn't split yet, or an API
            # caller): the marks are the characters; boxes only add names and positions.
            marked = sorted(set(split(p.prompt)[1]) | set(split(p.negative_prompt)[1]))
            known = {c[0]: c for c in characters}
            characters = [known.get(n, (n, "", "", "", None, None, SHARE)) for n in marked]
            if not has_marks(p.negative_prompt):  # the cards' Undesired Content, from the cards
                p.negative_prompt = merge(p.negative_prompt, {c[0]: c[3] for c in characters if c[3]})
        elif characters:
            p.prompt = merge(p.prompt, {c[0]: c[2] for c in characters})
            p.negative_prompt = merge(p.negative_prompt, {c[0]: c[3] for c in characters})
            # A separate hires prompt is NOT merged: Dynamic Prompts and Set Queue never roll
            # it, so it would carry raw wildcards. The hires pass reuses the first pass's rolls.
        if not characters:
            return

        auto = _auto(auto, characters, manual)
        numbers = [c[0] for c in characters]
        boxes = _places(characters, auto, manual)
        faces = {r: c[5] for r, c in enumerate(characters) if c[5] is not None}

        names = {c[0]: c[1] for c in characters if c[1]}
        # the prompts, names and positions go into the PNG info's prompt section (_with_characters)
        p._nai_chars = {"numbers": numbers, "boxes": boxes, "auto": auto, "manual": manual, "faces": faces, "names": names,
                        "shares": {c[0]: c[6] for c in characters}, "steps": p.steps, "images": {}}
        for n, _, _, _, _, face, _ in characters:
            if face is not None:
                p.extra_generation_params[f"Char {n} ADetailer face"] = FACES[face + 1]

    def before_process_batch(self, p, *args, batch_number=0, **kwargs):
        info = getattr(p, "_nai_chars", None)
        if not info:
            return
        start = batch_number * p.batch_size
        # read here, not in before_process: API override_settings are in effect by now
        info["lora"] = _lora_mode()
        lift = _lift_networks if info["lora"] == region_lora.MODES[0] else _keep_networks
        hr_prompts = getattr(p, "all_hr_prompts", None)
        hr_negatives = getattr(p, "all_hr_negative_prompts", None)
        for k in range(len(p.prompts)):
            g = start + k
            base, parts = lift(*split(p.prompts[k]))
            negative, ucs = split(p.negative_prompts[k])
            # "prompt"/"uc" are what was typed (the infotext); "encode*" what the model gets
            image = {"prompt": parts, "uc": ucs, "hr_prompt": parts, "hr_uc": ucs}
            p.prompts[k], p.negative_prompts[k] = base, negative
            p.all_prompts[g], p.all_negative_prompts[g] = base, negative
            if hr_prompts and g < len(hr_prompts) and has_marks(hr_prompts[g]):
                hr_base, image["hr_prompt"] = lift(*split(hr_prompts[g]))
                hr_prompts[g] = hr_base
            if hr_negatives and g < len(hr_negatives) and has_marks(hr_negatives[g]):
                hr_negatives[g], image["hr_uc"] = split(hr_negatives[g])
            # Forge's "Save Raw Comments" keeps an unstripped copy for the infotext; lifted
            # the same way, or get_hr_prompt sees base != base+<lora> and writes a Hires prompt
            raw = getattr(p, "_all_prompts_c", None)
            if raw and g < len(raw) and has_marks(raw[g]):
                raw[g], image["prompt_raw"] = lift(*split(raw[g]))
            raw = getattr(p, "_all_negative_prompts_c", None)
            if raw and g < len(raw) and has_marks(raw[g]):
                raw[g], image["uc_raw"] = split(raw[g])
            info["images"][g] = image
        # grids and the job-wide infotext read main_prompt
        for attr in ("main_prompt", "main_negative_prompt"):
            text = getattr(p, attr, None)
            if has_marks(text):
                setattr(p, attr, lift(*split(text))[0] if attr == "main_prompt" else split(text)[0])

    def process_batch(self, p, *args, batch_number=0, **kwargs):
        """Interaction tags: the plain action into each character, the who-does-what sentence
        into what gets encoded (not into the stored prompt -- it's rebuilt on paste). Custom
        boxes also say where each character is ("a boy on the left")."""
        info = getattr(p, "_nai_chars", None)
        if not info:
            return
        boxes = [info["boxes"][n] for n in info["numbers"]]
        start = batch_number * p.batch_size
        x = info["x"] = _card_mode()  # read here: API override_settings are in effect by now
        if x:
            p.extra_generation_params["Char cards"] = x["label"]
        for k in range(len(p.prompts)):
            image = info["images"].get(start + k)
            if image is None:
                continue
            for kind, prompts in (("prompt", p.prompts), ("hr_prompt", getattr(p, "hr_prompts", None))):
                texts = [image[kind].get(n, "") for n in info["numbers"]]
                # a card's own LoRA tags (Character LoRAs: Masked / Separate pass) aren't text
                image["loras_" + kind] = {n: _NETWORK.findall(t) for n, t in zip(info["numbers"], texts) if _NETWORK.search(t)}
                texts = [_NETWORK.sub("", t).strip(" ,") for t in texts]
                cleaned, phrases = translate_actions(texts, boxes)
                if not info["auto"]:
                    phrases = placement(texts, boxes) + phrases
                image["encode_" + kind] = {n: t for n, t in zip(info["numbers"], cleaned) if t}
                if phrases and prompts is not None and k < len(prompts):
                    prompts[k] = f"{prompts[k]}, {', '.join(phrases)}"
                if x.get("whole") and prompts is not None and k < len(prompts):
                    image["main_" + kind] = prompts[k]
                    if x["global"]:
                        prompts[k] = _join(prompts[k], *image["encode_" + kind].values())
            for kind, negatives in (("uc", p.negative_prompts), ("hr_uc", getattr(p, "hr_negative_prompts", None))):
                if x.get("whole") and negatives is not None and k < len(negatives):
                    image["main_" + kind] = negatives[k]
                    if x["global"]:
                        negatives[k] = _join(negatives[k], *(image[kind].get(n) for n in info["numbers"]))

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
        x = info.get("x") or {}
        steps = _schedule_steps(p, hr)

        def encode(numbers):
            if x.get("whole"):
                return _whole_regions(p, images, numbers, hr, x, steps)
            kind, uc_kind = ("encode_hr_prompt", "hr_uc") if hr else ("encode_prompt", "uc")
            texts = [image[key].get(n) for image in images for key in (kind, uc_kind) for n in numbers]
            encoded = _encode(p, texts, *steps)
            return [[(encoded.get(image[kind].get(n)), encoded.get(image[uc_kind].get(n))) for n in numbers] for image in images]

        regions = encode(info["numbers"])
        # Character LoRAs (Masked / Separate pass): one owner per character and LoRA set, with
        # the batch's images that have it (a wildcard can roll a different LoRA per image)
        loras, owners, sets = {}, {}, {}
        for k, image in enumerate(images):
            for n, own in (image.get("loras_hr_prompt" if hr else "loras_prompt") or {}).items():
                sets.setdefault((n, tuple(own)), set()).add(k)
        if info.get("lora", region_lora.MODES[0]) == region_lora.MODES[0]:
            sets = {}
        clip = p.sd_model.forge_objects.clip
        for (n, own), ks in sets.items():
            pairs, patched, problems = region_lora.load(unet, clip, list(own))
            for problem in problems:
                print(f"[Character Prompts] character {n} LoRA {problem}")
            o = (n, own)
            if pairs:
                loras[o] = pairs
                owners[o] = (n, None if len(ks) == len(images) else ks)
            if patched is not None:  # her text, read by her text-encoder LoRA
                p.sd_model.forge_objects.clip = patched
                try:
                    column = encode([n])
                finally:
                    p.sd_model.forge_objects.clip = clip
                r = info["numbers"].index(n)
                for k in ks:
                    regions[k][r] = column[k][0]
        denoiser = p.sampler.model_wrap_cfg
        session = RegionSession(
            _in_crop(p, [info["boxes"][n] for n in info["numbers"]]),
            regions,
            step=lambda: denoiser.step,
            patch=getattr(diffusion_model, "patch_spatial", 2),
            whole=bool(x.get("whole")),
            strength=x.get("strength", 1.0),
            shares=[info.get("shares", {}).get(n, SHARE) for n in info["numbers"]],
            numbers=info["numbers"],
        )
        unet = unet.clone()
        unet.set_transformer_option(anima_hooks.REGIONS_KEY, session)
        unet.add_conditioning_modifier(session.modifier)
        if loras:
            lora = region_lora.LoraSession(loras, session, info["lora"], owners)
            unet.set_transformer_option(region_lora.KEY, lora)
            if lora.mode == "Separate pass":
                unet.set_model_unet_function_wrapper(lora.wrapper(unet.model_options.get("model_function_wrapper")))
            p.extra_generation_params["Char LoRAs"] = lora.mode.lower()
        p.sd_model.forge_objects.unet = unet
        info["ran"] = True
        layout = "AI's Choice" if info["auto"] else info["manual"].lower()
        own = f", own LoRAs ({info['lora'].lower()}) for {', '.join(sorted({str(n) for n, _ in loras}))}" if loras else ""
        print(f"[Character Prompts] {len(info['numbers'])} characters, {layout}{own}{' (hires)' if hr else ''}")

    def postprocess(self, p, processed, *args):
        # not from inside ADetailer's own pass, which runs every script when "Apply only
        # selected scripts" is off: that would drop the matches for the faces still to come
        if not getattr(p, "_ad_inner", False):
            _faces["pass"] = None
