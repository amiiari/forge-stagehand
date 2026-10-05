"""Pure logic for Character Prompts: NovelAI-style multi-character prompting on Anima.

NovelAI's model was trained to read per-character captions and positions. Anima wasn't, so
this is regional prompting on its cross-attention: each character's region attends to the
base prompt's tokens plus that character's own, and everything else to the base alone.

Kept free of Forge imports so test_core.py can exercise it.
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------- markers
# Character boxes ride inside the main prompt from the start of processing until the
# per-batch split, so Set Queue words and Dynamic Prompts wildcards reach them too. (XYZ's
# Prompt S/R edits p.prompt before processing starts, so it never sees the boxes.) Characters come first and the base last, so a
# style appended to the prompt lands in the base, not in the last character's region.
# U+27E6/U+27E7: never typed, untouched by any of those, stripped before encoding.
#
# One line per character: Forge strips "# ..." and "// ..." comments to the end of the
# line, so a comment in one box must not swallow the boxes after it, or the base.
_MARK = re.compile(r"\s*⟦(\d+)⟧\s*")
# NovelAI's source#/target#/mutual# would read as comments too; a fullwidth ＃ rides through.
_HASH = re.compile(r"\b(source|target|mutual)#")
_FULLWIDTH_HASH = re.compile(r"\b(source|target|mutual)＃")


def merge(base: str, parts: dict[int, str]) -> str:
    """'⟦1⟧ part' / '⟦2⟧ part' / '⟦0⟧ base', a line each, for each non-empty part."""
    lines = []
    for n in sorted(parts):
        text = _HASH.sub(r"\1＃", parts[n].strip())
        if text:
            lines.append(f"⟦{n}⟧ {text}")
    return "\n".join(lines + [f"⟦0⟧ {base}"]) if lines else base


def split(text: str) -> tuple[str, dict[int, str]]:
    """Inverse of merge: (base, {n: part}). Anything outside a character's segment -- a
    style's prefix before the first mark, say -- belongs to the base."""
    pieces = _MARK.split(text)
    base = [pieces[0]]
    parts = {}
    for i in range(1, len(pieces), 2):
        n, segment = int(pieces[i]), pieces[i + 1]
        if n == 0:
            base.append(segment)
        else:
            parts[n] = _FULLWIDTH_HASH.sub(r"\1#", segment.strip())
    return " ".join(b.strip() for b in base if b.strip()), parts


def has_marks(text) -> bool:
    return isinstance(text, str) and _MARK.search(text) is not None


# ---------------------------------------------------------------------------- PNG info
# An image's PNG info carries its characters as lines in the prompt section, after the main
# prompt (and their Undesired Content after the main negative prompt):
#
#     masterpiece, 2girls, cafe
#
#     Character 1 (Ava): girl, black hair, ...
#     Character 2 (Mei) at 0.500 0.000 1.000 0.500: girl, orange hair, ...
#
# Readable, and complete: the position is there whenever it isn't AI's Choice. Whatever
# copies the prompt somewhere -- the paste button, the batch ADetailer / hires-fix tabs, an
# API caller -- carries the characters along, and Stagehand reads them back.
_WHOLE_INFOTEXT = re.compile(r"^(?:Negative prompt:|Steps: \d)", re.M)
_LABEL = re.compile(r"^Character (\d+)(?: \((.*?)\))?(?: at ((?:[\d.]+ ){3}[\d.]+|[A-E][1-5]))?:[ \t]?(.*)$")


def show(base: str, chars: list) -> str:
    """The main prompt, then a line per character: chars = [(number, name, box text or
    None, text)], empty texts left out."""
    lines = [f"Character {n}{f' ({name})' if name else ''}{f' at {box}' if box else ''}: {text.strip()}"
             for n, name, box, text in chars if text and text.strip()]
    if not lines:
        return base
    return "\n\n".join(part for part in (base.rstrip(), "\n".join(lines)) if part)


def read(text: str) -> tuple[str, dict]:
    """Inverse of show: (main prompt, {number: {"name", "box", "text"}}). A line after a
    character's that isn't another character continues it."""
    if not isinstance(text, str) or "Character " not in text:
        return text, {}
    if _WHOLE_INFOTEXT.search(text):
        return text, {}  # a whole PNG info pasted as a prompt: Forge's paste splits it first
    lines = text.split("\n")
    chars, current, first = {}, None, None
    for i, line in enumerate(lines):
        m = _LABEL.match(line)
        if m:
            first = i if first is None else first
            current = chars[int(m.group(1))] = {"name": m.group(2) or "", "box": m.group(3) or "", "text": m.group(4)}
        elif current is not None:
            current["text"] += "\n" + line
    if first is None:
        return text, {}
    for c in chars.values():
        c["text"] = c["text"].strip()
    return "\n".join(lines[:first]).rstrip(), chars


# ---------------------------------------------------------------------------- regions
# A character's place is a box (x0, y0, x1, y1) or a point (x, y), fractions of the image.
# A point is NovelAI's grid position: the character's center (see region_weights).
Box = tuple
GRID = 5  # NovelAI V4/V4.5's 5x5 position grid: columns A-E, rows 1-5
# A point is a character's center, and the image is split between the points: every spot
# belongs to the nearest one, with a narrow blend at the border (OWN_SIGMA). Two points side by
# side are the AI's Choice columns; stacked ones split top/bottom, diagonal ones diagonally.
# What was tested and lost (ab_grids/26-27): a soft Gaussian area per point -- looks leaked
# where kissing faces meet -- and a fade toward the edges, which starved heads (near the top)
# of their character, so hair colors changed.
OWN_SIGMA = 0.06


def auto_boxes(count: int) -> list[Box]:
    """NovelAI's "AI's Choice", done the regional-prompting way: equal columns, left to right."""
    return [(i / count, 0.0, (i + 1) / count, 1.0) for i in range(count)]


def center(shape) -> tuple:
    """A box's or a point's center."""
    return tuple(shape) if len(shape) == 2 else ((shape[0] + shape[2]) / 2, (shape[1] + shape[3]) / 2)


def parse_cell(text):
    """'C3' -> the point at that cell's center, or None."""
    m = re.fullmatch(r"\s*([A-Ea-e])([1-5])\s*", str(text or ""))
    if not m:
        return None
    return ((ord(m.group(1).upper()) - ord("A") + 0.5) / GRID, (int(m.group(2)) - 0.5) / GRID)


def format_cell(shape) -> str:
    """The grid cell a box's or point's center falls in: 'C3'."""
    x, y = center(shape)
    return f"{chr(ord('A') + min(int(x * GRID), GRID - 1))}{min(int(y * GRID), GRID - 1) + 1}"


def default_cells(count: int) -> list[str]:
    """Where Grid puts characters that have no cell yet: spread across the middle row, in card
    order, like AI's Choice -- or, with more characters than columns, alternating rows 2 and 4
    so no two share a cell."""
    def cell(k):
        row = 3 if count <= GRID else (2 if k % 2 == 0 else 4)
        return f"{chr(ord('A') + min(int((k + 0.5) / count * GRID), GRID - 1))}{row}"

    return [cell(k) for k in range(count)]


def _blur(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(sigma * 3))
    kernel = torch.exp(-(torch.arange(-radius, radius + 1).float() ** 2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    out = F.conv2d(F.pad(mask[None, None], (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    out = F.conv2d(F.pad(out, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
    return out[0, 0]


def crop_places(places: list, crop, full) -> list:
    """Positions (fractions of the whole image) in the frame of a crop of it: crop is
    (x, y, width, height) in pixels of an image of size full (width, height). Without a
    usable crop the positions come back unchanged."""
    if not crop or not full or crop[2] <= 0 or crop[3] <= 0:
        return places
    x, y, w, h = crop
    W, H = full

    def fx(v):
        return (v * W - x) / w

    def fy(v):
        return (v * H - y) / h

    return [(fx(pl[0]), fy(pl[1])) if len(pl) == 2 else (fx(pl[0]), fy(pl[1]), fx(pl[2]), fy(pl[3])) for pl in places]


def region_weights(boxes: list[Box], height: int, width: int, blur: float = 1.5, own: float = OWN_SIGMA) -> torch.Tensor:
    """[1 + len(boxes), height * width] blend weights; row 0 is the background (the base
    prompt alone). Every token's weights sum to 1, so overlapping places share it. A box is a
    blurred rectangle; points split the image between them, nearest point wins (see above)."""
    ys = (torch.arange(height) + 0.5) / height
    xs = (torch.arange(width) + 0.5) / width
    masks = [None] * len(boxes)
    points = [i for i, shape in enumerate(boxes) if len(shape) == 2]
    if points:
        d2 = torch.stack([(xs[None, :] - boxes[i][0]) ** 2 + (ys[:, None] - boxes[i][1]) ** 2 for i in points])
        ownership = torch.softmax(-d2 / (2 * own**2), 0)
        for k, i in enumerate(points):
            masks[i] = ownership[k]
    for i, shape in enumerate(boxes):
        if len(shape) == 2:
            continue
        x0, y0, x1, y1 = shape
        mask = ((ys[:, None] >= y0) & (ys[:, None] < y1) & (xs[None, :] >= x0) & (xs[None, :] < x1)).float()
        masks[i] = _blur(mask, blur) if blur > 0 else mask
    regions = torch.stack(masks) if masks else torch.zeros(0, height, width)
    background = (1 - regions.sum(0)).clamp(min=0)
    weights = torch.cat([background[None], regions])
    weights = weights / weights.sum(0, keepdim=True).clamp(min=1e-6)
    return weights.reshape(len(weights), height * width)


# ---------------------------------------------------------------------------- context
def real_length(tokens: torch.Tensor) -> int:
    """Anima zero-pads its text embeddings to 512 tokens; count the real ones."""
    nonzero = (tokens.abs().sum(-1) > 0).nonzero()
    return int(nonzero[-1, 0]) + 1 if len(nonzero) else 1


def region_context(base: torch.Tensor, extras: list) -> torch.Tensor:
    """The plain definition of a region's context: per row, the base prompt's real tokens,
    then extras[row] (a character's prompt on a positive row, its Undesired Content on a
    negative one) if any, re-padded with zeros. RegionSession builds the same thing in K/V
    space without re-projecting the base; the tests hold it to this.

    base is Anima's cross-attention context, [B, 1, tokens, 1024]; any leading shape works.
    """
    shape = base.shape
    flat = base.reshape(shape[0], -1, shape[-1])
    rows = []
    for b in range(flat.shape[0]):
        row = flat[b, : real_length(flat[b])]
        if extras[b] is not None:
            row = torch.cat([row, extras[b].to(row)], 0)
        rows.append(row)
    length = max(512, max(len(r) for r in rows))
    out = torch.stack([F.pad(r, (0, 0, 0, length - len(r))) for r in rows])
    return out.reshape(*shape[:-2], length, shape[-1])


def pick(schedule, step: int):
    """The tokens a prompt-editing schedule [(end_at_step, tokens), ...] has at this step."""
    if not schedule:
        return None
    return next((tokens for end, tokens in schedule if step <= end), schedule[-1][1])


def cat_schedules(a, b):
    """Two schedules' tokens end to end, switching whenever either one does."""
    if not a or not b:
        return a or b
    ends = sorted({end for end, _ in a} | {end for end, _ in b})
    return [(end, torch.cat([pick(a, end), pick(b, end)])) for end in ends]


def cross_kv(module, context):
    """k and v exactly as SelfCrossAttention.compute_qkv makes them for a cross-attention,
    without recomputing q (ControlLLLite's q_proj hook isn't called an extra time)."""
    heads, dim = module.n_heads, module.head_dim
    k = module.k_proj(context)
    v = module.v_proj(context)
    k = module.k_norm(k.reshape(*k.shape[:-1], heads, dim))
    v = module.v_norm(v.reshape(*v.shape[:-1], heads, dim))
    return k, v


class RegionSession:
    """Regional cross-attention for one sampling pass; lives in transformer_options.

    images[i][r] = (positive schedule, negative schedule) of region r in image i of the
    batch, either None when that character has nothing there. step() is the denoiser's
    current step, for prompt editing inside character boxes.

    A region's keys are the pass's own base keys (its real tokens) followed by the
    character's, projected once per block and cached: K/V projections act per token and
    zero padding projects to zero, so this equals projecting region_context() -- without
    re-projecting the 512-token base for every region of every block of every step.

    Settings > Stagehand: whole -- the schedules are each region's entire context, not extras
    after the base (the default; whole=False is the behaviour above); strength -- how far a
    region's attention replaces the base's (1 = all the way).
    """

    def __init__(self, boxes, images, step, patch=2, blur=1.5, whole=False, strength=1.0):
        self.boxes = list(boxes)
        self.images = images
        self.step = step
        self.patch = patch
        self.blur = blur
        self.whole = whole
        self.strength = strength
        self.grid = None
        self._weights = {}
        self._kv = {}
        self._lengths = (None, None)
        self._warned = False

    def modifier(self, model, x, timestep, uncond, cond, cond_scale, model_options, seed):
        """Conditioning modifier: records the token grid this pass samples on."""
        frames = x.shape[2] if x.ndim == 5 else 1
        self.grid = (frames, -(-x.shape[-2] // self.patch), -(-x.shape[-1] // self.patch))
        return model, x, timestep, uncond, cond, cond_scale, model_options, seed

    def weights(self, length, device, dtype):
        if self.grid is None:
            return None
        frames, height, width = self.grid
        plane = height * width
        if length % plane or length // plane < frames:
            return None
        total = length // plane
        key = (self.grid, total, device, dtype)
        if key not in self._weights:
            w = region_weights(self.boxes, height, width, self.blur)[:, None, :].expand(-1, frames, -1)
            if total > frames:
                # Forge's "[Anima] Enable Reference" appends reference frames on T after the
                # image's own; they get the base prompt alone
                extra = torch.zeros(len(w), total - frames, plane)
                extra[0] = 1
                w = torch.cat([w, extra], 1)
            self._weights[key] = w.reshape(len(w), total * plane).to(device=device, dtype=dtype)
        return self._weights[key]

    def _real_lengths(self, context, options_key=()):
        """Real base-token count per row: one GPU sync per model call, not one per block."""
        key = (context.data_ptr(), tuple(context.shape), self.step(), tuple(options_key))
        if self._lengths[0] != key:
            flat = context.reshape(context.shape[0], -1, context.shape[-1])
            real = flat.abs().sum(-1) > 0
            last = (real * torch.arange(1, real.shape[1] + 1, device=real.device)).amax(1)
            self._lengths = (key, last.clamp(min=1).tolist())
        return self._lengths[1]

    def _char_kv(self, module, tokens, like):
        key = (id(module), id(tokens), like.device, like.dtype)
        if key not in self._kv:
            self._kv[key] = cross_kv(module, tokens.to(device=like.device, dtype=like.dtype))
        return self._kv[key]

    @torch.compiler.disable
    def attend(self, module, q, k, v, context, options):
        """Replaces Anima's cross-attention after q/k/v: base attention everywhere, each
        character's own (base + its tokens) blended in over its region."""
        base = module.torch_attention_op(q, k, v, transformer_options=options)
        weights = self.weights(q.shape[1], q.device, base.dtype)
        cond = set(options.get("cond_indices") or [])
        uncond = set(options.get("uncond_indices") or [])
        batch = q.shape[0]
        if weights is None or len(cond) + len(uncond) != batch:
            if not self._warned:
                self._warned = True
                print("[Character Prompts] unexpected attention layout here -- characters skipped for this pass")
            return module.output_dropout(module.output_proj(base))

        step = self.step()
        images = len(self.images)
        lengths = self._real_lengths(context, tuple(options.get("cond_indices") or ()) + (-1,) + tuple(options.get("uncond_indices") or ()))
        heads, dim = k.shape[-2], k.shape[-1]
        k_rows, v_rows = k.reshape(batch, -1, heads, dim), v.reshape(batch, -1, heads, dim)
        out = None
        for r in range(1, len(weights)):
            rows, keys, values = [], [], []
            for b in range(batch):
                positive, negative = self.images[b % images][r - 1]
                tokens = pick(positive if b in cond else negative, step)
                if tokens is None:
                    continue
                k_char, v_char = self._char_kv(module, tokens, context)
                rows.append(b)
                if self.whole:
                    keys.append(k_char.to(k))
                    values.append(v_char.to(v))
                    continue
                keys.append(torch.cat([k_rows[b, : lengths[b]], k_char.to(k)], 0))
                values.append(torch.cat([v_rows[b, : lengths[b]], v_char.to(v)], 0))
            if not rows:
                continue
            length = max(512, max(len(t) for t in keys))
            k_r = torch.stack([F.pad(t, (0, 0, 0, 0, 0, length - len(t))) for t in keys])
            v_r = torch.stack([F.pad(t, (0, 0, 0, 0, 0, length - len(t))) for t in values])
            region = module.torch_attention_op(q[rows], k_r, v_r, transformer_options=options)
            if out is None:
                out = base.clone()
            out[rows] += self.strength * weights[r][None, :, None] * (region - base[rows])
        return module.output_dropout(module.output_proj(base if out is None else out))


# ---------------------------------------------------------------------------- ADetailer faces
def match_faces(faces: list, size: tuple, weights: torch.Tensor, picks: dict | None = None, one_to_one: bool = True) -> list:
    """Which character each of ADetailer's detections belongs to: [character index or None].

    faces are (x0, y0, x1, y1) in pixels of an image of `size` (width, height); weights is
    [characters, h, w], each character's share of every spot (its column or its box).

    one_to_one (faces, persons): each character gets one detection. picks {character: k} hand
    a character the k-th detection from the left; the rest get the best overall fit by how
    much of each detection lies in each character's region, ties going left to right in card
    order. Detections left over (more than there are characters) stay None. Otherwise (hands,
    several per character) each detection goes to the region it overlaps most."""
    out = [None] * len(faces)
    if not faces or not len(weights):
        return out
    _, h, w = weights.shape
    width, height = size
    fit = []
    for x0, y0, x1, y1 in faces:
        c0, r0 = min(int(x0 / width * w), w - 1), min(int(y0 / height * h), h - 1)
        c1, r1 = max(c0 + 1, -(-int(x1) * w // int(width))), max(r0 + 1, -(-int(y1) * h // int(height)))
        fit.append(weights[:, r0:r1, c0:c1].float().mean((1, 2)))
    fit = torch.stack(fit)  # [detections, characters]
    if not one_to_one:
        return fit.argmax(1).tolist()

    def from_left(js):
        return sorted(js, key=lambda j: faces[j][0] + faces[j][2])

    order = from_left(range(len(faces)))
    chars = list(range(len(weights)))
    for r, k in sorted((picks or {}).items()):
        if r in chars and 0 <= k < len(order) and out[order[k]] is None:
            out[order[k]] = r
            chars.remove(r)
    free = from_left(j for j in range(len(faces)) if out[j] is None)
    if chars and free:
        from scipy.optimize import linear_sum_assignment

        cost = [[-float(fit[j, r]) + 1e-6 * abs(a - b) for b, r in enumerate(chars)] for a, j in enumerate(free)]
        for a, b in zip(*linear_sum_assignment(cost)):
            out[free[a]] = chars[b]
    return out


def placement(prompts: list[str], boxes: list[Box]) -> list[str]:
    """'a boy on the left', 'a girl on the right': where the boxes put each character, in words
    for the base prompt -- Anima keeps a layout better with it than with the regions alone.
    Nothing when two boxes would get the same words."""
    labels = [position_label(box, boxes) for box in boxes]
    if len(set(labels)) < len(labels):
        return []
    return [f"a {subject(prompt)} {label}" for prompt, label in zip(prompts, labels)]


# ---------------------------------------------------------------------------- actions
# NovelAI's interaction syntax: source#hug on the one doing it, target#hug on the one it's
# done to, mutual#hug on both. Anima never saw it, so it becomes plain words: the action
# in each character's region, and a sentence in the base saying who does what to whom.
_ACTION = re.compile(r"\b(source|target|mutual)#\s*([^,\n]+?)\s*(?=,|\n|$)")


def _in_image(box: Box) -> str:
    """Where a box's or point's center sits in the image: 'on the left', 'at the top right', ..."""
    cx, cy = center(box)
    side = "left" if cx < 0.4 else "right" if cx > 0.6 else ""
    level = "top" if cy < 0.4 else "bottom" if cy > 0.6 else ""
    if level:
        return f"at the {level} {side}".rstrip()
    return f"on the {side}" if side else "in the middle"


def position_label(box: Box, boxes: list[Box]) -> str:
    """'on the left' / 'at the top right' / 'in the middle' ...: where the box sits in the
    image when that tells all the boxes apart (a pair stacked on the right is 'at the top
    right' and 'at the bottom right'); otherwise relative to the other boxes."""
    where = [_in_image(b) for b in boxes]
    if len(set(where)) == len(where):
        return _in_image(box)
    cx, cy = center(box)
    xs = sorted(center(b)[0] for b in boxes)
    ys = sorted(center(b)[1] for b in boxes)
    if xs[-1] - xs[0] >= ys[-1] - ys[0]:
        if cx <= xs[0]:
            return "on the left"
        return "on the right" if cx >= xs[-1] else "in the middle"
    if cy <= ys[0]:
        return "at the top"
    return "at the bottom" if cy >= ys[-1] else "in the middle"


def subject(prompt: str) -> str:
    """The character's noun, from the 'girl, ' / 'boy, ' that NovelAI prompts start with."""
    first = prompt.split(",", 1)[0].strip().lower()
    return first if first in ("girl", "boy") else "character"


def translate_actions(prompts: list[str], boxes: list[Box]) -> tuple[list[str], list[str]]:
    """(prompts with the tags made plain, phrases to add to the base prompt)."""
    found = []  # (role, action, character index)
    cleaned = []
    for i, prompt in enumerate(prompts):
        for role, action in _ACTION.findall(prompt):
            found.append((role, action.strip(), i))
        cleaned.append(_ACTION.sub(lambda m: m.group(2).strip(), prompt))

    def who(i):
        return f"the {subject(prompts[i])} {position_label(boxes[i], boxes)}"

    phrases = []
    for action in dict.fromkeys(a for _, a, _ in found):
        sources = [i for role, a, i in found if a == action and role == "source"]
        targets = [i for role, a, i in found if a == action and role == "target"]
        mutual = [i for role, a, i in found if a == action and role == "mutual"]
        phrases.append(action)
        for s in sources:
            for t in targets:
                phrases.append(f"{who(s)} is doing {action} to {who(t)}")
        if len(mutual) > 1:
            phrases.append(f"{' and '.join(who(i) for i in mutual)} are doing {action} together")
    return cleaned, phrases
