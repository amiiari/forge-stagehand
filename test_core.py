"""Self-check for the non-obvious logic. Run with Forge's python:

    venv\\Scripts\\python.exe extensions\\forge-stagehand\\test_core.py
"""

import numpy as np
import torch
from PIL import Image

from lib_precise_reference.ip_adapter import (
    AnimaIPAdapter,
    IPReference,
    IPSession,
    flatten,
    letterbox,
    lora_patch_source,
)


def test_ip_session():
    torch.manual_seed(0)
    adapter = AnimaIPAdapter(num_blocks=2, embed_dim=6, inner_dim=8)
    q = torch.randn(4, 12, 2, 4)  # [batch, T*H*W, heads, dim] -- the block's own query
    out = torch.randn(4, 1, 3, 4, 8)
    emb = torch.randn(4, 1, 8)
    options = {"cond_indices": [0, 1], "uncond_indices": [2, 3]}
    tokens, other = torch.randn(1, 5, 6), torch.randn(1, 7, 6)

    def run(*refs, options=options, out=out):
        session = IPSession(adapter, list(refs))
        session.capture(0, q, options)
        return session.apply(0, out, emb)

    full = run(IPReference(tokens, 1.0, 0.0))
    # fidelity 1: only the positive rows move; the negative pass never sees the reference
    assert not torch.allclose(full[:2], out[:2]) and torch.equal(full[2:], out[2:])
    # strength is linear in the injected output, and 0 is exactly nothing
    half = run(IPReference(tokens, 0.5, 0.0))
    assert torch.allclose(half - out, (full - out) * 0.5, atol=1e-6)
    assert torch.equal(run(IPReference(tokens, 0.0, 0.0)), out)
    # an unknown batch layout is left untouched rather than guessed at
    assert torch.equal(run(IPReference(tokens, 1.0, 0.0), options={**options, "uncond_indices": [2]}), out)
    # fidelity 0: the negative pass sees it equally, so CFG stops amplifying it
    both = run(IPReference(tokens, 1.0, 1.0))
    assert torch.allclose(both[:2], full[:2]) and not torch.equal(both[2:], out[2:])
    # two references add up rather than the later one winning
    second = run(IPReference(other, 1.0, 0.0))
    assert torch.allclose(run(IPReference(tokens, 1.0, 0.0), IPReference(other, 1.0, 0.0)) - out, (full - out) + (second - out), atol=1e-5)
    # a block that captured nothing passes its output through
    assert torch.equal(IPSession(adapter, []).apply(1, out, emb), out)
    # Forge's Anima reference frames: T=2 in the block output, still one embedding frame
    two = run(IPReference(tokens, 1.0, 0.0), out=torch.randn(4, 2, 3, 2, 8))
    assert two.shape == (4, 2, 3, 2, 8)


def test_kv_projected_once():
    adapter = AnimaIPAdapter(num_blocks=1, embed_dim=6, inner_dim=8)
    session = IPSession(adapter, [IPReference(torch.randn(1, 5, 6), 1.0, 0.0)])
    q = torch.randn(2, 12, 2, 4)
    for _ in range(3):  # three sampling steps
        session.capture(0, q, {"cond_indices": [0], "uncond_indices": [1]})
        session.apply(0, torch.randn(2, 1, 3, 4, 8), torch.randn(2, 1, 8))
    assert len(session.kv) == 1 and not session.pending
    session.clear()
    assert not session.kv


def test_lora_patch_source():
    lora = {
        "blocks.0.cross_attn.q_proj.lora_A.weight": torch.zeros(4, 8),
        "blocks.0.cross_attn.q_proj.lora_B.weight": torch.zeros(8, 4),
        "blocks.0.mlp.layer1.lora_A.weight": torch.zeros(4, 8),
        "blocks.0.mlp.layer1.lora_B.weight": torch.zeros(16, 4),
    }
    source, to_load = lora_patch_source(lora, 4.0)
    # every module the checkpoint trained is targeted -- not just cross_attn
    assert to_load == {
        "blocks.0.cross_attn.q_proj": "diffusion_model.blocks.0.cross_attn.q_proj.weight",
        "blocks.0.mlp.layer1": "diffusion_model.blocks.0.mlp.layer1.weight",
    }
    assert source["blocks.0.mlp.layer1.alpha"].item() == 4.0
    assert "blocks.0.mlp.layer1.alpha" not in lora_patch_source(lora, None)[0]


def test_flatten():
    # transparency becomes white, not the black a plain RGB conversion gives
    assert (flatten(np.zeros((4, 4, 4), np.uint8)) == 255).all()
    red = np.zeros((4, 4, 4), np.uint8)
    red[..., 0], red[..., 3] = 255, 255
    assert (flatten(red) == [255, 0, 0]).all()
    half = red.copy()
    half[..., 3] = 128
    assert (flatten(half)[0, 0] == [255, 127, 127]).all()
    # palette images with a transparent index, and plain RGB passing straight through
    pal = Image.new("P", (4, 4), 0)
    pal.info["transparency"] = 0
    assert (flatten(pal) == 255).all()
    rgb = np.random.randint(0, 255, (5, 7, 3), dtype=np.uint8)
    assert (flatten(rgb) == rgb).all()


def test_letterbox():
    wide = np.full((100, 400, 3), 200, np.uint8)
    out = letterbox(wide)
    assert out.shape == (512, 512, 3)
    # centred, padded with black above and below, content untouched in the middle
    assert out[:150].max() == 0 and out[-150:].max() == 0 and out[256, 256].tolist() == [200, 200, 200]
    assert out.flags.writeable  # torch.from_numpy warns on read-only arrays


def test_markers():
    from lib_precise_reference.characters import has_marks, merge, split

    parts = {1: "girl, blonde hair", 2: "boy, tall"}
    merged = merge("2girls, park,\nnight", parts)
    assert has_marks(merged) and split(merged) == ("2girls, park,\nnight", parts)
    # no characters: the prompt is untouched, so nothing changes for normal generations
    assert merge("1girl", {}) == "1girl" == merge("1girl", {1: "  "})
    assert split("1girl, solo") == ("1girl, solo", {}) and not has_marks("1girl")
    # a style wrapped around the whole thing lands in the base, not in a character
    styled = "masterpiece, " + merged + ", highres"
    assert split(styled) == ("masterpiece, 2girls, park,\nnight, highres", parts)
    # what Set Queue / Dynamic Prompts do inside a character survives the round trip
    rolled = merged.replace("blonde hair", "silver hair, twin braids")
    assert split(rolled)[1][1] == "girl, silver hair, twin braids"

    # Forge strips '# ...' / '// ...' comments to the end of the line: a comment in one box
    # must not eat the next box or the base, and NovelAI's source#hug must not read as one
    import re

    def strip_comments(text):  # modules/processing_scripts/comments.py
        return re.sub(r"[^\S\n]*(\#|\/\/).*", "", re.sub(r"\/\*.*?\*\/", "", text, flags=re.DOTALL))

    tagged = merge("2girls, park # outdoors", {1: "girl, red hair # note", 2: "girl, source#hug, smiling"})
    base, parts = split(strip_comments(tagged))
    assert base == "2girls, park" and parts == {1: "girl, red hair", 2: "girl, source#hug, smiling"}


def test_region_weights():
    from lib_precise_reference.characters import auto_boxes, region_weights

    weights = region_weights(auto_boxes(2), 4, 8, blur=0).reshape(3, 4, 8)
    assert torch.allclose(weights.sum(0), torch.ones(4, 8))
    assert weights[1, :, :4].eq(1).all() and weights[2, :, 4:].eq(1).all() and weights[0].eq(0).all()
    # outside every box the base prompt alone applies
    gap = region_weights([(0.0, 0.0, 0.25, 1.0)], 4, 8, blur=0).reshape(2, 4, 8)
    assert gap[0, :, 2:].eq(1).all() and gap[1, :, :2].eq(1).all()
    # soft edges still sum to 1 everywhere, and overlapping boxes share
    soft = region_weights([(0.0, 0.0, 0.6, 1.0), (0.4, 0.0, 1.0, 1.0)], 6, 10, blur=1.5)
    assert torch.allclose(soft.sum(0), torch.ones(60), atol=1e-5)
    assert torch.allclose(soft[1].reshape(6, 10)[:, 5], soft[2].reshape(6, 10)[:, 4], atol=1e-5)


def test_region_context():
    from lib_precise_reference.characters import real_length, region_context

    base = torch.zeros(2, 1, 512, 4)
    base[0, 0, :3], base[1, 0, :5] = 1.0, 2.0
    extra = torch.full((2, 4), 3.0)
    out = region_context(base, [extra, None])
    assert out.shape == (2, 1, 512, 4)
    # row 0: base's 3 real tokens then the character's 2; row 1 (nothing extra) unchanged
    assert real_length(out[0, 0]) == 5 and out[0, 0, 3:5].eq(3).all() and out[0, 0, :3].eq(1).all()
    assert torch.equal(out[1], base[1])
    # a long prompt grows the context past 512 rather than truncating it
    long = torch.zeros(1, 1, 512, 4)
    long[0, 0, :511] = 1.0
    assert region_context(long, [extra]).shape == (1, 1, 511 + 2, 4)


class FakeCross(torch.nn.Module):
    """The slice of Anima's SelfCrossAttention that RegionSession and cross_kv use: one
    head, identity projections, plain softmax attention."""

    is_SelfAttn = False
    n_heads = 1
    output_proj = output_dropout = k_proj = v_proj = k_norm = v_norm = torch.nn.Identity()

    def __init__(self, dim):
        super().__init__()
        self.head_dim = dim

    def compute_qkv(self, x, context, rope_emb=None):
        k = context.reshape(context.shape[0], -1, 1, context.shape[-1])
        return x[:, :, None, :], k, k

    @staticmethod
    def torch_attention_op(q, k, v, transformer_options=None):
        q, k, v = q[:, :, 0], k.reshape(k.shape[0], -1, k.shape[-1]), v.reshape(v.shape[0], -1, v.shape[-1])
        return torch.softmax(q @ k.transpose(1, 2), -1) @ v


def test_region_session():
    from lib_precise_reference.characters import RegionSession, region_context

    torch.manual_seed(0)
    module = FakeCross(4)
    context = torch.zeros(2, 1, 512, 4)
    context[:, 0, :3] = torch.randn(3, 4)
    x = torch.randn(2, 8, 4)  # 2 rows (cond, uncond) x a 2x4 token grid
    blonde = torch.randn(2, 4)
    options = {"cond_indices": [0], "uncond_indices": [1]}
    plain = module.torch_attention_op(*module.compute_qkv(x, context))
    qkv = module.compute_qkv(x, context)

    # character 1 owns the left half; it has a prompt but no Undesired Content
    session = RegionSession([(0.0, 0.0, 0.5, 1.0)], images=[[([(30, blonde)], None)]], step=lambda: 0, blur=0)
    session.modifier(None, torch.zeros(1, 4, 1, 4, 8), None, None, None, None, None, None)
    assert session.grid == (1, 2, 4)
    out = session.attend(module, *qkv, context, options).reshape(2, 2, 4, 4)

    ctx = region_context(context[:1], [blonde])
    left = module.torch_attention_op(*module.compute_qkv(x[:1], ctx)).reshape(2, 4, 4)
    assert torch.allclose(out[0, :, :2], left[:, :2], atol=1e-6)  # left: base + character
    assert torch.allclose(out[0, :, 2:], plain.reshape(2, 2, 4, 4)[0, :, 2:], atol=1e-6)  # right: base
    assert torch.allclose(out[1], plain[1].reshape(2, 4, 4), atol=1e-6)  # no UC: negative row untouched

    # prompt editing: the schedule entry for the current step is used
    late = torch.randn(2, 4)
    sched = RegionSession([(0.0, 0.0, 1.0, 1.0)], images=[[([(5, blonde), (30, late)], None)]], step=lambda: 10, blur=0)
    sched.modifier(None, torch.zeros(1, 4, 1, 4, 8), None, None, None, None, None, None)
    expect = module.torch_attention_op(*module.compute_qkv(x[:1], region_context(context[:1], [late])))
    assert torch.allclose(sched.attend(module, *qkv, context, options)[0], expect[0], atol=1e-6)

    # Forge's Anima reference frames ride along on T after the image: base prompt only there
    framed = session.weights(16, torch.device("cpu"), torch.float32)
    assert framed.shape == (2, 16) and framed[:, 8:].tolist() == [[1.0] * 8, [0.0] * 8]
    assert torch.equal(framed[:, :8], session.weights(8, torch.device("cpu"), torch.float32))
    assert session.weights(12, torch.device("cpu"), torch.float32) is None  # not whole frames

    # before the grid is known, or with an unexpected batch layout: plain attention
    fresh = RegionSession([(0.0, 0.0, 0.5, 1.0)], images=[[([(30, blonde)], None)]], step=lambda: 0)
    assert torch.allclose(fresh.attend(module, *qkv, context, options), plain)
    assert torch.allclose(session.attend(module, *qkv, context, {"cond_indices": [0]}), plain)


def test_translate_actions():
    from lib_precise_reference.characters import auto_boxes, translate_actions

    prompts = ["girl, blonde hair, source#hug", "girl, black hair, target#hug, smiling"]
    cleaned, phrases = translate_actions(prompts, auto_boxes(2))
    assert cleaned == ["girl, blonde hair, hug", "girl, black hair, hug, smiling"]
    assert phrases == ["hug", "the girl on the left is doing hug to the girl on the right"]
    mutual, phrases = translate_actions(["boy, mutual#holding hands", "girl, mutual#holding hands"], auto_boxes(2))
    assert mutual == ["boy, holding hands", "girl, holding hands"]
    assert phrases[-1] == "the boy on the left and the girl on the right are doing holding hands together"
    # nothing to translate: prompts pass through, nothing added to the base
    assert translate_actions(["girl, red hair"], auto_boxes(1)) == (["girl, red hair"], [])


def test_png_info_lines():
    from lib_precise_reference.characters import read, show

    base = "masterpiece, 2girls,\ncafe"
    chars = [(1, "Fran (sait0moriyama)", None, "girl, source#hug, [a::7]"), (2, "", "0.500 0.000 1.000 0.500", "girl,\nsecond line"), (3, "x", None, "")]
    text = show(base, chars)
    assert text == ("masterpiece, 2girls,\ncafe\n\nCharacter 1 (Fran (sait0moriyama)): girl, source#hug, [a::7]\n"
                    "Character 2 at 0.500 0.000 1.000 0.500: girl,\nsecond line")
    got_base, got = read(text)
    assert got_base == base
    assert got == {1: {"name": "Fran (sait0moriyama)", "box": "", "text": "girl, source#hug, [a::7]"},
                   2: {"name": "", "box": "0.500 0.000 1.000 0.500", "text": "girl,\nsecond line"}}
    # an Undesired Content section with no main negative prompt
    assert show("", [(2, "", None, "tan, dark skin")]) == "Character 2: tan, dark skin"
    assert read("Character 2: tan, dark skin") == ("", {2: {"name": "", "box": "", "text": "tan, dark skin"}})
    # prompts without characters pass through, including ones that merely mention the word
    assert read("a Character study, 1girl") == ("a Character study, 1girl", {})
    # a whole PNG info pasted as a prompt is left alone: its UC lines would become prompts
    whole = text + "\nNegative prompt: bad\nCharacter 1: blue eyes\nSteps: 20, Seed: 1"
    assert read(whole) == (whole, {})
    assert show("plain", []) == "plain"


def test_grid_points():
    from lib_precise_reference.characters import default_cells, format_cell, parse_cell, position_label, read, region_weights, show

    assert parse_cell("C3") == (0.5, 0.5) and parse_cell("a1") == (0.1, 0.1)
    assert parse_cell("F1") is None and parse_cell("C6") is None and parse_cell("") is None
    assert format_cell((0.5, 0.5)) == "C3" and format_cell((0.0, 0.0, 0.4, 1.0)) == "B3" and format_cell((0.999, 0.999)) == "E5"
    assert default_cells(1) == ["C3"] and default_cells(2) == ["B3", "D3"] and default_cells(3) == ["A3", "C3", "E3"]
    # a point owns its center, shares the space between two points, leaves far corners to the base
    w = region_weights([parse_cell("B3"), parse_cell("D3")], 11, 11).reshape(3, 11, 11)  # odd: has a middle
    assert w[1, 5, 3] > 0.9 and w[2, 5, 7] > 0.9
    assert abs(float(w[1, 5, 5] - w[2, 5, 5])) < 1e-6 and w[1, 5, 5] > 0.3
    assert w[0, 0, 0] > 0.5
    assert position_label(parse_cell("B3"), [parse_cell("B3"), parse_cell("D3")]) == "on the left"
    # PNG info lines carry a cell instead of a box
    text = show("base", [(1, "Ruby", "B2", "girl, black hair")])
    assert text == "base\n\nCharacter 1 (Ruby) at B2: girl, black hair"
    assert read(text)[1] == {1: {"name": "Ruby", "box": "B2", "text": "girl, black hair"}}


def test_has_image():
    import numpy as np

    from lib_precise_reference.ip_adapter import has_image

    # a UI upload is an array; comparing it to "" raised and silently disabled every reference
    assert has_image(np.zeros((4, 4, 4), np.uint8))
    assert has_image("C:/refs/a.png") and has_image("iVBORw0KGgo=")
    assert not has_image(None) and not has_image("")


def test_match_faces():
    from lib_precise_reference.characters import auto_boxes, match_faces, region_weights

    columns = region_weights(auto_boxes(2), 32, 32)[1:].reshape(2, 32, 32)
    left, right, far_right = (100, 100, 300, 300), (700, 120, 900, 320), (950, 50, 1000, 100)
    # by region, whatever order ADetailer sorted them in
    assert match_faces([right, left], (1024, 1024), columns) == [1, 0]
    # a hand-picked character takes the k-th face from the left; the other gets the rest
    assert match_faces([right, left], (1024, 1024), columns, {0: 1}) == [0, 1]
    # one to one: a small extra face in character 2's column stays unassigned
    assert match_faces([left, far_right, right], (1024, 1024), columns) == [0, None, 1]
    # fewer faces than characters: the face goes to the character it sits in
    assert match_faces([right], (1024, 1024), columns) == [1]
    assert match_faces([], (1024, 1024), columns) == []
    # two faces deep in one column (a hug drifting across): a tie, settled left to right
    a, b = (40, 100, 120, 180), (200, 100, 360, 260)
    assert match_faces([b, a], (1024, 1024), columns) == [1, 0]
    # two cards picking the same face: the first keeps it, the other falls back to auto;
    # a pick past the last face is ignored
    assert match_faces([left, right], (1024, 1024), columns, {0: 0, 1: 0}) == [0, 1]
    assert match_faces([left, right], (1024, 1024), columns, {1: 5}) == [0, 1]
    # many faces: nothing capped -- six characters each get the face in their own column
    six = region_weights(auto_boxes(6), 32, 64)[1:].reshape(6, 32, 64)
    faces = [(c * 200 + dx, 100, c * 200 + dx + 60, 160) for c in range(6) for dx in (20, 110)]
    picked = match_faces(faces, (1200, 600), six)
    assert [picked[2 * c] if picked[2 * c] is not None else picked[2 * c + 1] for c in range(6)] == list(range(6))
    assert picked.count(None) == 6
    # hands: several per character, each to the region it's in, picks ignored
    hands = [(50, 600, 100, 650), (400, 600, 450, 650), (600, 600, 650, 650), (900, 600, 950, 650)]
    assert match_faces(hands, (1024, 1024), columns, one_to_one=False) == [0, 0, 1, 1]


def test_placement():
    from lib_precise_reference.characters import auto_boxes, placement

    assert placement(["boy, tall", "girl, short"], auto_boxes(2)) == ["a boy on the left", "a girl on the right"]
    assert placement(["girl", "other thing"], [(0, 0, 1, 0.4), (0, 0.6, 1, 1)]) == ["a girl at the top", "a character at the bottom"]
    # boxes too alike to tell apart in words: say nothing
    assert placement(["girl", "boy"], [(0, 0, 1, 1), (0, 0, 1, 1)]) == []
    # one girl on the left, a pair stacked on the right (one carrying the other)
    stacked = [(0, 0.15, 0.45, 1), (0.45, 0.25, 1, 1), (0.5, 0, 1, 0.55)]
    assert placement(["girl, a", "girl, b", "girl, c"], stacked) == ["a girl on the left", "a girl at the bottom right", "a girl at the top right"]
    # four columns can't be told apart by thirds of the image: relative to each other
    assert placement(["girl"] * 4, auto_boxes(4)) == []


def test_position_words_in_actions():
    from lib_precise_reference.characters import translate_actions

    stacked = [(0.45, 0.25, 1, 1), (0.5, 0, 1, 0.55)]
    _, phrases = translate_actions(["girl, b, source#piggyback", "girl, c, target#piggyback"], stacked)
    assert phrases == ["piggyback", "the girl at the bottom right is doing piggyback to the girl at the top right"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all good")
