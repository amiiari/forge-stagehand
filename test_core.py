"""Self-check for the non-obvious logic. Run with Forge's python:

    venv\\Scripts\\python.exe extensions\\forge-stagehand\\test_core.py
"""

import os

import numpy as np
import torch
from PIL import Image

from lib_stagehand.ip_adapter import (
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


def test_ip_target():
    """A reference for one character goes only into her region, as Character Prompts weighs it."""
    torch.manual_seed(0)
    adapter = AnimaIPAdapter(num_blocks=1, embed_dim=6, inner_dim=8)
    q = torch.randn(2, 12, 2, 4)
    out = torch.randn(2, 1, 3, 4, 8)
    emb = torch.randn(2, 1, 8)
    options = {"cond_indices": [0], "uncond_indices": [1]}
    tokens = torch.randn(1, 5, 6)
    weights = torch.rand(3, 12)  # background, character 1, character 4

    class Regions:
        numbers = [1, 4]

        def weights(self, length, device, dtype):
            return weights.to(device=device, dtype=dtype)

    def run(target, regions=Regions()):
        session = IPSession(adapter, [IPReference(tokens, 1.0, 0.0, target)])
        session.capture(0, q, options, regions)
        return session.apply(0, out, emb)

    full = run(None) - out
    # the injected output is linear in the IP attention, so masking it scales each token
    mine = run(4) - out
    assert torch.allclose(mine, full * weights[2].view(1, 1, 3, 4, 1), atol=1e-6)
    assert torch.allclose(run(1) - out, full * weights[1].view(1, 1, 3, 4, 1), atol=1e-6)
    # no such character, or no regions at all: nothing
    assert torch.equal(run(2), out) and torch.equal(run(1, regions=None), out)
    # a whole-image reference ignores the regions
    assert torch.equal(run(None, regions=None), run(None))


def test_reference_args():
    """Per-card options (for, Hires, ADetailer): the current arg layout, the one before them,
    and images from before them pasted or re-run -- each means what it did when it was made."""
    pr = _script_module("precise_reference")
    card = ["img", "Character", 0.6, 0.75]
    empty = ["", "Character", 1.0, 1.0]
    # current: 4 cards, old panel-wide ADetailer, on, then (for, hires, adetailer) per card
    now = card + empty * 3 + [False, True] + ["Character 2", True, False] + ["Whole image", False, False] * 3
    assert pr._cards(now)[0] == ("img", "Character", 0.6, 0.75, 2, True, False)
    assert pr._cards(now)[1][4:] == (None, False, False)
    # a list without them gets the defaults (as Forge's API pads it), the old panel flag still counts
    assert pr._cards(card + empty * 3 + [True, True])[0][4:] == (None, False, True)
    assert pr._cards(card + empty * 3 + [False, True])[0][4:] == (None, False, False)
    assert pr._target("Character 3") == 3 and pr._target(3) == 3
    assert pr._target("Whole image") is None and pr._target(True) is None and pr._target(None) is None

    # an image from before them: its references were in hires (and ADetailer if it said so)
    old = pr._fill_defaults({"PR 1 image": "x.png", "PR in ADetailer": "True"})
    assert (old["PR 1 hires"], old["PR 1 ADetailer"], old["PR 1 for"]) == ("True", "True", "Whole image")
    assert (old["PR 2 hires"], old["PR 2 ADetailer"]) == ("False", "False")  # an empty card: the new defaults
    new = pr._fill_defaults({"PR 1 image": "x.png", "PR 1 hires": "False", "PR 1 ADetailer": "True", "PR 1 for": "Character 2"})
    assert (new["PR 1 hires"], new["PR 1 ADetailer"], new["PR 1 for"]) == ("False", "True", "Character 2")

    # re-run from PNG info (the batch tabs): the full current layout, the kept copy found again
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        kept = f.name
    try:
        args = pr.PreciseReference().args_from_infotext({"PR 1 image": kept, "PR 1 strength": "0.6", "PR 1 for": "Character 2", "PR 1 hires": "True", "PR 1 ADetailer": "False"})
        assert len(args) == pr.MAX_REFS * (pr.CARD_FIELDS + pr.EXTRA_FIELDS) + 2
        assert pr._cards(args)[0] == (kept, "Character", 0.6, 1.0, 2, True, False)
        assert pr.PreciseReference().args_from_infotext({"Steps": "30"}) is None
    finally:
        os.remove(kept)


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
    from lib_stagehand.characters import has_marks, merge, split

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
    from lib_stagehand.characters import auto_boxes, region_weights

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
    # one character in two places owns both; the gap between is the other's
    sheet = region_weights([((0.0, 0.0, 0.3, 1.0), (0.7, 0.0, 1.0, 1.0)), (0.3, 0.0, 0.7, 1.0)], 4, 10, blur=0).reshape(3, 4, 10)
    assert sheet[1, :, :3].eq(1).all() and sheet[1, :, 7:].eq(1).all() and sheet[2, :, 3:7].eq(1).all()
    # Grid: two dots of one character each claim their territory against the other's dot
    dots = region_weights([((0.1, 0.5), (0.9, 0.5)), (0.5, 0.5)], 4, 10, blur=0).reshape(3, 4, 10)
    assert dots[1, :, 0].gt(0.99).all() and dots[1, :, 9].gt(0.99).all() and dots[2, :, 5].gt(0.99).all()
    # overlap shares: 75 vs 25 splits the overlap 3:1; each one's own part and the
    # background are untouched; equal shares are exactly the plain split
    boxes = [(0.0, 0.0, 0.6, 1.0), (0.4, 0.0, 1.0, 1.0)]
    plain = region_weights(boxes, 4, 10, blur=0).reshape(3, 4, 10)
    lap = region_weights(boxes, 4, 10, blur=0, shares=[75, 25]).reshape(3, 4, 10)
    assert torch.allclose(lap[1, :, 5], torch.full((4,), 0.75)) and torch.allclose(lap[2, :, 5], torch.full((4,), 0.25))
    assert torch.equal(lap[:, :, :4], plain[:, :, :4]) and torch.equal(lap[:, :, 6:], plain[:, :, 6:])
    assert torch.equal(region_weights(boxes, 4, 10, shares=[60, 60]), region_weights(boxes, 4, 10))
    edge = region_weights([(0.0, 0.0, 0.5, 1.0)] * 2, 6, 10, blur=1.5, shares=[90, 10]).reshape(3, 6, 10)
    assert torch.allclose(edge[0], region_weights([(0.0, 0.0, 0.5, 1.0)] * 2, 6, 10, blur=1.5).reshape(3, 6, 10)[0])
    assert torch.allclose(edge.sum(0), torch.ones(6, 10), atol=1e-5)
    # 0% yields every overlap but still splits a spot both claim at 0
    zero = region_weights(boxes, 4, 10, blur=0, shares=[0, 100]).reshape(3, 4, 10)
    assert zero[2, :, 5].gt(0.99).all() and torch.allclose(zero.sum(0), torch.ones(4, 10))


def test_region_context():
    from lib_stagehand.characters import real_length, region_context

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
    from lib_stagehand.characters import RegionSession, region_context

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


def test_region_whole_and_strength():
    from lib_stagehand.characters import RegionSession, cat_schedules

    torch.manual_seed(0)
    module = FakeCross(4)
    context = torch.zeros(2, 1, 512, 4)
    context[:, 0, :3] = torch.randn(3, 4)
    x = torch.randn(2, 8, 4)
    one_pass = torch.randn(5, 4)  # a region's whole context: main prompt and card in one pass
    options = {"cond_indices": [0], "uncond_indices": [1]}
    qkv = module.compute_qkv(x, context)
    plain = module.torch_attention_op(*qkv)
    whole_ctx = torch.zeros(1, 1, 512, 4)
    whole_ctx[0, 0, :5] = one_pass
    alone = module.torch_attention_op(*module.compute_qkv(x[:1], whole_ctx))

    def session(**kw):
        s = RegionSession([(0.0, 0.0, 1.0, 1.0)], images=[[([(30, one_pass)], None)]], step=lambda: 10, blur=0, whole=True, **kw)
        s.modifier(None, torch.zeros(1, 4, 1, 4, 8), None, None, None, None, None, None)
        return s.attend(module, *qkv, context, options)

    # one card over the whole frame, whole context: exactly the one-pass prompt's attention
    assert torch.allclose(session()[0], alone[0], atol=1e-6)
    # strength 0.5: halfway between the base and the region
    assert torch.allclose(session(strength=0.5)[0], (plain[0] + alone[0]) / 2, atol=1e-6)

    a, b = torch.ones(2, 4), torch.zeros(3, 4)
    joined = cat_schedules([(5, a), (30, a * 2)], [(17, b), (30, b + 1)])
    assert [end for end, _ in joined] == [5, 17, 30] and [len(t) for _, t in joined] == [5, 5, 5]
    assert joined[1][1][:2].eq(2).all() and joined[1][1][2:].eq(0).all() and joined[2][1][2:].eq(1).all()
    assert cat_schedules(None, [(30, b)]) == [(30, b)]


def test_translate_actions():
    from lib_stagehand.characters import auto_boxes, translate_actions

    prompts = ["girl, blonde hair, source#hug", "girl, black hair, target#hug, smiling"]
    cleaned, phrases = translate_actions(prompts, auto_boxes(2))
    assert cleaned == ["girl, blonde hair, hug", "girl, black hair, hug, smiling"]
    assert phrases == ["hug", "the girl on the left is doing hug to the girl on the right"]
    mutual, phrases = translate_actions(["boy, mutual#holding hands", "girl, mutual#holding hands"], auto_boxes(2))
    assert mutual == ["boy, holding hands", "girl, holding hands"]
    assert phrases[-1] == "the boy on the left and the girl on the right are doing holding hands together"
    # nothing to translate: prompts pass through, nothing added to the base
    assert translate_actions(["girl, red hair"], auto_boxes(1)) == (["girl, red hair"], [])


def test_crop_places():
    from lib_stagehand.characters import crop_places, region_weights

    columns = [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0)]
    # inpaint "Only masked" around a face in the right half of a 1024x768 image
    crop = (580, 40, 220, 260)
    moved = crop_places(columns, crop, (1024, 768))
    w = region_weights(moved, 8, 8, blur=0)
    assert w[2].min() > 0.99, "the whole crop belongs to the right-hand character"
    assert crop_places([(0.75, 0.25)], (512, 0, 512, 384), (1024, 768)) == [(0.5, 0.5)]
    assert crop_places(columns, None, (1024, 768)) is columns  # not inpainting: unchanged
    # a character in two places: each one moves into the crop's frame
    assert crop_places([((0.0, 0.0, 0.5, 0.5), (0.75, 0.25))], (512, 0, 512, 384), (1024, 768)) == [((-1.0, 0.0, 0.0, 1.0), (0.5, 0.5))]


def _script_module(name="character_prompts"):
    """scripts/<name>.py with Forge stubbed out: only its pure helpers run."""
    import importlib.util
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock

    stubs = {name: MagicMock() for name in ("gradio", "modules", "modules.processing", "modules.prompt_parser",
                                            "modules.script_callbacks", "modules.scripts", "modules.sd_samplers",
                                            "modules.shared", "modules.paths_internal", "modules.processing_scripts",
                                            "modules.processing_scripts.comments", "backend", "backend.nn",
                                            "backend.nn.anima", "backend.args", "backend.patcher", "backend.patcher.lora",
                                            "lib_stagehand.adapter_runtime")}
    stubs["modules.paths_internal"].data_path = "."
    stubs["modules.processing_scripts.comments"].strip_comments = lambda text: text
    stubs["modules.scripts"].Script = type("Script", (), {})
    stubs["modules"].scripts = stubs["modules.scripts"]
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(f"{name}_under_test", f"scripts/{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_presets_tab():
    """The Presets tab: Save with a new name renames; it never overwrites another preset."""
    import tempfile

    cp = _script_module()
    real = cp.PRESETS
    cp.PRESETS = os.path.join(tempfile.mkdtemp(), "presets.json")
    try:
        cp._write_presets({"Sally": {"prompt": "girl, white hair", "uc": ""}, "Lily": {"prompt": "girl, purple hair", "uc": "hat"}})
        cp._tab_save("Sally", "Sally (modern)", "girl, white hair\nred eyes", "", "")
        assert cp._presets() == {"Lily": {"prompt": "girl, purple hair", "uc": "hat"}, "Sally (modern)": {"prompt": "girl, white hair\nred eyes", "uc": ""}}
        cp._tab_save("Lily", "Sally (modern)", "girl", "", "")  # taken: nothing changes
        assert cp._presets()["Sally (modern)"]["prompt"] == "girl, white hair\nred eyes" and "Lily" in cp._presets()
        cp._tab_save(None, "Kira", "girl, black hair", "", "")  # + New
        assert cp._preset_names("ki") == ["Kira"] and cp._tab_pick("Lily") == ("Lily", "girl, purple hair", "hat")
        cp._tab_delete("Kira", "")
        assert "Kira" not in cp._presets()
    finally:
        cp.PRESETS = real


def test_reference_actions():
    """A character card's add / delete / duplicate / reorder / undo, on the references (_act)."""
    import json

    pr = _script_module("precise_reference")

    def run(op, shown, cards):
        out = pr._act(json.dumps(op), shown, *[v for c in cards for v in c])
        size = len(pr.EMPTY)
        return out[0], [tuple(out[5 + i * size : 5 + (i + 1) * size]) for i in range(pr.MAX_REFS)]

    sally = ("s.png", "Character", 0.8, 0.6, "Character 1", True, False)
    whole = ("w.png", "Character", 1.0, 1.0, "Whole image", False, False)
    empty = pr.EMPTY
    shown, cards = run({"op": "add", "n": 2}, [True, True, False, False], [sally, whole, empty, empty])
    assert shown == [True, True, True, False] and cards[2][4] == "Character 2" and cards[2][0] is None
    shown, cards = run({"op": "copy", "from": 1, "to": 3}, [True, True, False, False], [sally, whole, empty, empty])
    assert shown == [True, True, True, False] and cards[2] == sally[:4] + ("Character 3",) + sally[5:]
    shown, cards = run({"op": "swap", "a": 1, "b": 2}, [True, True, True, False], [sally, whole, sally[:4] + ("Character 2",) + sally[5:], empty])
    assert [c[4] for c in cards[:3]] == ["Character 2", "Whole image", "Character 1"]
    shown, cards = run({"op": "drop", "n": 1}, [True, True, False, False], [sally, whole, empty, empty])
    assert shown == [False, True, False, False] and cards[0] == empty and cards[1] == whole
    # an undo puts targets back on shown cards only; a hidden card keeps a fresh one
    shown, cards = run({"op": "targets", "values": ["Character 4"] * 4}, [True, False, False, False], [sally, empty, empty, empty])
    assert [c[4] for c in cards[:2]] == ["Character 4", "Whole image"]
    assert run("not json", [True] + [False] * 3, [sally, empty, empty, empty])[1][0] == sally


def test_prompt_lines():
    """forge link forwards a slot's cards as PNG-info lines; they must read back as the same cards."""
    from lib_stagehand.characters import read

    cp = _script_module()
    script = cp.CharacterPrompts()

    def args(auto, cards, on=True, manual="Boxes", shares=()):
        flat = [auto]
        for i in range(cp.MAX_CHARS):
            flat += list(cards[i]) if i < len(cards) else [True, "", "", "", ""]
        return flat + [cp.FACES[0]] * cp.MAX_CHARS + [on, manual] + list(shares)

    ren = (True, "Ren", "girl, dark blue hair, source#hug", "blonde hair", "0.000 0.000 0.500 1.000")
    kira = (True, "", "girl, black hair\nwhite tips", "", "0.500 0.000 1.000 1.000")
    # nobody placed (the UI's AI's Choice arg is always on): no positions written, both cards and
    # Ren's Undesired Content come back
    unplaced = [(*ren[:4], ""), (*kira[:4], "")]
    pos, neg = script.prompt_lines("2girls, cafe", "bad hands", *args(True, unplaced))
    base, chars = read(pos)
    assert base == "2girls, cafe" and [c["text"] for c in chars.values()] == [ren[2], kira[2]], (base, chars)
    assert chars[1]["name"] == "Ren" and chars[1]["box"] == "" and chars[2]["box"] == ""
    nbase, ucs = read(neg)
    assert nbase == "bad hands" and ucs[1]["text"] == "blonde hair" and 2 not in ucs, (nbase, ucs)
    # someone dragged: placed, the others at their default columns, even with the arg on
    pos, _ = script.prompt_lines("2girls", "", *args(True, [ren, (*kira[:4], "")]))
    assert [c["box"] for c in read(pos)[1].values()] == [ren[4], "0.500 0.000 1.000 1.000"], pos
    # undo / redo hands the whole state back as JSON: cards shown, their fields, Boxes / Grid
    import json as _json

    state = {"manual": "Grid", "cards": [{"visible": True, "enabled": False, "name": "Ren", "prompt": "girl", "uc": "x",
                                          "box": "B3", "face": cp.FACES[2], "share": "70"}]}
    out = cp._restore(_json.dumps(state))
    assert out[0] == [True] + [False] * (cp.MAX_CHARS - 1), out[0]
    assert out[1 : 1 + cp.CARD_FIELDS] == [False, "Ren", "girl", "x", "B3", cp.FACES[2], 70], out[1 : 1 + cp.CARD_FIELDS]
    assert out[1 + cp.CARD_FIELDS : 1 + 2 * cp.CARD_FIELDS] == cp._card_values() and out[-1] == "Grid"
    assert len(out) == 1 + cp.MAX_CHARS * cp.CARD_FIELDS + cp.MAX_CHARS + 1
    # ...and anything it can't read changes nothing
    assert len(cp._restore("not json")) == len(out)
    # ...but only a position of the mode's kind counts: a Grid leftover in Boxes is nobody placed
    assert cp._auto(True, [(1, "", "girl", "", cp._parse_place("B3"), None, 50)], "Boxes")
    assert not cp._auto(True, [(1, "", "girl", "", cp._parse_place("B3"), None, 50)], "Grid")
    # Boxes: each card's box goes along
    pos, _ = script.prompt_lines("2girls", "", *args(False, [ren, kira]))
    assert [c["box"] for c in read(pos)[1].values()] == [ren[4], kira[4]], pos
    # Grid: the cell
    pos, _ = script.prompt_lines("2girls", "", *args(False, [(*ren[:4], "B3"), (*kira[:4], "D3")], manual="Grid"))
    assert [c["box"] for c in read(pos)[1].values()] == ["B3", "D3"], pos
    # one character in two places, and overlap shares: both travel in the lines
    two = "0.000 0.000 0.400 1.000 + 0.600 0.000 1.000 1.000"
    pos, _ = script.prompt_lines("1girl", "", *args(False, [(*ren[:4], two), kira], shares=[70, 30]))
    got = read(pos)[1]
    assert got[1]["box"] == two and got[1]["share"] == 70 and got[2]["share"] == 30, pos
    pos, _ = script.prompt_lines("1girl", "", *args(False, [(*ren[:4], "B3 + D3")], manual="Grid"))
    assert read(pos)[1][1]["box"] == "B3 + D3", pos
    # the place parser: several places, invalid parts dropped, one place stays a plain shape
    assert cp._parse_place(two) == ((0.0, 0.0, 0.4, 1.0), (0.6, 0.0, 1.0, 1.0))
    assert cp._parse_place("C3 + nonsense") == cp._parse_place("C3") == (0.5, 0.5)
    assert cp._format_place(cp._parse_place("B3 + D3")) == "B3 + D3"
    # a leftover of the other kind (Boxes <-> Grid) is ignored, as the editor ignores it
    mixed = [(1, "", "girl", "", cp._parse_place("B3 + 0.1 0.1 0.5 0.5"), None, 50)]
    assert cp._places(mixed, False, "Boxes") == {1: (0.1, 0.1, 0.5, 0.5)}
    assert cp._places(mixed, False, "Grid") == {1: (0.3, 0.5)}
    # Character Prompts switched off, or no card with text: untouched
    assert script.prompt_lines("x", "y", *args(True, [ren], on=False)) == ("x", "y")
    assert script.prompt_lines("x", "y", *args(True, [])) == ("x", "y")
    # a prompt that already carries its own character lines keeps them, as before_process would
    pasted = "x\n\nCharacter 1: girl, red hair"
    assert script.prompt_lines(pasted, "", *args(True, [ren])) == (pasted, "")


def test_png_info_lines():
    from lib_stagehand.characters import read, show

    base = "masterpiece, 2girls,\ncafe"
    chars = [(1, "Fran (sait0moriyama)", None, "girl, source#hug, [a::7]"), (2, "", "0.500 0.000 1.000 0.500", "girl,\nsecond line"), (3, "x", None, "")]
    text = show(base, chars)
    assert text == ("masterpiece, 2girls,\ncafe\n\nCharacter 1 (Fran (sait0moriyama)): girl, source#hug, [a::7]\n"
                    "Character 2 at 0.500 0.000 1.000 0.500: girl,\nsecond line")
    got_base, got = read(text)
    assert got_base == base
    assert got == {1: {"name": "Fran (sait0moriyama)", "box": "", "share": 50, "text": "girl, source#hug, [a::7]"},
                   2: {"name": "", "box": "0.500 0.000 1.000 0.500", "share": 50, "text": "girl,\nsecond line"}}
    # an overlap share is written only when it isn't the default, and read back
    shared = show("b", [(1, "Ava", "B3", "girl", 70), (2, "", "D3", "girl", 50)])
    assert shared == "b\n\nCharacter 1 (Ava) at B3, share 70%: girl\nCharacter 2 at D3: girl"
    assert {n: c["share"] for n, c in read(shared)[1].items()} == {1: 70, 2: 50}
    assert read("Character 1, share 0%: girl")[1][1]["share"] == 0
    # a multi-line card (a preset of appearance / outfit / proportions lines, a blank line
    # included) comes back exactly, through the PNG info and through the ⟦n⟧ merge
    from lib_stagehand.characters import merge, split

    card = "girl, red hair,\nblack suit, pencil skirt,\n\nlarge breasts,\nsmug"
    assert read(show("base", [(1, "Hana", None, card)]))[1][1]["text"] == card
    assert split(merge("base", {1: card}))[1][1] == card
    # an Undesired Content section with no main negative prompt
    assert show("", [(2, "", None, "tan, dark skin")]) == "Character 2: tan, dark skin"
    assert read("Character 2: tan, dark skin") == ("", {2: {"name": "", "box": "", "share": 50, "text": "tan, dark skin"}})
    # prompts without characters pass through, including ones that merely mention the word
    assert read("a Character study, 1girl") == ("a Character study, 1girl", {})
    # a whole PNG info pasted as a prompt is left alone: its UC lines would become prompts
    whole = text + "\nNegative prompt: bad\nCharacter 1: blue eyes\nSteps: 20, Seed: 1"
    assert read(whole) == (whole, {})
    assert show("plain", []) == "plain"


def test_grid_points():
    from lib_stagehand.characters import default_cells, format_cell, parse_cell, position_label, read, region_weights, show

    assert parse_cell("C3") == (0.5, 0.5) and parse_cell("a1") == (0.1, 0.1)
    assert parse_cell("F1") is None and parse_cell("C6") is None and parse_cell("") is None
    assert format_cell((0.5, 0.5)) == "C3" and format_cell((0.0, 0.0, 0.4, 1.0)) == "B3" and format_cell((0.999, 0.999)) == "E5"
    assert default_cells(1) == ["C3"] and default_cells(2) == ["B3", "D3"] and default_cells(3) == ["A3", "C3", "E3"]
    # more characters than columns: two rows, never two on one cell
    assert default_cells(6) == ["A2", "B4", "C2", "C4", "D2", "E4"] and len(set(default_cells(6))) == 6
    # the image is split between the points: nearest wins, an even split right on the border,
    # nothing left to the base alone (a fade starved heads near the top of their character)
    w = region_weights([parse_cell("B3"), parse_cell("D3")], 11, 11).reshape(3, 11, 11)  # odd: has a middle
    assert w[1, 5, 3] > 0.99 and w[2, 5, 7] > 0.99 and w[1, 0, 0] > 0.99 and w[2, 10, 10] > 0.99
    assert abs(float(w[1, 5, 5] - w[2, 5, 5])) < 1e-6 and w[1, 5, 5] > 0.49
    assert float(w[0].max()) < 1e-6
    # stacked in one column: top half / bottom half
    w = region_weights([parse_cell("C2"), parse_cell("C4")], 11, 11).reshape(3, 11, 11)
    assert w[1, 1, 5] > 0.99 and w[2, 9, 5] > 0.99 and w[1, 1, 0] > 0.99
    assert position_label(parse_cell("B3"), [parse_cell("B3"), parse_cell("D3")]) == "on the left"
    # PNG info lines carry a cell instead of a box
    text = show("base", [(1, "Ava", "B2", "girl, black hair")])
    assert text == "base\n\nCharacter 1 (Ava) at B2: girl, black hair"
    assert read(text)[1] == {1: {"name": "Ava", "box": "B2", "share": 50, "text": "girl, black hair"}}


def test_has_image():
    import numpy as np

    from lib_stagehand.ip_adapter import has_image

    # a UI upload is an array; comparing it to "" raised and silently disabled every reference
    assert has_image(np.zeros((4, 4, 4), np.uint8))
    assert has_image("C:/refs/a.png") and has_image("iVBORw0KGgo=")
    assert not has_image(None) and not has_image("")


def test_match_faces():
    from lib_stagehand.characters import auto_boxes, match_faces, region_weights

    columns = region_weights(auto_boxes(2), 32, 32)[1:].reshape(2, 32, 32)
    left, right, far_right = (100, 100, 300, 300), (700, 120, 900, 320), (950, 50, 1000, 100)
    # by region, whatever order ADetailer sorted them in
    assert match_faces([right, left], (1024, 1024), columns) == [1, 0]
    # a hand-picked character takes the k-th face from the left; the other gets the rest
    assert match_faces([right, left], (1024, 1024), columns, {0: 1}) == [0, 1]
    # an extra face (the same character drawn twice: multi-angle sheets) goes to the character
    # whose area it's in, after the one-to-one pass
    assert match_faces([left, far_right, right], (1024, 1024), columns) == [0, 1, 1]
    whole = region_weights(auto_boxes(1), 32, 32)[1:].reshape(1, 32, 32)
    assert match_faces([(1000, 300, 1500, 800), (300, 200, 450, 350)], (1600, 1280), whole) == [0, 0]
    # ...but not one outside every box: that's nobody's
    corner = region_weights([(0.0, 0.0, 0.4, 0.4)], 32, 32)[1:].reshape(1, 32, 32)
    assert match_faces([(50, 50, 150, 150), (800, 800, 900, 900)], (1024, 1024), corner) == [0, None]
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
    assert picked == [c for c in range(6) for _ in (0, 1)]
    # hands: several per character, each to the region it's in, picks ignored
    hands = [(50, 600, 100, 650), (400, 600, 450, 650), (600, 600, 650, 650), (900, 600, 950, 650)]
    assert match_faces(hands, (1024, 1024), columns, one_to_one=False) == [0, 0, 1, 1]
    # eyes the same way, however many each character shows: one covered, or three
    eyes = [(150, 200, 180, 220), (600, 200, 630, 220), (700, 200, 730, 220), (800, 200, 830, 220)]
    assert match_faces(eyes, (1024, 1024), columns, one_to_one=False) == [0, 1, 1, 1]


def test_placement():
    from lib_stagehand.characters import auto_boxes, placement

    assert placement(["boy, tall", "girl, short"], auto_boxes(2)) == ["a boy on the left", "a girl on the right"]
    assert placement(["girl", "other thing"], [(0, 0, 1, 0.4), (0, 0.6, 1, 1)]) == ["a girl at the top", "a character at the bottom"]
    # boxes too alike to tell apart in words: say nothing
    assert placement(["girl", "boy"], [(0, 0, 1, 1), (0, 0, 1, 1)]) == []
    # a character in two places: one position word would be wrong about the other
    assert placement(["girl", "boy"], [((0, 0, 0.3, 1), (0.7, 0, 1, 1)), (0.3, 0, 0.7, 1)]) == []
    # one girl on the left, a pair stacked on the right (one carrying the other)
    stacked = [(0, 0.15, 0.45, 1), (0.45, 0.25, 1, 1), (0.5, 0, 1, 0.55)]
    assert placement(["girl, a", "girl, b", "girl, c"], stacked) == ["a girl on the left", "a girl at the bottom right", "a girl at the top right"]
    # four columns can't be told apart by thirds of the image: relative to each other
    assert placement(["girl"] * 4, auto_boxes(4)) == []


def test_position_words_in_actions():
    from lib_stagehand.characters import translate_actions

    stacked = [(0.45, 0.25, 1, 1), (0.5, 0, 1, 0.55)]
    _, phrases = translate_actions(["girl, b, source#piggyback", "girl, c, target#piggyback"], stacked)
    assert phrases == ["piggyback", "the girl at the bottom right is doing piggyback to the girl at the top right"]


def test_region_lora():
    import torch.nn as nn

    from lib_stagehand import region_lora
    from lib_stagehand.characters import RegionSession

    # a 2x4 token grid split into two columns; character 2 has a LoRA on one Linear
    regions = RegionSession([(0, 0, 0.5, 1), (0.5, 0, 1, 1)], [[(None, None)] * 2], step=lambda: 0, blur=0, numbers=[1, 2])
    regions.grid = (1, 2, 4)
    linear = nn.Linear(3, 3, bias=False)
    down, up = torch.ones(1, 3), torch.ones(3, 1)
    x = torch.ones(1, 8, 3)
    session = region_lora.LoraSession({2: {linear: [(down, up, 0.5)]}}, regions, "Masked")
    delta = session.adjust(linear, x, torch.zeros(1, 8, 3))[0, :, 0].reshape(2, 4)
    assert torch.allclose(delta, torch.tensor([[0, 0, 1.5, 1.5]] * 2)), delta  # her column only
    # the text side: in full for her own text, nothing for anyone else's
    linear._rl_context = True
    assert not session.adjust(linear, x, torch.zeros(1, 8, 3)).any()
    session.context_owner = 2
    assert torch.allclose(session.adjust(linear, x, torch.zeros(1, 8, 3)), torch.full((1, 8, 3), 1.5))

    # Separate pass: base, then her call blended in over her region of the latent (4x8 here)
    session = region_lora.LoraSession({2: {}}, regions, "Separate pass")
    calls = []

    def apply_model(x, t, **c):
        calls.append(session.pass_owner)
        return torch.full((1, 1, 1, 4, 8), 1.0 if session.pass_owner else 0.0)

    out = session.wrapper()(apply_model, {"input": None, "timestep": None, "c": {}})
    assert calls == [None, 2] and session.pass_owner is None
    assert out[0, 0, 0, :, :4].eq(0).all() and out[0, 0, 0, :, 4:].eq(1).all(), out

    # hooked Linears read the current session; without one they're untouched
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn, self.cross_attn, self.mlp = nn.Linear(3, 3), nn.Module(), nn.Sequential(nn.Linear(3, 3))
            self.cross_attn.k_proj = nn.Linear(3, 3)

    block = Block()
    region_lora.hook_linears(block)
    region_lora.hook_linears(block)  # twice: still one hook
    assert block.cross_attn.k_proj._rl_context and not block.self_attn._rl_context
    plain = block.self_attn(x)
    region_lora.current = region_lora.LoraSession({2: {block.self_attn: [(down, up, 1.0)]}}, regions, "Masked")
    try:
        hooked = block.self_attn(x)
    finally:
        region_lora.current = None
    assert torch.allclose((hooked - plain)[0, :, 0].reshape(2, 4), torch.tensor([[0, 0, 3.0, 3.0]] * 2))

    # a batch of two images where a wildcard rolled character 2 a different LoRA each: rows
    # b % 2 == 0 are image 0 (cond and uncond rows alike), rows b % 2 == 1 image 1
    regions = RegionSession([(0, 0, 0.5, 1), (0.5, 0, 1, 1)], [[(None, None)] * 2] * 2, step=lambda: 0, blur=0, numbers=[1, 2])
    regions.grid = (1, 2, 4)
    linear = nn.Linear(3, 3, bias=False)
    owners = {(2, 0): (2, {0}), (2, 1): (2, {1})}
    session = region_lora.LoraSession({(2, 0): {linear: [(down, up, 1.0)]}, (2, 1): {linear: [(down, up, 2.0)]}}, regions, "Masked", owners)
    assert session.owner(2, 1) == (2, 1) and session.owner(1, 0) is None
    out = session.adjust(linear, torch.ones(4, 8, 3), torch.zeros(4, 8, 3))[:, :, 0].reshape(4, 2, 4)[:, 0, 2]
    assert out.tolist() == [3.0, 6.0, 3.0, 6.0], out  # each image its own LoRA, in her column
    session = region_lora.LoraSession({(2, 0): {}, (2, 1): {}}, regions, "Separate pass", owners)

    def apply_model(x, t, **c):
        return torch.full((2, 1, 1, 4, 8), 0.0 if session.pass_owner is None else 1.0 + session.pass_owner[1])

    out = session.wrapper()(apply_model, {"input": None, "timestep": None, "c": {}})
    assert out[:, 0, 0, 0, 4].tolist() == [1.0, 2.0] and not out[:, 0, 0, 0, :4].any(), out


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all good")
