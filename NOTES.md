# Stagehand: lab notes

How Stagehand works inside, and what the A/B testing found. The README is the user guide;
this is for anyone changing the code or wondering why it works the way it does. The evidence
grids aren't in the repo (several use other people's art), so results are described in words.

---

## The shared hook

Both features work on Anima's cross-attention, which `backend/nn/anima.py` exposes no patch
points for. `lib_stagehand/anima_hooks.py` wraps `SelfCrossAttention.forward` and
`Block.forward` once, for both features, so neither can silently disable the other, whatever
order the scripts load in. One wrapper computes q/k/v once and hands the same q to both:
recomputing it per feature costs a projection per block per step, and advances ControlLLLite's
per-call counter on `q_proj`. Sessions travel in `transformer_options`; a cross-attention
without one runs the original. Blocks are found by class, not by count.

---

## Character Prompts

### Regional attention

NovelAI's model was trained to read per-character captions. Anima wasn't, so this is regional
prompting on its cross-attention:

- Each character's region attends to the main prompt and its own card, read together in one
  pass, blended 65/35 with one prompt that holds every card (see "The card look" below).
- Everywhere else attends to that one prompt with every card.
- Undesired Content does the same on the negative pass.

Until 2026-10-05 a region attended to the main prompt's tokens plus its card's, encoded
separately, and everywhere else to the main prompt alone. Settings > Stagehand still gives
that: card reading off, strength 1.

Tested on two characters whose descriptions bleed into each other in one prompt (hair colors,
glasses and jackets swap). With Character Prompts each kept its own attributes on every seed,
even while hugging.

Findings along the way:

- Concatenating several separately encoded prompts makes blotchy backgrounds unless each
  character's EOS token is dropped.
- Raw cross-attention affinity is weak (contrast std ~0.02) and follows clothing, not whole
  bodies, so it can't find the characters by itself.
- A free phase followed by a regional one (no restart) locks in the attribute bleed from the
  free phase.
- Propagating masks through self-attention washes them out to uniform.

### Saturation ("it looks like a higher CFG")

Character Prompts images are more saturated and a bit glossier than the same text as one
prompt. Measured over 4 scenes x 4 seeds (2026-10-03):

| | mean saturation | pixels over 0.85 saturation | edge energy |
|---|---|---|---|
| Character Prompts, CFG 4.5 | 0.353 | 2.7% | 0.0406 |
| ... at CFG 4 / 3.5 | 0.348 / 0.343 | 2.5% / 2.2% | 0.0397 / 0.0391 |
| one prompt, CFG 4.5 / 6 / 7.5 | 0.316 / 0.313 / 0.310 | 1.0% / 1.3% / 1.6% | 0.0354 / 0.0363 / 0.0374 |
| each region encoded as main prompt + card together | 0.341 | 2.6% | 0.0380 |
| other cards' keys kept in each region with zero values | 0.340 | 1.6% | 0.0411 |

- It isn't CFG: CFG barely moves saturation on Anima, either way.
- It isn't the regional machinery: one card holding every character's text reproduces the
  plain prompt within rounding (~3/255). (With the style tags in the main prompt and the
  character in a card, the separate encoding alone moves a single-card image 23-30/255:
  see "The card look".)
- It's the split itself: each character's area attends to the main prompt and its own card
  only, so its tags get the attention they'd otherwise share with the other cards. Keeping
  the other cards' keys (values zeroed, so nothing of them leaks in) restores that share and
  halves the burnt pixels -- but weakens each card the same way: a character's weaker traits
  (Ren's dark blue hair) held on 1 of 4 seeds instead of 3. Not adopted.
- Hires fix and ADetailer don't change saturation.

### The card look (2026-10-04/05)

A card made images look different from the same tags typed into the prompt: heavier shading,
glossier skin, details drifting, "like the style LoRA at a very high strength". It isn't the
LoRA (applied once, at its weight) or the seed. The card was encoded on its own and glued onto
the main prompt inside the model, so the character never met the style tags in the text
encoder, and the model never saw two glued prompts in training. A single full-frame card,
same seed, against the same text as one prompt: 23-30/255 apart, up to 53.

Four fixes, 12 versions of 24 scene/seed pairs (288 images, base txt2img):

- **One pass**: each region reads its main prompt and card encoded together. A lone card is
  then the one-prompt image: 0.0/255 when nothing else blends in (1.5-11/255 of GPU rounding
  when it is blended at full strength). On groups it held traits like before (13/15) but
  stayed saturated (+7.8% against one prompt).
- **Regions for the first part of the steps**, then one prompt: traits leak back (5/15
  clean at 30%, no gain over the old way at 70%). Identity settles late, not only layout.
- **Strength**: each region's attention blended with one prompt holding every card. Below
  0.5 traits leak (3/15 at 0.3); 0.5-0.7 kept every major trait on 15/15, including Ren's
  dark blue hair, which the old way lost twice. Saturation +3-5% instead of +8.8%.
- **No card end-of-text**: worse (11/15, Ren's hair orange more often).

Then blind tests, ranked by eye with the versions shuffled:

- One card, 0 to 0.7 of the glued card mixed back in (8 images): the ranking fell almost in
  strength order; 0 (the one-prompt image) averaged 2.50 of 6, 0.7 averaged 4.81.
- Groups (12 images, 9 versions): one pass at 0.7 averaged 4.83 of 9, the same 0.7 without
  it 6.67, the old way 8.75 (last or second-to-last on 12/12).
- Groups, one pass at 0.65 / 0.75 / 0.85, over both rounds (24 images, 10 with interactions:
  kabedon, hugs, arm around neck, holding hands, headpat, back to back, feeding): average
  rank of 3 was 1.92 / 2.04 / 2.04; 0.65 first on 11. The three are often near-identical
  (5-8/255 on half the scenes); the scenes that preferred 0.85 were mostly Akira's (dark
  skin, large tattoo) and a three-way hug.

So the default is one pass at 0.65. A lone card is unaffected by the strength: it's always
the one-prompt image.

- **Hires fix** (x1.25, denoise 0.3) only refines what the first pass made. A lone card with
  hires is pixel-identical to the typed prompt with hires. The hires pass at strength 0,
  0.2, 0.35, 0.5 or 0.65 came out near-identical, so it simply uses the first pass's.
- **ADetailer** is unaffected: each face is repainted with one prompt (main prompt + its
  card) and no regions, whatever the setting.

### Positions

- **The default** ("AI's Choice") is equal columns in card order. A version where the model
  places the characters itself (a hidden preview pass plus person segmentation) lost: the
  segmentation sees two overlapping people as one, which is exactly when it would matter.
- **No AI's Choice switch since 2026-10-08** (the user's call): a character never dragged
  has an empty position field and stands in her column; dragging anyone writes positions, and
  then it's hand placement as before (position words included). So an image nobody placed
  comes out exactly as AI's Choice made it. `_auto` derives it; the arg stays for API callers
  (true: positions count when given). Only a position of the current mode's kind counts.
- **Editing** (stagehand.js "surfaces"): a small canvas per card (that character only) and
  the output-image overlay draw the same places from one `live` drag state, so they move
  together. A character's boxes are drawn as the union of their rectangles (coordinate-
  compressed cells, an edge only between a covered and an uncovered cell), so touching or
  overlapping places read as one shape; the name sits in the biggest piece, a dot in the
  others. Undo records the fields an action changed (before / after, once the page settles)
  and restores through the server (`_restore`), which owns which cards are shown -- undoing
  a drag doesn't throw away text typed since.
- **Grid**: each dot is a character's center; every spot of the image belongs to the nearest
  dot (a softmax over squared distances, `OWN_SIGMA` = 0.06, so the border blends narrowly).
  - Bunk beds with one character in C2 above the other in C4 came out right 3/3, where
    columns lost the top bunk's character 2/3.
  - In a piggyback both heads end up near the top, so a C2-above-C4 layout gave both girls
    the top character's look. Side by side works for that pose. Hence "place the cell where
    the head will be".
  - **Dropped:** a soft Gaussian area per dot, where neighbours blend: looks leaked where
    kissing faces meet (a blue-haired character lost her blue hair on 3/3 seeds). A fade
    toward the image's edges: heads near the top got less of their character.
- **Custom positions are also put into words** ("a boy on the left, a girl on the right"),
  in what gets encoded, not in the stored prompt. In a wide beach scene with small full-body
  figures it kept the order right on 3/3 seeds, against 2/3 for the boxes alone. Boxes are
  named by where they sit in the image when that tells them apart: a pair stacked on the right
  becomes "at the top right" / "at the bottom right", which made a shoulder-carry scene come
  out on 4/4 seeds together with a scene sentence in the main prompt.
- Small figures follow their boxes only loosely: the main prompt's count tags let the model
  draw a character anywhere, even inside the other's box, and then its own prompt doesn't
  reach it. Bigger figures (cowboy shot, upper body) stay in their boxes.

### Several places per character

A card's position field holds `place + place + ...`. Boxes: the character's mask is the union
(max) of her blurred boxes. Grid: each dot takes part in the nearest-dot softmax on its own,
and a character's mask is the sum of her dots' territories, so two dots of one character
split the image with the others' dots exactly as separate characters would. ADetailer then
gives every face inside any of her places her card (see "ADetailer per face"). Position words
are dropped when any character has more than one place. Not A/B tested yet on a sheet.

### Overlap share

Where places overlap, each spot used to be split evenly. Each card now has a share (percent,
50 by default); the characters' weights at a spot are scaled by their shares and renormalized
to the same total, so the background's part of a blurred edge, and a spot one character has
alone, don't change. 0% keeps a sliver (0.1) so a spot claimed only by 0% cards still splits.
It travels as `, share N%` in the character's line, written only when it isn't 50, so older
PNG info and API callers read as before.

Blind test (2026-10-06, five overlap-heavy poses x front 75% / even / front 25%, same seed):
no signal. 75% won twice, even twice, 25% never, and 75% also came last twice. The scenes
themselves failed: with big, heavily overlapping boxes both cards blend in the overlap -- a
lying girl wore the seated girl's boots and fishnets, at 25% she took her hairband; a hug
from behind shrank the front girl into a small figure inside the other's arms ("combined").
Even shares compute exactly the old weights, so this is the layout, not a regression. Kept
(harmless at 50), but it is not the fix for lap / carry poses; what is, is still open --
smaller places that overlap only where the bodies touch, and Grid dots on the heads, are
the next things to try.

### Interaction tags

`source#x` / `target#x` / `mutual#x` become plain words: the action goes into each
character's region, and a sentence like "the girl on the left is doing hug to the girl on the
right" goes into what gets encoded (not into the stored prompt; it's rebuilt on paste). The
`#` is protected from Forge's prompt comments.

**Tested** (two characters):

- Every pose composed and each character kept its own look: kiss, hug, holding hands, headpat,
  hug from behind, lap pillow, princess carry, sitting on lap. One slip: hug from behind, one
  seed, a blonde's hair came out brown. Regions guide rather than cut, so bodies cross into
  each other's half.
- Who does what: right on 3/3 for headpat, and swapping `source#` / `target#` swapped who pats
  whom. On the stacked poses it's right roughly half to two thirds of the time. Princess carry
  is the weakest (1 of 4 in one batch): the model makes the girl in the dress the one being
  carried, whatever the tags say. Some poses have a side the model puts the doer on: a
  braiding scene came out reversed 4/4 with the braider's card first, and right 4/4 with the
  cards swapped.
- **Tried, and not better:** naming each character by its look in the sentence instead of by
  position; telling each character's region its own role; the model placing the characters
  itself (above).

**Who-does-what A/B, 2026-10-02.** Doer: a white-haired girl in a blue sundress; the other: a
red-haired girl in a black hoodie and denim shorts (an outfit pairing that fights the doer).
AI's Choice, 4 seeds per cell; the count is how often the right girl does it.

| Pose | tags, doer's card 1st (left) | tags, doer's card 2nd (right) |
|---|---|---|
| princess carry | 0/4 | 0/4 |
| lap pillow | 0/4 | 1/4 |
| hug from behind | 0/4 | ~3/4 |
| braiding hair | 0/4 | 4/4 |
| piggyback | 3/4 | 0/4 |
| headpat | 0/4 | 1/4 |
| sitting on lap | 4/4 | 4/4 |

- Two habits, no single rule: **outfits cast roles** (the hoodie-and-shorts girl carried, sat,
  patted; the sundress girl sat on the lap) and **some poses have a side** (braiding and hug
  from behind put the doer on the right, piggyback on the left). So the column order is left
  to the user (swap the cards) instead of being reordered automatically.
- On carry, lap pillow, hug from behind and braiding, three more methods were tried (4 seeds each):
  - **A hand-written sentence naming hair colors** ("the girl with short white hair is
    braiding the long red hair of…") put the right *hair* on the doer (lap pillow and braiding
    4/4), but the outfits stayed with the image halves, so each girl wore the other's clothes.
  - **A Precise Reference of a pose image** (Strength 0.8): it replaced both characters' looks
    with the reference's, on every seed. The adapter carries looks, not composition.
  - **img2img from a correctly posed image** (denoise 0.8): about 1 in 4, with the init
    image's colors muddying the result.
- Why carry is so hard: in a princess carry the carried girl's legs and skirt lie across the
  carrier's half of the image, so with columns her lower body gets the carrier's outfit.

### ADetailer per face

ADetailer inpaints each detection with `[PROMPT]` = the image's prompt, which here is the main
prompt alone: without help, faces got repainted from it, and red eyes turned brown, freckles
went, a frown turned neutral. Stagehand wraps two of ADetailer's methods: `pred_preprocessing`
(one pass's sorted masks) matches the detections to characters, and `i2i_prompts_replace`
(just before each is inpainted) makes `[PROMPT]` the main prompt plus that character's prompt.

- Matching is `scipy.optimize.linear_sum_assignment` over how much of each mask lies in each
  character's region on Anima's token grid; ties go left to right in card order. Hands and
  eyes are many-to-one (a covered eye, or a third one, just counts as fewer or more).
- Faces past one per character used to stay unmatched and got the main prompt alone. On a
  one-character multi-angle sheet (2026-10-06) that repainted the close-up's face from the main
  prompt: red eyes where the card said black, lip piercings gone. Now a leftover face goes to
  whoever holds at least half of it.
- A failure in either wrapper falls back to ADetailer's own prompts; an ADetailer version
  without those methods gets a startup warning.

### Character LoRAs (2026-10-08)

Forge merges a LoRA into the weights, so a card's LoRA used to reach every character (its tags
moved to the main prompt). `lib_stagehand/region_lora.py` keeps a card's LoRA out of the
weights and adds its low-rank pair after each Linear of the DiT blocks' attention and MLP
(instance-level `forward` hooks, installed with the block tags; a module global holds the
model call's session, set by the Block wrapper). Two ways, a setting:

- **Masked**: one model call; each image token's LoRA change is scaled by that character's
  region weight (`RegionSession.weights`, the same as her prompt). The cross-attention's k/v
  project text, not image tokens: they get her LoRA in full while projecting her own context
  (`RegionSession._char_kv`), and none on the base context. Cost: a rank-r matmul pair and a
  mask multiply per layer per LoRA; measured 1.3-1.45x the whole job (Python hooks, unfused).
- **Separate pass**: Forge's `model_function_wrapper` runs the model once plain, then once per
  LoRA'd character with that LoRA on everywhere, and blends each prediction in over her region
  (token weights upsampled to the latent). Costs a full model call per LoRA'd character.

Text-encoder halves: her texts are encoded again with `forge_objects.clip` swapped for a clone
carrying the LoRA's TE patches, then swapped back. Checked: a plain image before and after a
job with a TE LoRA is pixel-identical (the clone's patches come off). Note that Forge maps
Anima TE keys as `lora_te_layers_*`; a LoRA saved as `lora_te1_layers_*` (Sally Whitemane's)
has its TE half ignored everywhere, Forge's own loading included. A renamed copy proved the
swap works (183k pixels changed).

ADetailer: in the per-character modes the face prompt carries her LoRA tags, so Forge applies
them whole to her face crop, and only hers. Only plain LoRA (no DoRA, LoKr, LoCon mid) on the
blocks' Linears is applied; anything else is skipped with a console line.

Blind test (the user's pipeline, 8 images: each pair of Sally Whitemane / Cissia / Fluorite and
all three, 2 seeds; ranks best first): Masked mean 1.75 (5 firsts, 3 lasts), Separate pass 1.75
(2 firsts, never last), Whole image 2.5 (1 first, 5 lasts). Masked's three lasts were all pairs
with Cissia, whose LoRA looked off in every version ("overall inaccurate"); with all three girls
Masked won both. Masked is the default: it ties, wins more often, and takes about half Separate
pass's time.

### Metadata and the ⟦n⟧ markers

- Before Set Queue and Dynamic Prompts run, the cards are merged into the prompt with `⟦n⟧`
  markers (`metadata.ini` orders this script first), so their words and wildcards roll inside
  the cards. They're split back out per image before anything is encoded; LoRA tags are
  lifted into the main prompt.
- The PNG info's prompt section gets `Character N (Name) at …: text` lines, written by
  wrapping `processing.create_infotext` (and ADetailer's own copy of it). `_clear_on_paste`
  reads them back. Forge's paste converts values with `type(component.value)`, so every
  Textbox needs `value=""`; the infotext key regex can't contain an apostrophe, so AI's
  Choice is inferred from the positions.
- The batch tabs re-run an image by calling `args_from_infotext(params)` on each script.

---

## Precise Reference

Runs the trained [LuciferTC/Anima-IP-Adapter](https://huggingface.co/LuciferTC/Anima-IP-Adapter)
("Character_Reference"): SigLIP2 encodes the reference, and each of Anima's 28 DiT blocks gets
an extra cross-attention onto it.

### What the A/B testing found

Grids over Lily (Duolingo), Wumpus 2D and 3D, Fern, a Mucha poster and a Van Gogh, all with
prompts that contradict the reference's scene:

- **Characters** come through at Strength 1 / Fidelity 1: hair, face, outfit. The prompt's
  scene and pose survive.
- **Plain backgrounds matter.** A reference on a dark or busy background gets its background
  and pose copied (2D Wumpus); the same character on white doesn't (3D Wumpus).
- **Style references** work on their own at about Strength 0.5. At 1.0 they copy the whole
  artwork (Mucha's frame, Van Gogh's river).
- **Character + Style together:** the character dominates and the style mostly tints colors
  (Fern's hair goes reddish next to Mucha).
- **Several cards add up:** three angles of one character at 1.0 each still looked right.
- **Blocks and steps:** the early half of the blocks contributes nothing and the late half
  carries everything. Cutting the step window short barely matters. Neither is exposed.
- No setting was found that separates a character from its art style, so the Type is only a
  Strength preset.

### How it hooks in

- The adapter's query is the block's own `q_norm(q_proj(x))`, from the shared hook.
- Its output is added, gated, after the whole block, which is where the adapter was trained to
  inject.

Three deliberate differences from the ComfyUI node:

- **All of the LoRA is loaded.** The checkpoint's LoRA covers `self_attn`, `cross_attn` and
  `mlp`; the node's regex loads only `cross_attn`. Here all of it goes through Forge's own
  `load_lora` / `add_patches`. Without the LoRA the adapter copies the reference's pose far
  more often.
- **Strength scales the injected output.** At small nonzero strengths that differs from the
  node, which scales the SigLIP tokens before projections that have biases.
- **Transparency is flattened onto white** before encoding. A plain RGB conversion turns it
  black, and the reference transfers that.

### Fidelity A/B (2026-10-06)

The friend's X/Y/Z complaint ("connected to the wrong parameter") was not wiring: each X/Y/Z
cell matched the same values set by hand pixel for pixel. Fidelity just changes little over
part of its range. Two other meanings were tried against the current one (strength 1, a
fox-girl reference, a cafe prompt that contradicts it, two seeds, fidelity 0-1):

- **current** (how much CFG amplifies the reference): reads like NovelAI's fidelity -- 0 keeps
  the gist (hair, ears), 1 copies outfit, markings and ornaments. Most of the change sits in
  one step, between 0.5 and 0.75 here (between 0 and 0.5 in the friend's grid).
- **pooled tokens** (SigLIP's grid averaged at low fidelity): noise below 0.75. Out of
  distribution for the adapter.
- **late blocks scaled by fidelity**: barely any effect at any value.

Picked by eye: the current one, on both seeds. Unsolved: an even slider (where the step
falls depends on the reference and the strength).

### Cost

About **1 s per image** (5.5 s vs 4.4 s at 768×1024, 30 steps, RTX 5090):

- The adapter LoRA's patch set gets a stable id, so between consecutive reference generations
  Forge keeps the patched weights instead of re-patching the whole model (~1.6 s).
- The adapter has one `ModelPatcher` for the process, so it isn't re-uploaded each run.

Any pass *without* the LoRA does force that re-patch: another job in between, or an ADetailer
face pass without a card ticked for it. With one ticked, the face pass was ~3 s/image faster
and kept the face closer to the reference (Fern's eyes stayed purple instead of drifting
darker).

### Per character, and which passes (2026-10-07)

Every card used to go over the whole image, so two characters' references blended into one.
Now a card is for the whole image or for one Character Prompts card:

- **Sampling:** the card's injected output is multiplied, per token, by that character's
  weight from `RegionSession.weights` -- the same blend her prompt gets, overlap shares
  included. The adapter's output is added after the block and is linear in it, so this is a
  clean mask (test_core `test_ip_target`). Hires reuses the regions, so it masks the same way.
- **ADetailer:** Character Prompts already matches each detection to a character; it now does
  so even when the ADetailer prompt doesn't use `[PROMPT]`, and tags the face's i2i with
  `_nai_character`. A character's card goes to her face only, unmasked (the crop is hers).
  Checked by pixels: the same image with Hsin's card ticked for ADetailer or not differed only
  inside her face box (23.7k pixels), and 0 pixels on the other girl's side.
- **Hires fix / ADetailer per card**, both off by default: a pose or composition card (the
  "pointing" meme moved two girls into its layout) should shape the first pass and nothing
  after. Batch Hires-Fix runs only Forge's hires pass, so it follows the Hires tick.
- A card for a character the image doesn't have is skipped -- pixel-identical to no reference
  (test_api).

Blind test (2026-10-07; Hsin + the silhouette girl through txt2img -> hires -> ADetailer,
ranked shuffled; `stagehand proof\Precise Reference\per-character blind test`, results.png):

- **Character scenes** (beach, cafe, office x 2 seeds): the same order on all 6 images --
  per-character first pass only, then per-character + hires + ADetailer, then no references,
  then blended (the old way) last. Blended was worse than no references at all. One note:
  the no-reference version "has the best style, but not accurate" (the references pull the
  art style toward theirs too).
- **Pointing scene** (the meme as a first-pass Style card at 0.5): "all really sucked" --
  nothing looked like the meme's pose. Pose card + per-character first pass ranked best of a
  bad set (1.5 of 5); pose card alone last. With Character Prompts' columns fixing the layout
  and one reference per girl, the card didn't carry the composition; TEST03 (no characters,
  same card) moved it only modestly. Consistent with the earlier finding that the adapter
  carries looks, not composition. Not a pose tool; untested: a higher pose strength, or
  without Character Prompts.

So the defaults stay: per-character cards, first pass only.

### History

- This started as training-free reference-attention injection (A1111's `reference_only`
  idea). It lost every comparison against the adapter, washed out at the settings that
  transfer anything, and on current Forge crashed every run (`noise_scaling` needs a tensor
  sigma). It was removed, along with its SD/SDXL support.
- **Fixed 2026-10-02: references dropped into the UI were never applied.** A UI upload arrives
  as an image array, and checking it against `""` raised an error that Forge's script runner
  swallowed. API calls, which send base64 text, worked, and that's what the earlier testing
  used. `test_ui.py` exists because of this.
