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

- **AI's Choice** is equal columns in card order. A version where the model places the
  characters itself (a hidden preview pass plus person segmentation) lost: the segmentation
  sees two overlapping people as one, which is exactly when it would matter.
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

### Overlap share

Where places overlap, each spot used to be split evenly. Each card now has a share (percent,
50 by default); the characters' weights at a spot are scaled by their shares and renormalized
to the same total, so the background's part of a blurred edge, and a spot one character has
alone, don't change. 0% keeps a sliver (0.1) so a spot claimed only by 0% cards still splits.
It travels as `, share N%` in the character's line, written only when it isn't 50, so older
PNG info and API callers read as before. Not A/B tested yet on the lap-pillow case it's for.

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

### Cost

About **1 s per image** (5.5 s vs 4.4 s at 768×1024, 30 steps, RTX 5090):

- The adapter LoRA's patch set gets a stable id, so between consecutive reference generations
  Forge keeps the patched weights instead of re-patching the whole model (~1.6 s).
- The adapter has one `ModelPatcher` for the process, so it isn't re-uploaded each run.

Any pass *without* the LoRA does force that re-patch: another job in between, or an ADetailer
face pass without "also in ADetailer". With it ticked, the face pass was ~3 s/image faster
and kept the face closer to the reference (Fern's eyes stayed purple instead of drifting
darker).

### History

- This started as training-free reference-attention injection (A1111's `reference_only`
  idea). It lost every comparison against the adapter, washed out at the settings that
  transfer anything, and on current Forge crashed every run (`noise_scaling` needs a tensor
  sigma). It was removed, along with its SD/SDXL support.
- **Fixed 2026-10-02: references dropped into the UI were never applied.** A UI upload arrives
  as an image array, and checking it against `""` raised an error that Forge's script runner
  swallowed. API calls, which send base64 text, worked, and that's what the earlier testing
  used. `test_ui.py` exists because of this.
