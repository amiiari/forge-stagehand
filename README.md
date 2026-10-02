# Stagehand

Multi-character prompting and image references for Forge Neo on **Anima**: who's in the
picture, where they stand, what they look like. It's modeled on NovelAI's versions of these
features and looks like them. Both panels sit under the prompt boxes in txt2img and img2img,
inside one **Stagehand** fold-out. Closed, its header says what's active ("2 characters ·
1 reference", or "off"), and it remembers whether you left it open:

- **Character Prompts**: a prompt box per character, placed by "AI's Choice" or by boxes you
  drag over the output image ([NovelAI's multi-character prompting](https://docs.novelai.net/en/image/multiplecharacters/)).
- **Precise Reference**: reference image cards with a type, Strength and Fidelity
  ([NovelAI's Precise Reference](https://docs.novelai.net/en/image/precisereference/)).

Each panel has a **How to use** fold-out under its title, for people who've never used
NovelAI (it's the `HELP` text at the top of each script). `javascript/stagehand.js` does the
browser side: the fold-out, splitting pasted characters into the cards, the position boxes.

Both work on Anima's cross-attention, which `backend/nn/anima.py` exposes no patch points for.
`lib_precise_reference/anima_hooks.py` wraps it once for both features, so neither can
silently disable the other, whatever order the scripts load in.

---

## Character Prompts

**+** adds a character. Each card has:

- an **On** checkbox: untick it to leave the character out of the next image without deleting
  it;
- a name;
- a **Face** dropdown for ADetailer (see below; leave it on auto);
- ↑ / ↓ to reorder, ⧉ to duplicate, 🗑 to delete;
- **Prompt** and **Undesired Content** tabs.

Up to 6 characters.

Write the scene, the style and the count tags (`2girls`, `1boy, 1girl`) in the main prompt,
like on NovelAI. Each box describes one character.

### How it works

NovelAI's model was trained to read per-character captions. Anima wasn't, so this is
regional prompting on its cross-attention:

- Each character's region attends to the main prompt's tokens plus its own, encoded
  separately.
- Everywhere else attends to the main prompt alone.
- Undesired Content does the same on the negative pass.

Tested on two characters whose descriptions bleed into each other in one prompt (hair colors,
glasses and jackets swap). With Character Prompts each kept its own attributes on every seed,
even while hugging.

### Positions

- **AI's Choice** (default) splits the image into equal columns, left to right in card order.
  Unlike NovelAI's, the model doesn't pick the places itself: a version that let it was tested
  and lost (see *Interaction tags*).
- **Turn it off** and a colored box per character appears in a dashed frame over the output
  area. Drag a box by its name tab to move it, and by its corner dot to resize. Clicks elsewhere
  reach the image. Boxes may overlap; the overlap is shared.
- **Positions are proportional:** they're stored as fractions of the image, so the same layout
  applies at any resolution, through hires fix, and with Set Queue's random resolutions.
  - The frame always has the shape of the *next* image (your width × height). It sits on the
    last output when that has the same shape.
  - Boxes guide where each character is; they don't cut it out. A body can spill past its box.
- **Custom boxes are also put into words:** a phrase like "a boy on the left, a girl on the
  right" goes into what gets encoded (not into the stored prompt). In a wide beach scene with
  small full-body figures it kept the order right on 3/3 seeds, against 2/3 for the boxes alone
  (`ab_grids/25_custom_boxes_position_words.png`).
  - Boxes are named by where they sit in the image when that tells them apart. A pair
    stacked on the right becomes "at the top right" / "at the bottom right", which made a
    shoulder-carry scene come out on 4/4 seeds together with a scene sentence in the main
    prompt. Otherwise they're named relative to each other, as with AI's Choice columns.
  - Small figures still follow their boxes only loosely. The main prompt's count tags
    (`1boy, 1girl`) let the model draw a character anywhere, even inside the other's box; then
    its own prompt doesn't reach it. Bigger figures (cowboy shot, upper body) stay in their boxes.

### Interaction tags

NovelAI's `source#hug` (the one doing it), `target#hug` (the one it's done to) and
`mutual#hug` (both) work. Anima never saw that syntax, so it becomes plain words:

- the action goes into each character's region;
- a sentence like "the girl on the left is doing hug to the girl on the right" goes into the
  main prompt.

The `#` is protected from Forge's prompt comments, which would otherwise eat it.

**What's been tested** (two characters):

- **Every pose composed, and each character kept its own look:** kiss, hug, holding hands,
  headpat, hug from behind, lap pillow, princess carry, sitting on lap. There was one slip:
  hug from behind, one seed, the blonde's hair came out brown. Regions guide rather than cut,
  so bodies cross into each other's half. Evidence:
  - six poses at 2 seeds: `ab_grids/23_interactions_one_prompt_vs_characters.png`;
  - hug: `ab_grids/11_interaction_hug.png`;
  - headpat: `ab_grids/19_three_characters_and_direction.png`;
  - five poses at 3 more seeds: the first column of `ab_grids/20_model_placed_vs_columns.png`.
- **Who does what:** right on 3/3 for headpat (swapping `source#` and `target#` swapped who
  pats whom). On the stacked poses (carry, lap pillow, sitting on lap, hug from behind) it's
  right roughly half to two thirds of the time.
  - Princess carry is the weakest: 1 of 4 in one batch. The model tends to make the girl in
    the dress the one being carried, whatever the tags say.
  - When a seed gets it backwards, re-roll it. When it's backwards on *every* seed, swap the two
    cards (↑ / ↓): some poses have a side the model puts the doer on. Kira braiding Ruby's hair
    came out reversed 4/4 with Kira's card first and right 4/4 with the cards swapped.
- **Tried, and not better** (`ab_grids/20_model_placed_vs_columns.png`,
  `21_columns_vs_phrasing_vs_model_placed.png`, `22_role_phrasing.png`). None of these beat
  the current method, so none was kept.
  - The model placing the characters itself (a hidden preview pass plus person
    segmentation): the segmentation sees two people who overlap as one, which is exactly when
    it would matter.
  - Naming each character by its look in the sentence instead of by position.
  - Telling each character's region its own role.

### Set Queue, wildcards, LoRAs, metadata

- **Set Queue words and Dynamic Prompts wildcards work inside the boxes.** Before Set Queue
  runs, the boxes are merged into the prompt with `⟦n⟧` markers (`metadata.ini` orders this
  script first). They're split back out before anything is encoded. Prompt editing (`[a:b:10]`)
  works in boxes too, switching on the same steps as the main prompt.
- **What doesn't reach the boxes:** XYZ Plot's Prompt S/R edits the main prompt before the
  boxes are merged in, so it only searches the main prompt. A box that's only a comment is
  ignored.
- **LoRA tags in a box apply to the whole image,** like everywhere else in Forge. They're moved
  into the main prompt.
- **Metadata: the characters are in the prompt section,** as lines after the main prompt (and
  their Undesired Content after the main negative prompt). That's what the PNG Info tab shows:

  ```
  masterpiece, 2girls, office

  Character 1 (Rebecca) at 0.000 0.000 0.480 1.000: girl, black hair, pink eyes, ...
  Character 2 (Hana) at 0.520 0.000 1.000 1.000: girl, red hair, red eyes, ...
  Negative prompt: worst quality, ...
  ```

  - The name is there when the card has one; the position (`at x0 y0 x1 y1`) whenever AI's
    Choice was off. Only `Char 1 ADetailer face` stays a parameter (when it isn't auto).
  - Pasting an image (or Send to txt2img / img2img) splits the lines back into the cards,
    switches them on, sets AI's Choice, and clears the cards if the image had none. Pasting the
    text into the prompt box does the same. Images from before this format, with
    `Char 1 prompt` parameters, still paste.
  - Anything that copies the prompt carries the characters with it. A prompt with
    `Character N:` lines in it *is* a character prompt: an API caller can send one, and the
    batch tabs re-run images from it (below).
  - A `#` comment in a box only affects that box.
- **Batch ADetailer and Batch Hires-Fix** (the `batch-adetailer` extension) work on these
  images. Both rebuild each image from its PNG info, so the characters, their positions and
  the Precise Reference come along. The batch extension has two small hooks for it
  (`has_characters`, `replay_script_args` in `batch_adetailer_shared.py`). Tested through
  the real UI on a folder: txt2img → Batch ADetailer (each face got its own character) →
  Batch Hires-Fix (`Character Prompts ... custom (hires)`, `Precise Reference +0.60`), with
  the characters and the reference path still in the final image's PNG info.
- **Krita:** Krita AI Diffusion has no per-character regions, so an image opened in Krita
  gets one plain prompt: the `Diffusion Metadata Guard` plugin drops the `Character N:`
  labels and turns `source#hug` into `hug` when it imports the prompt. That's for inpainting
  fixes; a Krita generation is an ordinary one-prompt generation.
- **Regenerating a pasted image** loses Set Queue's `Template:` field from the new image's
  metadata, though the image itself is right. Set Queue compares the merged prompt with the
  pasted base prompt.
- **ADetailer gives each face its own character.** Wherever an ADetailer prompt uses
  `[PROMPT]` (or is empty, which means the same), each detection's `[PROMPT]` becomes the main
  prompt plus the prompt of the character it belongs to. Its negative gets that character's
  Undesired Content the same way. Without this, faces got repainted from the main prompt alone:
  red eyes turned brown, freckles went, a frown turned neutral
  (`ab_grids/18_characters_with_adetailer.png`; fixed: `ab_grids/24_adetailer_per_face.png`).
  - **Automatic:** each character gets one detection, by how much of it lies in the
    character's column or box. Ties (both faces deep in one column, as in a hug) go left to
    right in card order.
    - Detections left over keep the main prompt (more faces than characters, e.g. someone in
      the background).
    - If a character's own face isn't detected, a leftover face can get its prompt.
  - **By hand:** each card's **Face** dropdown (`Face: auto` / `Face: 1st from left` / …) hands
    that character the n-th detection from the left, counted across the image. A duplicated
    card starts on auto.
  - **Hands** come several per character. With a hand model each hand goes to the character
    whose region it's in, and the Face picks don't apply. Person models work like faces.
  - **Left alone:**
    - an ADetailer prompt without `[PROMPT]`;
    - the Merge / Merge and Invert mask modes, where one mask covers every face;
    - non-Anima checkpoints, where the characters are ignored entirely.
  - The console prints which face, counted from the left, went to which character. If
    anything goes wrong here, ADetailer runs as it would without characters.
- **Inpaint "Only masked"** works differently: the boxes apply to the crop, not the whole
  image.
- **Set Queue's "Restore template"** (or pasting a merged prompt) puts the `⟦n⟧`-marked text
  back in the prompt box. The panel splits it back into the boxes by itself.

### Limits

- **Anima only.** Other checkpoints ignore the boxes, ADetailer's faces included, with a
  console message.
- **Speed:** about +25% generation time with two characters at 1280×1024.
  Each character's region runs its own attention on the positive pass, and on the negative
  pass only when it has Undesired Content.
- **Character count:** three characters stayed separate and in order on 3/3 seeds
  (`ab_grids/19_three_characters_and_direction.png`). Past ~4 it gets less reliable as regions
  shrink.

---

## Precise Reference

Runs the trained Anima IP-Adapter
([LuciferTC/Anima-IP-Adapter](https://huggingface.co/LuciferTC/Anima-IP-Adapter),
"Character_Reference"). SigLIP2 encodes the reference, and each of Anima's 28 DiT blocks gets
an extra cross-attention onto it. It needs a 28-block Anima (the base); the 40-block
`anima29B` is skipped with a console message.

### Models

Both go in `models\precise_reference\` (about 2 GB):

| File | From |
|---|---|
| `ip_adapter-Character_Reference-10.safetensors` | [LuciferTC/Anima-IP-Adapter](https://huggingface.co/LuciferTC/Anima-IP-Adapter) |
| `siglip2-base-patch16-512\` (`config.json`, `preprocessor_config.json`, `model.safetensors`) | [google/siglip2-base-patch16-512](https://huggingface.co/google/siglip2-base-patch16-512) |

### Controls

- **Type**: NovelAI's three. It only picks the starting Strength (Style = 0.5, the others
  1.0). The adapter has one mode, and no setting was found that separates a character from
  its art style, so Character and Character & Style behave the same.
- **Strength**: how much of the reference goes in. Negative pushes away from it.
  - At exactly 0 the card is ignored.
  - At any other value the adapter's own LoRA is applied at full strength. So 0.01 still
    changes the image slightly through the LoRA, even though the reference itself barely
    contributes.
- **Fidelity**: how hard the reference is to override with the prompt. Technically, how much
  CFG amplifies it: at 1 only the positive pass sees it, at 0 both passes do.

Settings go into the PNG info and restore on paste, **the images too**: the browser never says
where a dropped file came from, so each reference is kept as a copy in
`outputs\stagehand references\` (named by its content, so a reference is kept once however
often it's used), and that path goes into the PNG info as `PR 1 image`. Pasting the image, or
a batch tab re-running it, loads the reference back. XYZ Plot axes exist per card
(`[Precise Ref] Ref 1 strength`, ...).

**Fixed 2026-10-02: references dropped into the UI were never applied.** A UI upload arrives as
an image array, and checking it against `""` raised an error that Forge's script runner
swallowed, so every UI generation silently ran without its references (API calls, which send
base64 text, worked; that's what all the earlier testing used). If a reference "looked nothing
like" the result before, that's why.

**also in ADetailer** (off by default): ADetailer's face pass normally runs only the scripts
in its "Script names" setting, so it ignores the reference. Ticked, the reference joins that
pass for that generation, and the setting itself isn't changed. In testing it kept the face
closer to the reference (Fern's eyes stayed purple instead of drifting darker). It was also
~3 s/image faster, because the model isn't re-patched between the passes.

### What the A/B testing found

Grids over Lily (Duolingo), Wumpus 2D and 3D, Fern, a Mucha poster and a Van Gogh, all with
prompts that contradict the reference's scene:

- **Characters** come through at Strength 1 / Fidelity 1: hair, face, outfit. The prompt's
  scene and pose survive.
- **Plain backgrounds matter.** A reference on a dark or busy background gets its background
  and pose copied (2D Wumpus); the same character on white doesn't (3D Wumpus). If that
  happens, lower Fidelity (~0.6).
- **Style references** work on their own at about Strength 0.5. At 1.0 they copy the whole
  artwork (Mucha's frame, Van Gogh's river).
- **Character + Style together:** the character dominates and the style mostly tints colors
  (Fern's hair goes reddish next to Mucha). NovelAI warns that multiple references blend.
- **Blocks and steps:** the early half of the blocks contributes nothing and the late half
  carries everything. Cutting the step window short barely matters. Neither is exposed.

### How it hooks in

- The adapter's query is the block's own `q_norm(q_proj(x))`. The shared hook computes it once
  for both features.
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

About **1 s per image** (5.5 s vs 4.4 s at 768×1024, 30 steps, RTX 5090). Two things keep it
there:

- The adapter LoRA's patch set gets a stable id, so between consecutive reference generations
  Forge keeps the patched weights instead of re-patching the whole model (~1.6 s).
- The adapter has one `ModelPatcher` for the process, so it isn't re-uploaded each run.

Any pass *without* the LoRA does force that re-patch: another job in between, or an ADetailer
face pass without "also in ADetailer".

### Replaced

This extension started as training-free reference-attention injection (A1111's
`reference_only` idea). It lost every comparison against the adapter: washed out at the
settings that transfer anything. On current Forge it also crashed every run (`noise_scaling`
needs a tensor sigma). It was removed, along with its SD/SDXL support.

---

## Tests

```
venv\Scripts\python.exe extensions\forge-stagehand\test_core.py
```

What they cover:

- **Character Prompts:**
  - marker round trips, including Set Queue / wildcard rolls, styles and Forge's comment
    stripping;
  - region weights;
  - context building;
  - the regional attention against a hand-computed result, prompt editing, and its fallbacks;
  - interaction-tag translation and the position words for custom boxes;
  - the PNG info's character lines, written and read back;
  - matching ADetailer's detections to characters: regions, picks (incl. duplicate and
    out-of-range ones), left-to-right ties, many detections, hands.
- **Precise Reference:**
  - the adapter math (fidelity rows, linear strength, additive references, untouched unknown
    batch layouts, K/V projected once, reference frames);
  - the LoRA target map;
  - transparency flattening and letterboxing;
  - telling an uploaded image from "no image" (the UI bug above).
