# Stagehand

ahahah hi! i hope you find this repo useful, used opus 5.5 to make this extension after ironing out most of the details for how the logic would work!
anyways yeah, 2+ character generations on anima were possible but ugh still annoying, however 10x easier compared to the old CLIP infrastructure that illustrious and pony had.

so yeah, this extension is supposed to generally make things easier on forge neo for putting characters 
in the right spot! especially when it comes to doing 2+ character images, i've tested with 4 and it turned 
out pretty good! you can drag a character's dot thing and it'll snap to a 5x5 grid, however i doubt it would 
listen thaat much, but it does work for the most part! but yeah the normal feature with the ai's choice evenly 
distributes a regional prompting area depending on how many characters are actually in the image. if you'd 
rather be more specific (even more than the dragging thing) you can set it to boxes, where you can manually 
resize each box yourself etc. etc. yeah... it uses the same source# and target# notation that novel uses too (i hope)

i havent tried much NSFW stuff so yeah that might be a limitation. 

um theres also the precise reference or in forge terms, controlnet's ipadapter, so you throw in an image 
(i suggest decently large though LMFAO), and it will generate a small lora based on the input images and then 
apply it to the images, i honestly cant imagine using this feature much BUT on the off chance i do, i'd like 
to have it ahahaha, it definitely works! but ehhh if i had a commission of a character that anima didnt know 
i wouldnt really rely on it. 

in the settings i have opened up some? of the internals? with some parameters that you can edit since results 
on different models and parameters definitely would differ. i based the defaults on the best outcomes that i 
got myself after hours of A/B testing so yes! feel free to edit those if you'd like. 

some things you should know:
- THIS ONLY WORKS FOR ANIMA !!!!!!!!!!!!!!!!!!!!!!!!!!
- precise reference will not work well i think? with 2+ characters so just know that too
- this extension is definitely not perfect!!! just keep that in mind!

## thank you for visiting! and using even?! i hope this repo ends up being useful to you.

okay! anyways, i will let opus 5.5 take it from here with a more in depth explanation of the details and stuff 
i'd rather not type LMAO 

## Settings

**Settings → Stagehand → Precise Reference**: untick it (then Reload UI) to remove Precise
Reference completely: its section and pill under the prompt, its paste handling, its XYZ Plot
axes and its ADetailer hook. Character Prompts is unaffected.

## Install

Clone into Forge Neo's `extensions` folder and restart. Character Prompts needs nothing
else. Precise Reference needs two models in `models\precise_reference\` (about 2 GB):

| File | From |
|---|---|
| `ip_adapter-Character_Reference-10.safetensors` | [LuciferTC/Anima-IP-Adapter](https://huggingface.co/LuciferTC/Anima-IP-Adapter) |
| `siglip2-base-patch16-512\` (`config.json`, `preprocessor_config.json`, `model.safetensors`) | [google/siglip2-base-patch16-512](https://huggingface.co/google/siglip2-base-patch16-512) |

---

## Character Prompts

**+ Add character** adds a card (up to 6). Each card has:

- an **On** / **Off** pill in the character's color: Off leaves the character out of the next
  image without deleting it;
- a name;
- a **Face** dropdown for ADetailer (leave it on auto; see below);
- ↑ / ↓ to reorder, 💾 to save it as a preset, ⧉ to duplicate, 🗑 to delete;
- **Prompt** and **Undesired Content** tabs.

Write the scene, the style and the count tags (`2girls`, `1boy, 1girl`) in the main prompt,
like on NovelAI. Each card describes one character, starting with `girl` or `boy`. A card can
span several lines (appearance, outfit, proportions...); the line breaks are kept everywhere,
PNG info included.

### Presets

Name a card, then **💾** saves it as a preset: its prompt and Undesired Content exactly as
typed, line breaks included. Saving a card with the same name again updates the preset. Pick a
preset in the list next to **+ Add character**, then click **+ Add character** to add a card
filled with that character; **🗑** next to the list deletes the selected preset (it asks first). Presets live in
`stagehand character presets.json` in Forge's folder, so updating or reinstalling the
extension keeps them.

### Positions

- **AI's Choice** (default): equal columns, left to right in card order.
- **Turn it off** to place the characters yourself, in one of two ways (the switch next to
  AI's Choice; switching keeps each character where it was):
  - **Boxes**: drag a box by its name tab to move it, by its corner dot to resize. Boxes may
    overlap; the overlap is shared, by each card's **Overlap share %**.
  - **Grid**: NovelAI's 5×5 grid. Each character is a dot that snaps to a cell. Every part of
    the image belongs to the nearest dot, so stacked or diagonal dots give layouts columns
    can't (bunk beds, someone in the foreground).
    - **Put each dot where that character's head will be.** Looks are decided around the head.
    - Give each character its own cell; two on one cell merge.
    - A lone character's dot only changes the position words ("a girl on the left"). Use
      Boxes to confine one.
- **More than one place for a character** (with AI's Choice off): **＋** on her card adds
  another box (or dot, in Grid) beside her last one; drag it like any other, **×** on it
  removes it. Every place reads the same card. For multi-angle sheets (full body on the left,
  close-up on the right, one character) and compositions where one character spans two areas.
  Want different words per view? Duplicate the card (⧉) and edit it instead. A character with
  several places gets no position words ("a girl on the left").
- **Overlap share %** (on each card, with AI's Choice off): who wins where two places overlap.
  Every card starts at 50; the overlap is split in proportion, so 70 vs 30 gives 70/30 and
  80 vs 40 gives 2/3 vs 1/3. One character lying on another's lap: raise hers, lower the
  other's. A part of the image only one character has is the same at any share. The box or
  dot shows the share when it isn't 50.
- Positions are fractions of the image, so they survive any resolution, hires fix, and Set
  Queue's random resolutions. They guide where each character is; they don't cut it out.
- Custom positions are also put into words ("a boy on the left, a girl on the right"), which
  helps small figures keep their order.

### Who does what

NovelAI's interaction tags work: `source#hug` (the one doing it), `target#hug` (the one it's
done to), `mutual#hug` (both). They become a sentence like "the girl on the left is doing hug
to the girl on the right".

They steer; they don't guarantee. Shared actions (`mutual#`) are reliable. In one-sided
poses, two habits of the model often decide who does what, whatever the tags say:

- **Outfits cast roles.** The softer-looking outfit tends to get the passive role. With the doer
  in a sundress and the other girl in a hoodie and shorts, princess carry, lap pillow and
  headpat came out reversed on nearly every seed, in either card order.
- **Some poses have a side.** Braiding and hug from behind put the doer on the right;
  piggyback puts the carrier on the left.

When it's backwards:

- **re-roll** the seed;
- **swap the two cards** (↑ / ↓) if it's backwards on every seed, which fixes the poses with a
  side;
- if neither helps, it's the outfit habit: keep re-rolling, or fix it afterwards.

### Works with

- **Set Queue and Dynamic Prompts wildcards** inside the cards, and prompt editing
  (`[a:b:10]`), switching on the same steps as the main prompt.
- **LoRA tags** in a card apply to the whole image, like everywhere else in Forge.
- **Tag Autocomplete** (sd-webui-tagcomplete) suggests tags in each card's Prompt and Undesired
  Content boxes too. Its txt2img, img2img and negative prompt settings apply to them as to the
  main boxes.
- **Hires fix**, **img2img**, and inpainting. With "Only masked", positions stay where they are
  in the whole image: inpainting one character's face keeps it that character's.
- **ADetailer gives each face its own character.** Wherever ADetailer's prompt is `[PROMPT]`
  (or empty), each face gets the main prompt plus its character's prompt, and that
  character's Undesired Content. Faces are matched to characters by position; a card's
  **Face** dropdown (`1st from left`, …) overrides that. A character drawn more than once
  (a multi-angle sheet) gets every one of her faces. Hands and eyes (any ADetailer model with
  "hand" or "eye" in its name, e.g. `mediapipe_face_mesh_eyes_only`) go to whichever
  character's area they're in, however many each one shows. The console prints which face went
  to whom. Not with the Merge mask modes.
- **Batch ADetailer and Batch Hires-Fix** (the batch-adetailer extension) re-run images from
  their PNG info, characters and positions included.
- **XYZ Plot's Prompt S/R** only searches the main prompt.

### PNG info

The characters are lines in the prompt section, after the main prompt (their Undesired
Content after the main negative prompt):

```
masterpiece, 2girls, office

Character 1 (Ava) at 0.000 0.000 0.480 1.000: girl, black hair, pink eyes, ...
Character 2 (Mei) at C3, share 70%: girl, red hair, red eyes, ...
Character 3 (Ami) at B3 + D3: girl, black hair, glasses, ...
Negative prompt: worst quality, ...
```

- The name appears when the card has one; the position when AI's Choice was off (a box, or a
  grid cell, several joined by ` + `); the overlap share when it isn't 50.
- Pasting an image fills the cards back in, switches them on and sets the positions. So does
  pasting that prompt text into the prompt box. Send to txt2img / img2img carries the cards
  as they are.
- A prompt with `Character N:` lines in it *is* a character prompt, so API callers can send
  one. Tools that don't know Stagehand see the lines as ordinary prompt text, so a one-prompt
  tool still gets every character's description.

### Limits

- **Anima only.** On other checkpoints the characters are ignored, with a console message.
- **Speed:** about +25% generation time with two characters.
- **Character count:** three stay separate and in order; past ~4 it gets less reliable as each
  area shrinks.
- Regenerating a pasted image loses Set Queue's `Template:` field from the new image's PNG
  info (the image itself is right).

---

## Precise Reference

**+ Add reference** adds a card (up to 4). Drop an image in, then:

- **Type**: NovelAI's three. It only sets the starting Strength: 1.0 for Character and
  Character & Style, 0.5 for Style.
- **Strength**: how much of the reference goes in. 0 ignores the card; negative pushes away
  from it.
- **Fidelity**: how hard the reference is to override with the prompt.

Tips:

- Use it for a solo character or the whole image's look. **Several cards blend into one**:
  the same character from a few angles works well; two different characters become one.
- Cards add up, so lower each card's Strength when using three or four.
- A reference on a plain background transfers the character; on a busy one it also copies the
  background and pose. If that happens, lower Fidelity (~0.6).
- A style reference works best at about 0.5; at 1.0 it copies the whole artwork.

Each reference is kept as a copy in `outputs\stagehand references\`, and its path goes into
the PNG info, so pasting the image (or a batch tab re-running it) brings the reference back.

**also in ADetailer** (off by default) adds the reference to ADetailer's face pass for that
generation, which keeps the face closer to it.

It needs the base Anima (28 blocks); other sizes are skipped with a console message.

---

## API

Everything works through Forge's own API (start Forge with `--api`), on `/sdapi/v1/txt2img`
and `/sdapi/v1/img2img`. [`example_api.py`](example_api.py) is a complete, runnable example:
two characters placed by hand, one of them with a reference image.

### Characters: lines in the prompt (simplest)

Send the characters the way an image's PNG info has them: lines after the main prompt.

```
masterpiece, 2girls, lap pillow, on couch

Character 1 (Rin) at 0.000 0.450 0.650 1.000, share 80%: girl, long red hair, lying, head on lap
Character 2 (Aoi) at 0.350 0.000 1.000 1.000, share 20%: girl, short blue hair, sitting
```

- `Character N` numbers 1–6. `(Name)` is optional; it only labels the character.
- `at …` places the character, as fractions of the image (`x0 y0 x1 y1`) or a grid cell
  (`C3`). Several places for one character: `at B3 + D3`. Leave `at` out on every line for
  AI's Choice (columns, in number order).
- `, share N%` is the overlap share (50 if left out).
- A character's text may go on over several lines; the next `Character` line ends it.
- Undesired Content: the same lines after the main negative prompt, in `negative_prompt`
  (`Character 2: glasses`).
- Interaction tags, wildcards and LoRA tags work as in the UI.

### Characters: script args

Or send the cards as Character Prompts' args, as the UI does:

```jsonc
"alwayson_scripts": {"Character Prompts": {"args": [
    false,                                         // AI's Choice
    true, "Rin", "girl, red hair", "", "0 0 0.6 1",  // card 1: on, name, prompt, Undesired Content, position
    true, "", "girl, blue hair", "glasses", "0.4 0 1 1",
    true, "", "", "", "",  true, "", "", "", "",   // cards 3-6 (empty)
    true, "", "", "", "",  true, "", "", "", "",
    "Face: auto", "Face: auto", "Face: auto", "Face: auto", "Face: auto", "Face: auto",
    true, "Boxes",                                 // Character Prompts on, placement: "Boxes" or "Grid"
    70, 30, 50, 50, 50, 50                         // overlap shares (may be left out: 50 each)
]}}
```

A prompt that already has `Character N:` lines uses those and ignores the cards.

### Precise Reference

```jsonc
"alwayson_scripts": {"Precise Reference": {"args": [
    "<base64 PNG>", "Character", 1.0, 1.0,         // card 1: image, type, strength, fidelity
    "", "Character", 1.0, 1.0,                     // cards 2-4: "" = no image
    "", "Character", 1.0, 1.0,
    "", "Character", 1.0, 1.0,
    false, true                                    // also in ADetailer, on
]}}
```

The image is base64 (a `data:` URL works too) or a file path on the machine Forge runs on.
Type is `Character`, `Style` or `Character & Style`; strength −1 to 2; fidelity 0 to 1. The
image's PNG info reports what was applied (`PR 1 strength: …`).

### Per request

`override_settings` takes Stagehand's settings: `stagehand_cp_one_pass` (true/false) and
`stagehand_cp_region_strength` (0–1). ADetailer's per-character faces work over the API too,
whenever its prompt is `[PROMPT]` or empty.

## Tests

With Forge's venv python, from the extension folder:

```
python test_core.py            # the logic, no Forge needed
python test_ui.py [--batch]    # the real UI in headless Chrome; Forge must be running
python test_api.py             # the API: Forge must be running with --api
```

`test_ui.py` generates three small images: characters from the cards, pasting them back, and
a reference from a card. It also saves, re-adds and deletes a character preset (and never
leaves its test preset behind). `--batch` also runs them through Batch Hires-Fix, then Batch
ADetailer. `test_api.py` generates five small images (not saved) and checks that characters
sent as prompt lines and as script args, and a base64 reference, come back in the PNG info.

## License

[AGPL-3.0](LICENSE), like Forge.
