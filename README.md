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
- this extension is definitely not perfect!!! just keep that in mind!

## thank you for visiting! and using even?! i hope this repo ends up being useful to you.

okay! anyways, i will let opus 5.5 take it from here with a more in depth explanation of the details and stuff 
i'd rather not type LMAO 

## Settings

**Settings → Stagehand → Precise Reference**: untick it (then Reload UI) to remove Precise
Reference completely: its section and pill under the prompt, its paste handling, its XYZ Plot
axes and its ADetailer hook. Character Prompts is unaffected.

**Character Prompts: read each card together with the main prompt**, **card strength**: how
cards are read (see NOTES.md, "The card look"). **LoRAs typed in a card**: Masked (the default)
or Separate pass (only her), or Whole image (every character gets them, Forge's usual; see
"Works with" below). All three can be set per API request with `override_settings`.

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
- a name, which is also the preset search (below);
- **Overlap %**: who wins where two of the character's places overlap (below);
- ↑ / ↓ to reorder, ⧉ to duplicate, 💾 to save it as a preset, **⋯** for the rest, 🗑 to delete;
- **⋯** opens the **Face** pick for ADetailer (leave it on auto; see below). A dot on ⋯ means
  the face isn't auto;
- **Prompt**, **Undesired Content** (a dot when it has text) and **Reference** (her own
  reference images, with their count; see Precise Reference) tabs;
- a small map of her place in the image, with **+** for another place.

Write the scene, the style and the count tags (`2girls`, `1boy, 1girl`) in the main prompt,
like on NovelAI; switching a card off doesn't change the count, so keep it right yourself.
Each card describes one character, starting with `girl` or `boy`: that word is also what
the position words use ("a girl on the left"; without it, "a character on the left"). A card
can span several lines (appearance, outfit, proportions...); the line breaks are kept
everywhere, PNG info included.

### Presets

Name a card, then **💾** saves it as a preset: its prompt and Undesired Content exactly as
typed, line breaks included. Saving a card with the same name again updates the preset.
**Type in a card's name box** to find a preset, and pick it from the list: the card's prompt and
Undesired Content are replaced, while its place, references and number stay, so swapping one
character for another keeps the composition. Ctrl+Z undoes it. Typing a name that happens to
match a preset doesn't fill anything; only picking it does. **Presets**, a collapsed panel
under the cards, lists every preset, with a search: pick one to edit its prompt and Undesired
Content, change its name and Save to rename it, + New to add one, or Delete it (it asks
first). Presets live in `stagehand character
presets.json` in Forge's folder, so updating or reinstalling the extension keeps them.
Original characters, named like `Ruby (OC)` or `Fran (artist, OC)`, are listed first.

Several people on one Forge folder, each with their own settings file (`--ui-settings-file`,
as forge link's slots run): each gets their own presets, in `stagehand character
presets.json` beside their settings file, on top of the shared ones in Forge's folder. The
shared ones can't be changed or deleted there; saving one (or renaming it) makes your own copy.

### Positions

- **By default** the characters stand in equal columns, left to right in card order.
- **Drag one to place it yourself**: on the small canvas beside its card (that character
  only) or on the output image (everyone); the two stay in sync. A character you never
  dragged stays in her default column. Two ways (the Boxes / Grid switch; switching keeps each
  character where it was):
  - **Boxes**: drag a box from anywhere on it, resize it by any edge or corner. Its edges snap
    to the other boxes' edges and the image's borders and middle (hold Alt to place it
    freely). Boxes may overlap; the overlap is shared, by each card's **Overlap %**.
  - **Grid**: NovelAI's 5×5 grid. Each character is a dot that snaps to a cell. Every part of
    the image belongs to the nearest dot, so stacked or diagonal dots give layouts columns
    can't (bunk beds, someone in the foreground).
    - **Put each dot where that character's head will be.** Looks are decided around the head.
    - Give each character its own cell; two on one cell merge.
    - A lone character's dot only changes the position words ("a girl on the left"). Use
      Boxes to confine one.
- **Reset boxes** puts everyone back in the default columns; **Switch** swaps two characters'
  places (with just two, one ⇄ button); **on cards** / **on image** show or hide the maps on
  the cards and the boxes over the output image (off until you turn them on; a click on them
  still opens the picture, only a drag moves them). **Ctrl+Z / Ctrl+Y** undo and redo moves,
  Reset boxes, Switch and the card buttons (add, delete, ↑ ↓, duplicate), whenever you're not
  typing in a text box.
- **More than one place for a character**: **+** on her card's map adds a half-size box (or a dot,
  in Grid) beside her last one; drag it like any other, **×** on it (or click it and press Delete) removes it. Touching or
  overlapping places are drawn as one shape, the name in the biggest piece and a dot in the
  others. Every place reads the same card. For multi-angle sheets (full body on the left,
  close-up on the right, one character) and compositions where one character spans two areas.
  Want different words per view? Duplicate the card (⧉) and edit it instead. A character with
  several places gets no position words ("a girl on the left").
- **Overlap %** (beside each card's name): who wins where two places overlap.
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
- **LoRA tags** in a card stay on their own character (**Settings → Stagehand → Character
  LoRAs**):
  - **Masked** (the default): the LoRA changes only her area. About 1.3-1.45x the time (measured with hires fix
    and ADetailer: 45 s -> 60 s for two girls, 60 s -> 87 s for three).
  - **Separate pass**: one extra model run per character with a LoRA, kept only over her area.
    Slower: each LoRA'd character adds a full run (about 2.2x for two girls, 2.9x for three).
  - **Whole image**: like everywhere else in Forge, every character gets every card's LoRA.

  In a blind test (8 images, 2 and 3 girls) Masked and Separate pass tied, both ahead of Whole
  image; Masked won more often, Separate pass was never last.

  Either way her face in ADetailer gets only her own LoRA, and a LoRA's text-encoder part
  (when Forge recognizes it) reads only her card's text. Images record the mode
  ("Char LoRAs"). In a batch, each image gets the LoRAs its own card text rolled (a wildcard
  can pick a different one per image).
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

- The name appears when the card has one; the position when someone was placed by hand (a
  box, or a grid cell, several joined by ` + `); the overlap share when it isn't 50.
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

Up to 4 references in all, in two places:

- **For the whole image**: **+ Add reference** in the References section. Whole-image
  references work without any character cards (one character described in the main prompt).
- **For one character**: **+ Add reference** in the **Reference** tab of her Character Prompts
  card. It goes only into her part of the image (where her card places her), so each
  character can have her own references. It follows her card: ↑ / ↓ take it along, ⧉ copies
  it (while there's room), 🗑 deletes it with her. One for a character who isn't in the image
  (a pasted image whose card is gone) waits in the References section, saying so, and is
  skipped (console line and a note under the image).

Drop an image in, then:

- **Strength**: how much of the reference goes in: about 1 for a character, about 0.5 for an
  art style. 0 ignores the card; negative pushes away from it.
- **Fidelity**: how hard the reference is to override with the prompt. 0.6 by default: in a
  blind test it ranked best (1.0, the old default, ranked last). Raise it for a closer copy.
- **Also use in: Hires fix / ADetailer (faces)**: keep the reference in that pass too. Off (the
  default), it only guides the first pass, the one that draws the picture, and Hires fix and
  ADetailer refine without it -- in a blind test that ranked best on every image. Tick
  ADetailer when a face drifts away from the reference. In ADetailer a character's card goes only to her own face
  (matched as for her prompt). Batch Hires-Fix and Batch ADetailer follow the same ticks.

Tips:

- **Whole-image cards blend into one**: the same character from a few angles works well; two
  different characters become one. For two characters, give each her own cards.
- Cards add up, so lower each card's Strength when using three or four.
- A reference on a plain background transfers the character; on a busy one it also copies the
  background and pose. If that happens, lower Fidelity (~0.4).
- A style reference works best at about 0.5; at 1.0 it copies the whole artwork.
- It carries looks, not poses: a pose reference (a meme with two people pointing) didn't move
  the characters into its pose.

Each reference is kept as a copy in `outputs\stagehand references\`, and its path goes into
the PNG info, so pasting the image (or a batch tab re-running it) brings the reference back.

A character card in ADetailer keeps her face closer to the reference (and is ~3 s/image
faster than a face pass without the adapter).

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
  the default columns (in number order); a line without it among placed ones gets its column.
- `, share N%` is the overlap share (50 if left out).
- A character's text may go on over several lines; the next `Character` line ends it.
- Undesired Content: the same lines after the main negative prompt, in `negative_prompt`
  (`Character 2: glasses`).
- Interaction tags, wildcards and LoRA tags work as in the UI.

### Characters: script args

Or send the cards as Character Prompts' args, as the UI does:

```jsonc
"alwayson_scripts": {"Character Prompts": {"args": [
    true,                                          // (old AI's Choice) true: positions count when given;
                                                   // false: placed boxes even for empty ones
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
    false, true,                                   // (old: ADetailer for every card), on
    "Character 1", true, true,                     // card 1: for, in Hires fix, in ADetailer
    "Whole image", false, false,                   // cards 2-4
    "Whole image", false, false,
    "Whole image", false, false
]}}
```

The image is base64 (a `data:` URL works too) or a file path on the machine Forge runs on.
Type is `Character`, `Style` or `Character & Style` (recorded only: the UI no longer shows it,
since it changed nothing but the starting strength); strength −1 to 2; fidelity 0 to 1. `for`
is `Whole image` or `Character N` (N as in the character lines). The image's PNG info reports
what was applied (`PR 1 strength: …`, `PR 1 for: Character 1`, `PR 1 hires: True`).

An arg list without the last twelve values (written before they existed) still works: Forge
fills them with the defaults (whole image, neither pass), and the old ADetailer flag, when
true, still puts every card in ADetailer.

### Per request

`override_settings` takes Stagehand's settings: `stagehand_cp_one_pass` (true/false),
`stagehand_cp_region_strength` (0–1) and `stagehand_cp_lora` (`"Whole image"`, `"Masked"` or
`"Separate pass"`). ADetailer's per-character faces work over the API too,
whenever its prompt is `[PROMPT]` or empty.

## Tests

With Forge's venv python, from the extension folder:

```
python test_core.py            # the logic, no Forge needed
python test_ui.py [--batch]    # the real UI in headless Chrome; Forge must be running
python test_api.py             # the API: Forge must be running with --api
```

`test_ui.py` generates three small images: characters from the cards, pasting them back, and
a reference from a card. It also saves, re-adds, renames and deletes a character preset (and never
leaves its test preset behind). `--batch` also runs them through Batch Hires-Fix, then Batch
ADetailer. `test_api.py` generates five small images (not saved) and checks that characters
sent as prompt lines and as script args, and a base64 reference, come back in the PNG info.

## License

[AGPL-3.0](LICENSE), like Forge.
