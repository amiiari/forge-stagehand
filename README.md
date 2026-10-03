# Stagehand

Multi-character prompting and image references for Forge Neo on **Anima**: who's in the
picture, where they stand, what they look like. Modeled on NovelAI's versions of these
features, and laid out like them:

- **Character Prompts**: a prompt box per character, placed automatically, with boxes you drag
  over the output, or on NovelAI's 5×5 grid
  ([NovelAI's multi-character prompting](https://docs.novelai.net/en/image/multiplecharacters/)).
- **Precise Reference**: reference image cards with a type, Strength and Fidelity
  ([NovelAI's Precise Reference](https://docs.novelai.net/en/image/precisereference/)).

Both sit under the prompt boxes in txt2img and img2img, in one **Stagehand** fold-out. Its
header has an on/off pill per feature that works with the fold-out closed: off hides the
section and leaves the feature out of the next image, but keeps its cards. Each section ends
with a **How to use** fold-out.

Anima wasn't trained for either feature the way NovelAI's model was, so both are done at
generation time. [NOTES.md](NOTES.md) has how, and what the testing found.

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
- ↑ / ↓ to reorder, ⧉ to duplicate, 🗑 to delete;
- **Prompt** and **Undesired Content** tabs.

Write the scene, the style and the count tags (`2girls`, `1boy, 1girl`) in the main prompt,
like on NovelAI. Each card describes one character, starting with `girl` or `boy`. A card can
span several lines (appearance, outfit, proportions...); the line breaks are kept everywhere,
PNG info included.

### Presets

Name a card, then **💾** saves it as a preset: its prompt and Undesired Content exactly as
typed, line breaks included. Saving a card with the same name again updates the preset. Pick a
preset in the list next to **+ Add character**, then click it to add a card filled with that
character; **🗑** next to the list deletes the selected preset (it asks first). Presets live in
`stagehand character presets.json` in Forge's folder, so updating or reinstalling the
extension keeps them.

### Positions

- **AI's Choice** (default): equal columns, left to right in card order.
- **Turn it off** to place the characters yourself, in one of two ways (the switch next to
  AI's Choice; switching keeps each character where it was):
  - **Boxes**: drag a box by its name tab to move it, by its corner dot to resize. Boxes may
    overlap; the overlap is shared.
  - **Grid**: NovelAI's 5×5 grid. Each character is a dot that snaps to a cell. Every part of
    the image belongs to the nearest dot, so stacked or diagonal dots give layouts columns
    can't (bunk beds, someone in the foreground).
    - **Put each dot where that character's head will be.** Looks are decided around the head.
    - Give each character its own cell; two on one cell merge.
    - A lone character's dot only changes the position words ("a girl on the left"). Use
      Boxes to confine one.
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
- **Hires fix**, **img2img**, and inpainting. With "Only masked", positions stay where they are
  in the whole image: inpainting one character's face keeps it that character's.
- **ADetailer gives each face its own character.** Wherever ADetailer's prompt is `[PROMPT]`
  (or empty), each face gets the main prompt plus its character's prompt, and that
  character's Undesired Content. Faces are matched to characters by position; a card's
  **Face** dropdown (`1st from left`, …) overrides that. Hands go to whichever character's area
  they're in. The console prints which face went to whom. Not with the Merge mask modes.
- **Batch ADetailer and Batch Hires-Fix** (the batch-adetailer extension) re-run images from
  their PNG info, characters and positions included.
- **XYZ Plot's Prompt S/R** only searches the main prompt.

### PNG info

The characters are lines in the prompt section, after the main prompt (their Undesired
Content after the main negative prompt):

```
masterpiece, 2girls, office

Character 1 (Ava) at 0.000 0.000 0.480 1.000: girl, black hair, pink eyes, ...
Character 2 (Mei) at C3: girl, red hair, red eyes, ...
Negative prompt: worst quality, ...
```

- The name appears when the card has one; the position when AI's Choice was off (a box, or a
  grid cell).
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

## Tests

With Forge's venv python, from the extension folder:

```
python test_core.py            # the logic, no Forge needed
python test_ui.py [--batch]    # the real UI in headless Chrome; Forge must be running
```

`test_ui.py` generates three small images: characters from the cards, pasting them back, and
a reference from a card. It also saves, re-adds and deletes a character preset (and never
leaves its test preset behind). `--batch` also runs them through Batch ADetailer and Batch
Hires-Fix.

## License

[AGPL-3.0](LICENSE), like Forge.
