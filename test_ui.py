"""End-to-end check through Forge's real txt2img UI, in headless Chrome.

API tests can't see the UI's wiring, and that's where the worst bugs were: a reference
dropped into a card never reached the script, and pasted PNG info never refilled the cards.

    python test_ui.py [--batch] [--url http://127.0.0.1:7860]

Needs Forge running with this extension on an Anima checkpoint, and Chrome. Run it with
Forge's venv python (it has `websockets`). It generates three small images, saved to your
outputs like any others. --batch also runs Batch Hires-Fix, then Batch ADetailer (batch-adetailer's order), on a copy
in a temp folder (needs the batch-adetailer extension).
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets
from PIL import Image

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9336
CHAR1, CHAR2 = "girl, long red hair, black suit", "girl, short white hair, blue dress"


class Page:
    def __init__(self, ws, url):
        self.ws, self.url, self.n = ws, url, 0

    async def call(self, method, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.n:
                return msg.get("result", msg)

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    async def wait(self, expr, seconds=60):
        for _ in range(seconds * 2):
            if await self.js(expr):
                return True
            await asyncio.sleep(0.5)
        return False

    async def load(self):
        await self.call("Page.navigate", url=self.url + "/?__theme=dark")
        assert await self.wait("!!window.gradioApp && !!gradioApp().querySelector('#txt2img_prompt_container #nai_t2i_stagehand')", 120), \
            "the Stagehand fold-out never appeared under the prompt"
        await asyncio.sleep(2)

    async def type(self, selector, text):
        ok = await self.js(f"(() => {{ const t = gradioApp().querySelector({json.dumps(selector)}); if (!t) return false; "
                           f"t.value = {json.dumps(text)}; updateInput(t); return true; }})()")
        assert ok, f"no {selector}"

    async def generate(self):
        info = "(gradioApp().querySelector('#html_info_txt2img') || {}).innerText || ''"
        before = await self.js(info)
        await self.js("gradioApp().querySelector('#txt2img_generate').click()")
        for _ in range(900):
            await asyncio.sleep(1)
            text = await self.js(info)
            if text and text != before and "Steps:" in text:
                return text
        raise AssertionError("generation never finished")

    async def gallery_image(self, out):
        # the finished image, not whatever the gallery showed a moment earlier (a live preview
        # has no PNG info): the first one served with its parameters
        for _ in range(20):
            src = await self.js("(() => { const i = gradioApp().querySelector('#txt2img_gallery img'); return i ? i.src : null; })()")
            if src:
                with open(out, "wb") as f:
                    f.write(urllib.request.urlopen(src).read())
                if "Steps:" in Image.open(out).info.get("parameters", ""):
                    return
            await asyncio.sleep(0.5)
        assert src, "no image in the gallery"

    async def upload(self, selector, path):
        doc = await self.call("DOM.getDocument", depth=-1, pierce=True)
        node = await self.call("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector=selector)
        await self.call("DOM.setFileInputFiles", nodeId=node["nodeId"], files=[os.path.abspath(path)])


async def setup(page):
    await page.load()
    # small and quick: 12 steps at 768x768
    for key, value in (("steps", 12), ("width", 768), ("height", 768)):
        await page.type(f"#txt2img_{key} input[type=number]", str(value))
    await page.js("(() => { gradioApp().querySelector('#nai_t2i_stagehand').open = true; "
                  "for (const id of ['#nai_t2i_chars_on input', '#nai_t2i_pr_on input']) { const c = gradioApp().querySelector(id); if (c && !c.checked) c.click(); } })()")
    await asyncio.sleep(1)


async def check_characters(page, tmp):
    await setup(page)
    for k in (1, 2):
        await page.js("gradioApp().querySelector('#nai_t2i_chars .nai-add').click()")
        assert await page.wait(f"!!gradioApp().querySelector('#nai_t2i_char{k}_prompt textarea')", 10), f"+ didn't add card {k}"
        # a textbox typed into the moment it appears isn't wired up yet: its text would show
        # but never reach the server (a person can't type that fast)
        await asyncio.sleep(1.5)
    await page.type("#txt2img_prompt textarea", "2girls, park")
    await page.type("#nai_t2i_char1_prompt textarea", CHAR1)
    await page.type("#nai_t2i_char2_prompt textarea", CHAR2)
    info = await page.generate()
    assert f"Character 1: {CHAR1}" in info and f"Character 2: {CHAR2}" in info, f"characters missing from the PNG info:\n{info}"
    # Tag Autocomplete (when installed) only finds the main boxes; stagehand.js hands it the
    # cards', which follow its txt2img and negative prompt settings like the main boxes
    if await page.js("typeof addAutocompleteToArea === 'function' && !!window.TAC_CFG?.activeIn?.txt2img"):
        kinds = ["prompt", "uc"] if await page.js("TAC_CFG.activeIn.negativePrompts") else ["prompt"]
        attached = f"{json.dumps(kinds)}.every(k => gradioApp().querySelector(`#nai_t2i_char1_${{k}} textarea`)?.classList.contains('autocomplete'))"
        assert await page.wait(attached, 15), "Tag Autocomplete isn't attached to the character boxes"
        # what its settings and use counts go by: "n" means negative
        ids = await page.js("['prompt', 'uc'].map(k => getTextAreaIdentifier(gradioApp().querySelector(`#nai_t2i_char1_${k} textarea`)))")
        assert "txt2img" in ids[0] and "n" not in ids[0] and "n" in ids[1], f"Tag Autocomplete misreads the character boxes: {ids}"
        print("ok  Tag Autocomplete works in the character boxes")
    out = os.path.join(tmp, "ui_check.png")
    await page.gallery_image(out)
    assert f"Character 1: {CHAR1}" in Image.open(out).info.get("parameters", ""), "the saved image has no character lines"
    print("ok  characters from the cards reach the image and its PNG info")
    return info, out


async def drag(page, selector, dx, dy):
    """A real mouse drag (pointer events) from the middle of the element, by dx, dy pixels."""
    box = await page.js(f"(() => {{ const e = gradioApp().querySelector({json.dumps(selector)}); if (!e) return null; "
                        f"e.scrollIntoView({{block: 'center'}}); const r = e.getBoundingClientRect(); "
                        f"return [r.left + r.width / 2, r.top + r.height / 2]; }})()")
    assert box, f"nothing to drag at {selector}"
    x, y = box
    await page.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
    await page.call("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", clickCount=1)
    for k in range(1, 6):
        await page.call("Input.dispatchMouseEvent", type="mouseMoved", x=x + dx * k / 5, y=y + dy * k / 5, button="left")
        await asyncio.sleep(0.05)
    await page.call("Input.dispatchMouseEvent", type="mouseReleased", x=x + dx, y=y + dy, button="left", clickCount=1)
    await asyncio.sleep(0.8)  # the undo log closes an action once the page has settled


async def key(page, letter, shift=False):
    await page.js("document.activeElement?.blur()")
    for kind in ("keyDown", "keyUp"):
        await page.call("Input.dispatchKeyEvent", type=kind, key=letter, code=f"Key{letter.upper()}",
                        windowsVirtualKeyCode=ord(letter.upper()), modifiers=2 | (8 if shift else 0))
    await asyncio.sleep(1.2)  # restored through the server


async def check_positions(page):
    """The two characters from check_characters: their canvases, dragging, Reset / Switch, undo."""
    box = lambda n: f"(gradioApp().querySelector('#nai_t2i_char{n}_box textarea, #nai_t2i_char{n}_box input') || {{}}).value"  # noqa: E731
    assert await page.wait("!!gradioApp().querySelector('#nai_t2i_char1 .nai-mini [data-key=box1_0]') && "
                           "!!gradioApp().querySelector('#nai_t2i_char2 .nai-mini [data-key=box2_0]')", 10), "no small canvas on the cards"
    assert not await page.js("!!gradioApp().querySelector('#nai_t2i_char1 .nai-mini [data-key=box2_0]')"), \
        "a card's canvas shows another character"
    assert await page.js(box(1)) == "" and await page.js(box(2)) == "", "positions before anyone was placed"
    # 768x768: the canvas is 200x200, so 20px is 0.1 of the image
    await drag(page, "#nai_t2i_char1 .nai-mini [data-key=box1_0]", 20, 0)
    assert await page.js(box(1)) == "0.100 0.000 0.600 1.000", f"dragging on the canvas: {await page.js(box(1))!r}"
    assert await page.wait("(gradioApp().querySelector('#nai_t2i_positions [data-key=box1_0]') || {style: {}}).style.left === '10%'", 5), \
        "the output-image overlay didn't follow the canvas"
    # the top edge's grab strip, not just a corner
    await drag(page, "#nai_t2i_char1 .nai-mini [data-key=box1_0] [data-dir=n]", 0, 40)
    assert await page.js(box(1)) == "0.100 0.200 0.600 1.000", f"resizing by an edge: {await page.js(box(1))!r}"
    await key(page, "z")
    assert await page.js(box(1)) == "0.100 0.000 0.600 1.000", f"Ctrl+Z: {await page.js(box(1))!r}"
    await key(page, "z")
    assert await page.js(box(1)) == "", f"Ctrl+Z twice: {await page.js(box(1))!r}"
    await key(page, "y")
    assert await page.js(box(1)) == "0.100 0.000 0.600 1.000", f"Ctrl+Y: {await page.js(box(1))!r}"
    print("ok  dragging and resizing on a card's canvas moves the output overlay too; Ctrl+Z / Ctrl+Y")

    await page.js("(() => { const t = gradioApp().querySelector('#nai_t2i_chars_tools'); t.querySelector('.nai-switch-a').value = '1'; "
                  "t.querySelector('.nai-switch-b').value = '2'; t.querySelector('.nai-switch-go').click(); })()")
    await asyncio.sleep(0.8)
    assert (await page.js(box(1)), await page.js(box(2))) == ("0.500 0.000 1.000 1.000", "0.100 0.000 0.600 1.000"), \
        f"Switch: {await page.js(box(1))!r}, {await page.js(box(2))!r}"
    await page.js("gradioApp().querySelector('#nai_t2i_chars_tools .nai-reset').click()")
    await asyncio.sleep(0.8)
    assert (await page.js(box(1)), await page.js(box(2))) == ("", ""), "Reset left a position"
    # another place: half size, beside the first, drawn as one shape with the first
    await page.js("gradioApp().querySelector('#nai_t2i_char1 .nai-add-place').click()")
    await asyncio.sleep(0.8)
    assert await page.js(box(1)) == "0.000 0.000 0.500 1.000 + 0.500 0.250 0.750 0.750", f"＋: {await page.js(box(1))!r}"
    assert await page.js("gradioApp().querySelector('#nai_t2i_char1 .nai-mini [data-key=box1_0]').classList.contains('nai-labelled') && "
                         "!gradioApp().querySelector('#nai_t2i_char1 .nai-mini [data-key=box1_1]').classList.contains('nai-labelled')"), \
        "the name isn't on the biggest piece only"
    line = await page.js("gradioApp().querySelector('#nai_t2i_char1 .nai-mini .nai-union-line').getAttribute('d')")
    assert "M50.00 25.00V75.00" not in line and "M50.00 0.00V25.00" in line, f"not one outline: {line}"
    print("ok  Switch, Reset, and ＋ (a half-size place, one outline with the first)")

    # a card button is undoable too
    await page.js("gradioApp().querySelector('#nai_t2i_chars .nai-add').click()")
    assert await page.wait("getComputedStyle(gradioApp().querySelector('#nai_t2i_char3')).display !== 'none'", 10), "+ didn't add card 3"
    await asyncio.sleep(1)
    await key(page, "z")
    assert await page.wait("getComputedStyle(gradioApp().querySelector('#nai_t2i_char3')).display === 'none'", 10), "Ctrl+Z didn't take the card back"
    assert await page.js(box(1)) == "0.000 0.000 0.500 1.000 + 0.500 0.250 0.750 0.750", "undoing the card also undid a position"
    # click the extra place, Delete removes it; Ctrl+Z brings it back
    await drag(page, "#nai_t2i_char1 .nai-mini [data-key=box1_1]", 0, 0)
    for kind in ("keyDown", "keyUp"):
        await page.call("Input.dispatchKeyEvent", type=kind, key="Delete", code="Delete", windowsVirtualKeyCode=46)
    await asyncio.sleep(0.8)
    assert await page.js(box(1)) == "0.000 0.000 0.500 1.000", f"Delete: {await page.js(box(1))!r}"
    await key(page, "z")
    assert await page.js(box(1)) == "0.000 0.000 0.500 1.000 + 0.500 0.250 0.750 0.750", "Ctrl+Z didn't bring the place back"
    # the on-image toggle hides the overlay, the canvases stay
    await page.js("gradioApp().querySelector('#nai_t2i_chars_tools .nai-toggle[data-what=image]').click()")
    assert await page.wait("gradioApp().querySelector('#nai_t2i_positions').style.display === 'none'", 5), "the on-image toggle didn't hide the boxes"
    assert await page.js("!!gradioApp().querySelector('#nai_t2i_char1 .nai-mini')")
    await page.js("gradioApp().querySelector('#nai_t2i_chars_tools .nai-toggle[data-what=image]').click()")
    # leave the user's page as it was: one place, nobody placed
    await page.js("gradioApp().querySelector('#nai_t2i_chars_tools .nai-reset').click()")
    await asyncio.sleep(0.8)
    print("ok  undoing a card button; Delete on a picked place (and its undo); the on-image toggle")


async def check_paste(page, info):
    await setup(page)
    await page.type("#txt2img_prompt textarea", info)
    await page.js("gradioApp().querySelector('#paste').click()")
    assert await page.wait(f"(gradioApp().querySelector('#nai_t2i_char2_prompt textarea') || {{}}).value === {json.dumps(CHAR2)}", 20), \
        "pasted PNG info didn't refill the cards"
    prompt = await page.js("gradioApp().querySelector('#txt2img_prompt textarea').value")
    assert "Character 1" not in prompt, f"the character lines stayed in the prompt: {prompt!r}"
    print("ok  pasting PNG info splits the characters back into their cards")


async def check_reference(page, image):
    # the cards from check_paste are still filled: references and characters together
    if not await page.js("!!gradioApp().querySelector('#nai_t2i_pr')"):
        print("--  Precise Reference is switched off in Settings: reference check skipped")
        return
    await page.js("gradioApp().querySelector('#nai_t2i_pr .nai-add').click()")
    assert await page.wait("!!gradioApp().querySelector('#nai_t2i_pr1_image input[type=file]')", 10), "+ didn't add a reference card"
    await page.upload("#nai_t2i_pr1_image input[type=file]", image)
    assert await page.wait("!!gradioApp().querySelector('#nai_t2i_pr1_image img')", 20), "the reference never showed in its card"
    info = await page.generate()
    assert "PR 1 image:" in info, f"the reference wasn't applied (no 'PR 1 image' in the PNG info):\n{info}"
    print("ok  a reference dropped into a card is applied, and its file path is in the PNG info")

    # character 2's own (check_paste left two), from her card's Reference tab, in ADetailer
    await page.js("gradioApp().querySelector('#nai_t2i_char2 .nai-ref-add').click()")
    in_tab = "!!gradioApp().querySelector('#nai_t2i_char2 .nai-ref-slot #nai_t2i_pr2')"
    assert await page.wait(in_tab, 10), "+ Add reference in her Reference tab didn't put a card there"
    assert await page.js("gradioApp().querySelector('#nai_t2i_pr2_for textarea, #nai_t2i_pr2_for input').value") == "Character 2"
    await page.upload("#nai_t2i_pr2_image input[type=file]", image)
    assert await page.wait("!!gradioApp().querySelector('#nai_t2i_pr2_image img')", 20), "her reference never showed in its card"
    await page.js("gradioApp().querySelector('#nai_t2i_pr2_adetailer input').click()")
    await asyncio.sleep(1)
    info = await page.generate()
    assert "PR 2 for: Character 2" in info and "PR 2 ADetailer: True" in info and "PR 2 hires: False" in info, \
        f"her reference's For / ADetailer didn't reach the PNG info:\n{info}"
    assert "PR 1 for" not in info, "the whole-image reference picked up a character"
    print("ok  a character's reference from her Reference tab, ticked for ADetailer, is recorded as hers")

    # reordering the cards takes her reference along; deleting her card deletes it
    await page.js("gradioApp().querySelector('#nai_t2i_char2 .nai-up').click()")
    assert await page.wait("!!gradioApp().querySelector('#nai_t2i_char1 .nai-ref-slot #nai_t2i_pr2')", 10), "her reference didn't follow ↑"
    assert await page.js("gradioApp().querySelector('#nai_t2i_pr2_for textarea, #nai_t2i_pr2_for input').value") == "Character 1"
    await asyncio.sleep(1)
    await key(page, "z")
    assert await page.wait("!!gradioApp().querySelector('#nai_t2i_char2 .nai-ref-slot #nai_t2i_pr2')", 10), "Ctrl+Z after ↑ left her reference behind"
    await key(page, "y")
    assert await page.wait("!!gradioApp().querySelector('#nai_t2i_char1 .nai-ref-slot #nai_t2i_pr2')", 10), "Ctrl+Y after ↑ left her reference behind"
    # ⧉ (in ⋯) copies her reference to the copy
    await page.js("gradioApp().querySelector('#nai_t2i_char1 .nai-copy').click()")
    assert await page.wait("!!gradioApp().querySelector('#nai_t2i_char3 .nai-ref-slot #nai_t2i_pr3')", 10), "⧉ didn't copy her reference"
    assert await page.wait("!!gradioApp().querySelector('#nai_t2i_pr3_image img')", 10), "the copied reference has no image"
    await page.js("gradioApp().querySelector('#nai_t2i_char3 .nai-remove').click()")
    assert await page.wait("getComputedStyle(gradioApp().querySelector('#nai_t2i_pr3')).display === 'none'", 10)
    await page.js("gradioApp().querySelector('#nai_t2i_char1 .nai-remove').click()")
    assert await page.wait("getComputedStyle(gradioApp().querySelector('#nai_t2i_pr2')).display === 'none'", 10), \
        "deleting her card left her reference"
    assert await page.js("getComputedStyle(gradioApp().querySelector('#nai_t2i_pr1')).display !== 'none'"), "the whole-image one went too"
    print("ok  her reference follows ↑ / ↓ (and their undo), is copied by ⧉, and goes with her card")


PRESET_NAME = "stagehand test preset"
PRESET_TEXT = "girl, red hair,\nblack suit, pencil skirt,\n\nsmug"  # line breaks must survive
PRESETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "stagehand character presets.json")


def saved_presets():
    try:
        with open(PRESETS, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


async def pick_preset(page, n, name):
    """Picks a preset in card n's name box, as choosing it from the browser's list does."""
    name_box = f"gradioApp().querySelector('#nai_t2i_char{n} .nai-char-name input')"
    await page.js(f"{name_box}.focus()")
    listed = f"Array.from(gradioApp().querySelectorAll('#' + {name_box}.getAttribute('list') + ' option')).some(o => o.value === {json.dumps(name)})"
    assert await page.wait(listed, 10), f"{name!r} isn't in the name box's preset list"
    await page.js(f"(() => {{ const i = {name_box}; i.value = {json.dumps(name)}; "
                  "i.dispatchEvent(new InputEvent('input', {bubbles: true, inputType: 'insertReplacementText'})); })()")
    await asyncio.sleep(1)


async def check_presets(page):
    await setup(page)
    await page.js("gradioApp().querySelector('#nai_t2i_chars .nai-add').click()")
    assert await page.wait("!!gradioApp().querySelector('#nai_t2i_char1_prompt textarea')", 10)
    await asyncio.sleep(1.5)
    await page.type("#nai_t2i_char1 .nai-char-name input, #nai_t2i_char1 .nai-char-name textarea", PRESET_NAME)
    await page.type("#nai_t2i_char1_prompt textarea", PRESET_TEXT)
    await asyncio.sleep(1)
    await page.js("gradioApp().querySelector('#nai_t2i_char1 .nai-save-preset').click()")
    for _ in range(20):
        await asyncio.sleep(0.5)
        if PRESET_NAME in saved_presets():
            break
    assert saved_presets().get(PRESET_NAME, {}).get("prompt") == PRESET_TEXT, "💾 didn't save the card with its line breaks"

    await page.js("gradioApp().querySelector('#nai_t2i_chars .nai-add').click()")
    assert await page.wait("getComputedStyle(gradioApp().querySelector('#nai_t2i_char2')).display !== 'none'", 10)
    await asyncio.sleep(1.5)
    await page.type("#nai_t2i_char2_prompt textarea", "girl, to be replaced")
    await pick_preset(page, 2, PRESET_NAME)
    assert await page.wait(f"(gradioApp().querySelector('#nai_t2i_char2_prompt textarea') || {{}}).value === {json.dumps(PRESET_TEXT)}", 15), \
        "picking a preset in the name box didn't fill the card (line breaks included)"
    await key(page, "z")
    assert await page.wait("(gradioApp().querySelector('#nai_t2i_char2_prompt textarea') || {}).value === 'girl, to be replaced'", 10), \
        "Ctrl+Z didn't take the preset back"

    # the Stagehand Presets tab: opening it lists the new preset; rename it, then delete it
    await page.js("[...gradioApp().querySelectorAll('#tabs > .tab-nav button')].find(b => b.textContent.trim() === 'Stagehand Presets').click()")
    item = f"[...gradioApp().querySelectorAll('#stagehand_presets_list label')].find(l => l.textContent.trim() === {json.dumps(PRESET_NAME)})"
    assert await page.wait(f"!!{item}", 10), "the Presets tab doesn't list the new preset"
    await page.js(f"{item}.querySelector('input').click()")
    name_box = "#stagehand_presets input[placeholder^='what a card']"
    assert await page.wait(f"gradioApp().querySelector({json.dumps(name_box)}).value === {json.dumps(PRESET_NAME)}", 10), "picking it didn't fill the editor"
    renamed = PRESET_NAME + " renamed"
    await page.type(name_box, renamed)
    await page.js("[...gradioApp().querySelectorAll('#stagehand_presets button')].find(b => b.textContent.includes('Save')).click()")
    for _ in range(20):
        await asyncio.sleep(0.5)
        if renamed in saved_presets():
            break
    assert renamed in saved_presets() and PRESET_NAME not in saved_presets(), "Save with a new name didn't rename it"
    assert saved_presets()[renamed]["prompt"] == PRESET_TEXT
    await page.js("window.confirm = () => true")
    await page.js("[...gradioApp().querySelectorAll('#stagehand_presets button')].find(b => b.textContent.includes('Delete')).click()")
    for _ in range(20):
        await asyncio.sleep(0.5)
        if renamed not in saved_presets():
            break
    assert renamed not in saved_presets(), "🗑 Delete didn't delete it"
    await page.js("[...gradioApp().querySelectorAll('#tabs > .tab-nav button')].find(b => b.textContent.trim() === 'txt2img').click()")
    print("ok  presets: 💾 saves a card with its line breaks, the name box fills a card with it (Ctrl+Z undoes), "
          "the Presets tab renames and deletes it")


def drop_test_preset():
    presets = saved_presets()
    if [presets.pop(n, None) for n in (PRESET_NAME, PRESET_NAME + " renamed")] != [None, None]:
        with open(PRESETS, "w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)


# batch-adetailer's order: NrM.png -> Batch Hires-Fix -> NrM-hires.png -> Batch ADetailer -> NrM-hires-adetailer.png
BATCH_TABS = {
    "hires": ("Batch Hires-Fix", "load every base image in one folder", "-hires.png into each", "Run Batch Hires-Fix"),
    "adetailer": ("Batch ADetailer", "load every -hires image in one folder", "-adetailer.png into each", "Run Batch ADetailer"),
}


async def run_batch(page, kind, folder, expected):
    tab, folder_label, source_label, run_label = BATCH_TABS[kind]
    await page.load()
    find = "Array.from(gradioApp().querySelectorAll('label')).filter(l => l.offsetParent && l.textContent.includes(%s))"
    assert await page.js(f"(() => {{ const b = Array.from(gradioApp().querySelectorAll('.tab-nav button')).find(b => b.textContent.trim() === {json.dumps(tab)}); if (b) b.click(); return !!b; }})()"), f"no {tab} tab"
    await asyncio.sleep(2)
    assert await page.js(f"(() => {{ const l = {find % json.dumps(folder_label)}[0]; const t = l && (l.querySelector('textarea, input') || l.parentElement.querySelector('textarea, input')); if (!t) return false; t.value = {json.dumps(folder)}; updateInput(t); return true; }})()"), "no folder box"
    await asyncio.sleep(1)
    await page.js("Array.from(gradioApp().querySelectorAll('button')).filter(b => b.offsetParent && b.textContent.includes('Load Folder'))[0].click()")
    await asyncio.sleep(6)
    await page.js(f"(() => {{ const c = {find % json.dumps(source_label)}[0].querySelector('input[type=checkbox]'); if (!c.checked) c.click(); }})()")
    await asyncio.sleep(1)
    await page.js(f"Array.from(gradioApp().querySelectorAll('button')).filter(b => b.offsetParent && b.textContent.includes({json.dumps(run_label)}))[0].click()")
    path = os.path.join(folder, expected)
    for _ in range(900):
        await asyncio.sleep(1)
        if os.path.exists(path):
            break
    await asyncio.sleep(3)
    assert os.path.exists(path), f"{tab} never wrote {expected}"
    info = Image.open(path).info.get("parameters", "")
    assert f"Character 1: {CHAR1}" in info and f"Character 2: {CHAR2}" in info, f"{tab} lost the characters:\n{info}"
    print(f"ok  {tab} keeps the characters")


async def main(url, batch):
    tmp = tempfile.mkdtemp(prefix="stagehand_ui_")
    proc = subprocess.Popen(
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={PORT}",
         f"--user-data-dir={os.path.join(tmp, 'chrome')}", "--window-size=1700,1400", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                target = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        async with websockets.connect(target["webSocketDebuggerUrl"], max_size=80_000_000) as ws:
            page = Page(ws, url)
            await page.call("Page.enable")
            info, image = await check_characters(page, tmp)
            await check_positions(page)
            await check_paste(page, info)
            await check_reference(page, image)
            try:
                await check_presets(page)
            finally:
                drop_test_preset()  # never leave the test's preset in the user's list
            if batch:
                folder = os.path.join(tmp, "batch")
                os.makedirs(folder)
                shutil.copy(image, folder)
                await run_batch(page, "hires", folder, "ui_check-hires.png")
                await run_batch(page, "adetailer", folder, "ui_check-hires-adetailer.png")
    finally:
        proc.kill()
        time.sleep(1)
        shutil.rmtree(tmp, ignore_errors=True)
    print("all good")


if __name__ == "__main__":
    args = sys.argv[1:]
    url = args[args.index("--url") + 1] if "--url" in args else "http://127.0.0.1:7860"
    asyncio.run(main(url.rstrip("/"), "--batch" in args))
