"""End-to-end check through Forge's real txt2img UI, in headless Chrome.

API tests can't see the UI's wiring, and that's where the worst bugs were: a reference
dropped into a card never reached the script, and pasted PNG info never refilled the cards.

    python test_ui.py [--batch] [--url http://127.0.0.1:7860]

Needs Forge running with this extension on an Anima checkpoint, and Chrome. Run it with
Forge's venv python (it has `websockets`). It generates three small images, saved to your
outputs like any others. --batch also runs Batch ADetailer, then Batch Hires-Fix, on a copy
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
        src = await self.js("(() => { const i = gradioApp().querySelector('#txt2img_gallery img'); return i ? i.src : null; })()")
        assert src, "no image in the gallery"
        with open(out, "wb") as f:
            f.write(urllib.request.urlopen(src).read())

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
    # Tag Autocomplete (when installed) attaches to boxes that exist at page load; the cards'
    # boxes appear later and are handed to it by stagehand.js
    if await page.js("typeof addAutocompleteToArea === 'function'"):
        attached = "['prompt', 'uc'].every(k => gradioApp().querySelector(`#nai_t2i_char1_${k} textarea`)?.classList.contains('autocomplete'))"
        assert await page.wait(attached, 15), "Tag Autocomplete isn't attached to the character boxes"
        print("ok  Tag Autocomplete works in the character boxes")
    out = os.path.join(tmp, "ui_check.png")
    await page.gallery_image(out)
    assert f"Character 1: {CHAR1}" in Image.open(out).info.get("parameters", ""), "the saved image has no character lines"
    print("ok  characters from the cards reach the image and its PNG info")
    return info, out


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


PRESET_NAME = "stagehand test preset"
PRESET_TEXT = "girl, red hair,\nblack suit, pencil skirt,\n\nsmug"  # line breaks must survive
PRESETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "stagehand character presets.json")


def saved_presets():
    try:
        with open(PRESETS, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


async def pick_preset(page, name):
    await page.js("gradioApp().querySelector('#nai_t2i_chars_preset input').dispatchEvent(new Event('focus'))")
    await page.js("gradioApp().querySelector('#nai_t2i_chars_preset input').click()")
    found = f"Array.from(gradioApp().querySelectorAll('#nai_t2i_chars_preset li')).find(li => li.textContent.replace('✓', '').trim() === {json.dumps(name)})"
    assert await page.wait(f"!!{found}", 10), f"{name!r} isn't in the preset list"
    await page.js(f"{found}.dispatchEvent(new MouseEvent('mousedown', {{bubbles: true}}))")
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

    await pick_preset(page, PRESET_NAME)
    await page.js("gradioApp().querySelector('#nai_t2i_chars .nai-add').click()")
    assert await page.wait(f"(gradioApp().querySelector('#nai_t2i_char2_prompt textarea') || {{}}).value === {json.dumps(PRESET_TEXT)}", 15), \
        "+ Add character with a preset picked didn't fill the new card (line breaks included)"

    await pick_preset(page, PRESET_NAME)
    await page.js("window.confirm = () => true")
    await page.js("gradioApp().querySelector('#nai_t2i_chars .nai-delete-preset').click()")
    for _ in range(20):
        await asyncio.sleep(0.5)
        if PRESET_NAME not in saved_presets():
            break
    assert PRESET_NAME not in saved_presets(), "🗑 didn't delete the preset"
    print("ok  presets: 💾 saves a card with its line breaks, + adds it back, 🗑 deletes it")


def drop_test_preset():
    presets = saved_presets()
    if presets.pop(PRESET_NAME, None) is not None:
        with open(PRESETS, "w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)


BATCH_TABS = {
    "adetailer": ("Batch ADetailer", "load every base image in one folder", "-adetailer.png into each", "Run Batch ADetailer"),
    "hires": ("Batch Hires-Fix", "load every -adetailer image in one folder", "-hires.png into each", "Run Batch Hires-Fix"),
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
                await run_batch(page, "adetailer", folder, "ui_check-adetailer.png")
                await run_batch(page, "hires", folder, "ui_check-adetailer-hires.png")
    finally:
        proc.kill()
        time.sleep(1)
        shutil.rmtree(tmp, ignore_errors=True)
    print("all good")


if __name__ == "__main__":
    args = sys.argv[1:]
    url = args[args.index("--url") + 1] if "--url" in args else "http://127.0.0.1:7860"
    asyncio.run(main(url.rstrip("/"), "--batch" in args))
