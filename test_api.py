"""Stagehand through Forge's API: what API callers send must come out in the image.

    python test_api.py [--url http://127.0.0.1:7860] [--reference image.png]

Needs Forge running with --api on an Anima checkpoint. Generates five small images (not
saved). Checks the PNG info each one comes back with, which is how Stagehand reports what it
applied: characters as lines in the prompt, references as "PR n" parameters.
"""

import base64
import glob
import json
import os
import sys
import urllib.request

args = sys.argv[1:]
URL = (args[args.index("--url") + 1] if "--url" in args else "http://127.0.0.1:7860").rstrip("/")
MAX_CHARS, MAX_REFS = 6, 4


def txt2img(prompt, negative="worst quality", scripts=None):
    body = {"prompt": prompt, "negative_prompt": negative, "width": 512, "height": 512, "steps": 8,
            "cfg_scale": 4.5, "seed": 1234, "save_images": False, "send_images": True}
    if scripts:
        body["alwayson_scripts"] = scripts
    request = urllib.request.Request(f"{URL}/sdapi/v1/txt2img", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as r:
        return json.loads(json.load(r)["info"])["infotexts"][0]


def lines(info):
    return [line for line in info.split("\nNegative prompt:")[0].split("\n") if line.startswith("Character ")]


def card_args(cards, auto=False, manual="Boxes", shares=None):
    """Character Prompts' args: [AI's Choice] + 6 x [on, name, prompt, uc, position] + 6 ADetailer
    face picks + [on/off, placement] + 6 overlap shares (added fields go last)."""
    flat = [auto]
    for i in range(MAX_CHARS):
        flat += list(cards[i]) if i < len(cards) else [True, "", "", "", ""]
    flat += ["Face: auto"] * MAX_CHARS + [True, manual]
    return flat + (list(shares) + [50] * (MAX_CHARS - len(shares)) if shares is not None else [])


def check_prompt_lines():
    sent = ["Character 1 (Rin) at 0.000 0.000 0.600 1.000, share 70%: girl, red hair",
            "Character 2 at 0.400 0.000 1.000 1.000: girl, blue hair"]
    got = lines(txt2img("2girls, park\n\n" + "\n".join(sent), "worst quality\n\nCharacter 2: glasses"))
    assert got == sent, f"character lines in the prompt didn't come back as sent:\n{got}"
    # several places for one character, on the grid
    sheet = "Character 1 (Sil) at B3 + D3: girl, silver hair"
    got = lines(txt2img("1girl, multiple views\n\n" + sheet))
    assert got == [sheet], got
    print("ok  characters as prompt lines: positions, shares, several places, Undesired Content")


def check_script_args():
    cards = [(True, "Rin", "girl, red hair", "", "0.000 0.000 0.600 1.000"),
             (True, "", "girl, blue hair", "glasses", "0.400 0.000 1.000 1.000")]
    got = lines(txt2img("2girls, park", scripts={"Character Prompts": {"args": card_args(cards, shares=[70, 30])}}))
    assert got == ["Character 1 (Rin) at 0.000 0.000 0.600 1.000, share 70%: girl, red hair",
                   "Character 2 at 0.400 0.000 1.000 1.000, share 30%: girl, blue hair"], got
    # an arg list written before overlap shares existed still works (shares default to 50)
    got = lines(txt2img("2girls, park", scripts={"Character Prompts": {"args": card_args(cards)}}))
    assert got == ["Character 1 (Rin) at 0.000 0.000 0.600 1.000: girl, red hair",
                   "Character 2 at 0.400 0.000 1.000 1.000: girl, blue hair"], got
    print("ok  characters as alwayson_scripts args, with and without the overlap shares")


def check_reference(path):
    with open(path, "rb") as f:
        image = base64.b64encode(f.read()).decode()
    refs = [image, "Character", 0.8, 0.6] + ["", "Character", 1.0, 1.0] * (MAX_REFS - 1) + [False, True]
    info = txt2img("1girl, upper body", scripts={"Precise Reference": {"args": refs}})
    assert "PR 1 strength: 0.8" in info and "PR 1 fidelity: 0.6" in info, f"the reference wasn't applied:\n{info}"
    print("ok  a base64 reference through alwayson_scripts is applied")


def reference_image():
    if "--reference" in args:
        return args[args.index("--reference") + 1]
    here = os.path.dirname(os.path.abspath(__file__))
    kept = sorted(glob.glob(os.path.join(here, "..", "..", "outputs", "stagehand references", "*.png")))
    return kept[0] if kept else None


check_prompt_lines()
check_script_args()
path = reference_image()
if path:
    check_reference(path)
else:
    print("--  no reference image (pass --reference): Precise Reference not checked")
print("all good")
