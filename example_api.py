"""Stagehand over Forge's API: two characters, placed by hand, one with a reference image.

    python example_api.py [--url http://127.0.0.1:7860] [--reference face.png] [--out out.png]

Needs Forge Neo started with --api, on an Anima checkpoint, with Stagehand installed.
Plain Python, no packages. See README.md, "API", for every option.
"""

import argparse
import base64
import json
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:7860")
parser.add_argument("--reference", help="an image of character 1 for Precise Reference (optional)")
parser.add_argument("--out", default="stagehand_api_example.png")
args = parser.parse_args()

# The characters ride in the prompt, as lines after the main prompt -- the same lines an
# image's PNG info has. Positions are fractions of the image (x0 y0 x1 y1) or grid cells
# ("B3"); leave "at ..." out for the default columns (left to right in number order). ", share N%"
# decides who wins where places overlap (50 each if left out); several places for one
# character are joined with " + ".
prompt = """masterpiece, best quality, 2girls, lap pillow, on couch, living room

Character 1 (Rin) at 0.000 0.450 0.650 1.000, share 80%: girl, long red hair, white sweater, lying, head on lap, closed eyes
Character 2 (Aoi) at 0.350 0.000 1.000 1.000, share 20%: girl, short blue hair, black hoodie, sitting, smile, looking down"""

# Undesired Content per character: the same lines, after the main negative prompt.
negative = """worst quality, low quality

Character 2: glasses"""

payload = {
    "prompt": prompt,
    "negative_prompt": negative,
    "width": 1024,
    "height": 1024,
    "steps": 28,
    "cfg_scale": 4.5,
    "seed": -1,
}

if args.reference:
    with open(args.reference, "rb") as f:
        image = base64.b64encode(f.read()).decode()
    # 4 reference cards x [image, type, strength, fidelity], then [(old) ADetailer, on], then
    # 4 x [for, in Hires fix, in ADetailer]. An empty card is "". Type: "Character", "Style"
    # or "Character & Style". For: "Whole image" or "Character N" -- here only Rin gets it.
    empty = ["", "Character", 1.0, 1.0]
    payload["alwayson_scripts"] = {
        "Precise Reference": {"args": [image, "Character", 1.0, 1.0] + empty * 3 + [False, True]
                              + ["Character 1", True, True] + ["Whole image", False, False] * 3},
    }

request = urllib.request.Request(f"{args.url}/sdapi/v1/txt2img", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=1800) as response:
    result = json.load(response)

with open(args.out, "wb") as f:
    f.write(base64.b64decode(result["images"][0]))
info = json.loads(result["info"])["infotexts"][0]
print(f"saved {args.out}")
print("\n".join(line for line in info.split("\n") if line.startswith("Character")))
print(", ".join(part.strip() for part in info.split(",") if part.strip().startswith("PR ")))
