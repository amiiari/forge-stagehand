// Stagehand in the browser:
// - both panels (Character Prompts, Precise Reference) go under the prompt boxes, inside one
//   "Stagehand" fold-out. Its header has a pill per feature that switches it on or off -- also
//   while the fold-out is closed -- and an off feature's section is hidden;
// - a prompt that carries characters -- Set Queue's "Restore template" (marked ⟦n⟧), or text
//   pasted from an image's PNG info ("Character 1 (Ava): ...") -- is split into the cards;
// - with "AI's Choice" off, the characters are placed by hand over the output image: Boxes
//   (drag a box to move it, its corner to resize; "x0 y0 x1 y1", fractions of the image) or
//   Grid (NovelAI V4.5's 5x5 grid: drag a dot to a cell; "C3"). The result lands in the
//   card's hidden position field.
(() => {
    const TABS = [["txt2img", "t2i"], ["img2img", "i2i"]];
    const MAX = 6;
    const GRID = 5;
    const FEATURES = [["chars", "Character Prompts"], ["pr", "Precise Reference"]];
    let dragging = false;

    const el = (id) => gradioApp().getElementById(id);
    const field = (id) => el(id)?.querySelector("textarea, input");
    const checkbox = (id) => el(id)?.querySelector("input[type=checkbox]");

    const setText = (node, text) => node && node.textContent !== text && (node.textContent = text);

    function setValue(input, value) {
        if (!input) return;
        input.value = value;
        updateInput(input);
    }

    const featureOn = (id, feature) => checkbox(`nai_${id}_${feature}_on`)?.checked !== false;

    function manualOf(id) {
        return el(`nai_${id}_chars_manual`)?.querySelector("input:checked")?.value || "Boxes";
    }

    function setManual(id, value) {
        const input = el(`nai_${id}_chars_manual`)?.querySelector(`input[value="${value}"]`);
        if (input && !input.checked) input.click();
    }

    // Forge's prompt comments (/* */, # and // to the end of the line), when they're enabled
    function stripComments(text) {
        if (typeof opts === "undefined" || !opts.enable_prompt_comments) return text;
        return text.replace(/\/\*[\s\S]*?\*\//g, "").replace(/[^\S\n]*(#|\/\/).*/g, "");
    }

    function characters(id) {
        const out = [];
        if (!featureOn(id, "chars")) return out;
        for (let n = 1; n <= MAX; n++) {
            const card = el(`nai_${id}_char${n}`);
            // the card's own display: a hidden section or a closed fold-out doesn't count
            if (!card || getComputedStyle(card).display === "none") continue;
            const on = card.querySelector(".nai-char-on input");
            const prompt = field(`nai_${id}_char${n}_prompt`);
            if ((on && !on.checked) || !prompt || !stripComments(prompt.value).trim()) continue;
            const name = card.querySelector(".nai-char-name input, .nai-char-name textarea")?.value.trim();
            out.push({n, label: name || `${n}`, input: field(`nai_${id}_char${n}_box`)});
        }
        return out;
    }

    function references(id) {
        if (!featureOn(id, "pr")) return 0;
        let count = 0;
        for (let n = 1; n <= 4; n++) if (el(`nai_${id}_pr${n}_image`)?.querySelector("img")) count++;
        return count;
    }

    // ------------------------------------------------------------------ the fold-out
    function wrap(tab, id) {
        const negRow = el(`${tab}_neg_prompt_row`);
        const panels = [el(`nai_${id}_chars`), el(`nai_${id}_pr`)].filter(Boolean);
        if (!negRow || !panels.length) return;
        const box = document.createElement("details");
        box.id = `nai_${id}_stagehand`;
        box.className = "nai-stagehand";
        const key = `stagehand-open-${id}`;
        try {
            box.open = localStorage.getItem(key) === "1";
        } catch (e) {}
        const summary = document.createElement("summary");
        summary.innerHTML = '<span class="nai-stagehand-title">Stagehand</span>';
        for (const [feature, label] of FEATURES) {
            if (!el(`nai_${id}_${feature}_on`)) continue;
            const pill = document.createElement("button");
            pill.type = "button";
            pill.className = "nai-pill";
            pill.dataset.feature = feature;
            pill.title = `Switch ${label} on or off (an off section is hidden, its settings kept)`;
            pill.innerHTML = `<span class="nai-pill-dot"></span>${label}<span class="nai-pill-count"></span>`;
            // a click on the pill must not also open or close the fold-out
            pill.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                checkbox(`nai_${id}_${feature}_on`)?.click();
            });
            summary.appendChild(pill);
        }
        box.appendChild(summary);
        box.addEventListener("toggle", () => {
            try {
                localStorage.setItem(key, box.open ? "1" : "0");
            } catch (e) {}
        });
        negRow.after(box);
        panels.forEach((panel) => box.appendChild(panel));
    }

    function sync(id) {
        const box = el(`nai_${id}_stagehand`);
        if (!box) return;
        const counts = {chars: characters(id).length, pr: references(id)};
        for (const pill of box.querySelectorAll(":scope > summary .nai-pill")) {
            const feature = pill.dataset.feature;
            const on = featureOn(id, feature);
            pill.classList.toggle("nai-on", on);
            if (pill.getAttribute("aria-pressed") !== String(on)) pill.setAttribute("aria-pressed", String(on));
            const count = on && counts[feature] ? ` ${counts[feature]}` : "";
            const badge = pill.querySelector(".nai-pill-count");
            if (badge.textContent !== count) badge.textContent = count;
            const panel = el(`nai_${id}_${feature}`);
            if (panel) panel.style.display = on ? "" : "none";
        }
        // an open fold-out with both features off would be empty: say why (style.css)
        // (a feature hidden in Settings has no checkbox at all, so it doesn't count as on)
        box.classList.toggle("nai-all-off", !FEATURES.some(([feature]) => checkbox(`nai_${id}_${feature}_on`)?.checked));
        // the Boxes / Grid switch, and where the positions go, only matter with AI's Choice off
        const auto = checkbox(`nai_${id}_chars_auto`)?.checked;
        const manual = el(`nai_${id}_chars_manual`);
        if (manual) manual.style.display = auto ? "none" : "";
        const where = el(`nai_${id}_chars_where`);
        if (where) where.style.display = auto ? "none" : "block";
    }

    // ------------------------------------------------------------------ prompts with characters in them
    function splitMarks(text) {
        const pieces = text.split(/\s*⟦(\d+)⟧\s*/);
        const base = [pieces[0]];
        const chars = {};
        for (let i = 1; i < pieces.length; i += 2) {
            const n = Number(pieces[i]);
            if (n === 0) base.push(pieces[i + 1]);
            else chars[n] = {text: pieces[i + 1].trim().replace(/\b(source|target|mutual)＃/g, "$1#")};
        }
        return {base: base.map((s) => s.trim()).filter(Boolean).join(" "), chars};
    }

    // the PNG info's form; same pattern as characters.py's _LABEL
    const LABEL = /^Character (\d+)(?: \((.*?)\))?(?: at ((?:[\d.]+ ){3}[\d.]+|[A-E][1-5]))?:[ \t]?(.*)$/;

    function readLines(text) {
        // a whole PNG info (prompt, negative, parameters) is Forge's paste button's to split
        if (/^(Negative prompt:|Steps: \d)/m.test(text)) return null;
        const lines = text.split("\n");
        const chars = {};
        let current = null;
        let first = -1;
        lines.forEach((line, i) => {
            const m = line.match(LABEL);
            if (m) {
                if (first < 0) first = i;
                current = chars[m[1]] = {name: m[2] || "", box: m[3] || "", text: m[4]};
            } else if (current) {
                current.text += "\n" + line;
            }
        });
        if (first < 0) return null;
        Object.values(chars).forEach((c) => (c.text = c.text.trim()));
        return {base: lines.slice(0, first).join("\n").trimEnd(), chars};
    }

    // exact: from a PNG info, the text is the whole story -- positions (or none: AI's Choice)
    // and switched-on cards; a ⟦n⟧ template only carries the texts, the cards keep the rest
    function fill(tab, id, input, kind, parsed, exact) {
        // the pasted characters replace the cards: one the text doesn't mention is emptied
        for (let n = 1; n <= MAX; n++) {
            if (n in parsed.chars) continue;
            setValue(field(`nai_${id}_char${n}_${kind}`), "");
            if (kind === "prompt") setValue(field(`nai_${id}_char${n}_box`), "");
        }
        for (const [n, c] of Object.entries(parsed.chars)) {
            setValue(field(`nai_${id}_char${n}_${kind}`), c.text);
            if (kind !== "prompt") continue;
            if (c.name) setValue(el(`nai_${id}_char${n}`)?.querySelector(".nai-char-name input, .nai-char-name textarea"), c.name);
            if (c.box || exact) setValue(field(`nai_${id}_char${n}_box`), c.box);
            const cardOn = el(`nai_${id}_char${n}`)?.querySelector(".nai-char-on input");
            if (exact && cardOn && !cardOn.checked) cardOn.click();
        }
        if (kind === "prompt") {
            const places = Object.values(parsed.chars).map((c) => c.box).filter(Boolean);
            const auto = checkbox(`nai_${id}_chars_auto`);
            if (places.length) {
                if (auto?.checked) auto.click();
                setManual(id, places.some((p) => cell(p)) ? "Grid" : "Boxes");
            } else if (exact && auto && !auto.checked) {
                auto.click();
            }
            const on = checkbox(`nai_${id}_chars_on`);
            if (on && !on.checked) on.click();
        }
        setValue(input, parsed.base);
    }

    function unmerge(tab, id) {
        for (const [main, kind] of [[`${tab}_prompt`, "prompt"], [`${tab}_neg_prompt`, "uc"]]) {
            const input = field(main);
            if (input && /⟦\d+⟧/.test(input.value)) fill(tab, id, input, kind, splitMarks(input.value));
        }
    }

    function splitOnPaste(tab, id) {
        for (const [main, kind] of [[`${tab}_prompt`, "prompt"], [`${tab}_neg_prompt`, "uc"]]) {
            const input = field(main);
            // on paste only: while typing, "Character 1:" would be pulled out mid-sentence
            input?.addEventListener("paste", (e) => {
                // a tag pasted into a prompt that already has character lines mustn't reset the cards
                const pasted = e.clipboardData?.getData("text") || "";
                if (!pasted.split("\n").some((line) => LABEL.test(line))) return;
                setTimeout(() => {
                    const parsed = readLines(input.value);
                    if (parsed) fill(tab, id, input, kind, parsed, true);
                }, 0);
            });
        }
    }

    // ------------------------------------------------------------------ positions
    function parse(value) {
        const v = (value || "").trim().split(/[\s,]+/).map(Number);
        return v.length === 4 && v.every((x) => Number.isFinite(x)) && v[2] > v[0] && v[3] > v[1] ? v : null;
    }

    // "C3" -> [column 0-4, row 0-4], or null
    function cell(value) {
        const m = /^\s*([A-Ea-e])([1-5])\s*$/.exec(value || "");
        return m ? [m[1].toUpperCase().charCodeAt(0) - 65, Number(m[2]) - 1] : null;
    }

    const cellName = (col, row) => `${String.fromCharCode(65 + col)}${row + 1}`;
    const cellOf = (x, y) => cellName(Math.min(Math.floor(x * GRID), GRID - 1), Math.min(Math.floor(y * GRID), GRID - 1));
    // where Grid puts a character with no cell yet: across the middle row, in card order
    // (with more characters than columns, alternating rows 2 and 4; same as characters.default_cells)
    const defaultCell = (k, count) => cellName(Math.min(Math.floor(((k + 0.5) / count) * GRID), GRID - 1), count <= GRID ? 2 : k % 2 ? 3 : 1);

    // Switching Boxes <-> Grid keeps each character where it was: a box becomes the cell its
    // center is in, a cell becomes a box around that point.
    // Every card, also those switched off or empty, so none keeps the other kind of position.
    function convert(id, manual) {
        const count = Math.max(characters(id).length, 2);
        for (let n = 1; n <= MAX; n++) {
            const input = field(`nai_${id}_char${n}_box`);
            const value = input?.value || "";
            if (manual === "Grid") {
                const box = parse(value);
                if (box) setValue(input, cellOf((box[0] + box[2]) / 2, (box[1] + box[3]) / 2));
            } else {
                const at = cell(value);
                if (!at) continue;
                const x = (at[0] + 0.5) / GRID;
                const y = (at[1] + 0.5) / GRID;
                // centered on the point, so switching back lands on the same cell
                const hx = Math.min(0.5 / count, x, 1 - x);
                const hy = Math.min(0.45, y, 1 - y);
                setValue(input, [x - hx, y - hy, x + hx, y + hy].map((v) => v.toFixed(3)).join(" "));
            }
        }
    }

    // The canvas the positions are placed on: always the shape of the NEXT image (the current
    // width x height), since positions are fractions and scale with it. When the last output
    // has that shape the frame sits exactly on it (object-fit: contain); after a resolution
    // change it's a frame of the new shape, fitted in the gallery.
    function frame(tab) {
        const gallery = el(`${tab}_gallery`);
        if (!gallery) return null;
        const g = gallery.getBoundingClientRect();
        const aspect = (parseFloat(field(`${tab}_width`)?.value) || 1024) / (parseFloat(field(`${tab}_height`)?.value) || 1024);
        let img = null;
        let best = 0;
        gallery.querySelectorAll("img").forEach((i) => {
            const r = i.getBoundingClientRect();
            if (i.offsetParent && i.naturalWidth && r.width * r.height > best) {
                best = r.width * r.height;
                img = i;
            }
        });
        let left;
        let top;
        let width;
        let height;
        if (img && Math.abs(img.naturalWidth / img.naturalHeight / aspect - 1) < 0.02) {
            const r = img.getBoundingClientRect();
            width = Math.min(r.width, r.height * aspect);
            height = width / aspect;
            left = r.left - g.left + (r.width - width) / 2;
            top = r.top - g.top + (r.height - height) / 2;
        } else {
            width = Math.min(g.width * 0.9, g.height * 0.9 * aspect);
            height = width / aspect;
            left = (g.width - width) / 2;
            top = (g.height - height) / 2;
        }
        return {gallery, left, top, width, height};
    }

    function startDrag(event, div, input, mode, overlay) {
        event.preventDefault();
        event.stopPropagation();
        dragging = true;
        const start = mode === "dot" ? JSON.parse(div.dataset.at) : parse(input.value) || JSON.parse(div.dataset.box);
        const x0 = event.clientX;
        const y0 = event.clientY;
        const W = overlay.clientWidth;
        const H = overlay.clientHeight;
        let box = start.slice();
        const move = (e) => {
            const dx = (e.clientX - x0) / W;
            const dy = (e.clientY - y0) / H;
            if (mode === "dot") {
                box = [Math.min(Math.max(start[0] + dx, 0), 0.999), Math.min(Math.max(start[1] + dy, 0), 0.999)];
                placeDot(div, box);
            } else if (mode === "move") {
                const w = start[2] - start[0];
                const h = start[3] - start[1];
                const x = Math.min(Math.max(start[0] + dx, 0), 1 - w);
                const y = Math.min(Math.max(start[1] + dy, 0), 1 - h);
                box = [x, y, x + w, y + h];
                place(div, box);
            } else {
                box = [start[0], start[1], Math.min(Math.max(start[2] + dx, start[0] + 0.05), 1), Math.min(Math.max(start[3] + dy, start[1] + 0.05), 1)];
                place(div, box);
            }
        };
        const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            dragging = false;
            // a dot snaps to the cell it was dropped in
            setValue(input, mode === "dot" ? cellOf(box[0], box[1]) : box.map((v) => v.toFixed(3)).join(" "));
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
    }

    function place(div, box) {
        div.dataset.box = JSON.stringify(box);
        div.style.left = `${box[0] * 100}%`;
        div.style.top = `${box[1] * 100}%`;
        div.style.width = `${(box[2] - box[0]) * 100}%`;
        div.style.height = `${(box[3] - box[1]) * 100}%`;
    }

    function placeDot(div, at) {
        div.dataset.at = JSON.stringify(at);
        div.style.left = `${at[0] * 100}%`;
        div.style.top = `${at[1] * 100}%`;
    }

    function render(tab, id) {
        if (dragging) return;
        let overlay = el(`nai_${id}_positions`);
        const auto = checkbox(`nai_${id}_chars_auto`);
        const chars = characters(id);
        const open = el(`nai_${id}_stagehand`)?.open;
        const f = open && auto && !auto.checked && chars.length ? frame(tab) : null;
        if (!f) {
            if (overlay) overlay.style.display = "none";
            return;
        }
        if (!overlay || overlay.parentElement !== f.gallery) {
            overlay?.remove();
            overlay = document.createElement("div");
            overlay.id = `nai_${id}_positions`;
            f.gallery.style.position = "relative";
            f.gallery.appendChild(overlay);
        }
        const grid = manualOf(id) === "Grid";
        overlay.className = grid ? "nai-positions nai-grid" : "nai-positions";
        Object.assign(overlay.style, {display: "", left: `${f.left}px`, top: `${f.top}px`, width: `${f.width}px`, height: `${f.height}px`});

        const keep = new Set();
        chars.forEach((c, k) => {
            const kind = grid ? "dot" : "box";
            keep.add(`${kind}${c.n}`);
            let div = overlay.querySelector(`[data-key="${kind}${c.n}"]`);
            if (!div) {
                div = document.createElement("div");
                div.dataset.key = `${kind}${c.n}`;
                if (grid) {
                    div.className = `nai-pos-dot nai-c${c.n}`;
                    div.innerHTML = '<span class="nai-pos-label"></span>';
                    div.addEventListener("pointerdown", (e) => startDrag(e, div, c.input, "dot", overlay));
                } else {
                    div.className = `nai-pos-box nai-c${c.n}`;
                    div.innerHTML = '<span class="nai-pos-label"></span><span class="nai-pos-handle"></span>';
                    div.addEventListener("pointerdown", (e) => startDrag(e, div, c.input, e.target.classList.contains("nai-pos-handle") ? "resize" : "move", overlay));
                }
                overlay.appendChild(div);
            }
            if (grid) {
                const at = cell(c.input?.value) || cell(defaultCell(k, chars.length));
                setText(div.querySelector(".nai-pos-label"), `${c.label} · ${cellName(at[0], at[1])}`);
                placeDot(div, [(at[0] + 0.5) / GRID, (at[1] + 0.5) / GRID]);
            } else {
                setText(div.querySelector(".nai-pos-label"), c.label);
                // no saved position yet: AI's Choice columns, which is also what the backend uses
                place(div, parse(c.input?.value) || [k / chars.length, 0, (k + 1) / chars.length, 1]);
            }
        });
        overlay.querySelectorAll("[data-key]").forEach((d) => keep.has(d.dataset.key) || d.remove());
    }

    // Tag Autocomplete (when installed) attaches to the prompt boxes it finds when the page
    // loads; a card's boxes only exist once the card is shown, so they're handed to it as they
    // appear (once each: it skips a box it's already on, or one its settings leave out).
    const offered = new WeakSet();
    function autocomplete(id) {
        if (typeof addAutocompleteToArea !== "function" || typeof TAC_CFG === "undefined" || !TAC_CFG) return;
        for (let n = 1; n <= MAX; n++) {
            for (const kind of ["prompt", "uc"]) {
                const area = el(`nai_${id}_char${n}_${kind}`)?.querySelector("textarea");
                if (!area || offered.has(area)) continue;
                offered.add(area);
                try {
                    addAutocompleteToArea(area);
                } catch (e) {
                    console.warn("[Stagehand] Tag Autocomplete couldn't attach to a character box", e);
                }
            }
        }
    }

    // A card's controls only exist once the card is shown, so this runs with the layout loop.
    function tooltips(id) {
        const tips = {
            ".nai-char-on": "Switch this character off to leave it out of the next image without deleting it",
            ".nai-char-name": "Name (optional): shown in the image's PNG info and on its position box; a preset is saved under it",
            ".nai-char-face": "ADetailer: which detected face gets this character's prompt. auto matches them by position",
            ".nai-save-preset": "Save this character as a preset under its name (saving again updates it). Newlines are kept.",
            ".nai-preset": "Character presets: pick one, then + Add character adds a card filled with it",
            ".nai-delete-preset": "Delete the selected preset",
            ".nai-up": "Move this character up (earlier = further left with AI's Choice)",
            ".nai-down": "Move this character down (later = further right with AI's Choice)",
            ".nai-copy": "Duplicate this character",
            ".nai-remove": "Delete this character",
            ".nai-remove-ref": "Remove this reference",
            [`#nai_${id}_chars_auto`]: "On: the characters stand left to right in card order. Off: place them yourself, on the output image",
            [`#nai_${id}_chars_manual`]: "Boxes: drag and resize a box per character. Grid: NovelAI's 5x5 grid, a dot where each character's head goes",
            [`#nai_${id}_pr_adetailer`]: "Also use the references in ADetailer's face pass, for this generation",
        };
        const box = el(`nai_${id}_stagehand`);
        for (const [selector, tip] of Object.entries(tips)) {
            box?.querySelectorAll(selector).forEach((node) => node.title !== tip && (node.title = tip));
        }
    }

    onUiLoaded(() => {
        for (const [tab, id] of TABS) {
            wrap(tab, id);
            splitOnPaste(tab, id);
            el(`nai_${id}_chars_manual`)?.querySelectorAll("input[type=radio]").forEach((radio) => {
                radio.addEventListener("change", () => radio.checked && convert(id, radio.value));
            });
        }
        // Cheap and robust against everything that can change the layout (cards, toggles,
        // a new image, resizing); positions aren't re-placed while one is being dragged.
        setInterval(() => TABS.forEach(([tab, id]) => {
            unmerge(tab, id);
            sync(id);
            render(tab, id);
            tooltips(id);
            autocomplete(id);
        }), 400);
    });
})();
