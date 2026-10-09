// Stagehand in the browser:
// - both panels (Character Prompts, Precise Reference) go under the prompt boxes, inside one
//   "Stagehand" fold-out. Its header has a pill per feature that switches it on or off -- also
//   while the fold-out is closed -- and an off feature's section is hidden;
// - a prompt that carries characters -- Set Queue's "Restore template" (marked ⟦n⟧), or text
//   pasted from an image's PNG info ("Character 1 (Ava): ...") -- is split into the cards;
// - the characters are placed on "surfaces": a small canvas beside each card (that character
//   only) and an overlay on the output image (everyone), kept in sync. Boxes (drag a box from
//   anywhere, resize by any edge or corner; "x0 y0 x1 y1", fractions of the image) or Grid
//   (NovelAI V4.5's 5x5 grid: drag a dot to a cell; "C3"). The result lands in the card's
//   hidden position field; an empty one is the default column (the old AI's Choice);
// - Reset / Switch / the two surface toggles in the Characters header; Ctrl+Z / Ctrl+Y undo
//   and redo position and card changes (outside text boxes);
// - a card's name box searches the presets; its ⋯ opens the row with the less used controls;
// - Precise Reference's cards for a character live in her card's Reference tab.
(() => {
    const TABS = [["txt2img", "t2i"], ["img2img", "i2i"]];
    const MAX = 10;
    const GRID = 5;
    const SHARE = 50; // a card's default overlap share (characters.SHARE)
    const FEATURES = [["chars", "Character Prompts"], ["pr", "Precise Reference"]];
    const FACE_AUTO = "Face: auto"; // character_prompts.FACES[0]
    const SVG = "http://www.w3.org/2000/svg";
    const MINI = 200; // a card's canvas, long side in px
    // the place being dragged: {id, n, j, value} -- every surface draws it live from this
    let live = null;

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
            const share = Number(field(`nai_${id}_char${n}_share`)?.value ?? SHARE);
            // the box's tag says the share when it isn't the default: overlaps follow it
            const label = (name || `${n}`) + (share !== SHARE ? ` · ${share}%` : "");
            out.push({n, label, input: field(`nai_${id}_char${n}_box`)});
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
        syncTools(id);
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
    const PLACE = "(?:[\\d.]+ ){3}[\\d.]+|[A-E][1-5]";
    const LABEL = new RegExp(`^Character (\\d+)(?: \\((.*?)\\))?(?: at ((?:${PLACE})(?: \\+ (?:${PLACE}))*))?(?:, share (\\d{1,3})%)?:[ \\t]?(.*)$`);

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
                current = chars[m[1]] = {name: m[2] || "", box: m[3] || "", share: m[4] || "", text: m[5]};
            } else if (current) {
                current.text += "\n" + line;
            }
        });
        if (first < 0) return null;
        Object.values(chars).forEach((c) => (c.text = c.text.trim()));
        return {base: lines.slice(0, first).join("\n").trimEnd(), chars};
    }

    // exact: from a PNG info, the text is the whole story -- positions (or none: the default
    // columns) and switched-on cards; a ⟦n⟧ template only carries the texts, the cards keep the rest
    function fill(tab, id, input, kind, parsed, exact) {
        // the pasted characters replace the cards: one the text doesn't mention is emptied
        for (let n = 1; n <= MAX; n++) {
            if (n in parsed.chars) continue;
            setValue(field(`nai_${id}_char${n}_${kind}`), "");
            if (kind === "prompt") setValue(field(`nai_${id}_char${n}_box`), "");
            if (kind === "prompt") setValue(field(`nai_${id}_char${n}_share`), String(SHARE));
        }
        for (const [n, c] of Object.entries(parsed.chars)) {
            setValue(field(`nai_${id}_char${n}_${kind}`), c.text);
            if (kind !== "prompt") continue;
            if (c.name) setValue(el(`nai_${id}_char${n}`)?.querySelector(".nai-char-name input, .nai-char-name textarea"), c.name);
            if (c.box || exact) setValue(field(`nai_${id}_char${n}_box`), c.box);
            if (c.share || exact) setValue(field(`nai_${id}_char${n}_share`), c.share || String(SHARE));
            const cardOn = el(`nai_${id}_char${n}`)?.querySelector(".nai-char-on input");
            if (exact && cardOn && !cardOn.checked) cardOn.click();
        }
        if (kind === "prompt") {
            const places = Object.values(parsed.chars).map((c) => c.box).filter(Boolean);
            if (places.length) setManual(id, places.some((p) => cell(p.split("+")[0])) ? "Grid" : "Boxes");
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

    // A card's position field holds one place or several (one character in more than one spot:
    // multi-angle sheets, complex compositions), " + " between them.
    const places = (value) => (value || "").split("+").map((v) => v.trim()).filter(Boolean);
    const boxText = (box) => box.map((v) => v.toFixed(3)).join(" ");
    const column = (k, count) => [k / count, 0, (k + 1) / count, 1];

    // The places of the given kind ("Grid": cells, else boxes), as the backend reads them: a
    // leftover of the other kind is ignored. Empty: the character's default place.
    function ownPlaces(input, grid, k, count) {
        const own = places(input?.value).filter((v) => (grid ? cell(v) : parse(v)));
        return own.length ? own : [grid ? defaultCell(k, count) : boxText(column(k, count))];
    }

    // Switching Boxes <-> Grid keeps each character where it was: a box becomes the cell its
    // center is in, a cell becomes a box around that point.
    // Every card, also those switched off or empty, so none keeps the other kind of position.
    function convert(id, manual) {
        const count = Math.max(characters(id).length, 2);
        for (let n = 1; n <= MAX; n++) {
            const input = field(`nai_${id}_char${n}_box`);
            const list = places(input?.value);
            if (!list.length) continue;
            const converted = list.map((value) => {
                if (manual === "Grid") {
                    const box = parse(value);
                    return box ? cellOf((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) : value;
                }
                const at = cell(value);
                if (!at) return value;
                const x = (at[0] + 0.5) / GRID;
                const y = (at[1] + 0.5) / GRID;
                // centered on the point, so switching back lands on the same cell
                const hx = Math.min(0.5 / count, x, 1 - x);
                const hy = Math.min(0.45, y, 1 - y);
                return boxText([x - hx, y - hy, x + hx, y + hy]);
            });
            setValue(input, converted.join(" + "));
        }
    }

    // ＋ on a card: one more place for that character beside its last one -- a half-size box (so
    // it stands out from the merged outline instead of just widening it), or the next free
    // cell -- dragged from there like any other.
    function addPlace(id, n) {
        const chars = characters(id);
        const k = chars.findIndex((c) => c.n === n);
        const input = field(`nai_${id}_char${n}_box`);
        if (k < 0 || !input) return;
        const grid = manualOf(id) === "Grid";
        const list = ownPlaces(input, grid, k, chars.length);
        const last = list[list.length - 1];
        if (grid) {
            const taken = new Set(chars.flatMap((c, j) => ownPlaces(c.input, true, j, chars.length)));
            const [col, row] = cell(last);
            const order = Array.from({length: GRID * GRID}, (_, i) => (row * GRID + col + 1 + i) % (GRID * GRID));
            const free = order.map((i) => cellName(i % GRID, Math.floor(i / GRID))).find((c) => !taken.has(c));
            list.push(free || last);
        } else {
            const b = parse(last);
            const w = Math.max((b[2] - b[0]) / 2, 0.05);
            const h = Math.max((b[3] - b[1]) / 2, 0.05);
            const clamp = (v, size) => Math.min(Math.max(v, 0), 1 - size);
            // right of it, else left, else below, else above, else on its middle
            const y = clamp((b[1] + b[3]) / 2 - h / 2, h);
            const x = clamp((b[0] + b[2]) / 2 - w / 2, w);
            const box = b[2] + w <= 1 ? [b[2], y] : b[0] - w >= 0 ? [b[0] - w, y]
                : b[3] + h <= 1 ? [x, b[3]] : b[1] - h >= 0 ? [x, b[1] - h] : [x, y];
            list.push(boxText([box[0], box[1], box[0] + w, box[1] + h]));
        }
        setValue(input, list.join(" + "));
    }

    function removePlace(input, grid, k, count, j) {
        const list = ownPlaces(input, grid, k, count);
        if (list.length < 2) return;
        list.splice(j, 1);
        setValue(input, list.join(" + "));
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

    // ------------------------------------------------------------------ surfaces
    // The output-image overlay and each card's small canvas draw the same places; a drag on any
    // of them sets `live`, and every surface redraws from it, so they move together.
    const tabOf = (id) => TABS.find(([, i]) => i === id)[0];
    const aspectOf = (id) => (parseFloat(field(`${tabOf(id)}_width`)?.value) || 1024) / (parseFloat(field(`${tabOf(id)}_height`)?.value) || 1024);

    // a character's places as shapes -- boxes [x0, y0, x1, y1], or grid points [x, y] (cell
    // centers) -- with the one being dragged where the pointer has it
    function shapesOf(id, c, grid, k, count) {
        const list = ownPlaces(c.input, grid, k, count).map((v) => {
            if (!grid) return parse(v);
            const at = cell(v);
            return [(at[0] + 0.5) / GRID, (at[1] + 0.5) / GRID];
        });
        if (live && live.id === id && live.n === c.n && live.j < list.length) list[live.j] = live.value;
        return list;
    }

    // Stores a dropped place: the j-th of character n's places (the others read at drop time).
    function writePlace(id, n, j, text) {
        const chars = characters(id);
        const k = chars.findIndex((c) => c.n === n);
        if (k < 0) return;
        const now = ownPlaces(chars[k].input, manualOf(id) === "Grid", k, chars.length);
        now[j] = text;
        setValue(chars[k].input, now.join(" + "));
    }

    let frameRequested = false;
    function redraw(id) {
        if (frameRequested) return;
        frameRequested = true;
        requestAnimationFrame(() => {
            frameRequested = false;
            render(tabOf(id), id);
        });
    }

    // mode: "move", "dot", or the edges a handle moves ("n", "se", ...)
    function startDrag(event, surface, id, n, j, mode, start) {
        if (event.button || event.target.classList.contains("nai-pos-remove")) return;
        event.preventDefault();
        event.stopPropagation();
        track(id);
        const x0 = event.clientX;
        const y0 = event.clientY;
        const W = surface.clientWidth;
        const H = surface.clientHeight;
        const MIN = 0.05;
        live = {id, n, j, value: start.slice()};
        // a box's edges snap to the image's borders and middle and to every other place's edges
        // (Alt held: no snapping), when within SNAP screen px
        const SNAP = 6;
        const xs = [0, 0.5, 1];
        const ys = [0, 0.5, 1];
        if (mode !== "dot") {
            const chars = characters(id);
            chars.forEach((ch, k) => ownPlaces(ch.input, false, k, chars.length).forEach((v, i) => {
                const other = parse(v);
                if (other && !(ch.n === n && i === j)) {
                    xs.push(other[0], other[2]);
                    ys.push(other[1], other[3]);
                }
            }));
        }
        // how far to shift so the nearest of `edges` meets a target (0: none close enough)
        const pull = (edges, targets, size) => {
            let shift = 0;
            let gap = SNAP / size;
            for (const at of edges) for (const t of targets) {
                if (Math.abs(t - at) < gap) {
                    gap = Math.abs(t - at);
                    shift = t - at;
                }
            }
            return shift;
        };
        const move = (e) => {
            const dx = (e.clientX - x0) / W;
            const dy = (e.clientY - y0) / H;
            const snap = !e.altKey;
            let [a, b, c, d] = start;
            if (mode === "dot") {
                live.value = [Math.min(Math.max(a + dx, 0), 0.999), Math.min(Math.max(b + dy, 0), 0.999)];
            } else if (mode === "move") {
                const w = c - a;
                const h = d - b;
                let x = Math.min(Math.max(a + dx, 0), 1 - w);
                let y = Math.min(Math.max(b + dy, 0), 1 - h);
                if (snap) {
                    x = Math.min(Math.max(x + pull([x, x + w], xs, W), 0), 1 - w);
                    y = Math.min(Math.max(y + pull([y, y + h], ys, H), 0), 1 - h);
                }
                live.value = [x, y, x + w, y + h];
            } else {
                if (mode.includes("w")) a = Math.min(Math.max(a + dx, 0), c - MIN);
                if (mode.includes("e")) c = Math.max(Math.min(c + dx, 1), a + MIN);
                if (mode.includes("n")) b = Math.min(Math.max(b + dy, 0), d - MIN);
                if (mode.includes("s")) d = Math.max(Math.min(d + dy, 1), b + MIN);
                if (snap) {
                    if (mode.includes("w")) a = Math.min(a + pull([a], xs, W), c - MIN);
                    if (mode.includes("e")) c = Math.max(c + pull([c], xs, W), a + MIN);
                    if (mode.includes("n")) b = Math.min(b + pull([b], ys, H), d - MIN);
                    if (mode.includes("s")) d = Math.max(d + pull([d], ys, H), b + MIN);
                }
                live.value = [a, b, c, d];
            }
            redraw(id);
        };
        const up = (e) => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            const v = live.value;
            const click = Math.hypot(e.clientX - x0, e.clientY - y0) < 4;
            const moved = !click && v.some((x, i) => x !== start[i]);
            live = null;
            // a dot snaps to the cell it was dropped in
            if (moved) writePlace(id, n, j, mode === "dot" ? cellOf(v[0], v[1]) : boxText(v));
            redraw(id);
            // a click on the output image's boxes is a click on the image: it opens it
            if (click && surface.classList.contains("nai-positions")) {
                surface.style.visibility = "hidden";
                const under = document.elementFromPoint(e.clientX, e.clientY);
                surface.style.visibility = "";
                under?.click();
            }
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
    }

    const pct = (v) => `${v * 100}%`;

    function place(div, box) {
        div.style.left = pct(box[0]);
        div.style.top = pct(box[1]);
        div.style.width = pct(box[2] - box[0]);
        div.style.height = pct(box[3] - box[1]);
    }

    // One character's boxes as one shape: the cells of the grid their edges make, filled where
    // a box covers them, with an outline only between a covered cell and an uncovered one -- so
    // touching or overlapping places read as one continuous shape, in SVG path data (0-100).
    function union(boxes) {
        const xs = [...new Set(boxes.flatMap((b) => [b[0], b[2]]))].sort((a, b) => a - b);
        const ys = [...new Set(boxes.flatMap((b) => [b[1], b[3]]))].sort((a, b) => a - b);
        const covered = ys.slice(1).map((_, r) => xs.slice(1).map((_, q) => {
            const cx = (xs[q] + xs[q + 1]) / 2;
            const cy = (ys[r] + ys[r + 1]) / 2;
            return boxes.some((b) => b[0] < cx && cx < b[2] && b[1] < cy && cy < b[3]);
        }));
        const on = (q, r) => r >= 0 && r < covered.length && q >= 0 && q < covered[r].length && covered[r][q];
        const P = (v) => (v * 100).toFixed(2);
        let fill = "";
        let line = "";
        covered.forEach((row, r) => row.forEach((c, q) => {
            if (!c) return;
            const [l, t, rt, bt] = [P(xs[q]), P(ys[r]), P(xs[q + 1]), P(ys[r + 1])];
            fill += `M${l} ${t}H${rt}V${bt}H${l}Z`;
            if (!on(q, r - 1)) line += `M${l} ${t}H${rt}`;
            if (!on(q, r + 1)) line += `M${l} ${bt}H${rt}`;
            if (!on(q - 1, r)) line += `M${l} ${t}V${bt}`;
            if (!on(q + 1, r)) line += `M${rt} ${t}V${bt}`;
        }));
        return [fill, line];
    }

    const HANDLES = ["n", "s", "e", "w", "ne", "nw", "se", "sw"];

    // Draws characters on a surface: everyone (the overlay) or one (`only`, a card's canvas).
    // Elements are keyed, so the one under the pointer survives every redraw.
    // the place last clicked: Delete / Backspace removes it while its character has another (deleteKey)
    let picked = null;

    function drawSurface(surface, id, chars, grid, only) {
        surface.classList.toggle("nai-grid", grid);
        let svg = surface.querySelector(":scope > svg");
        if (!svg) {
            svg = document.createElementNS(SVG, "svg");
            svg.setAttribute("viewBox", "0 0 100 100");
            svg.setAttribute("preserveAspectRatio", "none");
            surface.prepend(svg);
        }
        const keep = new Set();
        let paths = "";
        chars.forEach((c, k) => {
            if (only && c.n !== only) return;
            const shapes = shapesOf(id, c, grid, k, chars.length);
            const remove = '<span class="nai-pos-remove" title="Remove this place">×</span>';
            if (!grid) {
                const [fill, line] = union(shapes);
                paths += `<path class="nai-union-fill nai-c${c.n}" d="${fill}"/><path class="nai-union-line nai-c${c.n}" d="${line}"/>`;
            }
            // the name once, small, in the middle of the biggest piece; a dot marks the others
            const area = (b) => (b[2] - b[0]) * (b[3] - b[1]);
            const big = grid ? 0 : shapes.reduce((best, b, j) => (area(b) > area(shapes[best]) ? j : best), 0);
            shapes.forEach((shape, j) => {
                const key = `${grid ? "dot" : "box"}${c.n}_${j}`;
                keep.add(key);
                let div = surface.querySelector(`:scope > [data-key="${key}"]`);
                if (!div) {
                    div = document.createElement("div");
                    div.dataset.key = key;
                    div.className = `${grid ? "nai-pos-dot" : "nai-pos-box"} nai-c${c.n}`;
                    div.innerHTML = '<span class="nai-pos-label"></span><span class="nai-pos-mark"></span>'
                        + (j ? remove : "") + (grid ? "" : HANDLES.map((h) => `<span class="nai-pos-h" data-dir="${h}"></span>`).join(""));
                    div.addEventListener("pointerdown", (e) => {
                        const mine = shapesOf(id, c, manualOf(id) === "Grid", characters(id).findIndex((x) => x.n === c.n), characters(id).length);
                        if (mine.length > 1) {
                            picked = {id, n: c.n, j};
                            div.classList.add("nai-pos-picked");
                        }
                        const now = mine[j];
                        if (now) startDrag(e, surface, id, c.n, j, grid ? "dot" : e.target.dataset.dir || "move", now);
                    });
                    div.querySelector(".nai-pos-remove")?.addEventListener("click", (e) => {
                        e.stopPropagation();
                        track(id);
                        const all = characters(id);
                        const at = all.findIndex((x) => x.n === c.n);
                        if (at >= 0) removePlace(all[at].input, manualOf(id) === "Grid", at, all.length, j);
                    });
                    surface.appendChild(div);
                }
                if (grid) {
                    div.style.left = pct(shape[0]);
                    div.style.top = pct(shape[1]);
                    setText(div.querySelector(".nai-pos-label"), `${c.label} · ${cellOf(shape[0], shape[1])}`);
                } else {
                    place(div, shape);
                    div.classList.toggle("nai-labelled", j === big);
                    setText(div.querySelector(".nai-pos-label"), c.label);
                }
            });
        });
        if (svg.dataset.paths !== paths) {
            svg.dataset.paths = paths;
            svg.innerHTML = paths;
        }
        surface.querySelectorAll(":scope > [data-key]").forEach((d) => keep.has(d.dataset.key) || d.remove());
    }

    // the two surface toggles, remembered per browser: on cards starts on, on image off (its
    // boxes tint the picture you're looking at)
    const toggledOn = (id, what) => {
        let saved = null;
        try {
            saved = localStorage.getItem(`stagehand-${what}-${id}`);
        } catch (e) {}
        return saved === null ? what !== "image" : saved !== "0";
    };

    function render(tab, id) {
        const chars = characters(id);
        const open = el(`nai_${id}_stagehand`)?.open;
        const grid = manualOf(id) === "Grid";

        // the output image
        let overlay = el(`nai_${id}_positions`);
        const f = open && chars.length && toggledOn(id, "image") ? frame(tab) : null;
        if (!f) {
            if (overlay) overlay.style.display = "none";
        } else {
            if (!overlay || overlay.parentElement !== f.gallery) {
                overlay?.remove();
                overlay = document.createElement("div");
                overlay.id = `nai_${id}_positions`;
                overlay.className = "nai-surface nai-positions";
                f.gallery.style.position = "relative";
                f.gallery.appendChild(overlay);
            }
            Object.assign(overlay.style, {display: "", left: `${f.left}px`, top: `${f.top}px`, width: `${f.width}px`, height: `${f.height}px`});
            drawSurface(overlay, id, chars, grid);
        }

        // each card's own canvas, the image's shape, MINI px on its long side
        const aspect = aspectOf(id);
        const [w, h] = aspect >= 1 ? [MINI, MINI / aspect] : [MINI * aspect, MINI];
        const canvases = open && toggledOn(id, "canvases");
        for (let n = 1; n <= MAX; n++) {
            const card = el(`nai_${id}_char${n}`);
            if (!card) continue;
            let mini = card.querySelector(":scope > .nai-mini");
            const on = canvases && chars.some((c) => c.n === n);
            card.classList.toggle("nai-has-mini", !!on);
            if (!on) {
                mini?.remove();
                continue;
            }
            if (!mini) {
                mini = document.createElement("div");
                mini.className = "nai-surface nai-mini";
                // another place for this character (trackCards makes it an undo step)
                const more = document.createElement("button");
                more.type = "button";
                more.className = "nai-add-place";
                more.textContent = "+";
                more.addEventListener("click", (e) => {
                    e.stopPropagation();
                    addPlace(id, n);
                });
                mini.appendChild(more);
                card.appendChild(mini);
            }
            if (mini.style.width !== `${w}px` || mini.style.height !== `${h}px`) Object.assign(mini.style, {width: `${w}px`, height: `${h}px`});
            card.style.setProperty("--nai-mini-w", `${w}px`);
            card.style.setProperty("--nai-mini-h", `${h}px`);
            drawSurface(mini, id, chars, grid, n);
        }
    }

    // ------------------------------------------------------------------ Reset, Switch, toggles
    function tools(id) {
        const host = el(`nai_${id}_chars_tools`);
        if (!host || host.querySelector(".nai-tools-row")) return;
        const row = document.createElement("span");
        row.className = "nai-tools-row";
        row.innerHTML = '<button type="button" class="nai-tool nai-reset">↺ Reset boxes</button>'
            + '<span class="nai-switch"><select class="nai-switch-a"></select><span class="nai-switch-arrow">⇄</span>'
            + '<select class="nai-switch-b"></select><button type="button" class="nai-tool nai-switch-go">Switch</button></span>'
            + '<button type="button" class="nai-toggle" data-what="canvases">▣ on cards</button>'
            + '<button type="button" class="nai-toggle" data-what="image">▣ on image</button>';
        host.appendChild(row);
        row.querySelector(".nai-reset").addEventListener("click", () => {
            track(id);
            // every card, also those off or empty, so none keeps a place to come back with
            for (let n = 1; n <= MAX; n++) setValue(field(`nai_${id}_char${n}_box`), "");
        });
        row.querySelector(".nai-switch-go").addEventListener("click", () => {
            const a = Number(row.querySelector(".nai-switch-a").value);
            const b = Number(row.querySelector(".nai-switch-b").value);
            const chars = characters(id);
            const ka = chars.findIndex((c) => c.n === a);
            const kb = chars.findIndex((c) => c.n === b);
            if (ka < 0 || kb < 0 || ka === kb) return;
            track(id);
            // written out, defaults included: their columns follow card order, not the card
            const grid = manualOf(id) === "Grid";
            const pa = ownPlaces(chars[ka].input, grid, ka, chars.length);
            const pb = ownPlaces(chars[kb].input, grid, kb, chars.length);
            setValue(chars[ka].input, pb.join(" + "));
            setValue(chars[kb].input, pa.join(" + "));
        });
        row.querySelectorAll(".nai-toggle").forEach((button) => button.addEventListener("click", () => {
            try {
                localStorage.setItem(`stagehand-${button.dataset.what}-${id}`, toggledOn(id, button.dataset.what) ? "0" : "1");
            } catch (e) {}
            syncTools(id);
            render(tabOf(id), id);
        }));
    }

    function syncTools(id) {
        const row = el(`nai_${id}_chars_tools`)?.querySelector(".nai-tools-row");
        if (!row) return;
        row.querySelectorAll(".nai-toggle").forEach((button) => {
            const on = toggledOn(id, button.dataset.what);
            button.classList.toggle("nai-on", on);
            if (button.getAttribute("aria-pressed") !== String(on)) button.setAttribute("aria-pressed", String(on));
        });
        // the switch lists the characters as they are; a pick that's gone falls back to 1st / 2nd.
        // With just two there's nothing to pick: one ⇄ button (style.css hides the lists).
        const chars = characters(id);
        const options = chars.map((c) => `<option value="${c.n}">${c.label.replace(/[&<>"]/g, (ch) => `&#${ch.charCodeAt(0)};`)}</option>`).join("");
        const sw = row.querySelector(".nai-switch");
        sw.style.display = chars.length >= 2 ? "" : "none";
        sw.classList.toggle("nai-switch-two", chars.length === 2);
        setText(row.querySelector(".nai-switch-go"), chars.length === 2 ? "⇄ Switch" : "Switch");
        ["a", "b"].forEach((which, i) => {
            const select = row.querySelector(`.nai-switch-${which}`);
            if (select.dataset.options !== options) {
                const kept = select.value;
                select.dataset.options = options;
                select.innerHTML = options;
                select.value = chars.some((c) => String(c.n) === kept) ? kept : String(chars[i]?.n ?? "");
            }
            if (chars.length === 2 && select.value !== String(chars[i].n)) select.value = String(chars[i].n);
        });
    }

    // ------------------------------------------------------------------ undo / redo
    // An action (a drag, Reset, Switch, a card button) records the fields it changed, before and
    // after, once the page settles (card buttons go through the server). Undo puts back only
    // those, so text typed since stays; the whole state is restored through the server
    // (_restore), which also owns which cards are shown.
    const undoLog = {};
    const hist = (id) => (undoLog[id] ??= {undo: [], redo: [], pending: null});

    function snapshot(id) {
        const out = {manual: manualOf(id)};
        for (let n = 1; n <= MAX; n++) {
            const card = el(`nai_${id}_char${n}`);
            const p = `c${n}.`;
            out[p + "visible"] = !!card && getComputedStyle(card).display !== "none";
            out[p + "enabled"] = card?.querySelector(".nai-char-on input")?.checked ?? true;
            out[p + "name"] = card?.querySelector(".nai-char-name input, .nai-char-name textarea")?.value ?? "";
            out[p + "prompt"] = field(`nai_${id}_char${n}_prompt`)?.value ?? "";
            out[p + "uc"] = field(`nai_${id}_char${n}_uc`)?.value ?? "";
            out[p + "box"] = field(`nai_${id}_char${n}_box`)?.value ?? "";
            out[p + "face"] = card?.querySelector(".nai-char-face input")?.value || FACE_AUTO;
            out[p + "share"] = Number(field(`nai_${id}_char${n}_share`)?.value ?? SHARE);
        }
        // who each reference is for: reordering cards moves their references along
        out["pr.targets"] = JSON.stringify(refTargets(id));
        return out;
    }

    const changed = (a, b) => Object.keys(a).filter((k) => a[k] !== b[k]);

    function track(id) {
        const h = hist(id);
        if (h.pending) settle(id, true);
        h.pending = {before: snapshot(id), since: Date.now(), last: null, stable: 0};
    }

    // closes the pending action once nothing has changed for a moment (or now, when forced)
    function settle(id, force) {
        const h = hist(id);
        const p = h.pending;
        if (!p || (live && live.id === id && !force)) return;
        const now = snapshot(id);
        const keys = changed(p.before, now);
        if (!force) {
            if (!keys.length) {
                if (Date.now() - p.since > 4000) h.pending = null; // the action changed nothing
                return;
            }
            if (!p.last || changed(p.last, now).length) {
                p.last = now;
                p.stable = Date.now();
                return;
            }
            if (Date.now() - p.stable < 350) return;
        }
        h.pending = null;
        if (!keys.length) return;
        const entry = {before: {}, after: {}};
        keys.forEach((k) => {
            entry.before[k] = p.before[k];
            entry.after[k] = now[k];
        });
        h.undo.push(entry);
        if (h.undo.length > 100) h.undo.shift();
        h.redo = [];
    }

    function restore(id, values) {
        if ("pr.targets" in values) prAction(id, {op: "targets", values: JSON.parse(values["pr.targets"])});
        const state = {...snapshot(id), ...values};
        const cards = [];
        for (let n = 1; n <= MAX; n++) {
            const card = {};
            for (const key of ["visible", "enabled", "name", "prompt", "uc", "box", "face", "share"]) card[key] = state[`c${n}.${key}`];
            cards.push(card);
        }
        setValue(field(`nai_${id}_chars_restore`), JSON.stringify({manual: state.manual, cards, at: Date.now()}));
    }

    function step(id, back) {
        const h = hist(id);
        settle(id, true);
        const entry = (back ? h.undo : h.redo).pop();
        if (!entry) return false;
        (back ? h.redo : h.undo).push(entry);
        restore(id, back ? entry.before : entry.after);
        return true;
    }

    // which tab's Stagehand is on screen
    const activeId = () => TABS.map(([, id]) => id).find((id) => el(`nai_${id}_stagehand`)?.offsetParent);

    const TYPING = "textarea, select, [contenteditable=''], [contenteditable=true], input:not([type=checkbox]):not([type=radio]):not([type=range]):not([type=button])";

    function undoKeys() {
        document.addEventListener("keydown", (e) => {
            if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
            const key = e.key.toLowerCase();
            if (key !== "z" && key !== "y") return;
            // a text box keeps its own undo
            if (e.target.closest?.(TYPING)) return;
            const id = activeId();
            if (id && step(id, key === "z" && !e.shiftKey)) e.preventDefault();
        });
    }

    function deleteKey() {
        const unpick = () => {
            picked = null;
            document.querySelectorAll(".nai-pos-picked").forEach((d) => d.classList.remove("nai-pos-picked"));
        };
        // capture: runs before a place's own pointerdown picks it
        document.addEventListener("pointerdown", unpick, true);
        document.addEventListener("keydown", (e) => {
            if (!picked || (e.key !== "Delete" && e.key !== "Backspace") || e.target.closest?.(TYPING)) return;
            const {id, n, j} = picked;
            const all = characters(id);
            const at = all.findIndex((x) => x.n === n);
            const grid = manualOf(id) === "Grid";
            if (at < 0 || ownPlaces(all[at].input, grid, at, all.length).length < 2) return;
            e.preventDefault();
            track(id);
            removePlace(all[at].input, grid, at, all.length, j);
            unpick();
        });
    }

    // the card buttons and the Boxes / Grid switch are actions too
    function trackCards(id) {
        el(`nai_${id}_chars`)?.addEventListener("click", (e) => {
            if (e.target.closest?.(".nai-add, .nai-remove, .nai-up, .nai-down, .nai-copy, .nai-add-place, .nai-char-on, .nai-manual input, .nai-ref-add")) track(id);
            const n = Number(e.target.closest?.(".nai-char-card")?.id.match(/_char(\d+)$/)?.[1]);
            if (!n) return;
            // her references go where she goes (precise_reference.py's _act)
            if (e.target.closest(".nai-up") && n > 1) prAction(id, {op: "swap", a: n, b: n - 1});
            if (e.target.closest(".nai-down") && n < MAX) prAction(id, {op: "swap", a: n, b: n + 1});
            if (e.target.closest(".nai-remove")) prAction(id, {op: "drop", n});
            if (e.target.closest(".nai-ref-add")) prAction(id, {op: "add", n});
            if (e.target.closest(".nai-copy")) copyRefsAfter(id, n);
            if (e.target.closest(".nai-more-btn")) e.target.closest(".nai-char-card").classList.toggle("nai-more-open");
        }, true);
    }

    // ------------------------------------------------------------------ references on the cards
    const shownCard = (node) => !!node && getComputedStyle(node).display !== "none";
    const refTargets = (id) => [1, 2, 3, 4].map((i) => field(`nai_${id}_pr${i}_for`)?.value ?? "Whole image");
    const refOwner = (value) => Number(/^\s*(?:Character )?(\d+)\s*$/.exec(value || "")?.[1]) || null;

    function prAction(id, action) {
        const input = field(`nai_${id}_pr_action`);
        // no reference for a character (most of the time): nothing to move, no round trip
        if (!input || (action.op !== "add" && action.op !== "targets" && !refTargets(id).some(refOwner))) return;
        setValue(input, JSON.stringify({...action, at: Date.now()}));
    }

    // ⧉ adds the copy through the server; her references follow once the new card shows
    function copyRefsAfter(id, from) {
        const before = new Set([...Array(MAX)].map((_, k) => k + 1).filter((k) => shownCard(el(`nai_${id}_char${k}`))));
        const started = Date.now();
        const look = setInterval(() => {
            const fresh = [...Array(MAX)].map((_, k) => k + 1).find((k) => !before.has(k) && shownCard(el(`nai_${id}_char${k}`)));
            if (fresh || Date.now() - started > 4000) clearInterval(look);
            if (fresh && refTargets(id).some((t) => refOwner(t) === from)) prAction(id, {op: "copy", from, to: fresh});
        }, 150);
    }

    // Each reference card sits where its target says: in that character's Reference tab while
    // her card is shown, else in the References section (one for a character who isn't there
    // says so: it's skipped at generation).
    const homes = {};
    function placeRefs(id) {
        const on = featureOn(id, "pr");
        for (let i = 1; i <= 4; i++) {
            const card = el(`nai_${id}_pr${i}`);
            if (!card) continue;
            if (!homes[`${id}${i}`]) {
                const anchor = document.createComment(`reference ${i}`);
                card.before(anchor);
                homes[`${id}${i}`] = anchor;
            }
            const n = refOwner(field(`nai_${id}_pr${i}_for`)?.value);
            const owner = n && el(`nai_${id}_char${n}`);
            const slot = on && shownCard(owner) ? owner.querySelector(".nai-ref-slot") : null;
            if (slot && card.parentElement !== slot) slot.appendChild(card);
            if (!slot && card.previousSibling !== homes[`${id}${i}`]) homes[`${id}${i}`].after(card);
            const note = n && !slot ? `for Character ${n}, who isn't in the image: skipped` : "";
            if (card.dataset.orphan !== note) card.dataset.orphan = note;
        }
    }

    // ------------------------------------------------------------------ the cards' small things
    const presetsOf = (id) => {
        try {
            return JSON.parse(field(`nai_${id}_chars_presets`)?.value || "{}");
        } catch (e) {
            return {};
        }
    };

    // The name box is the preset search: a native list of every preset under it; picking one
    // (not typing a name that happens to match) replaces the card's text, as one undo step.
    function presetSearch(id) {
        const panel = el(`nai_${id}_chars`);
        if (!panel) return;
        const presets = presetsOf(id);
        let list = panel.querySelector(":scope > datalist");
        if (!list) {
            list = document.createElement("datalist");
            list.id = `nai_${id}_preset_names`;
            panel.appendChild(list);
        }
        // original characters ("Ruby (OC)") first, then A-Z
        const oc = (n) => (/[(,]\s*OC\)$/.test(n) ? 0 : 1);
        const names = Object.keys(presets).sort((a, b) => oc(a) - oc(b) || a.localeCompare(b, undefined, {sensitivity: "base"}));
        const key = JSON.stringify(names);
        if (list.dataset.key !== key) {
            list.dataset.key = key;
            list.replaceChildren(...names.map((name) => Object.assign(document.createElement("option"), {value: name})));
        }
        for (let n = 1; n <= MAX; n++) {
            const card = el(`nai_${id}_char${n}`);
            const input = card?.querySelector(".nai-char-name input");
            if (!input) continue;
            if (input.getAttribute("list") !== list.id) input.setAttribute("list", list.id);
            if (!input.dataset.search) {
                input.dataset.search = "1";
                input.dataset.last = input.value;
                input.addEventListener("focus", () => (input.dataset.last = input.value));
                input.addEventListener("input", (e) => {
                    const preset = presetsOf(id)[input.value];
                    const picked = preset && (!e.inputType || e.inputType === "insertReplacementText");
                    if (picked) {
                        track(id);
                        hist(id).pending.before[`c${n}.name`] = input.dataset.last;
                        setValue(field(`nai_${id}_char${n}_prompt`), preset.prompt || "");
                        setValue(field(`nai_${id}_char${n}_uc`), preset.uc || "");
                    }
                    input.dataset.last = input.value;
                });
            }
        }
    }

    // A dot on Undesired Content when it has text, the count on Reference, and a dot on ⋯ when
    // something in it isn't the default -- so nothing set is out of sight.
    function cardBadges(id) {
        for (let n = 1; n <= MAX; n++) {
            const card = el(`nai_${id}_char${n}`);
            if (!shownCard(card)) continue;
            const uc = (field(`nai_${id}_char${n}_uc`)?.value || "").trim() ? "•" : "";
            const refsShown = [...card.querySelectorAll(".nai-ref-slot > .nai-ref-card")].filter(shownCard).length;
            for (const tab of card.querySelectorAll(".nai-char-tabs button[role=tab], .nai-char-tabs .tab-nav > button")) {
                const text = tab.textContent;
                const badge = text.startsWith("Undesired") ? uc : text.startsWith("Reference") ? (refsShown ? String(refsShown) : "") : null;
                if (badge !== null && tab.dataset.badge !== badge) tab.dataset.badge = badge;
                if (text.startsWith("Reference")) tab.style.display = featureOn(id, "pr") ? "" : "none";
            }
            const face = card.querySelector(".nai-char-face input")?.value || FACE_AUTO;
            card.querySelector(".nai-more-btn")?.classList.toggle("nai-set", face !== FACE_AUTO);
        }
    }

    // Tag Autocomplete (when installed) only finds the main prompt boxes, so the cards' boxes
    // are handed to it once its setup is done (and its CSS is in). It tells boxes apart by an
    // identifier it reads with includes("txt2img"), includes("img2img") and includes("n") for
    // negative, and would call the cards generic third-party boxes: they get identifiers of
    // their own instead (no other "n"), so its txt2img / img2img / negative prompt settings
    // and its positive / negative use counts apply to them as to the main boxes.
    const tacIds = new WeakMap();
    let tacAttached = false;
    function attachAutocomplete() {
        if (tacAttached || !window.TAC_CFG || window.tacLoading) return;
        if (typeof addAutocompleteToArea !== "function" || typeof getTextAreaIdentifier !== "function") return;
        tacAttached = true;
        const identifier = getTextAreaIdentifier;
        window.getTextAreaIdentifier = (area) => tacIds.get(area) ?? identifier(area);
        // every card's boxes exist from page load, hidden or not
        for (const [tab, id] of TABS) {
            for (let n = 1; n <= MAX; n++) {
                for (const kind of ["prompt", "uc"]) {
                    const area = field(`nai_${id}_char${n}_${kind}`);
                    if (!area) continue;
                    tacIds.set(area, `.${tab}-char${n}${kind === "uc" ? "-n" : ""}`);
                    try {
                        addAutocompleteToArea(area);
                    } catch (e) {
                        console.warn("[Stagehand] Tag Autocomplete couldn't attach to a character box", e);
                    }
                }
            }
        }
    }

    // A card's controls only exist once the card is shown, so this runs with the layout loop.
    function tooltips(id) {
        const tips = {
            ".nai-char-on": "Switch this character off to leave it out of the next image without deleting it",
            ".nai-char-name": "Name (optional): shown in the image's PNG info and on its position box; a preset is saved under it. Type to find a preset: picking one replaces this card's text (Ctrl+Z undoes it)",
            ".nai-char-face": "ADetailer: which detected face gets this character's prompt. auto matches them by position",
            ".nai-save-preset": "Save this character as a preset under its name (saving again updates it). Newlines are kept.",
            ".nai-more-btn": "More: which ADetailer face is this character's (a dot: it isn't auto)",
            ".nai-share": "Where two places overlap, they split it in proportion: 70 vs 30 gives 70/30. 50 each by default",
            ".nai-ref-add": "A reference image for this character only: her part of the image, and her face in ADetailer",
            ".nai-up": "Move this character up (earlier = further left in the default columns)",
            ".nai-down": "Move this character down (later = further right in the default columns)",
            ".nai-copy": "Duplicate this character (her references too, while there's room for them)",
            ".nai-add-place": "Another place for this character (the same card in each): multi-angle sheets, complex compositions. × on a place (or click it and press Delete) removes it",
            ".nai-remove": "Delete this character",
            ".nai-remove-ref": "Remove this reference",
            ".nai-ref-hires": "Keep this reference while Hires fix redraws the image. Off (default): only the first pass uses it, which looked best in testing",
            ".nai-ref-adetailer": "Keep this reference while ADetailer repaints faces, e.g. when a face drifts from the reference. A character's reference goes only to her own face",
            [`#nai_${id}_chars_manual`]: "Boxes: drag and resize a box per character. Grid: NovelAI's 5x5 grid, a dot where each character's head goes",
            ".nai-reset": "Everyone back in the default columns, left to right in card order (Ctrl+Z undoes it)",
            ".nai-switch": "Swap two characters' places",
            ".nai-toggle[data-what=canvases]": "Show or hide the small map beside each character",
            ".nai-toggle[data-what=image]": "Show or hide the boxes over the output image (hide them to click the picture)",
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
            tools(id);
            trackCards(id);
            el(`nai_${id}_chars_manual`)?.querySelectorAll("input[type=radio]").forEach((radio) => {
                radio.addEventListener("change", () => radio.checked && convert(id, radio.value));
            });
        }
        undoKeys();
        deleteKey();
        // the Presets panel re-reads the file when opened: the other tab, or another person, may have saved one
        document.addEventListener("click", (e) => {
            const panel = e.target.closest?.(".nai-presets > .label-wrap")?.parentElement;
            if (panel) setTimeout(() => panel.querySelector(".nai-presets-refresh")?.click(), 0);
        });
        setInterval(() => TABS.forEach(([, id]) => settle(id)), 100);
        // Cheap and robust against everything that can change the layout (cards, toggles,
        // a new image, resizing); a drag redraws on its own (redraw).
        setInterval(() => {
            TABS.forEach(([tab, id]) => {
                unmerge(tab, id);
                sync(id);
                placeRefs(id);
                render(tab, id);
                presetSearch(id);
                cardBadges(id);
                tooltips(id);
            });
            attachAutocomplete();
        }, 400);
    });
})();
