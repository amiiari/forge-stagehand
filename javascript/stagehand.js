// Stagehand in the browser:
// - both panels (Character Prompts, Precise Reference) go under the prompt boxes, inside one
//   "Stagehand" fold-out that says what's active while it's closed;
// - a prompt that carries characters -- Set Queue's "Restore template" (marked ⟦n⟧), or text
//   pasted from an image's PNG info ("Character 1 (Ruby): ...") -- is split into the cards;
// - with "AI's Choice" off, a box per character is drawn over the output image -- NovelAI V5's
//   positioning, on Forge's gallery. Drag a box to move it, its corner to resize; the result
//   lands in the card's hidden position field as "x0 y0 x1 y1" (fractions of the image).
(() => {
    const TABS = [["txt2img", "t2i"], ["img2img", "i2i"]];
    const MAX = 6;
    let dragging = false;

    const el = (id) => gradioApp().getElementById(id);
    const field = (id) => el(id)?.querySelector("textarea, input");

    function setValue(input, value) {
        if (!input) return;
        input.value = value;
        updateInput(input);
    }

    function characters(id) {
        const out = [];
        for (let n = 1; n <= MAX; n++) {
            const card = el(`nai_${id}_char${n}`);
            // computed style, not offsetParent: a closed fold-out hides cards that still count
            if (!card || getComputedStyle(card).display === "none") continue;
            const on = card.querySelector(".nai-char-on input");
            const prompt = field(`nai_${id}_char${n}_prompt`);
            if ((on && !on.checked) || !prompt || !prompt.value.trim()) continue;
            const name = card.querySelector(".nai-char-name input, .nai-char-name textarea")?.value.trim();
            out.push({n, label: name || `${n}`, input: field(`nai_${id}_char${n}_box`)});
        }
        return out;
    }

    function references(id) {
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
        box.innerHTML = '<summary><span class="nai-stagehand-title">Stagehand</span><span class="nai-stagehand-count"></span></summary>';
        box.addEventListener("toggle", () => {
            try {
                localStorage.setItem(key, box.open ? "1" : "0");
            } catch (e) {}
        });
        negRow.after(box);
        panels.forEach((panel) => box.appendChild(panel));
    }

    function summarize(id) {
        const count = el(`nai_${id}_stagehand`)?.querySelector(".nai-stagehand-count");
        if (!count) return;
        const c = characters(id).length;
        const r = references(id);
        const parts = [];
        if (c) parts.push(`${c} character${c > 1 ? "s" : ""}`);
        if (r) parts.push(`${r} reference${r > 1 ? "s" : ""}`);
        const text = parts.length ? parts.join(" · ") : "off";
        if (count.textContent !== text) count.textContent = text;
        count.classList.toggle("nai-active", parts.length > 0);
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
    const LABEL = /^Character (\d+)(?: \((.*?)\))?(?: at ((?:[\d.]+ ){3}[\d.]+))?:[ \t]?(.*)$/;

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

    function fill(tab, id, input, kind, parsed) {
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
            if (c.box) setValue(field(`nai_${id}_char${n}_box`), c.box);
        }
        if (kind === "prompt" && Object.values(parsed.chars).some((c) => c.box)) {
            const auto = field(`nai_${id}_chars_auto`);
            if (auto?.checked) auto.click();
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
            input?.addEventListener("paste", () => setTimeout(() => {
                const parsed = readLines(input.value);
                if (parsed) fill(tab, id, input, kind, parsed);
            }, 0));
        }
    }

    // ------------------------------------------------------------------ position boxes
    function parse(value) {
        const v = (value || "").trim().split(/[\s,]+/).map(Number);
        return v.length === 4 && v.every((x) => Number.isFinite(x)) && v[2] > v[0] && v[3] > v[1] ? v : null;
    }

    // The canvas the boxes are placed on: always the shape of the NEXT image (the current
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
        const start = parse(input.value) || JSON.parse(div.dataset.box);
        const x0 = event.clientX;
        const y0 = event.clientY;
        const W = overlay.clientWidth;
        const H = overlay.clientHeight;
        let box = start.slice();
        const move = (e) => {
            const dx = (e.clientX - x0) / W;
            const dy = (e.clientY - y0) / H;
            if (mode === "move") {
                const w = start[2] - start[0];
                const h = start[3] - start[1];
                const x = Math.min(Math.max(start[0] + dx, 0), 1 - w);
                const y = Math.min(Math.max(start[1] + dy, 0), 1 - h);
                box = [x, y, x + w, y + h];
            } else {
                box = [start[0], start[1], Math.min(Math.max(start[2] + dx, start[0] + 0.05), 1), Math.min(Math.max(start[3] + dy, start[1] + 0.05), 1)];
            }
            place(div, box);
        };
        const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            dragging = false;
            setValue(input, box.map((v) => v.toFixed(3)).join(" "));
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

    function render(tab, id) {
        if (dragging) return;
        let overlay = el(`nai_${id}_positions`);
        const auto = field(`nai_${id}_chars_auto`);
        const chars = characters(id);
        const f = auto && !auto.checked && chars.length ? frame(tab) : null;
        if (!f) {
            if (overlay) overlay.style.display = "none";
            return;
        }
        if (!overlay || overlay.parentElement !== f.gallery) {
            overlay?.remove();
            overlay = document.createElement("div");
            overlay.id = `nai_${id}_positions`;
            overlay.className = "nai-positions";
            f.gallery.style.position = "relative";
            f.gallery.appendChild(overlay);
        }
        Object.assign(overlay.style, {display: "", left: `${f.left}px`, top: `${f.top}px`, width: `${f.width}px`, height: `${f.height}px`});

        const keep = new Set();
        chars.forEach((c, k) => {
            keep.add(String(c.n));
            let div = overlay.querySelector(`[data-n="${c.n}"]`);
            if (!div) {
                div = document.createElement("div");
                div.dataset.n = c.n;
                div.className = `nai-pos-box nai-c${c.n}`;
                div.innerHTML = '<span class="nai-pos-label"></span><span class="nai-pos-handle"></span>';
                div.addEventListener("pointerdown", (e) => startDrag(e, div, c.input, e.target.classList.contains("nai-pos-handle") ? "resize" : "move", overlay));
                overlay.appendChild(div);
            }
            div.querySelector(".nai-pos-label").textContent = c.label;
            // no saved position yet: AI's Choice columns, which is also what the backend uses
            place(div, parse(c.input?.value) || [k / chars.length, 0, (k + 1) / chars.length, 1]);
        });
        overlay.querySelectorAll(".nai-pos-box").forEach((d) => keep.has(d.dataset.n) || d.remove());
    }

    onUiLoaded(() => {
        for (const [tab, id] of TABS) {
            wrap(tab, id);
            splitOnPaste(tab, id);
            el(`nai_${id}_chars`)?.querySelectorAll(".nai-char-on").forEach((on) => {
                on.title = "Untick to leave this character out of the next image without deleting it";
            });
        }
        // Cheap and robust against everything that can change the layout (cards, toggles,
        // a new image, resizing); boxes aren't re-placed while one is being dragged.
        setInterval(() => TABS.forEach(([tab, id]) => {
            unmerge(tab, id);
            render(tab, id);
            summarize(id);
        }), 400);
    });
})();
