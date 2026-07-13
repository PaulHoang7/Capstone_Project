"""Labeling UI — Flask server with pre-filled text + keyboard shortcuts.

Workflow:
- Start: `python labeling_server.py --pool <pre_ocr_dir> --port 8080`
- Browser: http://localhost:8080
- For each bubble: see crop + pre-filled text → edit → Ctrl+Enter = save+next
- Auto-saves progress every edit → JSONL file
- Supports resume: progress persists across sessions

Target workflow: 15-20s per bubble with pre-fill.
"""
import argparse
import json
import re
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, request, send_file, render_template_string
from PIL import Image


app = Flask(__name__)
LOCK = Lock()

POOL_DIR: Path = None
OUTPUT_JSONL: Path = None
STATE = {"pages_index": [], "labeled_ids": set()}


HTML_UI = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>VN Manga OCR Labeler</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 0; }
  #app { display: flex; min-height: 100vh; }
  #left { flex: 0 0 40%; padding: 16px; background: #222; }
  #left img { width: 100%; border-radius: 4px; }
  #right { flex: 1; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  h2 { margin: 0 0 10px 0; color: #4da6ff; font-size: 18px; }
  #status { background: #333; padding: 12px; border-radius: 4px; font-size: 14px; }
  #status.error { background: #5a1a1a; color: #ff8888; }
  .crop { max-width: 100%; max-height: 240px; background: #000; border-radius: 4px; object-fit: contain; display: block; }
  .bubble-meta { color: #aaa; font-size: 13px; display: flex; justify-content: space-between; padding: 4px 0; }
  textarea {
    width: 100%; background: #111; color: #fff; border: 2px solid #555;
    padding: 12px; font-family: monospace; font-size: 16px;
    border-radius: 4px; resize: vertical; min-height: 80px;
  }
  textarea:focus { border-color: #4da6ff; outline: none; }
  textarea.changed { border-color: #ffc107; }
  .controls { display: flex; gap: 8px; flex-wrap: wrap; }
  button {
    padding: 10px 16px; border: none; border-radius: 4px;
    font-weight: 600; cursor: pointer; font-size: 14px;
  }
  .btn-save { background: #4caf50; color: white; }
  .btn-skip { background: #757575; color: white; }
  .btn-prev { background: #444; color: white; }
  .hint { color: #888; font-size: 11px; margin-top: 4px; }
  .progress { background: #333; height: 6px; border-radius: 3px; overflow: hidden; }
  .progress-fill { background: linear-gradient(90deg, #4caf50, #4da6ff); height: 100%; transition: width 0.3s; }
  #prefill-row { font-size: 12px; color: #ff9800; padding: 6px 10px; background: #2a1a00; border-radius: 3px; }
</style>
</head>
<body>
<div id="app">
  <div id="left">
    <h2>🖼 Page</h2>
    <img id="page-img" src="" alt="Page loading...">
  </div>
  <div id="right">
    <div id="status">⏳ Loading bubble data... (1717 bubbles)</div>
    <div class="progress"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
    <div class="bubble-meta">
      <span id="series-info">—</span>
      <span id="bubble-info">—</span>
    </div>
    <img class="crop" id="crop" src="" alt="Bubble crop">
    <div id="prefill-row"><b>Pre-fill:</b> <span id="prefill-text">—</span></div>
    <textarea id="text-input" placeholder="Edit text here..." autofocus></textarea>
    <div class="controls">
      <button class="btn-prev" onclick="prevBubble()">◀ Previous</button>
      <button class="btn-save" onclick="saveAndNext()">💾 Save + Next (Ctrl+Enter)</button>
      <button class="btn-skip" onclick="skipAndNext()">⏭ Skip</button>
    </div>
    <div class="hint">Ctrl+Enter = save+next • Ctrl+S = skip • Ctrl+← = prev</div>
  </div>
</div>

<script>
let state = {bubbles: [], idx: 0, total: 0, labeled: 0};
let originalText = "";

function setStatus(msg, isError) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = isError ? 'error' : '';
}

async function loadState() {
  try {
    const r = await fetch('/api/state');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    state.bubbles = data.bubbles || [];
    if (state.idx === 0 || state.idx === undefined) state.idx = data.idx || 0;
    state.total = data.total || 0;
    state.labeled = data.labeled || 0;
    if (!state.bubbles.length) {
      setStatus('❌ No bubbles in pool');
      return;
    }
    renderCurrent();
  } catch (e) {
    setStatus('❌ Failed to load: ' + e.message, true);
    console.error(e);
  }
}

function renderCurrent() {
  const b = state.bubbles[state.idx];
  if (!b) { setStatus('🎉 All done!'); return; }
  document.getElementById('page-img').src = '/api/page-image?path=' + encodeURIComponent(b.image);
  document.getElementById('crop').src = '/api/bubble-crop?page_id=' + encodeURIComponent(b.page_id) + '&order=' + b.order;
  document.getElementById('series-info').textContent = (b.series || '') + ' · ' + (b.page_name || '');
  document.getElementById('bubble-info').textContent = `bubble #${b.order} · ${state.idx+1}/${state.total}`;
  document.getElementById('prefill-text').textContent = b.text_prefill || '(empty)';
  const ti = document.getElementById('text-input');
  ti.value = b.text_labeled != null ? b.text_labeled : (b.text_prefill || '');
  ti.classList.remove('changed');
  originalText = ti.value;
  ti.focus(); ti.select();
  const pct = state.total ? (state.labeled / state.total * 100).toFixed(1) : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
  setStatus(`Labeled ${state.labeled}/${state.total} (${pct}%)`);
}

async function saveAndNext(skip) {
  const b = state.bubbles[state.idx];
  if (!b) return;
  const text = document.getElementById('text-input').value.trim();
  try {
    const r = await fetch('/api/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({page_id: b.page_id, order: b.order, text: text, skipped: !!skip})
    });
    if (!r.ok) throw new Error('Save failed: HTTP ' + r.status);
  } catch (e) {
    setStatus('❌ Save error: ' + e.message, true);
    return;
  }
  state.labeled++;
  state.bubbles[state.idx].text_labeled = text;
  state.bubbles[state.idx].skipped = !!skip;
  if (state.idx < state.bubbles.length - 1) state.idx++;
  else { setStatus('🎉 Done all ' + state.total + ' bubbles!'); return; }
  renderCurrent();
}

function skipAndNext() { saveAndNext(true); }
function prevBubble() { if (state.idx > 0) { state.idx--; renderCurrent(); } }

document.addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); saveAndNext(false); }
  else if (e.ctrlKey && e.key === 's') { e.preventDefault(); skipAndNext(); }
  else if (e.ctrlKey && e.key === 'ArrowLeft') { e.preventDefault(); prevBubble(); }
});

document.getElementById('text-input').addEventListener('input', e => {
  if (e.target.value !== originalText) e.target.classList.add('changed');
  else e.target.classList.remove('changed');
});

loadState();
</script>
</body>
</html>
"""


def load_pool():
    """Load all bubbles from pool into a flat list with metadata."""
    bubbles = []
    with open(POOL_DIR / "index.json") as f:
        pages = json.load(f)

    # Also load existing labels from JSONL
    labeled_map = {}
    if OUTPUT_JSONL.exists():
        with open(OUTPUT_JSONL) as f:
            for line in f:
                item = json.loads(line)
                labeled_map[(item["page_id"], item["order"])] = item

    for page_meta in pages:
        page_id = f"{page_meta['series']}_{page_meta['page_idx']:03d}_{page_meta['name']}"
        with open(page_meta["json"]) as f:
            page = json.load(f)
        for b in page.get("bubbles", []):
            key = (page_id, b["order"])
            labeled = labeled_map.get(key)
            bubbles.append({
                "page_id": page_id,
                "series": page_meta["series"],
                "page_name": page_meta["name"],
                "image": page.get("image"),
                "order": b["order"],
                "bbox": b["bbox"],
                "text_prefill": b["text_prefill"],
                "text_labeled": labeled["text"] if labeled else None,
                "skipped": labeled["skipped"] if labeled else False,
            })
    STATE["pages_index"] = pages
    return bubbles


@app.route("/")
def home():
    return HTML_UI


@app.route("/api/state")
def api_state():
    bubbles = load_pool()
    labeled = sum(1 for b in bubbles if b["text_labeled"] is not None or b["skipped"])
    # Find next unlabeled
    idx = 0
    for i, b in enumerate(bubbles):
        if b["text_labeled"] is None and not b["skipped"]:
            idx = i
            break
    return jsonify({
        "bubbles": bubbles,
        "idx": idx,
        "total": len(bubbles),
        "labeled": labeled,
    })


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json()
    with LOCK:
        # Append or update in JSONL (simple approach: rewrite entire file)
        existing = {}
        if OUTPUT_JSONL.exists():
            with open(OUTPUT_JSONL) as f:
                for line in f:
                    item = json.loads(line)
                    existing[(item["page_id"], item["order"])] = item
        existing[(data["page_id"], data["order"])] = data
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for key, item in existing.items():
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return jsonify({"ok": True})


@app.route("/api/page-image")
def api_page_image():
    path = request.args.get("path")
    if not path or not Path(path).exists():
        return "Not found", 404
    return send_file(path)


@app.route("/api/bubble-crop")
def api_bubble_crop():
    page_id = request.args.get("page_id")
    order = int(request.args.get("order"))

    # Find page in index
    bubbles = load_pool()
    target = next((b for b in bubbles if b["page_id"] == page_id and b["order"] == order), None)
    if not target:
        return "Not found", 404

    from io import BytesIO
    img = Image.open(target["image"])
    x1, y1, x2, y2 = target["bbox"]
    pad = 6
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(img.width, x2 + pad); y2 = min(img.height, y2 + pad)
    crop = img.crop((x1, y1, x2, y2))
    # Upscale small crops for readability
    if crop.width < 400:
        scale = 400 / crop.width
        new_w = int(crop.width * scale); new_h = int(crop.height * scale)
        crop = crop.resize((new_w, new_h), Image.LANCZOS)

    buf = BytesIO()
    crop.save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


def main():
    global POOL_DIR, OUTPUT_JSONL
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="/mnt/nfs-data/tin_dataset/comic/labeling_pool",
                   help="Pre-OCR pool directory (output of preocr_all_series.py)")
    p.add_argument("--output", default="/mnt/nfs-data/tin_dataset/comic/labels.jsonl",
                   help="JSONL to save labels")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    POOL_DIR = Path(args.pool)
    OUTPUT_JSONL = Path(args.output)
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    print(f"Pool: {POOL_DIR}")
    print(f"Labels: {OUTPUT_JSONL}")
    print(f"Server: http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
