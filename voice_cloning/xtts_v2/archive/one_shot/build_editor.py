"""Tier 3: Interactive HTML editor for manual OCR correction.

Shows per-page: image + bubble crops + editable text fields. User can:
  - Review crop + text side-by-side
  - Edit text inline
  - Mark bubble as delete (typo detection)
  - Export corrected JSON via download button

Usage: Open index.html in browser → edit → click "Export JSON" for each page →
       save downloaded JSON over original in the CV output dir → re-run synth.
"""
import argparse, json, shutil
from pathlib import Path
from PIL import Image


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>OCR Manual Editor — 3-Tier Tier 3</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 20px; }}
  h1 {{ margin: 0 0 10px 0; }}
  .help {{ color: #aaa; font-size: 13px; margin-bottom: 20px; }}
  .page {{ display: flex; gap: 24px; margin-bottom: 40px; padding: 16px; background: #2a2a2a; border-radius: 8px; }}
  .page-img {{ flex: 0 0 36%; position: sticky; top: 20px; align-self: flex-start; }}
  .page-img img {{ width: 100%; border-radius: 4px; }}
  .page-info {{ flex: 1; display: flex; flex-direction: column; gap: 10px; }}
  .page-info h2 {{ margin: 0; color: #4da6ff; display: flex; justify-content: space-between; align-items: center; }}
  .export-btn {{
    background: #4caf50; color: white; border: none; padding: 8px 16px;
    border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: 600;
  }}
  .export-btn:hover {{ background: #45a049; }}
  .bubble {{
    display: grid;
    grid-template-columns: 150px 1fr;
    gap: 12px;
    padding: 8px;
    margin: 6px 0;
    background: #333;
    border-radius: 4px;
    border-left: 3px solid #4da6ff;
    align-items: center;
  }}
  .bubble .crop {{
    max-width: 150px; max-height: 120px; object-fit: contain;
    background: #000; border-radius: 3px;
  }}
  .bubble-right {{ display: flex; flex-direction: column; gap: 4px; }}
  .meta-row {{ display: flex; gap: 8px; align-items: center; font-size: 12px; }}
  .char-id {{
    padding: 2px 8px; background: #4da6ff; color: white;
    border-radius: 3px; font-weight: 600;
  }}
  .bubble.unknown .char-id {{ background: #888; }}
  .bubble.sound_effect .char-id {{ background: #e67e22; }}
  .order {{ color: #999; font-family: monospace; }}
  .text-input {{
    width: 100%; background: #222; color: #fff; border: 1px solid #555;
    padding: 6px 10px; border-radius: 3px; font-family: monospace;
    font-size: 14px; box-sizing: border-box; resize: vertical; min-height: 36px;
  }}
  .text-input:focus {{ border-color: #4da6ff; outline: none; }}
  .text-input.edited {{ border-color: #ffc107; background: #2a2a1a; }}
  .pass1 {{ color: #888; font-size: 11px; font-style: italic; }}
  .delete-btn {{
    background: #c62828; color: white; border: none; padding: 4px 8px;
    border-radius: 3px; cursor: pointer; font-size: 11px;
  }}
  .bubble.deleted {{ opacity: 0.3; text-decoration: line-through; }}
  .summary {{ background: #333; padding: 10px; border-radius: 4px; font-size: 12px; color: #aaa; }}
</style>
</head>
<body>
<h1>✏️ OCR Manual Editor (Tier 3)</h1>
<p class="help">Edit text directly in each field. When done with a page, click <b>Export JSON</b> to download corrected JSON. Save to <code>{out_json_dir}</code> (overwrite original) then re-run synth.</p>
{pages_html}
<script>
// Track edits per page
function markEdited(el) {{
  el.classList.add('edited');
}}
function deleteBubble(btn) {{
  const bubble = btn.closest('.bubble');
  bubble.classList.toggle('deleted');
  bubble.dataset.deleted = bubble.classList.contains('deleted');
}}

function exportPage(pageId) {{
  const bubbles = document.querySelectorAll(`.page[data-page-id="${{pageId}}"] .bubble`);
  const out = [];
  bubbles.forEach(b => {{
    if (b.dataset.deleted === 'true') return;
    const txt = b.querySelector('.text-input').value.trim();
    const originalData = JSON.parse(b.dataset.original);
    originalData.text = txt;
    out.push(originalData);
  }});
  const meta = JSON.parse(document.querySelector(`.page[data-page-id="${{pageId}}"]`).dataset.meta);
  meta.bubbles = out;
  const blob = new Blob([JSON.stringify(meta, null, 2)], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = pageId + '.json';
  a.click();
}}

// Auto-resize textareas
document.querySelectorAll('.text-input').forEach(el => {{
  el.addEventListener('input', e => {{
    markEdited(e.target);
    e.target.style.height = 'auto';
    e.target.style.height = e.target.scrollHeight + 'px';
  }});
  // Initial size
  el.style.height = 'auto';
  el.style.height = el.scrollHeight + 'px';
}});
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pages-dir", required=True)
    p.add_argument("--cv-json-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "crops").mkdir(exist_ok=True)

    with open(Path(args.cv_json_dir) / "index.json") as f:
        pages_meta = json.load(f)

    pages_html = []
    for meta in pages_meta:
        page_idx = meta["page_idx"]
        page_name = meta["name"]
        page_id = f"{page_idx:03d}_{page_name}"

        # Copy page image
        src_images = list(Path(args.pages_dir).glob(f"{page_name}.*"))
        if not src_images:
            continue
        src_img_path = src_images[0]
        dst_img_name = f"{page_id}{src_img_path.suffix}"
        dst_img = out_dir / "images" / dst_img_name
        if not dst_img.exists():
            shutil.copy(src_img_path, dst_img)

        with open(meta["json"]) as f:
            page_result = json.load(f)
        bubbles = sorted(page_result.get("bubbles", []), key=lambda b: b.get("order", 0))

        img = Image.open(src_img_path)

        bubble_items = []
        for b in bubbles:
            order = b.get("order", 0)
            text = b.get("text", "").strip()
            pass1_text = b.get("text_pass1")  # from Tier 2
            bbox = b.get("bbox", [0, 0, 0, 0])
            char_id = b.get("qwen_speaker") or b.get("speaker_id") or "unknown"

            crop_name = f"{page_id}_b{order:03d}.png"
            crop_path = out_dir / "crops" / crop_name
            if bbox and len(bbox) == 4 and not crop_path.exists():
                try:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    pad = 5
                    x1 = max(0, x1-pad); y1 = max(0, y1-pad)
                    x2 = min(img.width, x2+pad); y2 = min(img.height, y2+pad)
                    img.crop((x1, y1, x2, y2)).save(crop_path)
                except Exception:
                    pass

            css_class = "bubble"
            if char_id == "unknown":
                css_class += " unknown"
            elif char_id == "sound_effect":
                css_class += " sound_effect"

            original_data = json.dumps(b, ensure_ascii=False).replace('"', "&quot;")
            pass1_html = f'<div class="pass1">pass1: {pass1_text.replace(chr(34), "&quot;")}</div>' if pass1_text and pass1_text != text else ""
            bubble_items.append(f'''
<div class="{css_class}" data-original="{original_data}">
  <img class="crop" src="crops/{crop_name}" alt="bubble">
  <div class="bubble-right">
    <div class="meta-row">
      <span class="order">#{order:02d}</span>
      <span class="char-id">{char_id}</span>
      <button class="delete-btn" onclick="deleteBubble(this)">Delete</button>
    </div>
    <textarea class="text-input" rows="2">{text.replace('<', '&lt;').replace('>', '&gt;')}</textarea>
    {pass1_html}
  </div>
</div>''')

        page_meta_for_js = json.dumps({
            "image": page_result.get("image"),
            "size": page_result.get("size"),
        }, ensure_ascii=False).replace('"', "&quot;")

        page_html = f"""
<div class="page" data-page-id="{page_id}" data-meta="{page_meta_for_js}">
  <div class="page-img"><img src="images/{dst_img_name}" alt="Page"></div>
  <div class="page-info">
    <h2>
      <span>Page {page_idx}: {page_name}</span>
      <button class="export-btn" onclick="exportPage('{page_id}')">💾 Export JSON</button>
    </h2>
    <div class="summary">{len(bubbles)} bubbles. Edit text below, then click Export to download corrected JSON.</div>
    {"".join(bubble_items)}
  </div>
</div>"""
        pages_html.append(page_html)

    html = HTML_TEMPLATE.format(
        pages_html="\n".join(pages_html),
        out_json_dir=args.cv_json_dir,
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"Editor at: file://{(out_dir/'index.html').absolute()}")
    print(f"Pages: {len(pages_html)}")


if __name__ == "__main__":
    main()
