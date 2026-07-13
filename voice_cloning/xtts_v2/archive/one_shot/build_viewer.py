"""Build an HTML viewer: comic page + audio + bubble crops + per-bubble audio.

Shows detected bubble images side-by-side with OCR text + character voice assignment
+ per-bubble synthesized audio (individual playback), so user can verify:
  - OCR correctness (compare crop vs text)
  - TTS quality per bubble (isolate hallucination sources)
  - Speaker assignment
"""
import argparse, json, shutil
from pathlib import Path
from PIL import Image


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Comic + XTTS Audio Viewer</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 20px; }}
  h1 {{ margin: 0 0 20px 0; }}
  .page {{ display: flex; gap: 24px; margin-bottom: 40px; padding: 16px; background: #2a2a2a; border-radius: 8px; }}
  .page-img {{ flex: 0 0 36%; position: sticky; top: 20px; align-self: flex-start; }}
  .page-img img {{ width: 100%; border-radius: 4px; }}
  .page-info {{ flex: 1; display: flex; flex-direction: column; gap: 12px; }}
  .page-info h2 {{ margin: 0; color: #4da6ff; }}
  .page-audio {{ background: #333; padding: 10px; border-radius: 4px; }}
  audio {{ width: 100%; }}
  .bubble-list {{ font-size: 13px; line-height: 1.5; }}
  .bubble {{
    display: grid;
    grid-template-columns: 150px 1fr;
    gap: 12px;
    padding: 8px;
    margin: 8px 0;
    background: #333;
    border-radius: 4px;
    border-left: 3px solid #4da6ff;
    align-items: center;
  }}
  .bubble .crop {{
    max-width: 150px;
    max-height: 120px;
    object-fit: contain;
    background: #000;
    border-radius: 3px;
  }}
  .bubble-right {{ display: flex; flex-direction: column; gap: 4px; }}
  .bubble .meta-row {{ display: flex; gap: 8px; align-items: center; }}
  .bubble .char-id {{
    display: inline-block; padding: 2px 8px; background: #4da6ff;
    color: white; border-radius: 3px; font-size: 11px; font-weight: 600;
  }}
  .bubble.unknown .char-id {{ background: #888; }}
  .bubble.sound_effect .char-id {{ background: #e67e22; }}
  .bubble .order {{ color: #666; font-size: 11px; font-family: monospace; }}
  .bubble .text {{
    color: #ddd; background: #222; padding: 4px 8px; border-radius: 3px;
    font-family: monospace; font-size: 13px;
  }}
  .bubble.empty {{ opacity: 0.4; }}
  .bubble audio {{ width: 100%; height: 30px; }}
  .meta {{ color: #888; font-size: 12px; }}
</style>
</head>
<body>
<h1>🎬 Comic + XTTS FT Audio Viewer</h1>
<p class="meta">Model: XTTS fine-tuned on VieNeu-TTS-140h (heldout cos_sim 0.71). Bubble crops + individual audio per bubble — isolate OCR vs TTS issues.</p>
{pages_html}
<script>
document.querySelectorAll('audio').forEach(a => {{
  a.addEventListener('play', () => {{
    document.querySelectorAll('audio').forEach(o => {{ if (o !== a) o.pause(); }});
  }});
}});
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pages-dir", required=True, help="Original comic images")
    p.add_argument("--synth-dir", required=True, help="XTTS output dir (pages/*.wav + bubbles/*.wav)")
    p.add_argument("--cv-json-dir", required=True, help="CV JSON dir (has index.json)")
    p.add_argument("--out", required=True, help="Output HTML viewer dir")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "crops").mkdir(exist_ok=True)
    (out_dir / "audio").mkdir(exist_ok=True)
    (out_dir / "bubble_audio").mkdir(exist_ok=True)

    with open(Path(args.cv_json_dir) / "index.json") as f:
        pages_meta = json.load(f)

    synth_pages_dir = Path(args.synth_dir) / "pages"
    synth_bubbles_dir = Path(args.synth_dir) / "bubbles"
    pages_html = []

    for meta in pages_meta:
        page_idx = meta["page_idx"]
        page_name = meta["name"]

        src_images = list(Path(args.pages_dir).glob(f"{page_name}.*"))
        if not src_images:
            continue
        src_img_path = src_images[0]
        dst_img_name = f"{page_idx:03d}_{page_name}{src_img_path.suffix}"
        dst_img = out_dir / "images" / dst_img_name
        if not dst_img.exists():
            shutil.copy(src_img_path, dst_img)

        src_audios = list(synth_pages_dir.glob(f"{page_idx:03d}_{page_name}.wav"))
        if not src_audios:
            continue
        dst_audio_name = f"{page_idx:03d}_{page_name}.wav"
        dst_audio = out_dir / "audio" / dst_audio_name
        if not dst_audio.exists():
            shutil.copy(src_audios[0], dst_audio)

        with open(meta["json"]) as f:
            page_result = json.load(f)
        bubbles = sorted(page_result.get("bubbles", []), key=lambda b: b.get("order", 0))

        # Open page image for cropping
        img = Image.open(src_img_path)

        bubble_items = []
        for b in bubbles:
            order = b.get("order", 0)
            text = b.get("text", "").strip()
            bbox = b.get("bbox", [0, 0, 0, 0])
            char_id = b.get("qwen_speaker") or b.get("speaker_id") or "unknown"

            # Save bubble crop
            crop_name = f"{page_idx:03d}_{page_name}_b{order:03d}.png"
            crop_path = out_dir / "crops" / crop_name
            if bbox and len(bbox) == 4 and not crop_path.exists():
                try:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    crop = img.crop((x1, y1, x2, y2))
                    crop.save(crop_path)
                except Exception:
                    crop_path = None

            # Copy bubble audio — match by prefix (speaker suffix varies)
            bubble_wav_pattern = f"{page_idx:03d}_{page_name}_b{order:03d}_*.wav"
            bubble_wavs = list(synth_bubbles_dir.glob(bubble_wav_pattern))
            bubble_audio_rel = None
            if bubble_wavs:
                bubble_audio_name = f"{page_idx:03d}_{page_name}_b{order:03d}.wav"
                dst_b_audio = out_dir / "bubble_audio" / bubble_audio_name
                if not dst_b_audio.exists():
                    shutil.copy(bubble_wavs[0], dst_b_audio)
                bubble_audio_rel = f"bubble_audio/{bubble_audio_name}"

            css_class = "bubble"
            if not text:
                css_class += " empty"
            if char_id == "unknown":
                css_class += " unknown"
            elif char_id == "sound_effect":
                css_class += " sound_effect"

            crop_html = (f'<img class="crop" src="crops/{crop_name}" alt="bubble crop">'
                         if crop_path else '<div class="crop" style="background:#444"></div>')
            audio_html = (f'<audio controls preload="none"><source src="{bubble_audio_rel}" type="audio/wav"></audio>'
                          if bubble_audio_rel else '')

            bubble_items.append(f'''
<div class="{css_class}">
  {crop_html}
  <div class="bubble-right">
    <div class="meta-row">
      <span class="order">#{order:02d}</span>
      <span class="char-id">{char_id}</span>
    </div>
    <div class="text">{text or "<em>(empty)</em>"}</div>
    {audio_html}
  </div>
</div>''')

        page_html = f"""
<div class="page">
  <div class="page-img">
    <img src="images/{dst_img_name}" alt="Page {page_idx}">
    <div class="page-audio" style="margin-top:8px">
      <div class="meta" style="margin-bottom:4px">Full page audio</div>
      <audio controls preload="metadata"><source src="audio/{dst_audio_name}" type="audio/wav"></audio>
    </div>
  </div>
  <div class="page-info">
    <h2>Page {page_idx}: {page_name}</h2>
    <div class="meta">{len(bubbles)} bubbles detected</div>
    <div class="bubble-list">
      {"".join(bubble_items)}
    </div>
  </div>
</div>
"""
        pages_html.append(page_html)

    html = HTML_TEMPLATE.format(pages_html="\n".join(pages_html))
    out_html = out_dir / "index.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Viewer: file://{out_html.absolute()}")
    print(f"Pages: {len(pages_html)}")


if __name__ == "__main__":
    main()
