"""Build a flat check bundle per page: image + transcript txt + audio.

Output structure:
    out_dir/
      006_013_img_00008.webp   ← page image
      006_013_img_00008.txt    ← all bubbles ordered, speaker + text
      006_013_img_00008.wav    ← page audio
      ...
"""
import argparse, json, shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pages-dir", required=True)
    p.add_argument("--synth-dir", required=True)
    p.add_argument("--cv-json-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(args.cv_json_dir) / "index.json") as f:
        pages_meta = json.load(f)

    synth_pages_dir = Path(args.synth_dir) / "pages"

    for meta in pages_meta:
        page_idx = meta["page_idx"]
        page_name = meta["name"]
        stem = f"{page_idx:03d}_{page_name}"

        # Copy page image
        src_images = list(Path(args.pages_dir).glob(f"{page_name}.*"))
        if src_images:
            shutil.copy(src_images[0], out_dir / f"{stem}{src_images[0].suffix}")

        # Copy page audio
        src_audios = list(synth_pages_dir.glob(f"{stem}.wav"))
        if src_audios:
            shutil.copy(src_audios[0], out_dir / f"{stem}.wav")

        # Build transcript txt
        with open(meta["json"]) as f:
            page_result = json.load(f)
        bubbles = sorted(page_result.get("bubbles", []), key=lambda b: b.get("order", 0))

        lines = [f"=== {stem} ===", f"Total bubbles: {len(bubbles)}", ""]
        for b in bubbles:
            order = b.get("order", 0)
            text = b.get("text", "").strip() or "(empty)"
            char_id = b.get("qwen_speaker") or b.get("speaker_id") or "unknown"
            speaker_desc = (b.get("qwen_speaker_desc") or "").strip()
            desc_part = f" [{speaker_desc}]" if speaker_desc else ""
            lines.append(f"[{order:02d}] {char_id}{desc_part}: \"{text}\"")

        txt_path = out_dir / f"{stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # Build summary index
    index_lines = ["=== Doraemon Check Bundle ===", ""]
    for meta in pages_meta:
        stem = f"{meta['page_idx']:03d}_{meta['name']}"
        index_lines.append(f"  {stem}  ({meta.get('n_bubbles','?')} bubbles)")
    (out_dir / "INDEX.txt").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    print(f"Bundle at: {out_dir}")
    print(f"Per page: <name>.webp, <name>.txt, <name>.wav")


if __name__ == "__main__":
    main()
