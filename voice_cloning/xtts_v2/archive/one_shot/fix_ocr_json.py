"""Post-process CV JSONs: Vietnamese spell correction + dedupe duplicate bubbles.

Pipeline:
  raw CV JSON  ──→  VN spell correction (bmd1905)  ──→  dedupe  ──→  fixed JSON

Usage:
    python fix_ocr_json.py --in <raw cv json dir> --out <fixed json dir>
"""
import argparse, json, shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="bmd1905/vietnamese-correction-v2")
    p.add_argument("--no-correct", action="store_true", help="Skip correction, only dedupe")
    p.add_argument("--no-dedupe",  action="store_true", help="Skip dedupe, only correct")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load correction model
    corrector = None
    if not args.no_correct:
        print(f"Loading VN correction model: {args.model}")
        from transformers import pipeline
        corrector = pipeline("text2text-generation", model=args.model, device=0, max_length=256)

    def correct_text(text: str) -> str:
        if not corrector or not text.strip():
            return text
        # Model was trained on lowercase-like inputs. Keep original case.
        original_upper = text.isupper()
        inp = text.lower() if original_upper else text
        try:
            out = corrector(inp, max_length=256, num_beams=4)
            corrected = out[0]["generated_text"].strip()
            # If model returned empty, fallback to original
            if not corrected:
                return text
            # Restore UPPERCASE if original was
            if original_upper:
                corrected = corrected.upper()
            return corrected
        except Exception as e:
            print(f"  correction failed: {e}")
            return text

    # Read index
    with open(in_dir / "index.json") as f:
        pages_meta = json.load(f)

    new_pages_meta = []
    for meta in pages_meta:
        in_json = Path(meta["json"])
        out_json = out_dir / in_json.name
        with open(in_json) as f:
            page = json.load(f)

        bubbles = page.get("bubbles", [])
        fixed = []
        seen_texts_norm = set()

        for b in sorted(bubbles, key=lambda x: x.get("order", 0)):
            text = (b.get("text") or "").strip()
            if not text:
                fixed.append(b)
                continue

            # Correction
            orig_text = text
            if corrector:
                text = correct_text(text)

            # Dedupe — normalize (lowercase, trim punctuation) for comparison
            if not args.no_dedupe:
                import re
                norm = re.sub(r'[^\w\s]', '', text.lower()).strip()
                if norm in seen_texts_norm:
                    print(f"  [skip dup] order={b.get('order')} text={text!r}")
                    continue
                if len(norm) >= 3:
                    seen_texts_norm.add(norm)

            new_b = dict(b)
            new_b["text"] = text
            if text != orig_text:
                new_b["text_original"] = orig_text
            fixed.append(new_b)

        page["bubbles"] = fixed
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(page, f, ensure_ascii=False, indent=2)

        new_meta = dict(meta)
        new_meta["json"] = str(out_json)
        new_meta["n_bubbles"] = len(fixed)
        new_pages_meta.append(new_meta)

        print(f"{in_json.name}: {len(bubbles)} → {len(fixed)} bubbles")

    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(new_pages_meta, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Fixed JSONs at {out_dir}")


if __name__ == "__main__":
    main()
