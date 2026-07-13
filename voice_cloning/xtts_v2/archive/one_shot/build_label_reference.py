"""Build label reference folder — examples of good labeling format.

Picks representative pages across all 5 series. For each page includes:
  - page.webp       : comic page image
  - page.txt        : labels (id + prefill + suggested_correct + skip notes)
  - page.wav        : per-page audio synthesized with current pipeline

User reviews this to understand what "good label" looks like before starting.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="/mnt/nfs-data/tin_dataset/comic/labeling_pool")
    p.add_argument("--out", default="/home/bes/Desktop/Tin/label_reference")
    p.add_argument("--audio-src",
                   default="/home/bes/Desktop/Tin/demo_audio/doraemon_rules_xtts/pages",
                   help="Source for Doraemon audio if available")
    p.add_argument("--pages-per-series", type=int, default=2)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load rule corrector to suggest better text
    sys.path.insert(0, "/home/bes/Desktop/Tin/Capstone_project/voice_cloning/xtts_v2")
    from vn_correct_rules import correct as rule_correct

    with open(Path(args.pool) / "index.json") as f:
        pages_meta = json.load(f)

    # Group by series
    by_series = {}
    for m in pages_meta:
        by_series.setdefault(m["series"], []).append(m)

    # Pick middle pages (skip covers) per series
    picks = []
    for series, pages in by_series.items():
        # Prefer pages with 5-15 bubbles (meaningful content)
        ranked = sorted(
            [p for p in pages if 5 <= p.get("n_bubbles", 0) <= 15],
            key=lambda p: p["page_idx"],
        )
        # Pick 2 from middle of chapter
        if len(ranked) >= args.pages_per_series:
            start = len(ranked) // 3
            picks.extend(ranked[start : start + args.pages_per_series])
        else:
            picks.extend(ranked)

    print(f"Building reference with {len(picks)} pages ({args.pages_per_series}/series)")

    audio_src_dir = Path(args.audio_src)

    for idx, meta in enumerate(picks, start=1):
        series = meta["series"]
        page_name = meta["name"]
        stem = f"{idx:02d}_{series}_{page_name}"

        # Copy page image
        with open(meta["json"]) as f:
            page = json.load(f)
        img_src = Path(page["image"])
        shutil.copy(img_src, out_dir / f"{stem}{img_src.suffix}")

        # Build labels txt
        lines = [f"=== {stem} ===", f"Series: {series}  Page: {page_name}",
                 f"Total bubbles detected: {len(page.get('bubbles', []))}",
                 "", "Format: [id] BBOX → prefill (Qwen-VL raw) → suggested_correct", ""]
        for b in sorted(page.get("bubbles", []), key=lambda x: x.get("order", 0)):
            order = b["order"]
            prefill = b.get("text_prefill", "").strip()
            # Apply rule correction as "suggested correct"
            suggested, applied = rule_correct(prefill) if prefill else ("", [])
            is_suspicious = len(prefill) <= 2 or not any(c.isalpha() for c in prefill)
            skip_note = " [SKIP? watermark/noise]" if is_suspicious else ""
            changed = " ⭐" if suggested != prefill else ""
            lines.append(f"[{order:02d}]  prefill:    {prefill!r}")
            lines.append(f"      suggested: {suggested!r}{changed}{skip_note}")
            if applied:
                lines.append(f"      rules:     {', '.join(applied[:3])}")
            lines.append("")
        (out_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        # Copy audio if doraemon (we have rules synth for doraemon only)
        # For other series audio hasn't been generated yet — note in README
        if series == "doraemon":
            # Our doraemon_rules_xtts was on pages skip_pages=4, max=10
            # Page mapping: idx 1-10 maps to 008-017_img_00003..012
            # Find matching audio by name
            audio_candidates = list(audio_src_dir.glob(f"*{page_name}*.wav"))
            if audio_candidates:
                shutil.copy(audio_candidates[0], out_dir / f"{stem}.wav")

    # README
    readme = """# Label Reference Examples

Mục đích: cho bạn xem format label CHUẨN trước khi bắt đầu label 1600 bubbles trong CSV.

## Files per page

- `XX_series_page.webp` — trang comic gốc
- `XX_series_page.txt`  — labels với 3 layer:
  - `prefill`   : Text Qwen-VL đọc tự động (có thể sai)
  - `suggested` : Text đã qua rule-correction (tham khảo — có thể vẫn chưa hoàn hảo)
  - `⭐`         : Dòng này có correction so với prefill
  - `[SKIP?]`   : Gợi ý skip nếu là watermark/noise (không phải thoại thật)
- `XX_doraemon_page.wav` — audio trang đó (chỉ Doraemon có sẵn, các series khác chưa synth)

## Cách dùng

1. **Nhìn ảnh** page để hiểu context
2. **Đọc text** prefill và suggested — so sánh
3. **Nghe audio** (Doraemon) để biết tone đọc
4. **Hiểu khi nào skip**: bubble detect sai → crop không phải text thoại thật

## Mapping sang CSV label task của bạn

Khi label file CSV chính:
- Column `prefill` = như prefill trong txt này
- Column `corrected_text` = giống suggested nếu bạn đồng ý, hoặc tự sửa
- Column `skip` = đánh 'x' nếu là [SKIP?] noise

## Chất lượng label mục tiêu

- Chính xác dấu thanh điệu (sắc/huyền/hỏi/ngã/nặng)
- Đủ dấu chữ Ă Â Đ Ê Ô Ơ Ư
- Giữ UPPERCASE nếu comic dùng hoa
- Skip các crop không phải text (objects, backgrounds)
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"\n✓ Reference folder: {out_dir}")
    print(f"  Total files: {len(list(out_dir.iterdir()))}")


if __name__ == "__main__":
    main()
