"""Rule-based Vietnamese UPPERCASE manga OCR correction.

Deterministic regex substitution dictionary for common Qwen-VL OCR errors.
Safe: only fixes known patterns, never creates new errors.
Covers common tone-mark errors (huyền ↔ nặng, hỏi ↔ ngã, etc.) and
specific words/phrases from manga context.
"""
import re
import json
import argparse
from pathlib import Path


# -----------------------------------------------------------------------------
# Correction rules — ordered by frequency + importance
# Format: (regex_pattern, replacement)
# Uses word boundaries to avoid substring issues.
# -----------------------------------------------------------------------------
CORRECTIONS = [
    # --- CẦU / CẬU confusion (huyền vs nặng) — most common error ---
    (r"\bCẦU LÀ AI\b", "CẬU LÀ AI"),
    (r"\bCẦU NÀY\b", "CẬU NÀY"),
    (r"\bCẦU ẤY\b", "CẬU ẤY"),
    (r"\bCẦU ĐI\b", "CẬU ĐI"),
    (r"\bCẦU MUỐN\b", "CẬU MUỐN"),
    (r"\bCẦU BIẾT\b", "CẬU BIẾT"),
    (r"\bCẦU SẼ\b", "CẬU SẼ"),
    (r"\bCẦU PHẢI\b", "CẬU PHẢI"),
    (r"\bCẦU CÓ\b", "CẬU CÓ"),
    (r"\bC-CẦU\b", "C-CẬU"),
    (r"\bCỦA CẦU\b", "CỦA CẬU"),
    (r"\bCHO CẦU\b", "CHO CẬU"),
    (r"\bTỚI CẦU\b", "TỚI CẬU"),
    (r"\bVỚI CẦU\b", "VỚI CẬU"),
    (r"\bVỀ CẦU\b", "VỀ CẬU"),
    (r"\bCHÚ CẦU\b", "CHÚ CẬU"),

    # --- BẤT / BẬT (sắc vs nặng) ---
    (r"\bBẬT NGỜ\b", "BẤT NGỜ"),
    (r"\bBẬT CẦN\b", "BẤT CẦN"),
    (r"\bBẬT CỨ\b", "BẤT CỨ"),
    (r"\bBẬT CHỢT\b", "BẤT CHỢT"),
    (r"\bBẤT NHẸP\b", "BẬT NHẸP"),  # rare reverse case

    # --- TỪ / TƯ (huyền vs no tone) ---
    (r"\bTƯ ĐẦU\b", "TỪ ĐÂU"),
    (r"\bTƯ ĐẾN\b", "TỪ ĐẾN"),

    # --- ĐÂU / ĐẦU (no tone vs huyền) ---
    (r"\bĐẦU ĐẾN\b", "ĐÂU ĐẾN"),
    (r"\bTỪ ĐẦU ĐẾN\b", "TỪ ĐÂU ĐẾN"),
    (r"\bỞ ĐẦU\b", "Ở ĐÂU"),

    # --- ĐUỔI / ĐUÔI (hook vs no hook) ---
    (r"\bĐEN ĐUÔI\b", "ĐEN ĐỦI"),   # actually "đen đủi" (bad luck)
    (r"\bĐEN ĐỦI\b", "ĐEN ĐỦI"),
    (r"\bĐUỔI THEO\b", "ĐUỔI THEO"),
    (r"\bĐUÔI THEO\b", "ĐUỔI THEO"),

    # --- TỚ / TỐ ---
    (r"\bTỐ ĐẾN\b", "TỚ ĐẾN"),
    (r"\bTỐ LÀ\b", "TỚ LÀ"),
    (r"\bTỐ SẼ\b", "TỚ SẼ"),
    (r"\bTỐ RA\b", "TỚ RA"),

    # --- CẤP ĐỘ / CẤP ĐỒ ---
    (r"\bCẤP ĐỒ\b", "CẤP ĐỘ"),

    # --- CHỨ / CHƯ ---
    (r"\bBIẾT CHƯ\b", "BIẾT CHỨ"),
    (r"\bBIẾT CHÚ\b", "BIẾT CHỨ"),
    (r"\bLÀ CHƯ\b", "LÀ CHỨ"),
    (r"\bMÀ CHƯ\b", "MÀ CHỨ"),
    (r"\bĐƯỢC CHÚ\b", "ĐƯỢC CHỨ"),   # context manga — "được chứ" more common than "được chú"
    (r"\bVẬY CHƯ\b", "VẬY CHỨ"),
    (r"\bCHỨ?!\b", "CHỨ!"),

    # --- PHÁN / PHẢN ---
    (r"\bPHẢN CHỨ\b", "PHÁN CHỨ"),

    # --- ĐIÊN / ĐIỆN (no nặng vs nặng) — context "crazy" ---
    (r"\bĐỒ ĐIỆN\b", "ĐỒ ĐIÊN"),
    (r"\bCON ĐIỆN\b", "CON ĐIÊN"),
    (r"\bTHẰNG ĐIỆN\b", "THẰNG ĐIÊN"),

    # --- ÁC / ẢC ---
    (r"\bẢC MỘNG\b", "ÁC MỘNG"),
    (r"\bẢC ĐỘC\b", "ÁC ĐỘC"),
    (r"\bẢC QUỶ\b", "ÁC QUỶ"),

    # --- RẮC / RÁC ---
    (r"\bRÁC RỐI\b", "RẮC RỐI"),

    # --- LỪNG / LŨNG ---
    (r"\bLŨNG DANH\b", "LỪNG DANH"),
    (r"\bLŨNG LẪY\b", "LỪNG LẪY"),

    # --- HỘC / HỌC ---
    (r"\bHỌC BÀN\b", "HỘC BÀN"),
    (r"\bHỌC TỦ\b", "HỘC TỦ"),

    # --- ĐÙA / ĐUA ---
    (r"\bĐỪNG ĐUA\b", "ĐỪNG ĐÙA"),
    (r"\bĐUA CHÚ\b", "ĐÙA CHÚ"),
    (r"\bĐUA MÀ\b", "ĐÙA MÀ"),
    (r"\bĐUA THÔI\b", "ĐÙA THÔI"),

    # --- ƯỚT / UỐT ---
    (r"\bUỐT NHẸP\b", "ƯỚT NHẸP"),
    (r"\bUỐT ÁT\b", "ƯỚT ÁT"),
    (r"\bUỐT NHOẸT\b", "ƯỚT NHOẸT"),

    # --- TOẸT / TOỆT ---
    (r"\bLÁO TOỆT\b", "LÁO TOẸT"),
    (r"\bKHỐI TOỆT\b", "KHỐI TOẸT"),

    # --- ĐƯƠNG NHIÊN / ĐƯỜNG NHIỆN ---
    (r"\bĐƯỜNG NHIỆN\b", "ĐƯƠNG NHIÊN"),
    (r"\bĐƯỜNG NHIÊN\b", "ĐƯƠNG NHIÊN"),

    # --- DĨ NHIÊN / DỊ NHIÊN ---
    (r"\bDỊ NHIÊN\b", "DĨ NHIÊN"),

    # --- TƯỞNG TƯỢNG ---
    (r"\bTƯỞNG TƯỞNG\b", "TƯỞNG TƯỢNG"),

    # --- CHUA / CHƯA / CHÚC ---
    # Context: manga greetings
    (r"\bCHUA CHẮC\b", "CHƯA CHẮC"),
    (r"\bCHUA MỪNG\b", "CHÚC MỪNG"),

    # --- THẾT / HÉT ---
    (r"\bĐỪNG CÓ THẾT\b", "ĐỪNG CÓ HÉT"),
    (r"\bTHẾT LÊN\b", "HÉT LÊN"),

    # --- Character name normalization (OCR variants) ---
    (r"\bNÔ-BI-TA\b", "NOBITA"),
    (r"\bĐÔ-RÊ-MON\b", "DORAEMON"),
    (r"\bSI-ZU-KA\b", "SHIZUKA"),
    (r"\bSU-NÊ-Ô\b", "SUNEO"),
    (r"\bCHA-I-AN\b", "JAIAN"),

    # --- Common OCR hallucinations (Greek letters, odd chars) ---
    (r"σ[\s\-]*", ""),       # stray Greek sigma
    (r"ơ[\s\-]*Ở", "Ờ-Ở"),   # "ơ-Ở..." → "Ờ-Ở..." stutter

    # --- Extra whitespace cleanup ---
    (r"\s+", " "),
    (r"^\s+|\s+$", ""),
]


def correct(text: str) -> tuple[str, list[str]]:
    """Apply all rules to text. Returns (corrected_text, list_of_applied_rules)."""
    applied = []
    out = text
    for pattern, replacement in CORRECTIONS:
        new_out = re.sub(pattern, replacement, out)
        if new_out != out:
            applied.append(f"{pattern} → {replacement}")
            out = new_out
    return out.strip(), applied


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_dir / "index.json") as f:
        pages_meta = json.load(f)

    total_changes = 0
    new_index = []
    for meta in pages_meta:
        in_json = Path(meta["json"])
        with open(in_json) as f:
            page = json.load(f)

        changes = 0
        for b in page.get("bubbles", []):
            t = (b.get("text") or "").strip()
            if not t:
                continue
            new_t, applied = correct(t)
            if new_t != t:
                b["text_before_rule"] = t
                b["text"] = new_t
                b["rules_applied"] = applied
                changes += 1
                print(f"  {in_json.stem} #{b.get('order'):02d}: {t!r}\n         → {new_t!r}")

        total_changes += changes
        out_json = out_dir / in_json.name
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(page, f, ensure_ascii=False, indent=2)
        new_meta = dict(meta)
        new_meta["json"] = str(out_json)
        new_index.append(new_meta)

    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(new_index, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {total_changes} corrections across {len(pages_meta)} pages.")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
