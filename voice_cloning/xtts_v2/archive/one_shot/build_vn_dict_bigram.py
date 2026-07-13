"""Build Vietnamese dictionary + bigram LM for strict OCR correction.

Outputs (saved to /mnt/nfs-data/tin_dataset/ocr_corrector/):
  - vn_words.txt: UPPERCASE VN single-word dictionary (set of valid words)
  - vn_bigrams.json: P(w2|w1) bigram counts from VieNeu + manga corpus
  - manga_whitelist.txt: manga-specific vocabulary (names, SFX)
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


UNDERTHESEA_DICT = "/home/bes/miniconda3/envs/xtts_ft/lib/python3.10/site-packages/underthesea/corpus/data/Viet74K.txt"
VIENEU_TEXT     = "/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/raw_text.tsv"
OUT_DIR         = Path("/mnt/nfs-data/tin_dataset/ocr_corrector")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. VN word dictionary — from Viet74K + VieNeu tokens
# -----------------------------------------------------------------------------
print("[1/3] Building VN word dictionary...")
words = set()

# Load Viet74K — single words + split compounds into their parts
with open(UNDERTHESEA_DICT, encoding="utf-8") as f:
    for line in f:
        w = line.strip()
        if not w:
            continue
        # Keep both compound and its parts
        # Split on spaces + dashes + underscores
        for part in re.split(r"[\s\-_]+", w):
            part = part.strip()
            if part and not part.isdigit():
                words.add(part.upper())

print(f"  From Viet74K (single + split): {len(words)} words")

# Augment from VieNeu corpus — more modern vocab
print("  Extracting tokens from VieNeu corpus...")
token_freq = Counter()
with open(VIENEU_TEXT, encoding="utf-8") as f:
    next(f)  # header
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        text = parts[4]
        # Tokenize: strip punct, keep VN chars + basic latin
        tokens = re.findall(r"[\w\u00C0-\u1EF9]+", text, flags=re.UNICODE)
        for t in tokens:
            token_freq[t.upper()] += 1

# Add tokens that appear ≥5 times (filters out typos/rare proper nouns)
for tok, count in token_freq.items():
    if count >= 5:
        words.add(tok)

print(f"  After VieNeu augment: {len(words)} words")

# -----------------------------------------------------------------------------
# 2. Manga-specific whitelist (names, SFX, common exclamations)
# -----------------------------------------------------------------------------
print("[2/3] Building manga whitelist...")
MANGA_WHITELIST = {
    # Character names
    "NOBITA", "DORAEMON", "SHIZUKA", "SUNEO", "JAIAN", "GIAN", "DORAMI",
    "CONAN", "KUDO", "KUDOU", "SHINICHI", "RAN", "MORI", "AGASA", "HARIBO",
    "GOKU", "VEGETA", "BULMA", "KRILLIN", "PICCOLO", "GOHAN", "TRUNKS",
    "NARUTO", "SASUKE", "SAKURA", "KAKASHI", "HINATA", "ITACHI", "SAI",
    "LUFFY", "ZORO", "NAMI", "SANJI", "USOPP", "CHOPPER", "ROBIN",
    # Sound effects
    "RẦM", "BÙM", "BỤP", "BÒM", "BƯỚC", "CẠCH", "CÁCH", "TA-DA",
    "HAHA", "HIHI", "HỨ", "Ờ", "À", "Á", "Ế", "HÁ", "HẢ", "HẢ!",
    "AA", "AAA", "AAAA", "HÉ", "ỒI", "ÚI", "ÁI", "CHÀ", "CHÁ",
    "UA", "UÀ", "UÁ", "UẢ", "HƯỚ", "XỒ", "CỘC", "KÍT",
    # Common exclamations
    "ĐI", "THÔI", "NÀO", "NHÉ", "NHA", "À", "Ạ", "Ơ", "ỦA",
    "CÚT", "ĐỒ", "LÃO", "THẰNG", "CON", "ĐỨA",
    # Manga-common words (may not be in Viet74K)
    "LÁO", "TOẸT", "QUẮC", "KHOE", "KHOANG",
}
(OUT_DIR / "manga_whitelist.txt").write_text(
    "\n".join(sorted(MANGA_WHITELIST)), encoding="utf-8"
)
words.update(MANGA_WHITELIST)
print(f"  Manga whitelist: {len(MANGA_WHITELIST)} entries")

# Save dictionary
(OUT_DIR / "vn_words.txt").write_text("\n".join(sorted(words)), encoding="utf-8")
print(f"  FINAL VN dict: {len(words)} words → {OUT_DIR}/vn_words.txt")

# -----------------------------------------------------------------------------
# 3. Bigram LM from VieNeu corpus
# -----------------------------------------------------------------------------
print("[3/3] Building bigram LM from VieNeu corpus...")
bigram_counts = Counter()   # (w1, w2) → count
unigram_counts = Counter()  # w1 → count

with open(VIENEU_TEXT, encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        text = parts[4].upper()
        tokens = re.findall(r"[\w\u00C0-\u1EF9]+", text)
        tokens = ["<S>"] + tokens + ["</S>"]
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            bigram_counts[(w1, w2)] += 1
            unigram_counts[w1] += 1

print(f"  Unigrams: {len(unigram_counts)} unique")
print(f"  Bigrams: {len(bigram_counts)} unique")

# Save as compact format: P(w2|w1) = count(w1,w2) / count(w1)
# Flatten for JSON: { "w1": { "w2": count, ... }, ... }
bigram_data = {}
for (w1, w2), c in bigram_counts.items():
    if c < 2:  # drop hapax
        continue
    if w1 not in bigram_data:
        bigram_data[w1] = {}
    bigram_data[w1][w2] = c

# Keep top-20 successors per w1 to bound size
for w1 in list(bigram_data.keys()):
    sorted_next = sorted(bigram_data[w1].items(), key=lambda x: -x[1])[:20]
    bigram_data[w1] = dict(sorted_next)

with open(OUT_DIR / "vn_bigrams.json", "w", encoding="utf-8") as f:
    json.dump(bigram_data, f, ensure_ascii=False, indent=2)
print(f"  Bigrams saved: {OUT_DIR}/vn_bigrams.json")

# Save unigram counts too
with open(OUT_DIR / "vn_unigrams.json", "w", encoding="utf-8") as f:
    json.dump({w: c for w, c in unigram_counts.most_common() if c >= 2},
              f, ensure_ascii=False)
print(f"  Unigrams saved: {OUT_DIR}/vn_unigrams.json")

print("\nDone. Resources at:", OUT_DIR)
