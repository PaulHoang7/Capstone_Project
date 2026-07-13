"""Vietnamese character vocabulary for CTC supervision on XTTS GPT.

Design (NFC-normalized, lowercased):
  0  - blank (CTC special)
  1  - space
  2  - apostrophe / hyphen cluster  (', -)
  3  - generic punct cluster        (. , ? ! : ; … " ( ) [ ])
  4-29  - 26 lowercase Latin letters: a b c d ... z
  30-35 - Vietnamese consonants:      â ă đ ê ô ơ ư
  36-71 - 6 vowels × 6 tones = 36 (a/â/ă/e/ê/i/o/ô/ơ/u/ư/y × ngang/sắc/huyền/hỏi/ngã/nặng)

Actually use a simpler enumeration: every NFC vowel-with-tone glyph gets its own
ID. Total expected: ~70 tokens.

The vocab is fully deterministic — built by enumerating all printable Vietnamese
characters that can appear in cleaned VieNeu text. NO learned tokenizer.
"""
from __future__ import annotations
import unicodedata
from functools import lru_cache


# Base letters (lowercase, Latin + Vietnamese-specific consonants)
_BASE_CHARS = list("abcdefghijklmnopqrstuvwxyzâăđêôơư")

# Vietnamese vowels that take tone marks (NFC composed forms)
# Each base vowel × {ngang, sắc, huyền, hỏi, ngã, nặng}
_VOWEL_BASES = ["a", "â", "ă", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y"]
_TONE_MARKS  = ["", "́", "̀", "̉", "̃", "̣"]  # sắc, huyền, hỏi, ngã, nặng


def _build_tone_chars() -> list[str]:
    """All NFC vowel+tone composed characters used in Vietnamese."""
    out = []
    for v in _VOWEL_BASES:
        for t in _TONE_MARKS:
            composed = unicodedata.normalize("NFC", v + t)
            if composed not in out:
                out.append(composed)
    return out


# Build full vocab — IDs are fixed by enumeration order
_VOCAB = ["<blank>", " ", "'-", ".,?!:;…\"()[]"]
# 4..29 — Latin letters
_VOCAB.extend(_BASE_CHARS)
# Then tone characters (some overlap with base; dedup)
for c in _build_tone_chars():
    if c not in _VOCAB and c not in _BASE_CHARS:
        _VOCAB.append(c)

VOCAB: tuple[str, ...] = tuple(_VOCAB)
VOCAB_SIZE = len(VOCAB)
BLANK_ID = 0

# Char → ID lookup. Multi-char tokens (clusters) are NOT keyed here; they're
# handled in `encode()` per-input-char.
_CHAR2ID = {c: i for i, c in enumerate(VOCAB)}
# Override cluster char→ID
for c in "'-":
    _CHAR2ID[c] = _VOCAB.index("'-")
for c in ".,?!:;…\"()[]":
    _CHAR2ID[c] = _VOCAB.index(".,?!:;…\"()[]")


def encode(text: str) -> list[int]:
    """Encode raw Vietnamese text → list of token IDs.

    Pipeline: NFC normalize → lowercase → map each char → drop unknowns.
    Returns IDs in order; CTC handles the alignment with audio frames.
    """
    text = unicodedata.normalize("NFC", text).lower()
    ids = []
    for c in text:
        if c in _CHAR2ID:
            ids.append(_CHAR2ID[c])
        # else: silently skip unknown chars (digits, foreign letters → handled
        # by upstream text normalization in the dataset pipeline)
    return ids


def decode(ids: list[int]) -> str:
    """Decode IDs back to a string (debugging only). Blank → ''."""
    out = []
    for i in ids:
        if i == BLANK_ID:
            continue
        out.append(VOCAB[i])
    return "".join(out)


@lru_cache(maxsize=4)
def vocab_info() -> dict:
    return {
        "size": VOCAB_SIZE,
        "blank_id": BLANK_ID,
        "tokens": list(VOCAB),
    }


if __name__ == "__main__":
    # Smoke test
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"First 30 tokens: {VOCAB[:30]}")
    print()
    samples = [
        "Đô-rê-mon là một chú mèo máy.",
        "Tớ đến từ thế kỷ 22.",
        "hả ? !",
        "Nhân vật chính: Nobita (10 tuổi).",
    ]
    for s in samples:
        ids = encode(s)
        rec = decode(ids)
        print(f"  {s!r}")
        print(f"    → {len(ids)} tokens: {ids[:30]}…" if len(ids) > 30 else f"    → {len(ids)} tokens: {ids}")
        print(f"    ↩ decoded: {rec!r}")
        print()
