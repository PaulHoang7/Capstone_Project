"""
Tone extraction utilities for Vietnamese IPA text.

Vietnamese tones encoded as digits in the VieNeu-TTS IPA text:
  0 = padding/space/punctuation
  1 = ngang (sometimes explicit, often unmarked — default for no digit)
  2 = sắc
  4 = hỏi
  5 = ngã
  6 = nặng
  7 = rare/special (appears ~28 times in val set)

NOTE: Tone 3 (huyền) does NOT appear in VieNeu-TTS sea-g2p output.
Syllables that would be huyền are unmarked (no digit) and default to
tone 1 in extraction. This is a dataset characteristic, not a bug.

The IPA text contains tone numbers inline, e.g.:
  "mˈo6t̪  kˈəːn  bˈaː5w"
   ^^^6     ^^^1    ^^^5

These functions extract a per-character tone sequence that is aligned
with the output of text_to_sequence / cleaned_text_to_sequence from
vits2_pytorch/text/__init__.py.
"""

import re
import sys
import os

# Add vits2_pytorch to path so we can import its text processing
_VITS2_DIR = os.path.join(os.path.dirname(__file__), '../../vits2_pytorch')
if _VITS2_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_VITS2_DIR))

from text.symbols import symbols
from text import _clean_text

# Build symbol lookup once
_SYMBOL_SET = set(symbols)

# Tone digits that can appear in IPA text (1-7; 3 absent in VieNeu-TTS)
_TONE_DIGITS = set('1234567')

# Number of distinct tones (0=pad + tones 1-7)
N_TONES = 8


def extract_tones_per_position(cleaned_text: str) -> list:
    """Assign a tone ID (0-7) to each character position in cleaned IPA text.

    Algorithm:
    - Split into syllable groups (non-space sequences separated by spaces)
    - For each syllable group, find the first digit [1-7] -> that's the tone
    - If no digit found, default to tone 1 (ngang/unmarked)
    - Each character in the syllable group gets that tone
    - Space characters get tone 0

    Args:
        cleaned_text: Vietnamese IPA text (already cleaned/normalized)

    Returns:
        List of ints, same length as cleaned_text. Each int is 0-7.
    """
    tone_per_pos = [0] * len(cleaned_text)

    i = 0
    while i < len(cleaned_text):
        if cleaned_text[i] == ' ':
            tone_per_pos[i] = 0
            i += 1
        else:
            # Collect syllable group (non-space sequence)
            start = i
            while i < len(cleaned_text) and cleaned_text[i] != ' ':
                i += 1
            syllable = cleaned_text[start:i]

            # Find tone digit in syllable
            tone = 1  # default: ngang (no explicit tone number)
            for c in syllable:
                if c in _TONE_DIGITS:
                    tone = int(c)
                    break

            # Assign tone to all characters in this syllable group
            for j in range(start, i):
                tone_per_pos[j] = tone

    assert len(tone_per_pos) == len(cleaned_text)
    return tone_per_pos


def cleaned_text_to_tone_sequence(cleaned_text: str) -> list:
    """Extract tone sequence aligned with cleaned_text_to_sequence() output.

    This function mirrors vits2_pytorch/text/__init__.py:cleaned_text_to_sequence()
    exactly: it iterates character-by-character over cleaned_text, and for each
    character that exists in the symbol vocabulary, it outputs the corresponding
    tone ID. Characters not in the vocabulary are skipped (same as the symbol
    sequence function).

    This guarantees:
        len(cleaned_text_to_tone_sequence(t)) == len(cleaned_text_to_sequence(t))

    Args:
        cleaned_text: Vietnamese IPA text (already cleaned/normalized)

    Returns:
        List of ints (0-7), same length as cleaned_text_to_sequence(cleaned_text)
    """
    tone_per_pos = extract_tones_per_position(cleaned_text)

    tone_sequence = []
    for idx, symbol in enumerate(cleaned_text):
        if symbol in _SYMBOL_SET:
            tone_sequence.append(tone_per_pos[idx])

    return tone_sequence


def text_to_tone_sequence(text: str, cleaner_names: list) -> list:
    """Extract tone sequence from raw text, applying cleaners first.

    Mirrors vits2_pytorch/text/__init__.py:text_to_sequence() but returns
    tone IDs instead of symbol IDs.

    Args:
        text: Raw Vietnamese IPA text
        cleaner_names: List of cleaner function names to apply

    Returns:
        List of ints (0-7), same length as text_to_sequence(text, cleaner_names)
    """
    clean_text = _clean_text(text, cleaner_names)
    return cleaned_text_to_tone_sequence(clean_text)


def extract_tone_label_from_vietnamese(vietnamese_text: str) -> list:
    """Extract tone labels from Vietnamese Unicode text (not IPA).

    Used for evaluation: comparing synthesized audio transcription against
    expected tones. Vietnamese diacritics map deterministically to tones:
      - No diacritic: tone 1 (ngang)
      - Acute accent (á): tone 2 (sắc)
      - Grave accent (à): tone 3 (huyền)
      - Hook above (ả): tone 4 (hỏi)
      - Tilde (ã): tone 5 (ngã)
      - Dot below (ạ): tone 6 (nặng)

    Args:
        vietnamese_text: Vietnamese text in Unicode (e.g., "Xin chào bạn")

    Returns:
        List of (syllable, tone) tuples for each syllable
    """
    import unicodedata

    # Combining diacritical marks for Vietnamese tones
    TONE_MARKS = {
        '\u0301': 2,  # combining acute accent -> sắc
        '\u0300': 3,  # combining grave accent -> huyền
        '\u0309': 4,  # combining hook above -> hỏi
        '\u0303': 5,  # combining tilde -> ngã
        '\u0323': 6,  # combining dot below -> nặng
    }

    results = []
    syllables = vietnamese_text.strip().split()

    for syllable in syllables:
        # Skip punctuation-only tokens
        if all(c in ',.!?;:…"\'()-' for c in syllable):
            continue

        # Decompose to NFD to separate base chars and combining marks
        decomposed = unicodedata.normalize('NFD', syllable)

        tone = 1  # default ngang
        for char in decomposed:
            if char in TONE_MARKS:
                tone = TONE_MARKS[char]
                break

        results.append((syllable, tone))

    return results
