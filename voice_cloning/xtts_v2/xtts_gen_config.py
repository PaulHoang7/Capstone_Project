"""Single source of truth for XTTS hallucination-control config.

All XTTS pipelines should import from this module:
  - comic_pipeline_xtts.py        (Capstone CLI demo)
  - synthesize_xtts.py            (Capstone batch synth from JSON)
  - web/scripts/synth_batch.py    (live web demo)
  - web/scripts/render_featured_xtts.py  (pre-rendered featured pairs)

Config rationale (battle-tested in web production):
  - temperature=0.5      : tighter than vanilla 0.7 → less token drift
  - repetition_penalty=15.0 : aggressive anti-loop (viXTTS default 2.0 loops
                            on short VN bubbles; 15.0 prevents this)
  - top_k=5              : biggest single hallucination fix (was 30) per
                            project notes — restricts vocab to top candidates
  - top_p=0.75           : tight nucleus
  - length_penalty=1.0   : neutral
"""
from __future__ import annotations
import os
import re
import sys
from typing import Tuple

import numpy as np

# Capstone_project là parents[2] của file này (.../voice_cloning/xtts_v2/xtts_gen_config.py)
_CP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _CP_ROOT not in sys.path:
    sys.path.insert(0, _CP_ROOT)
from Capstone_project.text_norm.vn_numbers import expand_vn_numbers

# ─── Generation config ────────────────────────────────────────────────
# Apply with: xtts.inference(**XTTS_GEN_CONFIG)
#
# Two configs:
#   LONG  — tight sampling for long text (kill hallucination, suppress loops)
#   SHORT — relaxed sampling for short text (1-4 words) where tight sampling
#           starves the decoder of acoustic diversity → "không tròn vành"
#           (muddled / clipped articulation on comic bubble exclamations).
XTTS_GEN_CONFIG = dict(
    temperature=0.5,
    length_penalty=1.0,
    repetition_penalty=15.0,
    top_k=5,
    top_p=0.75,
)

XTTS_GEN_CONFIG_SHORT = dict(
    temperature=0.7,          # ↑ more variability for articulation
    length_penalty=1.0,
    repetition_penalty=5.0,   # ↓ allow natural phoneme repetition (e.g. "đi đi")
    top_k=30,                 # ↑ wider candidate pool → richer articulation
    top_p=0.85,
)

SHORT_TEXT_WORD_THRESHOLD = 4  # text with ≤ this many words → use SHORT config


def gen_config_for_text(text: str) -> dict:
    """Pick the appropriate generation config based on text length.

    Short bubbles (≤4 words) need more sampling diversity to articulate
    clearly. Long bubbles need tight sampling to prevent hallucination loops.
    """
    return (XTTS_GEN_CONFIG_SHORT
            if len(text.split()) <= SHORT_TEXT_WORD_THRESHOLD
            else XTTS_GEN_CONFIG)

SAMPLE_RATE = 24000

# Duration cap defaults — truncate hallucinated tails post-synthesis.
DEFAULT_CHARS_PER_SEC = 6.0
DEFAULT_DURATION_BUFFER = 0.8
DEFAULT_MIN_DURATION = 1.0


# ─── Text preprocessing ───────────────────────────────────────────────
FOREIGN_NAME_MAP = {
    # Conan series
    "Conan Doyle": "Cô-nan Đoi-lờ", "conan doyle": "cô-nan đoi-lờ",
    "Holmes": "Hôn-mơ", "holmes": "hôn-mơ",
    "Watson": "Oát-sân", "watson": "oát-sân",
    "Shinichi": "Si-ni-chi", "shinichi": "si-ni-chi",
    "Sonoko": "Sô-nô-cô", "sonoko": "sô-nô-cô",
    "Ran": "Ran", "ran": "ran",
    "Conan": "Cô-nan", "conan": "cô-nan",
    "Afghanistan": "A-phơ-ga-ni-xtan", "afghanistan": "a-phơ-ga-ni-xtan",
    "tennis": "ten-nít",
    # Doraemon
    "Doraemon": "Đô-rê-mon", "doraemon": "đô-rê-mon", "DORAEMON": "Đô-rê-mon",
    "Nobita": "Nô-bi-ta", "nobita": "nô-bi-ta", "NOBITA": "Nô-bi-ta",
    "Shizuka": "Si-zu-ka", "shizuka": "si-zu-ka", "SHIZUKA": "Si-zu-ka",
    "Suneo": "Su-nê-ô", "suneo": "su-nê-ô", "SUNEO": "Su-nê-ô",
    "Suneko": "Su-nê-kô", "SUNEKO": "Su-nê-kô",
    "Jaian": "Cha-i-an", "jaian": "cha-i-an", "JAIAN": "Cha-i-an",
    "Gian": "Chi-an", "gian": "chi-an",
    "Dorami": "Đô-ra-mi", "dorami": "đô-ra-mi",
    "Mineko": "Mi-nê-kô", "mineko": "mi-nê-kô",
    "Tokyo": "Tô-ky-ô", "tokyo": "tô-ky-ô",
    # Naruto / other
    "Naruto": "Na-ru-tô", "naruto": "na-ru-tô",
    "Sasuke": "Sa-su-kê", "sasuke": "sa-su-kê",
    "Sakura": "Sa-ku-ra", "sakura": "sa-ku-ra",
    # Common OCR misreads
    "Xê-cô": "Sê-kô",
}


def clean_for_xtts(text: str) -> str:
    """Preprocess raw text to reduce XTTS hallucination triggers.

    1. Phoneticize foreign names (XTTS loops on Latin tokens)
    2. Replace ellipsis '...' / '…' / '..' → ',' (XTTS loops on ellipsis)
    3. Collapse repeated punctuation (!! → !, ?? → ?)
    4. Lowercase (XTTS trained on lowercase newspaper text)
    5. Add terminal period for very short utterances (helps pacing)
    """
    out = text
    out = expand_vn_numbers(out)
    for foreign, viet in FOREIGN_NAME_MAP.items():
        out = out.replace(foreign, viet)
    out = out.replace("…", ", ").replace("...", ", ").replace("..", ", ")
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"([!?,]){2,}", r"\1", out)
    out = out.lower()
    words = out.split()
    if len(words) <= 2 and out and out.rstrip()[-1:] not in ".!?:":
        out = out.rstrip() + "."
    return out


def cap_duration_by_text(
    wav: np.ndarray,
    text: str,
    sample_rate: int = SAMPLE_RATE,
    chars_per_sec: float = DEFAULT_CHARS_PER_SEC,
    buffer_sec: float = DEFAULT_DURATION_BUFFER,
    min_sec: float = DEFAULT_MIN_DURATION,
    fade_sec: float = 0.1,
) -> Tuple[np.ndarray, bool]:
    """Truncate audio if duration > expected based on text length.

    Returns (possibly_truncated_wav, was_truncated). Applies a fade-out at
    the truncation point to avoid an audible click.
    """
    raw_dur = len(wav) / sample_rate
    text_len_chars = len(text)
    max_dur = max(min_sec, text_len_chars / chars_per_sec + buffer_sec)
    if raw_dur <= max_dur:
        return wav, False
    max_samples = int(max_dur * sample_rate)
    truncated = wav[:max_samples].astype(np.float32, copy=True)
    fade_samples = min(int(fade_sec * sample_rate), len(truncated))
    if fade_samples > 0:
        fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        truncated[-fade_samples:] *= fade
    return truncated, True


# ─── Whisper forced-alignment trim ─────────────────────────────────────
# Cuts the audio precisely after the LAST word that matches a token from
# the target bubble text. Eliminates tail hallucination (XTTS reading
# extra content past the bubble).
WHISPER_TRIM_BUFFER_SEC = 0.18  # safety buffer past last-word end_time
DEFAULT_WHISPER_MODEL = "small"  # tradeoff: small=fast+OK, medium=slow+better
WHISPER_SR = 16000  # faster-whisper expects 16kHz mono float32


def _normalize_word_for_match(w: str) -> str:
    """Strip diacritics + lowercase + remove punct — for fuzzy matching."""
    import unicodedata
    if not w:
        return ""
    w = w.lower().strip()
    w = unicodedata.normalize("NFD", w)
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")
    w = re.sub(r"[^a-z0-9]", "", w)
    return w


def _tokenize_target_text(text: str) -> list:
    out = []
    for w in re.findall(r"\S+", text):
        n = _normalize_word_for_match(w)
        if n:
            out.append(n)
    return out


def _find_last_target_end(target_tokens: list, whisper_words: list):
    """Return end_time of the last whisper word matching target's last token,
    falling back to the latest whisper word matching ANY target token."""
    if not target_tokens or not whisper_words:
        return None
    last_target = target_tokens[-1]
    target_set = set(target_tokens)
    for ww in reversed(whisper_words):
        if _normalize_word_for_match(ww["word"]) == last_target:
            return ww["end"]
    for ww in reversed(whisper_words):
        if _normalize_word_for_match(ww["word"]) in target_set:
            return ww["end"]
    return None


def load_whisper_trimmer(model_size: str = DEFAULT_WHISPER_MODEL, device: str = "cuda"):
    """Load faster-whisper for tail-hallucination trimming. Call once at
    pipeline init — model is reusable across all bubbles.

    Returns the WhisperModel or None if faster-whisper is unavailable.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[whisper-trim] faster-whisper not installed — skipping (pip install faster-whisper)")
        return None
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def trim_with_whisper(
    wav: np.ndarray,
    text: str,
    asr,
    sample_rate: int = SAMPLE_RATE,
    buffer_sec: float = WHISPER_TRIM_BUFFER_SEC,
) -> Tuple[np.ndarray, bool]:
    """Trim tail-hallucination by finding the last bubble-text word in the
    audio via Whisper word-level timestamps, then cutting at end_time + buffer.

    Returns (trimmed_wav, was_trimmed). Falls back to original audio if:
      - asr is None
      - text too short (≤4 words) — Whisper unreliable on sub-2s audio,
        and these bubbles rarely have tail hallucination worth trimming
      - no target tokens after normalization
      - no whisper word matches target text (safer to keep original)
    """
    if asr is None:
        return wav, False
    # Skip Whisper for short text — alignment unreliable, can clip articulation
    if len(text.split()) <= SHORT_TEXT_WORD_THRESHOLD:
        return wav, False
    target_tokens = _tokenize_target_text(text)
    if not target_tokens:
        return wav, False

    # Resample to 16kHz for faster-whisper if needed
    if sample_rate != WHISPER_SR:
        # Use linear resampling — good enough for ASR (Whisper internally also resamples)
        import math
        ratio = WHISPER_SR / sample_rate
        new_len = int(math.ceil(len(wav) * ratio))
        idx = np.linspace(0, len(wav) - 1, new_len).astype(np.int64)
        asr_input = wav[idx].astype(np.float32)
    else:
        asr_input = wav.astype(np.float32)

    try:
        segments, _ = asr.transcribe(
            asr_input, language="vi", word_timestamps=True,
            vad_filter=False, beam_size=1, condition_on_previous_text=False,
        )
        whisper_words = []
        for seg in segments:
            if seg.words:
                whisper_words.extend([
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in seg.words
                ])
    except Exception as exc:
        print(f"[whisper-trim] ASR failed: {exc} — keeping original")
        return wav, False

    end_time = _find_last_target_end(target_tokens, whisper_words)
    if end_time is None:
        return wav, False

    end_samples = int((end_time + buffer_sec) * sample_rate)
    end_samples = min(end_samples, len(wav))
    if end_samples >= len(wav):
        return wav, False
    return wav[:end_samples].astype(np.float32, copy=True), True
