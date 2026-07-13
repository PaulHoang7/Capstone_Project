"""
End-to-End Comic Voice-Over Pipeline.

Input : path to a comic page image (or directory of pages)
Output: per-page folder with:
  - bubble_<N>.wav          → synthesised audio for each bubble
  - summary.json            → bubble_id, text, speaker_char_id, emotion, audio_path

Pipeline flow:
  1. YOLOv8          → detect bubbles, characters, panels
  2. OCR             → read Vietnamese text from each bubble
  3. Face clustering → match characters to CharacterDB
  4. Speaker Attr    → assign each bubble to a character (MLP → rule-based fallback)
  5. TTS             → Phase-3 zero-shot voice cloning per character
  6. Save            → wav files + summary JSON

Usage:
    cd /home/bes/Desktop/Tin
    # Single page (auto-assign voices from CharacterDB)
    python Capstone_project/scripts/pipeline_end_to_end.py \\
        --image /path/to/page.jpg \\
        --tts-config  Capstone_project/configs/vits2_vieneu_clone_phase3.json \\
        --tts-ckpt    /mnt/nfs-data/tin_dataset/vits2_logs/vieneu_clone_phase3/G_latest.pth \\
        --spk-ckpt    /mnt/nfs-data/tin_dataset/checkpoints/speaker_encoder/speaker_encoder_best.pth \\
        --out-dir     /tmp/comic_voiceover

    # Chapter directory with persistent CharacterDB
    python Capstone_project/scripts/pipeline_end_to_end.py \\
        --image /path/to/chapter/ \\
        --char-db /tmp/chapter1_db.json \\
        --ref-audio-dir /path/to/ref_wavs/ \\
        --out-dir /tmp/chapter1_out

    # Use GPU 1 for TTS, skip face clustering, rule-based speaker
    python Capstone_project/scripts/pipeline_end_to_end.py \\
        --image page.jpg --tts-device cuda:1 --no-face --rule-based

Reference audio directory format (--ref-audio-dir):
    ref_wavs/
      char_0.wav   (or char_0_ref.wav)
      char_1.wav
      ...
    If a character has no matching wav, a random VieNeu-TTS speaker is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from scipy.io.wavfile import write as wav_write

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve()
_TIN_ROOT    = _HERE.parents[2]          # /home/bes/Desktop/Tin
_VITS2_ROOT  = _TIN_ROOT / "vits2_pytorch"

for _p in (str(_TIN_ROOT), str(_VITS2_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Default paths ──────────────────────────────────────────────────────────────
DEFAULT_YOLO_WEIGHTS   = "/mnt/nfs-data/tin_dataset/checkpoints/yolo_comic.pt"
DEFAULT_TTS_CONFIG     = str(_TIN_ROOT / "Capstone_project/configs/vits2_vieneu_clone_phase3.json")
DEFAULT_TTS_CKPT       = "/mnt/nfs-data/tin_dataset/vits2_logs/vieneu_clone_phase3/G_latest.pth"
DEFAULT_SPK_CKPT       = "/mnt/nfs-data/tin_dataset/checkpoints/speaker_encoder/speaker_encoder_best.pth"
DEFAULT_SPEAKER_MODEL  = "/mnt/nfs-data/tin_dataset/comic/speaker_attribution/speaker_mlp.pt"
DEFAULT_SPEAKER_SCALER = "/mnt/nfs-data/tin_dataset/comic/speaker_attribution/scaler.pkl"

# Fallback speaker IDs when no reference audio is available
# (uses emb_g lookup from Phase-2 checkpoint; Phase-3 ignores sid → random fallback)
_FALLBACK_SPEAKER_IDS  = list(range(0, 193, 10))   # 20 evenly-spaced speakers

SAMPLE_RATE_ECAPA = 16_000
SAMPLE_RATE_TTS   = 24_000


# ──────────────────────────────────────────────────────────────────────────────
# TTS Engine
# ──────────────────────────────────────────────────────────────────────────────

class TTSEngine:
    """Wraps Phase-3 VITS2 model + ECAPA speaker encoder for zero-shot TTS.

    Usage:
        engine = TTSEngine(config_path, tts_ckpt, spk_ckpt, device)
        audio  = engine.synthesise(text, ref_wav_path_or_embedding)
    """

    def __init__(
        self,
        config_path: str,
        tts_ckpt: str,
        spk_ckpt: str,
        device: str = "cuda",
    ):
        import utils
        import commons as vits_commons
        from text.symbols import symbols
        from Capstone_project.models import build_synthesizer
        from Capstone_project.voice_cloning.speaker_encoder import ECAPASpeakerEncoder

        self.device  = torch.device(device)
        self._commons = vits_commons

        # ── Config ─────────────────────────────────────────────────────────────
        self.hps = utils.get_hparams_from_file(config_path)
        log.info(f"TTS config: {config_path}")

        # ── VITS2 model (use build_synthesizer — handles use_mel_posterior_encoder) ──
        self.net_g = build_synthesizer(self.hps, len(symbols)).to(self.device)
        self.net_g.eval()
        utils.load_checkpoint(tts_ckpt, self.net_g, None)
        log.info(f"TTS checkpoint: {tts_ckpt}")

        # ── Speaker encoder ────────────────────────────────────────────────────
        gin_channels = self.hps.model.gin_channels
        self.spk_enc = ECAPASpeakerEncoder(
            n_speakers     = self.hps.data.n_speakers,
            projection_dim = gin_channels,
            device         = device,
        ).to(self.device)
        ckpt = torch.load(spk_ckpt, map_location=self.device)
        state = ckpt.get("model", ckpt)
        self.spk_enc.load_state_dict(state, strict=False)
        self.spk_enc.eval()
        log.info(f"Speaker encoder: {spk_ckpt}")

        # ── Text processing helpers ────────────────────────────────────────────
        from text import text_to_sequence
        from Capstone_project.tone_encoder.tone_utils import text_to_tone_sequence
        self._text_to_seq  = text_to_sequence
        self._text_to_tone = text_to_tone_sequence
        self._add_blank    = self.hps.data.add_blank
        self._cleaners     = self.hps.data.text_cleaners
        self._sr           = self.hps.data.sampling_rate

        # ── Vietnamese G2P (raw text → IPA phonemes) ──────────────────────────
        # The model is trained on pre-phonemized IPA text from VieNeu-TTS.
        # Raw Vietnamese must be converted to IPA before text_to_sequence().
        try:
            from phonemizer import phonemize as _phonemize
            from phonemizer.separator import Separator
            self._phonemize = _phonemize
            self._phon_sep = Separator(phone="", word="  ", syllable="")
            self._has_g2p = True
            log.info("G2P: phonemizer (espeak-vi) ready")
        except ImportError:
            self._has_g2p = False
            log.warning(
                "phonemizer not installed — raw Vietnamese text will NOT be "
                "phonemized. Install with: pip install phonemizer"
            )

    # ── Public helpers ─────────────────────────────────────────────────────────

    def vietnamese_to_ipa(self, text: str) -> str:
        """Convert raw Vietnamese text to IPA matching VieNeu-TTS format.

        Pipeline: raw Vietnamese → espeak IPA → add stress marks (ˈ/ˌ)
        to match the phoneme format the model was trained on.
        """
        # Skip if text is already IPA (contains IPA-only symbols)
        _IPA_ONLY = set("əɛɜː")
        if any(c in _IPA_ONLY for c in text):
            return text

        if not self._has_g2p:
            return text

        try:
            ipa = self._phonemize(
                text,
                language="vi",
                backend="espeak",
                separator=self._phon_sep,
                preserve_punctuation=True,
                strip=True,
            )
            if ipa and ipa != text:
                ipa = self._add_stress_marks(ipa)
                log.debug(f"G2P: \"{text[:30]}\" → \"{ipa[:30]}\"")
                return ipa
        except Exception as exc:
            log.warning(f"G2P failed: {exc}")

        return text

    @staticmethod
    def _add_stress_marks(ipa: str) -> str:
        """Add VieNeu-TTS-style stress marks to espeak IPA output.

        VieNeu-TTS places ˈ (primary) or ˌ (secondary) after the onset
        consonant(s), before the nucleus vowel of each syllable.
        espeak omits these.  We insert ˈ by default (covers ~90% of
        syllables in the training data).
        """
        _VOWELS = set("aeiouəɛɔæɪʊɯʌyøœ")

        words = ipa.split("  ")
        result = []
        for word in words:
            # Skip punctuation-only tokens
            if not any(c in _VOWELS for c in word):
                result.append(word)
                continue

            # Insert ˈ before the first vowel (after onset consonants)
            new = []
            stress_placed = False
            for ch in word:
                if ch in _VOWELS and not stress_placed:
                    new.append("ˈ")
                    stress_placed = True
                new.append(ch)
            result.append("".join(new))

        return "  ".join(result)

    def embed_reference(self, ref_wav_path: str) -> torch.Tensor:
        """Load a reference wav and return speaker embedding [1, 256, 1]."""
        wav, sr = torchaudio.load(ref_wav_path)
        if sr != SAMPLE_RATE_ECAPA:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE_ECAPA)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.to(self.device)          # [1, T]
        wav_lens = torch.ones(1, device=self.device)
        with torch.no_grad():
            emb = self.spk_enc.encode(wav, wav_lens)          # [1, 192]
            g   = self.spk_enc.projection(emb).unsqueeze(-1)  # [1, 256, 1]
        return g

    def get_text_and_tone(self, text: str):
        """Convert raw Vietnamese text → aligned (x, tone) LongTensors."""
        text = self.vietnamese_to_ipa(text)
        text_norm = self._text_to_seq(text, self._cleaners)
        tone_norm = self._text_to_tone(text, self._cleaners)
        if self._add_blank:
            text_norm = self._commons.intersperse(text_norm, 0)
            tone_norm = self._commons.intersperse(tone_norm, 0)
        assert len(text_norm) == len(tone_norm), (
            f"text/tone length mismatch: {len(text_norm)} vs {len(tone_norm)}"
        )
        return (
            torch.LongTensor(text_norm).to(self.device),
            torch.LongTensor(tone_norm).to(self.device),
        )

    def synthesise(
        self,
        text: str,
        g: torch.Tensor,
        noise_scale:   float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale:  float = 1.0,
        max_len:       int   = 500,
    ) -> np.ndarray:
        """Synthesise audio for text conditioned on speaker embedding g.

        Args:
            text:  Raw Vietnamese text (will be cleaned/phonemised internally).
            g:     Speaker embedding [1, 256, 1] from embed_reference().
            Returns: float32 numpy array, shape [T], sample rate = 24 kHz.
        """
        if not text.strip():
            return np.zeros(0, dtype=np.float32)

        # Split long sentences to avoid OOM (especially on CPU).
        # Vietnamese punctuation + common delimiters serve as split points.
        import re
        MAX_CHARS = 60
        if len(text) > MAX_CHARS:
            parts = re.split(r'(?<=[.!?,;:])(?:\s+)', text)
            if len(parts) == 1:
                # No punctuation — split on whitespace near the midpoint
                words = text.split()
                mid = len(words) // 2
                parts = [" ".join(words[:mid]), " ".join(words[mid:])]
            segments = []
            for part in parts:
                if not part.strip():
                    continue
                segments.append(self._synthesise_one(
                    part.strip(), g, noise_scale, noise_scale_w,
                    length_scale, max_len,
                ))
            return np.concatenate(segments) if segments else np.zeros(0, np.float32)

        return self._synthesise_one(
            text, g, noise_scale, noise_scale_w, length_scale, max_len,
        )

    def _synthesise_one(
        self, text: str, g: torch.Tensor,
        noise_scale: float, noise_scale_w: float,
        length_scale: float, max_len: int,
    ) -> np.ndarray:
        """Synthesise a single (short) text segment."""
        x, tone = self.get_text_and_tone(text)
        x    = x.unsqueeze(0)
        tone = tone.unsqueeze(0)
        x_len = torch.LongTensor([x.shape[1]]).to(self.device)

        with torch.no_grad():
            audio = self.net_g.infer(
                x, x_len,
                sid          = None,
                tone         = tone,
                g            = g,
                noise_scale  = noise_scale,
                noise_scale_w= noise_scale_w,
                length_scale = length_scale,
                max_len      = max_len,
            )[0][0, 0].float().cpu().numpy()

        return audio


# ──────────────────────────────────────────────────────────────────────────────
# Voice Registry — per-character speaker embeddings
# ──────────────────────────────────────────────────────────────────────────────

class VoiceRegistry:
    """Maps char_id → speaker embedding, backed by a reference audio directory.

    Priority order for each new character:
      1. ref_audio_dir/<char_id>.wav  (user-provided reference)
      2. ref_audio_dir/<char_id>_ref.wav
      3. Auto-assign: cycle through fallback speaker IDs (simple lookup)

    The registry is lazy: embeddings are computed on first request, then cached.
    """

    def __init__(
        self,
        engine: TTSEngine,
        ref_audio_dir: Optional[str] = None,
    ):
        self._engine        = engine
        self._ref_dir       = Path(ref_audio_dir) if ref_audio_dir else None
        self._registry: dict[str, torch.Tensor] = {}  # char_id → g [1,256,1]
        self._auto_idx      = 0

    def get(self, char_id: str) -> torch.Tensor:
        """Return speaker embedding for char_id, computing if needed."""
        if char_id not in self._registry:
            self._registry[char_id] = self._build_embedding(char_id)
        return self._registry[char_id]

    def register_from_wav(self, char_id: str, wav_path: str) -> None:
        """Manually register a reference wav for a character."""
        log.info(f"  Voice registry: {char_id} ← {wav_path}")
        self._registry[char_id] = self._engine.embed_reference(wav_path)

    def _build_embedding(self, char_id: str) -> torch.Tensor:
        """Try to find reference audio; fall back to emb_g lookup."""
        if self._ref_dir is not None:
            for suffix in ("", "_ref"):
                candidate = self._ref_dir / f"{char_id}{suffix}.wav"
                if candidate.exists():
                    log.info(f"  Voice registry: {char_id} ← {candidate.name}")
                    return self._engine.embed_reference(str(candidate))

        # No reference audio — use emb_g from the TTS model's learned lookup
        # (only works if Phase-2 checkpoint is used; Phase-3 sets n_speakers>0 too)
        fallback_sid = _FALLBACK_SPEAKER_IDS[
            self._auto_idx % len(_FALLBACK_SPEAKER_IDS)
        ]
        self._auto_idx += 1
        log.warning(
            f"  Voice registry: no ref audio for '{char_id}' → "
            f"auto-assign sid={fallback_sid}"
        )
        with torch.no_grad():
            g = self._engine.net_g.emb_g(
                torch.LongTensor([fallback_sid]).to(self._engine.device)
            ).unsqueeze(-1)              # [1, 256, 1]
        return g

    def summary(self) -> None:
        log.info(f"Voice registry: {len(self._registry)} characters registered")
        for cid in self._registry:
            log.info(f"  {cid}")


# ──────────────────────────────────────────────────────────────────────────────
# Per-page TTS pass
# ──────────────────────────────────────────────────────────────────────────────

def synthesise_page(
    page_result: dict,
    engine: TTSEngine,
    voice_reg: VoiceRegistry,
    out_dir: Path,
    noise_scale:   float = 0.667,
    noise_scale_w: float = 0.8,
    length_scale:  float = 1.0,
) -> list[dict]:
    """Synthesise audio for all bubbles in one page result dict.

    Returns list of summary dicts (one per bubble), also saves wavs to out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    page_name = Path(page_result["image"]).stem
    summaries = []

    bubbles = page_result.get("bubbles", [])
    log.info(f"  Synthesising {len(bubbles)} bubbles for page '{page_name}'…")

    for bubble in sorted(bubbles, key=lambda b: b.get("order", 0)):
        order      = bubble.get("order", 0)
        text       = bubble.get("text", "").strip()
        speaker_id = bubble.get("speaker_id")
        char_id    = speaker_id or "unknown"

        bubble_id  = f"{page_name}_bubble_{order:03d}"
        wav_name   = f"{bubble_id}.wav"
        wav_path   = out_dir / wav_name

        # ── Emotion (placeholder — extend with emotion detection module later) ──
        emotion = "neutral"

        if not text:
            log.debug(f"  [{order:3d}] {char_id:10s}  <empty text — skip>")
            summaries.append({
                "bubble_id":       bubble_id,
                "order":           order,
                "text":            text,
                "speaker_char_id": char_id,
                "emotion":         emotion,
                "audio_path":      None,
                "attribution":     bubble.get("attribution", "none"),
                "speaker_conf":    bubble.get("speaker_conf", 0.0),
            })
            continue

        # ── Get speaker embedding ──────────────────────────────────────────────
        g = voice_reg.get(char_id)

        # ── Synthesise ────────────────────────────────────────────────────────
        t0 = time.time()
        try:
            audio = engine.synthesise(
                text,
                g,
                noise_scale   = noise_scale,
                noise_scale_w = noise_scale_w,
                length_scale  = length_scale,
            )
        except Exception as exc:
            log.error(f"  [{order:3d}] TTS failed for '{text[:40]}': {exc}")
            summaries.append({
                "bubble_id":       bubble_id,
                "order":           order,
                "text":            text,
                "speaker_char_id": char_id,
                "emotion":         emotion,
                "audio_path":      None,
                "error":           str(exc),
            })
            continue

        elapsed = time.time() - t0
        dur     = len(audio) / SAMPLE_RATE_TTS

        # ── Save wav ──────────────────────────────────────────────────────────
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        wav_write(str(wav_path), SAMPLE_RATE_TTS, audio_int16)

        log.info(
            f"  [{order:3d}] {char_id:10s}  "
            f"{dur:.2f}s audio  ({elapsed:.2f}s)  \"{text[:35]}\""
        )

        summaries.append({
            "bubble_id":       bubble_id,
            "order":           order,
            "text":            text,
            "speaker_char_id": char_id,
            "emotion":         emotion,
            "audio_path":      str(wav_path),
            "duration_s":      round(dur, 3),
            "attribution":     bubble.get("attribution", "none"),
            "speaker_conf":    bubble.get("speaker_conf", 0.0),
        })

    return summaries


# ──────────────────────────────────────────────────────────────────────────────
# Continuous page audio — all bubbles concatenated in reading order
# ──────────────────────────────────────────────────────────────────────────────

def concatenate_page_audio(
    summaries: list[dict],
    out_path: Path,
    pause_between_speakers: float = 0.45,
    pause_same_speaker:     float = 0.25,
) -> dict | None:
    """Concatenate per-bubble wavs into one continuous audio file per page.

    Inserts a short silence between bubbles. A longer pause is used when the
    speaker changes, so the listener can perceive the voice switch.

    Returns a dict with metadata about the combined file, or None if no audio.
    """
    from scipy.io.wavfile import read as wav_read

    segments: list[np.ndarray] = []
    prev_speaker = None
    total_dur = 0.0

    for s in sorted(summaries, key=lambda x: x.get("order", 0)):
        wav_path = s.get("audio_path")
        if not wav_path or not Path(wav_path).exists():
            continue

        sr, data = wav_read(wav_path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32767.0

        current_speaker = s.get("speaker_char_id", "unknown")

        # Insert silence gap
        if segments:
            if current_speaker != prev_speaker:
                gap = pause_between_speakers
            else:
                gap = pause_same_speaker
            silence = np.zeros(int(gap * SAMPLE_RATE_TTS), dtype=np.float32)
            segments.append(silence)
            total_dur += gap

        segments.append(data)
        total_dur += len(data) / SAMPLE_RATE_TTS
        prev_speaker = current_speaker

    if not segments:
        return None

    combined = np.concatenate(segments)
    combined_int16 = (combined * 32767).clip(-32768, 32767).astype(np.int16)
    wav_write(str(out_path), SAMPLE_RATE_TTS, combined_int16)

    log.info(
        f"  Page audio: {total_dur:.2f}s → {out_path.name}"
    )
    return {
        "audio_path":  str(out_path),
        "duration_s":  round(total_dur, 3),
        "num_bubbles": sum(1 for s in summaries if s.get("audio_path")),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> None:
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect pages ───────────────────────────────────────────────────────
    image_path = Path(args.image)
    extensions = (".jpg", ".jpeg", ".png", ".webp")
    if image_path.is_dir():
        pages = sorted(p for p in image_path.iterdir()
                       if p.suffix.lower() in extensions)
    elif image_path.is_file():
        pages = [image_path]
    else:
        log.error(f"Image path not found: {image_path}")
        sys.exit(1)
    log.info(f"Found {len(pages)} page(s) to process")

    # ── 2. Load CV models ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Loading CV models…")

    # Build a namespace that comic_pipeline.init_pipeline() expects
    import types
    cv_args = types.SimpleNamespace(
        weights        = args.yolo_weights,
        lang           = args.lang,
        no_ocr         = args.no_ocr,
        no_face        = args.no_face,
        paddle_only    = args.paddle_only,
        no_gpu         = args.no_gpu,
        char_db        = args.char_db,
        speaker_model  = args.speaker_model,
        speaker_scaler = args.speaker_scaler,
    )

    from Capstone_project.scripts.comic_pipeline import init_pipeline, process_page as cv_process_page
    cv_models = init_pipeline(cv_args)

    # ── 3. Load TTS engine ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Loading TTS engine…")
    engine = TTSEngine(
        config_path = args.tts_config,
        tts_ckpt    = args.tts_ckpt,
        spk_ckpt    = args.spk_ckpt,
        device      = args.tts_device,
    )

    # ── 4. Build voice registry ────────────────────────────────────────────────
    voice_reg = VoiceRegistry(engine, ref_audio_dir=args.ref_audio_dir)

    # Pre-register if ref_audio_dir given
    if args.ref_audio_dir:
        ref_dir = Path(args.ref_audio_dir)
        for wav_file in sorted(ref_dir.glob("*.wav")):
            char_id = wav_file.stem.replace("_ref", "")
            voice_reg.register_from_wav(char_id, str(wav_file))

    # ── 5. Process each page ───────────────────────────────────────────────────
    log.info("=" * 60)
    all_page_summaries = []
    total_t0 = time.time()

    for page_path in pages:
        page_t0 = time.time()
        log.info(f"Processing: {page_path.name}")

        # ── CV pipeline ──────────────────────────────────────────────────────
        try:
            page_result = cv_process_page(
                image_path          = page_path,
                yolo_model          = cv_models["yolo"],
                ocr_pipeline        = cv_models["ocr"],
                face_extractor      = cv_models["extractor"],
                char_db             = cv_models["char_db"],
                speaker_model_path  = cv_models["speaker_model"],
                speaker_scaler_path = cv_models["speaker_scaler"],
                direction           = args.direction,
                rule_based          = args.rule_based,
                use_gpu             = not args.no_gpu,
                yolo_conf           = args.yolo_conf,
            )
        except Exception as exc:
            log.error(f"CV pipeline failed on {page_path.name}: {exc}", exc_info=True)
            continue

        # ── TTS pass ──────────────────────────────────────────────────────────
        page_out_dir = out_root / page_path.stem
        summaries = synthesise_page(
            page_result,
            engine,
            voice_reg,
            out_dir       = page_out_dir,
            noise_scale   = args.noise_scale,
            noise_scale_w = args.noise_scale_w,
            length_scale  = args.length_scale,
        )

        # ── Concatenate continuous page audio ─────────────────────────────
        page_audio_path = page_out_dir / f"{page_path.stem}_full.wav"
        page_audio_info = concatenate_page_audio(
            summaries, page_audio_path,
            pause_between_speakers = args.pause_between,
            pause_same_speaker     = args.pause_same,
        )

        # ── Save per-page summary JSON ────────────────────────────────────────
        page_summary = {
            "page":       str(page_path),
            "width":      page_result["width"],
            "height":     page_result["height"],
            "direction":  page_result["direction"],
            "characters": page_result["characters"],
            "bubbles":    summaries,
            "page_audio": page_audio_info,
            "process_s":  round(time.time() - page_t0, 2),
        }
        summary_path = page_out_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(page_summary, f, ensure_ascii=False, indent=2)
        log.info(f"  Summary saved: {summary_path}")

        # ── Print quick table ─────────────────────────────────────────────────
        n_voiced = sum(1 for s in summaries if s.get("audio_path"))
        log.info(
            f"  Page done: {n_voiced}/{len(summaries)} bubbles voiced  "
            f"({time.time() - page_t0:.1f}s)"
        )
        _print_page_table(summaries)

        all_page_summaries.append(page_summary)

    # ── 6. Save chapter-level summary ─────────────────────────────────────────
    chapter_json = out_root / "chapter_summary.json"
    with open(chapter_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_pages": len(all_page_summaries),
                "total_time_s": round(time.time() - total_t0, 2),
                "pages": all_page_summaries,
            },
            f, ensure_ascii=False, indent=2,
        )

    # ── 7. Save CharacterDB if face clustering was used ────────────────────────
    if cv_models.get("char_db") is not None:
        db_path = out_root / "character_db.json"
        cv_models["char_db"].save(db_path)
        cv_models["char_db"].summary()

    voice_reg.summary()

    log.info("=" * 60)
    log.info(f"Done. Output: {out_root}")
    log.info(f"Chapter summary: {chapter_json}")
    log.info(f"Total time: {time.time() - total_t0:.1f}s")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_page_table(summaries: list[dict]) -> None:
    print(f"\n  {'Order':>5}  {'Speaker':>10}  {'Emotion':>9}  {'Dur':>5}  Text")
    print(f"  {'─'*5}  {'─'*10}  {'─'*9}  {'─'*5}  {'─'*40}")
    for s in summaries:
        dur  = f"{s.get('duration_s', 0):.2f}s" if s.get("audio_path") else "–"
        text = s.get("text", "")[:38] + ("…" if len(s.get("text", "")) > 38 else "")
        print(
            f"  {s['order']:>5}  {s['speaker_char_id']:>10}  "
            f"{s['emotion']:>9}  {dur:>5}  \"{text}\""
        )
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end Comic Voice-Over: image → voiced audio per bubble",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input / output
    p.add_argument("--image",   required=True,
                   help="Comic page image or directory of pages")
    p.add_argument("--out-dir", default="/tmp/comic_voiceover",
                   help="Output directory (wav files + summary JSON)")

    # CV pipeline
    p.add_argument("--yolo-weights",  default=DEFAULT_YOLO_WEIGHTS)
    p.add_argument("--char-db",       default=None,
                   help="Existing CharacterDB JSON (for chapter continuity)")
    p.add_argument("--speaker-model",  default=DEFAULT_SPEAKER_MODEL)
    p.add_argument("--speaker-scaler", default=DEFAULT_SPEAKER_SCALER)
    p.add_argument("--lang",      default="vi", choices=["vi", "ja"])
    p.add_argument("--direction", default="ltr", choices=["ltr", "rtl"])
    p.add_argument("--yolo-conf", type=float, default=0.25)

    # TTS
    p.add_argument("--tts-config",    default=DEFAULT_TTS_CONFIG,
                   help="VITS2 Phase-3 config JSON")
    p.add_argument("--tts-ckpt",      default=DEFAULT_TTS_CKPT,
                   help="VITS2 Phase-3 generator checkpoint (.pth)")
    p.add_argument("--spk-ckpt",      default=DEFAULT_SPK_CKPT,
                   help="ECAPA speaker encoder checkpoint (.pth)")
    p.add_argument("--tts-device",    default="cuda",
                   help="PyTorch device for TTS (cuda / cuda:0 / cpu)")
    p.add_argument("--ref-audio-dir", default=None,
                   help="Directory with per-character reference wavs "
                        "(char_id.wav or char_id_ref.wav)")

    # TTS synthesis params
    p.add_argument("--noise-scale",   type=float, default=0.667)
    p.add_argument("--noise-scale-w", type=float, default=0.8)
    p.add_argument("--length-scale",  type=float, default=1.0)

    # Continuous audio params
    p.add_argument("--pause-between", type=float, default=0.45,
                   help="Silence (seconds) between different speakers")
    p.add_argument("--pause-same",    type=float, default=0.25,
                   help="Silence (seconds) between same speaker's consecutive bubbles")

    # Flags
    p.add_argument("--no-ocr",       action="store_true", help="Skip OCR")
    p.add_argument("--no-face",      action="store_true", help="Skip face clustering")
    p.add_argument("--rule-based",   action="store_true",
                   help="Rule-based speaker attribution (no MLP)")
    p.add_argument("--paddle-only",  action="store_true",
                   help="Use PaddleOCR only (no VietOCR)")
    p.add_argument("--no-gpu",       action="store_true",
                   help="Run CV models on CPU")
    p.add_argument("--verbose",      action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info("=" * 60)
    log.info("Comic Voice-Over Pipeline")
    log.info(f"  input : {args.image}")
    log.info(f"  output: {args.out_dir}")
    log.info(f"  tts   : {args.tts_ckpt}")
    log.info("=" * 60)

    run_pipeline(args)


if __name__ == "__main__":
    main()
