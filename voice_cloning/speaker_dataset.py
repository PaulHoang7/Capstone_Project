"""
Dataset for ECAPA-TDNN speaker encoder training on VieNeu-TTS.

Filelist format (one line per utterance):
    /path/to/wav|speaker_id|ipa_text

DataLoader strategy:
    - N speakers × M utterances per batch (GE2E / AAM style)
    - Random crop to fixed duration
    - Resample 24 kHz → 16 kHz (SpeechBrain ECAPA standard)
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, Sampler


# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE_IN  = 24_000   # VieNeu-TTS native
SAMPLE_RATE_OUT = 16_000   # SpeechBrain ECAPA standard
MAX_SECS        = 6.0
MIN_SECS        = 1.5
MAX_SAMPLES     = int(MAX_SECS * SAMPLE_RATE_OUT)
MIN_SAMPLES     = int(MIN_SECS * SAMPLE_RATE_OUT)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpeakerDataset(Dataset):
    """
    Flat dataset: one item = (wav_tensor, speaker_id).
    wav_tensor is already resampled to 16 kHz and length-cropped/padded.
    """

    def __init__(
        self,
        filelist_path: str,
        held_out_speakers: Optional[set] = None,
        max_samples: int = MAX_SAMPLES,
        min_samples: int = MIN_SAMPLES,
    ):
        self.max_samples = max_samples
        self.min_samples = min_samples

        # Parse filelist — collect raw items first
        raw_items: List[Tuple[str, int]] = []
        with open(filelist_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 2:
                    continue
                wav_path, sid_str = parts[0], parts[1]
                sid = int(sid_str)
                if held_out_speakers and sid in held_out_speakers:
                    continue
                raw_items.append((wav_path, sid))

        # Remap speaker IDs to contiguous 0..N-1 to avoid out-of-bounds in classifier
        unique_sids = sorted(set(sid for _, sid in raw_items))
        self.sid_remap: Dict[int, int] = {orig: new for new, orig in enumerate(unique_sids)}

        self.items: List[Tuple[str, int]] = []   # (wav_path, remapped_speaker_id)
        self.speaker_to_items: Dict[int, List[int]] = defaultdict(list)
        for wav_path, sid in raw_items:
            remapped = self.sid_remap[sid]
            idx = len(self.items)
            self.items.append((wav_path, remapped))
            self.speaker_to_items[remapped].append(idx)

        self.speakers = sorted(self.speaker_to_items.keys())
        print(
            f"[SpeakerDataset] {len(self.items)} utterances, "
            f"{len(self.speakers)} speakers  ({filelist_path})"
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        wav_path, sid = self.items[idx]
        wav, rel_len = self._load_wav(wav_path)
        return wav, sid, rel_len

    def _load_wav(self, path: str) -> Tuple[torch.Tensor, float]:
        """Returns (wav [max_samples], rel_len) where rel_len = actual / max_samples."""
        try:
            # Use soundfile — works with torchaudio nightly (no torchcodec needed)
            data, sr = sf.read(path, dtype="float32", always_2d=False)
            if data.ndim == 2:
                data = data.mean(axis=1)  # stereo → mono
            wav = torch.from_numpy(data)
        except Exception:
            return torch.zeros(self.max_samples), 1.0

        # Resample 24kHz → 16kHz using torchaudio functional (no I/O, just math)
        if sr != SAMPLE_RATE_OUT:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE_OUT)

        # Crop or pad to max_samples
        if len(wav) >= self.max_samples:
            start = random.randint(0, len(wav) - self.max_samples)
            wav = wav[start: start + self.max_samples]
            rel_len = 1.0
        else:
            actual_len = len(wav)
            wav = F.pad(wav, (0, self.max_samples - actual_len))
            rel_len = actual_len / self.max_samples  # <1.0 → norm ignores padding

        return wav, rel_len


# ── N×M Batch Sampler ─────────────────────────────────────────────────────────

class NMBatchSampler(Sampler):
    """
    Sample N speakers × M utterances per batch.
    Ensures every batch has exactly N*M items from N different speakers.
    """

    def __init__(
        self,
        speaker_to_items: Dict[int, List[int]],
        n_speakers: int = 16,
        m_utterances: int = 4,
        n_batches: int = 1000,
    ):
        self.speaker_to_items = speaker_to_items
        self.n_speakers = n_speakers
        self.m_utterances = m_utterances
        self.n_batches = n_batches

        # Only keep speakers with enough utterances
        self.eligible = [
            spk for spk, items in speaker_to_items.items()
            if len(items) >= m_utterances
        ]
        assert len(self.eligible) >= n_speakers, (
            f"Need ≥{n_speakers} eligible speakers, got {len(self.eligible)}"
        )

    def __iter__(self):
        for _ in range(self.n_batches):
            chosen_speakers = random.sample(self.eligible, self.n_speakers)
            batch = []
            for spk in chosen_speakers:
                utts = random.sample(self.speaker_to_items[spk], self.m_utterances)
                batch.extend(utts)
            yield batch

    def __len__(self):
        return self.n_batches


def collate_fn(batch):
    wavs     = torch.stack([item[0] for item in batch])              # [B, T]
    sids     = torch.LongTensor([item[1] for item in batch])         # [B]
    wav_lens = torch.FloatTensor([item[2] for item in batch])        # [B] relative lengths
    return wavs, sids, wav_lens


def build_dataloaders(
    train_filelist: str,
    val_filelist: str,
    held_out_speakers: set,
    n_speakers: int = 16,
    m_utterances: int = 4,
    n_batches_train: int = 500,
    n_batches_val: int = 50,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, SpeakerDataset]:
    """Build train and val DataLoaders + held-out evaluation dataset."""

    train_ds = SpeakerDataset(train_filelist, held_out_speakers=held_out_speakers)
    val_ds   = SpeakerDataset(val_filelist,   held_out_speakers=held_out_speakers)

    train_sampler = NMBatchSampler(
        train_ds.speaker_to_items, n_speakers, m_utterances, n_batches_train
    )
    val_sampler = NMBatchSampler(
        val_ds.speaker_to_items, n_speakers, m_utterances, n_batches_val
    )

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )

    # Held-out dataset (unseen speakers, for zero-shot EER evaluation)
    held_out_ds = SpeakerDataset(
        train_filelist, held_out_speakers=None
    )
    # Override: keep ONLY held-out speakers
    held_out_ds.items = [
        item for item in held_out_ds.items if item[1] in held_out_speakers
    ]
    held_out_ds.speaker_to_items = defaultdict(list)
    for i, (_, sid) in enumerate(held_out_ds.items):
        held_out_ds.speaker_to_items[sid].append(i)
    held_out_ds.speakers = sorted(held_out_ds.speaker_to_items.keys())

    return train_loader, val_loader, held_out_ds
