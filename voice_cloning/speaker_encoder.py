"""
ECAPA-TDNN Speaker Encoder — Phase 1: Fine-tune on VieNeu-TTS.

Strategy:
    Step A: Freeze encoder, train classifier only on pre-trained embeddings.
            Verify embeddings are discriminative (loss should drop fast).
    Step B: Unfreeze last 2 SE-Res2Blocks, fine-tune encoder + classifier.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECAPASpeakerEncoder(nn.Module):
    def __init__(
        self,
        pretrained_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        n_speakers: int = 193,
        projection_dim: int = 256,
        device: str = "cuda:0",
    ):
        super().__init__()
        if device == "cuda":
            device = "cuda:0"
        self.device_str = device
        self.emb_dim = 192

        # ── Load SpeechBrain ECAPA-TDNN ──────────────────────────────────────
        from speechbrain.inference.speaker import EncoderClassifier

        sb_save = os.path.join(
            os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
            "speechbrain_ecapa",
        )
        self._sb = EncoderClassifier.from_hparams(
            source=pretrained_source, savedir=sb_save,
            run_opts={"device": device},
        )

        # ── Classification head ──────────────────────────────────────────────
        self.classifier = nn.Linear(self.emb_dim, n_speakers)

        # ── Projection 192 → 256 for VITS2 gin_channels ─────────────────────
        self.projection = nn.Linear(self.emb_dim, projection_dim)

        n_train = sum(p.numel() for p in self.classifier.parameters())
        n_train += sum(p.numel() for p in self.projection.parameters())
        print(f"[ECAPASpeakerEncoder] classifier+projection params: {n_train:,}")

    def encode(self, wav: torch.Tensor, wav_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract 192-d speaker embedding using SpeechBrain's full pipeline.

        Args:
            wav:      [B, T] raw 16 kHz waveform
            wav_lens: [B]    relative lengths [0, 1]
        Returns:
            [B, 192]
        """
        if wav_lens is None:
            wav_lens = torch.ones(wav.shape[0], device=wav.device)

        # Use SpeechBrain's official encode_batch — guaranteed correct pipeline
        emb = self._sb.encode_batch(wav, wav_lens)  # [B, 1, 192]
        if emb.dim() == 3:
            emb = emb.squeeze(1)
        return emb  # [B, 192]

    def project(self, emb: torch.Tensor) -> torch.Tensor:
        """192-d → [B, 256, 1] for VITS2 gin injection."""
        return self.projection(emb).unsqueeze(-1)

    def forward(self, wav, speaker_ids=None, wav_lens=None):
        """Training forward: encode → classify."""
        # encode_batch handles everything internally (features + norm + encoder)
        emb = self.encode(wav, wav_lens)

        # Detach embeddings — classifier trains on frozen features
        # (encoder fine-tuning added later in Phase B)
        emb = emb.detach()

        loss = None
        if speaker_ids is not None:
            logits = self.classifier(emb)
            loss = F.cross_entropy(logits, speaker_ids)
        return emb, loss


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_encoder(model: ECAPASpeakerEncoder, path: str, epoch: int, eer: float):
    torch.save({
        "epoch": epoch, "eer": eer,
        "classifier_state": model.classifier.state_dict(),
        "projection_state": model.projection.state_dict(),
    }, path)
    print(f"[save] {path}  epoch={epoch}  EER={eer:.4f}")


def load_encoder(model: ECAPASpeakerEncoder, path: str, strict: bool = True) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.classifier.load_state_dict(ckpt["classifier_state"])
    model.projection.load_state_dict(ckpt["projection_state"])
    print(f"[load] {path}  epoch={ckpt.get('epoch')}  EER={ckpt.get('eer')}")
    return ckpt
