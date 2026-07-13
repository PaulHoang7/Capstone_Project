"""CTC auxiliary head for XTTS GPT — forces text-audio alignment.

Plugs into the XTTS GPT decoder: after the last transformer block, the GPT
hidden states [B, T_audio, D] are projected to Vietnamese character logits
[B, T_audio, V_chars]. Trained with CTC loss against the target text's
character sequence.

This adds an explicit per-frame alignment supervision signal that the
audio-token-only loss lacks. Expected to fix the semantic drift / length
mismatch failure mode measured 2026-05-19 (75% of VieNeu heldout samples).
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CTCHeadConfig:
    gpt_hidden_dim: int = 1024     # XTTS-v2 GPT hidden size
    vocab_size: int = 97           # from vn_char_vocab.VOCAB_SIZE
    dropout: float = 0.1
    blank_id: int = 0


class CTCAlignmentHead(nn.Module):
    """v2 — 3-layer MLP head: LN → Linear(D→D) → GELU → Dropout → Linear(D→D/2)
    → GELU → Dropout → Linear(D/2→V_chars). More capacity than 1-layer
    Linear; deeper for alignment learning on long sentences."""

    def __init__(self, cfg: CTCHeadConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.gpt_hidden_dim
        Dh = D // 2
        self.proj = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(D, Dh),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(Dh, cfg.vocab_size),
        )
        # Initialize last linear to small values so untrained head doesn't
        # disrupt main loss at warm-up
        nn.init.normal_(self.proj[-1].weight, std=0.01)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, gpt_hidden: torch.Tensor) -> torch.Tensor:
        """gpt_hidden: [B, T_audio, D] → logits [B, T_audio, V_chars]"""
        return self.proj(gpt_hidden)


class CTCAlignmentHeadV3(nn.Module):
    """v3 — 5-layer deeper MLP for stronger alignment supervision.

    Architecture:
      LN → Linear(D→D) → GELU → Drop
         → Linear(D→D) → GELU → Drop      ← extra block vs v2
         → Linear(D→D/2) → GELU → Drop
         → Linear(D/2→D/4) → GELU → Drop  ← smoother dimensional reduction
         → Linear(D/4→V_chars)

    Rationale (project memory: CTC v1/v2 → alignment 25%→36.7% with 3-layer):
      - Deeper head = more capacity to learn fine-grained char↔audio-token
        alignment, especially for short utterances (comic bubbles)
      - Smoother dim reduction (D→D→D→D/2→D/4→V) avoids the abrupt
        1024→512→97 collapse of v2
      - v3 adds ~3M params (still tiny vs 400M XTTS GPT)
    """

    def __init__(self, cfg: CTCHeadConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.gpt_hidden_dim
        Dh = D // 2          # 512
        Dq = D // 4          # 256
        self.proj = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(D, Dh),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(Dh, Dq),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(Dq, cfg.vocab_size),
        )
        # Small-init last layer (same logic as v2 — keep warmup stable)
        nn.init.normal_(self.proj[-1].weight, std=0.01)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, gpt_hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(gpt_hidden)


def make_ctc_head(cfg: CTCHeadConfig, version: str = "v2") -> nn.Module:
    """Factory — selects CTC head architecture by version string."""
    if version == "v2":
        return CTCAlignmentHead(cfg)
    if version == "v3":
        return CTCAlignmentHeadV3(cfg)
    raise ValueError(f"Unknown CTC head version: {version!r} (expected 'v2' or 'v3')")


def ctc_loss(
    logits: torch.Tensor,            # [B, T_audio, V]
    target_char_ids: torch.Tensor,   # [B, max_target_len] (padded)
    audio_lengths: torch.Tensor,     # [B] real T per sample
    target_lengths: torch.Tensor,    # [B] real target char count
    blank_id: int = 0,
    zero_infinity: bool = True,
) -> torch.Tensor:
    """Compute CTC loss. Returns a scalar.

    Note: pytorch's F.ctc_loss expects log_probs shape [T, B, V] and targets
    padded. input_lengths must be the actual T per sample. Mean reduction
    across the batch (not per-frame).
    """
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # [T, B, V]
    return F.ctc_loss(
        log_probs       = log_probs,
        targets         = target_char_ids,
        input_lengths   = audio_lengths,
        target_lengths  = target_lengths,
        blank           = blank_id,
        reduction       = "mean",
        zero_infinity   = zero_infinity,
    )


def collate_ctc_targets(
    text_list: list[str],
    encode_fn,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-encode texts → (padded_ids [B, max_len], lengths [B])."""
    encoded = [encode_fn(t) for t in text_list]
    lengths = torch.LongTensor([len(e) for e in encoded])
    max_len = max(lengths.max().item(), 1)
    padded = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
    for i, e in enumerate(encoded):
        padded[i, :len(e)] = torch.LongTensor(e)
    return padded, lengths


if __name__ == "__main__":
    # Smoke test — verify forward + loss don't crash
    from vn_char_vocab import encode, VOCAB_SIZE, BLANK_ID

    B, T, D = 2, 200, 1024  # batch, audio tokens, gpt hidden
    cfg = CTCHeadConfig(gpt_hidden_dim=D, vocab_size=VOCAB_SIZE, blank_id=BLANK_ID)
    head = CTCAlignmentHead(cfg)
    gpt_h = torch.randn(B, T, D)
    logits = head(gpt_h)
    print(f"Head forward: {gpt_h.shape} → {logits.shape}")

    texts = ["Đô-rê-mon là một chú mèo máy.", "hả ?"]
    targets, t_lens = collate_ctc_targets(texts, encode, pad_id=BLANK_ID)
    audio_lens = torch.LongTensor([T, T])
    print(f"Targets shape: {targets.shape}, lengths: {t_lens.tolist()}")

    loss = ctc_loss(logits, targets, audio_lens, t_lens, blank_id=BLANK_ID)
    print(f"CTC loss (random init): {loss.item():.4f}")

    # Sanity: loss should be finite, gradient flows
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in head.parameters() if p.grad is not None)
    print(f"Grad norm: {grad_norm:.4f}  (should be > 0)")
    print("Smoke test passed.")
