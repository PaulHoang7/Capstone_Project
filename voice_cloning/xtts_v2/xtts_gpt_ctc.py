"""XTTS GPT subclass that adds a CTC auxiliary head on the mel-side hidden
states. Trains alignment between mel positions and the target Vietnamese
character sequence (vn_char_vocab).

Why this file lives outside coqui_tts:
  - Keeps the upstream package unmodified
  - Forward-only addition: vanilla XTTS inference is untouched
  - Wrapper subclass overrides the training-time forward path

Usage:
    from xtts_gpt_ctc import XttsGPTWithCTC, make_ctc_targets
    model = XttsGPTWithCTC(**xtts_gpt_kwargs, ctc_vocab_size=97, lambda_ctc=0.3)
    # ... load XTTS checkpoint (will warn about missing ctc_head — ok)
    loss_text, loss_mel, loss_ctc, mel_logits = model.forward_with_ctc(
        text_inputs, text_lengths, audio_codes, wav_lengths,
        cond_mels=cond_mels, ctc_targets=ctc_targets, ctc_target_lens=ctc_lens,
    )
    total_loss = loss_text + loss_mel + model.lambda_ctc * loss_ctc
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make coqui_tts importable
_COQUI = Path(__file__).parent / "coqui_tts"
if str(_COQUI) not in sys.path:
    sys.path.insert(0, str(_COQUI))

from TTS.tts.layers.xtts.gpt import GPT  # noqa: E402

from ctc_aux_head import CTCAlignmentHead, CTCHeadConfig, ctc_loss, make_ctc_head  # noqa: E402


class XttsGPTWithCTC(GPT):
    """GPT + CTC auxiliary head on mel-side hidden states.

    The only forward path that uses CTC is `forward_with_ctc(...)`. The
    inherited `forward(...)` is unchanged so eval / inference / vanilla
    training still work.
    """

    def __init__(
        self,
        *gpt_args,
        ctc_vocab_size: int = 97,
        ctc_blank_id: int = 0,
        ctc_dropout: float = 0.1,
        lambda_ctc: float = 0.3,
        ctc_head_version: str = "v2",   # "v2" (3-layer) or "v3" (5-layer deeper)
        **gpt_kwargs,
    ):
        super().__init__(*gpt_args, **gpt_kwargs)
        self.ctc_head = make_ctc_head(
            CTCHeadConfig(
                gpt_hidden_dim=self.model_dim,
                vocab_size=ctc_vocab_size,
                dropout=ctc_dropout,
                blank_id=ctc_blank_id,
            ),
            version=ctc_head_version,
        )
        self.ctc_head_version = ctc_head_version
        self.ctc_blank_id = ctc_blank_id
        self.lambda_ctc = lambda_ctc

    # ─────────────────────────────────────────────────────────────
    # Hidden-state-returning version of get_logits. This duplicates
    # ~30 lines from GPT.get_logits to also yield enc_mel for CTC.
    # ─────────────────────────────────────────────────────────────
    def _get_logits_and_mel_hidden(
        self,
        text_emb,
        mel_emb,
        cond_latents,
        attn_mask_cond,
        attn_mask_text,
        attn_mask_mel,
    ):
        emb = torch.cat([cond_latents, text_emb, mel_emb], dim=1)
        offset = cond_latents.shape[1]

        attn_mask = None
        if attn_mask_text is not None:
            attn_mask = torch.cat([attn_mask_text, attn_mask_mel], dim=1)
            attn_mask_c = torch.ones(
                cond_latents.shape[0], offset, dtype=torch.bool, device=emb.device
            )
            attn_mask = torch.cat([attn_mask_c, attn_mask], dim=1)

        gpt_out = self.gpt(
            inputs_embeds=emb,
            return_dict=True,
            attention_mask=attn_mask,
        )
        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)

        text_enc = enc[:, : text_emb.shape[1]]
        mel_enc  = enc[:, -mel_emb.shape[1] :]

        # Standard heads
        text_logits = self.text_head(text_enc).permute(0, 2, 1)
        mel_logits  = self.mel_head(mel_enc).permute(0, 2, 1)

        return text_logits, mel_logits, mel_enc

    # ─────────────────────────────────────────────────────────────
    # Training forward with CTC. Mirrors GPT.forward up to the point
    # where get_logits is called; then computes CTC additionally.
    # ─────────────────────────────────────────────────────────────
    def forward_with_ctc(
        self,
        text_inputs,
        text_lengths,
        audio_codes,
        wav_lengths,
        cond_mels=None,
        cond_idxs=None,
        cond_lens=None,
        ctc_targets: torch.Tensor = None,           # [B, max_chars] padded with blank_id
        ctc_target_lengths: torch.Tensor = None,    # [B]
    ):
        # Replicate the preprocessing from GPT.forward
        assert self.max_conditioning_inputs > 0 or cond_mels is None
        max_text_len = text_lengths.max()
        code_lengths = torch.ceil(wav_lengths / self.code_stride_len).long() + 3

        if cond_lens is not None:
            if self.use_perceiver_resampler:
                cond_lens = cond_lens // self.perceiver_cond_length_compression
            else:
                cond_lens = cond_lens // self.code_stride_len
        if cond_idxs is not None:
            for idx in range(cond_idxs.size(0)):
                if self.use_perceiver_resampler:
                    cond_idxs[idx] = cond_idxs[idx] // self.perceiver_cond_length_compression
                else:
                    cond_idxs[idx] = cond_idxs[idx] // self.code_stride_len

        max_mel_len = code_lengths.max()
        if max_mel_len > audio_codes.shape[-1]:
            audio_codes = F.pad(audio_codes, (0, max_mel_len - audio_codes.shape[-1]))

        text_inputs = F.pad(text_inputs[:, :max_text_len], (0, 1), value=self.stop_text_token)
        audio_codes = F.pad(audio_codes[:, :max_mel_len], (0, 1), value=self.stop_audio_token)
        audio_codes = self.set_mel_padding(audio_codes, code_lengths - 3)

        text_inputs, text_targets = self.set_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token
        )
        audio_codes, mel_targets = self.set_inputs_and_targets(
            audio_codes, self.start_audio_token, self.stop_audio_token
        )

        attn_mask_cond = torch.ones(
            cond_mels.shape[0], cond_mels.shape[-1], dtype=torch.bool,
            device=text_inputs.device,
        )
        attn_mask_text = torch.ones(text_inputs.shape, dtype=torch.bool, device=text_inputs.device)
        attn_mask_mel  = torch.ones(audio_codes.shape, dtype=torch.bool, device=text_inputs.device)
        for idx, l in enumerate(text_lengths):
            attn_mask_text[idx, l + 1 :] = 0
        for idx, l in enumerate(code_lengths):
            attn_mask_mel[idx, l + 1 :] = 0

        # Build embeddings
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_emb  = self.mel_embedding(audio_codes)   + self.mel_pos_embedding(audio_codes)

        # Conditioning latents — must match original GPT.forward exactly:
        # cond_latents = self.get_style_emb(cond_mels).transpose(1, 2)
        # → shape (B, T_cond, D) where T_cond=32 if perceiver resampler used.
        cond_latents = self.get_style_emb(cond_mels).transpose(1, 2)

        # Get logits AND mel hidden state
        text_logits, mel_logits, mel_enc = self._get_logits_and_mel_hidden(
            text_emb, mel_emb, cond_latents,
            attn_mask_cond, attn_mask_text, attn_mask_mel,
        )

        # Mask padding in targets
        for idx, l in enumerate(text_lengths):
            text_targets[idx, l + 1 :] = -1
        for idx, l in enumerate(code_lengths):
            mel_targets[idx, l + 1 :] = -1

        # Main losses (cross-entropy on next-token)
        loss_text = F.cross_entropy(
            text_logits, text_targets.long(),
            ignore_index=-1, label_smoothing=self.label_smoothing,
        ).mean()
        loss_mel = F.cross_entropy(
            mel_logits, mel_targets.long(),
            ignore_index=-1, label_smoothing=self.label_smoothing,
        ).mean()

        # ── CTC auxiliary loss on mel-side hidden states ───────────
        # mel_enc: [B, T_mel, D] → CTC logits [B, T_mel, V_chars]
        loss_ctc = torch.tensor(0.0, device=text_inputs.device)
        if ctc_targets is not None and ctc_target_lengths is not None:
            ctc_logits = self.ctc_head(mel_enc)             # [B, T_mel, V]
            audio_lengths = code_lengths.clamp(max=mel_enc.shape[1])
            loss_ctc = ctc_loss(
                ctc_logits,
                ctc_targets,
                audio_lengths,
                ctc_target_lengths,
                blank_id=self.ctc_blank_id,
            )

        return loss_text, loss_mel, loss_ctc, mel_logits


def make_ctc_targets(
    texts: list[str],
    encode_fn,
    pad_id: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-encode raw VN texts → (padded_ids [B, max_len], lengths [B])."""
    encoded = [encode_fn(t) for t in texts]
    lengths = torch.LongTensor([len(e) for e in encoded])
    max_len = max(int(lengths.max().item()), 1)
    padded = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
    for i, e in enumerate(encoded):
        padded[i, : len(e)] = torch.LongTensor(e)
    return padded.to(device), lengths.to(device)


if __name__ == "__main__":
    # Minimal smoke test — only verifies __init__ and shape compatibility.
    # Does NOT call forward_with_ctc with real inputs (needs full XTTS config).
    from vn_char_vocab import VOCAB_SIZE, BLANK_ID, encode

    print(f"Creating XttsGPTWithCTC with VN vocab size {VOCAB_SIZE}...")
    model = XttsGPTWithCTC(
        # Minimal valid XTTS GPT config (matches xtts_v2 production)
        layers=30,
        model_dim=1024,
        heads=16,
        max_text_tokens=402,
        max_mel_tokens=605,
        max_prompt_tokens=70,
        number_text_tokens=6681,
        start_text_token=261,
        stop_text_token=0,
        num_audio_tokens=1026,
        start_audio_token=1024,
        stop_audio_token=1025,
        use_perceiver_resampler=True,
        # CTC additions
        ctc_vocab_size=VOCAB_SIZE,
        ctc_blank_id=BLANK_ID,
        lambda_ctc=0.3,
    )
    print(f"  CTC head: {model.ctc_head}")
    print(f"  λ_ctc: {model.lambda_ctc}")
    n_params_ctc = sum(p.numel() for p in model.ctc_head.parameters())
    n_params_total = sum(p.numel() for p in model.parameters())
    print(f"  CTC head params: {n_params_ctc:,} ({100*n_params_ctc/n_params_total:.2f}% of model)")

    # Sample CTC targets
    texts = ["Đô-rê-mon là chú mèo máy.", "hả ?"]
    targets, lens = make_ctc_targets(texts, encode, pad_id=BLANK_ID)
    print(f"  CTC targets: shape={tuple(targets.shape)}, lengths={lens.tolist()}")

    print("Smoke test passed.")
