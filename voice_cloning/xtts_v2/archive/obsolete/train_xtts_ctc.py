"""Training script for XTTS-FT + CTC auxiliary head on Vietnamese characters.

Mirrors train_xtts_vieneu.py but:
  1. Replaces xtts.gpt with XttsGPTWithCTC (adds ~100K params)
  2. Adds VN char ids to the batch via format_batch_on_device override
  3. Adds loss_ctc to train_step

Run on the same dataset (VieNeu-TTS-140h XTTS splits). Initialize from
the existing xtts_vieneu_ft checkpoint so we only need to learn the CTC
head + slightly adapt the GPT (low LR).

Usage:
    conda run -n xtts_env --no-capture-output python train_xtts_ctc.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# Coqui_tts on sys.path
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "coqui_tts"))

import torch  # noqa: E402

# coqui_tts imports
from trainer import Trainer, TrainerArgs                              # noqa: E402
from TTS.config.shared_configs import BaseDatasetConfig                # noqa: E402
from TTS.tts.datasets import load_tts_samples                          # noqa: E402
from TTS.tts.layers.xtts.trainer.gpt_trainer import (                  # noqa: E402
    GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig,
)

# Local imports
from vn_char_vocab import encode as vn_encode, VOCAB_SIZE, BLANK_ID    # noqa: E402
from xtts_gpt_ctc import XttsGPTWithCTC                                # noqa: E402
from ctc_aux_head import ctc_loss                                      # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Paths (mirror train_xtts_vieneu.py)
# ─────────────────────────────────────────────────────────────────────
VIXTTS_DIR      = "/mnt/nfs-data/tin_dataset/checkpoints/vixtts"
VIXTTS_AUX_DIR  = "/mnt/nfs-data/tin_dataset/checkpoints/vixtts_aux"   # has dvae.pth + mel_stats.pth
RESUME_FROM_FT  = "/mnt/nfs-data/tin_dataset/checkpoints/xtts_vieneu_ft"
SPLITS_DIR      = "/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/xtts_splits"
OUT_PATH        = "/mnt/nfs-data/tin_dataset/experiments/xtts_vieneu_ctc_v2"   # iter v2: deeper head + λ=0.5
LANGUAGE        = "vi"

# Training hyperparameters
BATCH_SIZE       = 4
GRAD_ACUMM_STEPS = 64
LR               = 1e-5
LAMBDA_CTC       = 0.5          # iter v2: stronger CTC weight (was 0.3)
CTC_WARMUP_STEPS = 2000     # λ_ctc = 0 for the first N steps
EVAL_EVERY       = 5000
SAVE_EVERY       = 5000
MAX_STEPS        = 30000


# ─────────────────────────────────────────────────────────────────────
# Custom trainer subclass — overrides forward + train_step to include CTC
# ─────────────────────────────────────────────────────────────────────
class GPTTrainerCTC(GPTTrainer):
    """GPTTrainer + CTC auxiliary head. Drop-in replacement."""

    def __init__(self, config: GPTTrainerConfig, lambda_ctc: float = LAMBDA_CTC):
        super().__init__(config)
        # Swap xtts.gpt for XttsGPTWithCTC with same hyperparameters
        old_gpt = self.xtts.gpt
        new_gpt = XttsGPTWithCTC(
            **self._gpt_kwargs_from_old(old_gpt),
            ctc_vocab_size=VOCAB_SIZE,
            ctc_blank_id=BLANK_ID,
            lambda_ctc=lambda_ctc,
        )
        # Copy pretrained weights (everything except ctc_head)
        missing, unexpected = new_gpt.load_state_dict(
            old_gpt.state_dict(), strict=False,
        )
        print(f"[ctc-trainer] Restored XTTS GPT weights. "
              f"Missing (new): {len(missing)} keys (expected: ctc_head.*). "
              f"Unexpected: {len(unexpected)}")
        self.xtts.gpt = new_gpt
        self.lambda_ctc = lambda_ctc
        self.step = 0  # tracks for λ_ctc warmup

    @staticmethod
    def _gpt_kwargs_from_old(old):
        """Extract constructor kwargs from existing GPT instance."""
        # NOTE: not all attributes are exposed; this is a best-effort copy.
        # In practice we should re-init from the config that was used to
        # build the original GPT. Adjust if init fails.
        return dict(
            layers=old.layers,
            model_dim=old.model_dim,
            heads=old.heads,
            # Reverse engineer constructor inputs from stored attrs:
            #   stored max_text_tokens = input + 2  → input = stored - 2
            #   stored max_mel_tokens  = input + 2 + max_conditioning_inputs
            max_text_tokens=(old.max_text_tokens - 2) if old.max_text_tokens > 0 else old.max_text_tokens,
            max_mel_tokens=(old.max_mel_tokens - 2 - old.max_conditioning_inputs)
                           if old.max_mel_tokens > 0 else old.max_mel_tokens,
            max_conditioning_inputs=old.max_conditioning_inputs,
            max_prompt_tokens=old.max_prompt_tokens,
            number_text_tokens=old.number_text_tokens,
            start_text_token=old.start_text_token,
            stop_text_token=old.stop_text_token,
            num_audio_tokens=old.num_audio_tokens,
            start_audio_token=old.start_audio_token,
            stop_audio_token=old.stop_audio_token,
            use_perceiver_resampler=old.use_perceiver_resampler,
            perceiver_cond_length_compression=old.perceiver_cond_length_compression,
            code_stride_len=old.code_stride_len,
            label_smoothing=old.label_smoothing,
        )

    def format_batch_on_device(self, batch):
        """Add `ctc_targets` + `ctc_target_lengths` derived from raw text.

        XTTS dataset doesn't keep raw text strings in the batch (only the
        tokenized `text_inputs`). We decode tokens back to UTF-8 via the
        XTTS BPE tokenizer, then re-encode as VN char IDs for CTC.
        """
        batch = super().format_batch_on_device(batch)
        text_inputs  = batch["text_inputs"]      # [B, max_T_text] (token IDs)
        text_lengths = batch["text_lengths"]     # [B]
        tokenizer    = self.xtts.tokenizer

        encoded = []
        for i in range(text_inputs.shape[0]):
            L = int(text_lengths[i].item())
            valid_ids = text_inputs[i, :L]
            raw_text  = tokenizer.decode(valid_ids)
            encoded.append(vn_encode(raw_text))

        lengths = torch.LongTensor([len(e) for e in encoded])
        max_len = max(int(lengths.max().item()), 1)
        padded = torch.full(
            (len(encoded), max_len), BLANK_ID,
            dtype=torch.long, device=text_inputs.device,
        )
        for i, e in enumerate(encoded):
            padded[i, : len(e)] = torch.tensor(e, dtype=torch.long, device=padded.device)
        batch["ctc_targets"] = padded
        batch["ctc_target_lengths"] = lengths.to(padded.device)
        return batch

    def forward(
        self, text_inputs, text_lengths, audio_codes, wav_lengths,
        cond_mels, cond_idxs, cond_lens,
        ctc_targets=None, ctc_target_lengths=None,
    ):
        return self.xtts.gpt.forward_with_ctc(
            text_inputs, text_lengths, audio_codes, wav_lengths,
            cond_mels=cond_mels, cond_idxs=cond_idxs, cond_lens=cond_lens,
            ctc_targets=ctc_targets,
            ctc_target_lengths=ctc_target_lengths,
        )

    def train_step(self, batch, criterion):
        # Dry-run safety: abort after DRYRUN_STEPS to validate integration
        # without committing to the full 3-5 day run. Set via env var.
        _dryrun = int(os.environ.get("DRYRUN_STEPS", "0"))
        if _dryrun > 0 and self.step >= _dryrun:
            print(f"[dryrun] reached step {self.step} ≥ {_dryrun}, exiting cleanly")
            sys.exit(0)

        loss_dict = {}
        loss_text, loss_mel, loss_ctc, _ = self.forward(
            text_inputs   = batch["text_inputs"],
            text_lengths  = batch["text_lengths"],
            audio_codes   = batch["audio_codes"],
            wav_lengths   = batch["wav_lengths"],
            cond_mels     = batch["cond_mels"],
            cond_idxs     = batch["cond_idxs"],
            cond_lens     = batch["cond_lens"],
            ctc_targets         = batch.get("ctc_targets"),
            ctc_target_lengths  = batch.get("ctc_target_lengths"),
        )

        # Linear warmup: λ_ctc ramps 0.05 → self.lambda_ctc over CTC_WARMUP_STEPS.
        # A small floor (0.05) keeps gradient flowing into ctc_head from step 1,
        # so the head actually learns during warmup (hard zero would freeze it).
        self.step += 1
        ramp = min(1.0, self.step / max(1, CTC_WARMUP_STEPS))
        eff_lambda = 0.05 + (self.lambda_ctc - 0.05) * ramp

        loss_dict["loss_text_ce"] = loss_text * self.args.gpt_loss_text_ce_weight
        loss_dict["loss_mel_ce"]  = loss_mel  * self.args.gpt_loss_mel_ce_weight
        loss_dict["loss_ctc_raw"] = loss_ctc.detach()           # raw value — monitor even during warmup
        loss_dict["loss_ctc"]     = loss_ctc * eff_lambda       # weighted (goes into total)
        loss_dict["loss"]         = (
            loss_dict["loss_text_ce"]
            + loss_dict["loss_mel_ce"]
            + loss_dict["loss_ctc"]
        )
        return {"model_outputs": None}, loss_dict


# ─────────────────────────────────────────────────────────────────────
# Main — mirror train_xtts_vieneu.py with our trainer subclass
# ─────────────────────────────────────────────────────────────────────
def main():
    SPEAKER_REFERENCE = ["/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/wavs/capybara1812_1003_193.wav"]

    config_dataset_train = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="vieneu_train",
        path=SPLITS_DIR,
        meta_file_train="train.csv",
        meta_file_val="val.csv",
        language=LANGUAGE,
    )
    DATASETS_CONFIG_LIST = [config_dataset_train]

    # Mirror train_xtts_vieneu.py model_args + audio_config
    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        debug_loading_failures=False,
        max_wav_length=255995,
        max_text_length=200,
        mel_norm_file=os.path.join(VIXTTS_AUX_DIR, "mel_stats.pth"),
        dvae_checkpoint=os.path.join(VIXTTS_AUX_DIR, "dvae.pth"),
        xtts_checkpoint=os.path.join(RESUME_FROM_FT, "model.pth"),
        tokenizer_file=os.path.join(VIXTTS_DIR, "vocab.json"),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name="xtts_ctc_vieneu",
        project_name="xtts_ctc",
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=4,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=SAVE_EVERY,
        save_n_checkpoints=2,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=LR,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        epochs=1000,
        run_eval=True,
        run_eval_steps=EVAL_EVERY,
        save_best_after=5000,
        test_sentences=[
            {"text": "Đô-rê-mon là một chú mèo máy đến từ thế kỷ 22.",
             "speaker_wav": SPEAKER_REFERENCE, "language": LANGUAGE},
            {"text": "hả ?",
             "speaker_wav": SPEAKER_REFERENCE, "language": LANGUAGE},
        ],
    )

    # Use our subclass, NOT GPTTrainer
    model = GPTTrainerCTC(config=config, lambda_ctc=LAMBDA_CTC)
    print(f"[ctc-trainer] λ_ctc = {LAMBDA_CTC}, warmup = {CTC_WARMUP_STEPS} steps")

    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=0.01,
    )
    print(f"[data] train={len(train_samples)} eval={len(eval_samples)}")

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
