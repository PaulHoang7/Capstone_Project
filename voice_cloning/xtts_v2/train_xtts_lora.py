"""Phase B2: LoRA fine-tune on XTTS GPT decoder.

Adds rank-16 LoRA adapters to c_attn + c_proj across all 30 GPT2 transformer
layers. Resumes from xtts_vieneu_ft baseline (the same checkpoint that fed
B1 CTC training). Freezes all base params; trains only LoRA delta matrices.

Memory: ~12-15 GB VRAM peak (vs 30+ GB for full GPT FT) — fits next to
external processes like the pfedrec ray cluster.

Output: experiments/xtts_vieneu_lora_v1 — separate dir so v3 + LoRA can be
compared independently in 4-way ablation (baseline / v2 / v3 / lora).

Usage:
    conda run -n xtts_env python train_xtts_lora.py
    # dry-run smoke test (3 steps):
    DRYRUN_STEPS=3 conda run -n xtts_env python train_xtts_lora.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "coqui_tts"))

import torch  # noqa: E402

from trainer import Trainer, TrainerArgs                              # noqa: E402
from TTS.config.shared_configs import BaseDatasetConfig                # noqa: E402
from TTS.tts.datasets import load_tts_samples                          # noqa: E402
from TTS.tts.layers.xtts.trainer.gpt_trainer import (                  # noqa: E402
    GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig,
)
from peft import LoraConfig, get_peft_model                            # noqa: E402


# ─── Paths ───────────────────────────────────────────────────────────
VIXTTS_DIR      = "/mnt/nfs-data/tin_dataset/checkpoints/vixtts"
VIXTTS_AUX_DIR  = "/mnt/nfs-data/tin_dataset/checkpoints/vixtts_aux"
RESUME_FROM_FT  = "/mnt/nfs-data/tin_dataset/checkpoints/xtts_vieneu_ft"
SPLITS_DIR      = "/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/xtts_splits"
OUT_PATH        = "/mnt/nfs-data/tin_dataset/experiments/xtts_vieneu_lora_v1"
LANGUAGE        = "vi"

# ─── LoRA hyperparameters ────────────────────────────────────────────
LORA_RANK       = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.1
LORA_TARGETS    = ["c_attn", "c_proj"]   # HF GPT2 attention modules

# ─── Training hyperparameters ────────────────────────────────────────
BATCH_SIZE       = 4
GRAD_ACUMM_STEPS = 64
LR               = 1e-4         # higher than full FT (LoRA tolerates)
EVAL_EVERY       = 5000
SAVE_EVERY       = 5000
MAX_STEPS        = 30000        # LoRA converges faster than full FT


class GPTTrainerLoRA(GPTTrainer):
    """GPTTrainer with LoRA adapters on the inner HF GPT2.

    Apply LoRA in __init__ AFTER the parent loads the pretrained xtts_checkpoint
    — so the base weights are the well-trained FT, and only the freshly-init'd
    LoRA matrices are trainable.
    """

    def __init__(self, config: GPTTrainerConfig):
        super().__init__(config)

        # Wrap inner HF GPT2 with LoRA.
        # NOTE: task_type=None (not CAUSAL_LM) — XTTS uses GPT2Model with custom
        # loss outside, and CAUSAL_LM task_type makes peft wrap with LM-head logic
        # that passes `labels` kwarg → GPT2Model.forward rejects it.
        peft_cfg = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGETS,
            lora_dropout=LORA_DROPOUT,
            bias="none",
        )
        # self.xtts.gpt.gpt is the HF GPT2 model (GPT2InferenceModel)
        self.xtts.gpt.gpt = get_peft_model(self.xtts.gpt.gpt, peft_cfg)

        # Freeze everything except LoRA params
        for name, p in self.xtts.named_parameters():
            if "lora_" not in name.lower():
                p.requires_grad_(False)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[lora] trainable: {n_trainable/1e6:.2f}M / {n_total/1e6:.0f}M "
              f"({100*n_trainable/n_total:.3f}%)")

        # Track for DRYRUN safety
        self._dry_count = 0

    def train_step(self, batch, criterion):
        # Dry-run safety: abort after DRYRUN_STEPS for smoke test
        _dryrun = int(os.environ.get("DRYRUN_STEPS", "0"))
        if _dryrun > 0 and self._dry_count >= _dryrun:
            print(f"[dryrun] reached step {self._dry_count} ≥ {_dryrun}, exiting cleanly")
            sys.exit(0)
        self._dry_count += 1
        return super().train_step(batch, criterion)


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

    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000,
    )

    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name="xtts_lora_vieneu",
        project_name="xtts_lora",
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
        lr_scheduler_params={"milestones": [10000, 20000, 25000], "gamma": 0.5, "last_epoch": -1},
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

    model = GPTTrainerLoRA(config=config)
    print(f"[lora-trainer] rank={LORA_RANK}, alpha={LORA_ALPHA}, "
          f"targets={LORA_TARGETS}, max_steps={MAX_STEPS}")

    train_samples, eval_samples = load_tts_samples(
        [config_dataset_train],
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
