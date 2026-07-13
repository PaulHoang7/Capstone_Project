"""Fine-tune viXTTS on VieNeu-TTS-140h with heldout evaluation.

Starting point: viXTTS (capleaf/viXTTS) — already pretrained on ~1000h Vietnamese
Fine-tune target: VieNeu-TTS-140h (193 speakers, 140h) — adds speaker diversity

Strategy:
  - GPT decoder fully trainable (quality fine-tune, not LoRA)
  - HiFi-GAN vocoder FROZEN (preserve audio quality)
  - Low LR (5e-6) to avoid catastrophic forgetting of viXTTS's Vietnamese knowledge
  - Heldout speakers 0, 10, 20, ..., 190 NEVER seen in training

Target: heldout cos_sim > 0.55 (vs VITS2+Dual-Path baseline of 0.43).
"""
import os
from pathlib import Path
import urllib.request

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
VIXTTS_DIR       = "/mnt/nfs-data/tin_dataset/checkpoints/vixtts"
CHECKPOINTS_DIR  = "/mnt/nfs-data/tin_dataset/checkpoints/vixtts_aux"
SPLITS_DIR       = "/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/xtts_splits"
OUT_PATH         = "/mnt/nfs-data/tin_dataset/experiments/xtts_vieneu_ft"

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(OUT_PATH, exist_ok=True)

# -----------------------------------------------------------------------------
# Download DVAE + mel_stats (not bundled with viXTTS)
# -----------------------------------------------------------------------------
DVAE_URL      = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_STATS_URL = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
DVAE_CKPT     = os.path.join(CHECKPOINTS_DIR, "dvae.pth")
MEL_STATS     = os.path.join(CHECKPOINTS_DIR, "mel_stats.pth")

for url, path in [(DVAE_URL, DVAE_CKPT), (MEL_STATS_URL, MEL_STATS)]:
    if not os.path.exists(path):
        print(f"[download] {url} -> {path}")
        urllib.request.urlretrieve(url, path)

XTTS_CHECKPOINT = os.path.join(VIXTTS_DIR, "model.pth")
TOKENIZER_FILE  = os.path.join(VIXTTS_DIR, "vocab.json")

# -----------------------------------------------------------------------------
# Hyperparameters (quality fine-tune — not a quick LoRA)
# -----------------------------------------------------------------------------
RUN_NAME         = "viXTTS_VieNeu_FT"
PROJECT_NAME     = "XTTS_VN"
BATCH_SIZE       = 3
GRAD_ACUMM_STEPS = 84   # effective batch = 252 (recommended for XTTS stability)
LR               = 5e-6
LANGUAGE         = "vi"

# Heldout reference (a single VieNeu sample for test sentences during training)
SPEAKER_REFERENCE = ["/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/wavs/capybara1812_1003_193.wav"]

# -----------------------------------------------------------------------------
# Dataset config — use 'coqui' formatter (audio_file|text|speaker_name)
# -----------------------------------------------------------------------------
config_dataset_train = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="vieneu_train",
    path=SPLITS_DIR,
    meta_file_train="train.csv",
    meta_file_val="val.csv",
    language=LANGUAGE,
)

DATASETS_CONFIG_LIST = [config_dataset_train]

# -----------------------------------------------------------------------------
# Model args
# -----------------------------------------------------------------------------
def main():
    model_args = GPTArgs(
        max_conditioning_length=132300,   # 6s @ 22050 Hz
        min_conditioning_length=66150,    # 3s
        debug_loading_failures=False,
        max_wav_length=255995,            # ~11.6s
        max_text_length=200,
        mel_norm_file=MEL_STATS,
        dvae_checkpoint=DVAE_CKPT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # viXTTS — Vietnamese pretrained
        tokenizer_file=TOKENIZER_FILE,
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
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="viXTTS fine-tune on VieNeu-TTS-140h (193 speakers, 140h)",
        dashboard_logger="tensorboard",
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=4,
        eval_split_max_size=256,
        print_step=50,
        plot_step=500,
        log_model_step=5000,
        save_step=10000,
        save_n_checkpoints=3,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=LR,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "Xin chào, đây là hệ thống lồng tiếng tự động cho truyện tranh.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "Tôi đã nói đi nói lại nhiều lần, chúng ta phải hành động ngay bây giờ.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "Mã và mả, hai từ khác nhau nhưng nghe rất giống.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
    )

    model = GPTTrainer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=0.01,
    )
    print(f"[data] train={len(train_samples)}  eval={len(eval_samples)}")

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
