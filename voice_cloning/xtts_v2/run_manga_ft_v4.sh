#!/usr/bin/env bash
# Manga LoRA v4: warm-start v2 + 2824 hard-example mới (Conan/Doraemon/Shin hi-res
# v5 mining, đã label). Tổng train 8771. Nhắm giảm lỗi dấu thanh (chẩn đoán: 2.4%).
set -euo pipefail
PY=/home/bes/miniconda3/envs/comic_ocr/bin/python
CKPT=/mnt/nfs-data/tin_dataset/checkpoints
cd /home/bes/Desktop/Tin/Capstone_project/voice_cloning/xtts_v2
"$PY" finetune_qwen_lora.py \
    --csv /home/bes/Desktop/Tin/csv/manga_train_v4.csv \
    --crops-dir /home/bes/Desktop/Tin/labeling_task_v4/bubble_crops \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --resume-adapter "$CKPT/qwen25vl_7b_manga_lora_v3hires/final" \
    --output-dir "$CKPT/qwen25vl_7b_manga_lora_v4" \
    --epochs 2 --lr 5e-5 --batch-size 1 --grad-accum 8 \
    --valid-pct 0.03 --eval-every 400 --save-every 800
echo "##### MANGA LoRA v4 TRAIN DONE #####"
