#!/usr/bin/env bash
# Manga LoRA v5clean: train SẠCH từ v1 (KHÔNG warm từ v3hires → tránh double-exposure
# csv cũ làm drift như v4). Full combined 8771 (csv cũ + v5 đã sửa), 3 epoch.
# Phép thử: nhãn v5 có giúp khi train đúng cách không? So v3hires (1.59%).
set -euo pipefail
PY=/home/bes/miniconda3/envs/comic_ocr/bin/python
CKPT=/mnt/nfs-data/tin_dataset/checkpoints
cd /home/bes/Desktop/Tin/Capstone_project/voice_cloning/xtts_v2
"$PY" finetune_qwen_lora.py \
    --csv /home/bes/Desktop/Tin/csv/manga_train_v4.csv \
    --crops-dir /home/bes/Desktop/Tin/labeling_task_v4/bubble_crops \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --resume-adapter "$CKPT/qwen25vl_7b_manga_lora_v1/best" \
    --output-dir "$CKPT/qwen25vl_7b_manga_lora_v5clean" \
    --epochs 3 --lr 5e-5 --batch-size 1 --grad-accum 8 \
    --valid-pct 0.03 --eval-every 400 --save-every 99999
echo "##### MANGA LoRA v5clean TRAIN DONE #####"
