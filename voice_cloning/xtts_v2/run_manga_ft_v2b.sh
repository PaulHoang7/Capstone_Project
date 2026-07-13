#!/usr/bin/env bash
# Manga LoRA v2 RE-RUN (lần trước bị ngắt ở ~400 steps). Train tới hết rồi eval `final`.
# Baseline v1 đã có (csv/naruto_heldout_pred_v1.csv); bỏ qua bước đó.
set -euo pipefail

PY=/home/bes/miniconda3/envs/comic_ocr/bin/python
HERE=/home/bes/Desktop/Tin/Capstone_project/voice_cloning/xtts_v2
CSVD=/home/bes/Desktop/Tin/csv
CKPT=/mnt/nfs-data/tin_dataset/checkpoints
V1=$CKPT/qwen25vl_7b_manga_lora_v1/best
V2=$CKPT/qwen25vl_7b_manga_lora_v2
MCROPS=/home/bes/Desktop/Tin/labeling_task_v4/bubble_crops
NCROPS=/home/bes/Desktop/Tin/zip/naruto_mining/crops
cd "$HERE"

echo "==== [1/3] TRAIN warm-start v1 -> v2 (tới hết) ===="
"$PY" finetune_qwen_lora.py \
    --csv "$CSVD/manga_train_v2.csv" --crops-dir "$MCROPS" \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --resume-adapter "$V1" --output-dir "$V2" \
    --epochs 2 --lr 5e-5 --batch-size 1 --grad-accum 8 \
    --valid-pct 0.03 --eval-every 400 --save-every 99999

echo "==== [2/3] EVAL v2/final trên heldout naruto mới (100) ===="
"$PY" eval_ocr_bubble.py --csv "$CSVD/naruto_heldout_new.csv" --crops-dir "$NCROPS" \
    --adapter "$V2/final" --out-csv "$CSVD/naruto_heldout_pred_v2final.csv"

echo "==== [3/3] EVAL v2/final trên heldout cũ 49 (anti-forgetting) ===="
"$PY" eval_ocr_bubble.py --csv /home/bes/Desktop/Tin/labeling_task_v4/heldout_manga_50.csv \
    --crops-dir "$MCROPS" --adapter "$V2/final" --out-csv "$CSVD/heldout_old_pred_v2final.csv"

echo "ALL DONE."
