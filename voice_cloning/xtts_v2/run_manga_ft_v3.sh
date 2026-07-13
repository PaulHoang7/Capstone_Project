#!/usr/bin/env bash
# Manga LoRA v3: old 5680 + 240 hi-res Doraemon/OnePiece (người-kiểm) + 61 dup lỗi.
# KHÔNG pseudo-label (đã hại v2). Image paths tuyệt đối -> crops-dir bỏ qua cho new rows.
# Eval: heldout hi-res 60 (30+30) + heldout cũ 49 (anti-forgetting).
set -euo pipefail
PY=/home/bes/miniconda3/envs/comic_ocr/bin/python
HERE=/home/bes/Desktop/Tin/Capstone_project/voice_cloning/xtts_v2
CSVD=/home/bes/Desktop/Tin/csv
CKPT=/mnt/nfs-data/tin_dataset/checkpoints
V1=$CKPT/qwen25vl_7b_manga_lora_v1/best
V3=$CKPT/qwen25vl_7b_manga_lora_v3hires
MCROPS=/home/bes/Desktop/Tin/labeling_task_v4/bubble_crops
cd "$HERE"

echo "==== [1/4] BASELINE v1 trên heldout hi-res 60 ===="
$PY eval_ocr_bubble.py --csv "$CSVD/manga_heldout_hires.csv" --crops-dir "$MCROPS" \
    --adapter "$V1" --out-csv "$CSVD/hires_heldout_v1.csv"

echo "==== [2/4] TRAIN warm-start v1 -> v3hires ===="
$PY finetune_qwen_lora.py \
    --csv "$CSVD/manga_train_v3.csv" --crops-dir "$MCROPS" \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --resume-adapter "$V1" --output-dir "$V3" \
    --epochs 2 --lr 5e-5 --batch-size 1 --grad-accum 8 \
    --valid-pct 0.03 --eval-every 400 --save-every 99999

echo "==== [3/4] EVAL v3/final trên heldout hi-res 60 ===="
$PY eval_ocr_bubble.py --csv "$CSVD/manga_heldout_hires.csv" --crops-dir "$MCROPS" \
    --adapter "$V3/final" --out-csv "$CSVD/hires_heldout_v3.csv"

echo "==== [4/4] EVAL v3/final trên heldout cũ 49 (anti-forgetting) ===="
$PY eval_ocr_bubble.py --csv /home/bes/Desktop/Tin/labeling_task_v4/heldout_manga_50.csv \
    --crops-dir "$MCROPS" --adapter "$V3/final" --out-csv "$CSVD/old_heldout_v3.csv"
echo "ALL DONE."
