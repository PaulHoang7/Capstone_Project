#!/usr/bin/env bash
# Manga LoRA v2: thêm 137 naruto-hires hard examples (mined+labeled+QC) + 43 dup lỗi
# + 203 agree pseudo-labels vào 5680 label cũ. Warm-start v1 -> v2 (2ep, lr 5e-5).
# Eval kép: heldout naruto mới (100) + heldout cũ 49 (kiểm tra không quên 7 series).
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

echo "==== [1/4] BASELINE v1 trên heldout naruto mới (100) ===="
"$PY" eval_ocr_bubble.py --csv "$CSVD/naruto_heldout_new.csv" --crops-dir "$NCROPS" \
    --adapter "$V1" --out-csv "$CSVD/naruto_heldout_pred_v1.csv"

echo "==== [2/4] TRAIN warm-start v1 -> v2 ===="
"$PY" finetune_qwen_lora.py \
    --csv "$CSVD/manga_train_v2.csv" --crops-dir "$MCROPS" \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --resume-adapter "$V1" --output-dir "$V2" \
    --epochs 2 --lr 5e-5 --batch-size 1 --grad-accum 8 \
    --valid-pct 0.03 --eval-every 400 --save-every 800

echo "==== [3/4] EVAL v2 trên heldout naruto mới ===="
"$PY" eval_ocr_bubble.py --csv "$CSVD/naruto_heldout_new.csv" --crops-dir "$NCROPS" \
    --adapter "$V2/final" --out-csv "$CSVD/naruto_heldout_pred_v2.csv"

echo "==== [4/4] EVAL v2 trên heldout cũ 49 (anti-forgetting check) ===="
"$PY" eval_ocr_bubble.py --csv /home/bes/Desktop/Tin/labeling_task_v4/heldout_manga_50.csv \
    --crops-dir "$MCROPS" --adapter "$V2/final" --out-csv "$CSVD/heldout_old_pred_v2.csv"

echo "DONE."
