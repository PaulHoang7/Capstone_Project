#!/usr/bin/env bash
# Round 3b (nhẹ tay, trị overfit ở v4): retrain trên GT đã CHUẨN HOÁ QUY ƯỚC (in hoa, '!' tách space, bỏ space
# trước dấu khác, '…'->'...', loại dòng Cyrillic). Giả thuyết: 62% label TĐĐV
# lệch quy ước là nguồn nhiễu chính khiến output dao động case/dấu câu.
# Warm-start v2 -> v4_norm, cùng held-out split cũ (GT eval cũng đã normalize).
set -euo pipefail

# Gọi thẳng binary của env (pyenv shim đè PATH làm `conda run` trỏ nhầm python 3.7)
PY=/home/bes/miniconda3/envs/comic_ocr/bin/python
HERE=/home/bes/Desktop/Tin/Capstone_project/voice_cloning/xtts_v2
CROPS=/home/bes/Desktop/Tin/labeling_task_v4/bubble_crops_vn
CSVD=/home/bes/Desktop/Tin/csv
TRAIN=$CSVD/ocr_ft_train_norm.csv
EVAL=$CSVD/ocr_ft_eval_norm.csv
CKPT=/mnt/nfs-data/tin_dataset/checkpoints
V2=$CKPT/qwen25vl_7b_vncomic_uppercase_lora_v2/best
V4=$CKPT/qwen25vl_7b_vncomic_uppercase_lora_v4b_norm
MODEL=Qwen/Qwen2.5-VL-7B-Instruct
cd "$HERE"

echo "==== [1/2] TRAIN warm-start v2 -> v4_norm trên GT chuẩn hoá ===="
"$PY" finetune_qwen_lora.py \
    --csv "$TRAIN" --crops-dir "$CROPS" \
    --model-id "$MODEL" --resume-adapter "$V2" --output-dir "$V4" \
    --epochs 1 --lr 2e-5 --batch-size 1 --grad-accum 8 \
    --valid-pct 0.05 --eval-every 300 --save-every 600

echo "==== [2/2] EVAL v4_norm/final trên held-out (GT normalize) ===="
"$PY" eval_ocr_bubble.py \
    --csv "$EVAL" --crops-dir "$CROPS" --adapter "$V4/final" \
    --out-csv "$CSVD/eval_pred_v4b_norm.csv"

echo "DONE."
