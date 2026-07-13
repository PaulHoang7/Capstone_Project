#!/usr/bin/env bash
# Targeted VN-comic OCR fine-tune: BEFORE (v2) -> TRAIN (warm-start v2 -> v3) -> AFTER (v3).
# Honest measurement: eval set = v2's OWN held-out valid rows (v2 never trained on them),
# and v3's train CSV excludes them — so the before/after delta is unbiased for both models.
#
# Run on the GPU box (bes-ai-machine-02). 7B in bf16 + LoRA fits in ~24 GB.
set -euo pipefail

ENV=comic_ocr
HERE=/home/bes/Desktop/Tin/Capstone_project/voice_cloning/xtts_v2
CROPS=/home/bes/Desktop/Tin/labeling_task_v4/bubble_crops_vn   # VN crops (NOT bubble_crops = manga, same filenames/different images!)
TRAIN=/home/bes/Desktop/Tin/csv/ocr_ft_train.csv
EVAL=/home/bes/Desktop/Tin/csv/ocr_ft_eval.csv
CKPT=/mnt/nfs-data/tin_dataset/checkpoints
V2=$CKPT/qwen25vl_7b_vncomic_uppercase_lora_v2/best
V3=$CKPT/qwen25vl_7b_vncomic_uppercase_lora_v3
MODEL=Qwen/Qwen2.5-VL-7B-Instruct

cd "$HERE"

echo "===================================================================="
echo "[1/3] BEFORE — eval v2 on held-out ($EVAL)"
echo "===================================================================="
conda run -n $ENV python eval_ocr_bubble.py \
    --csv "$EVAL" --crops-dir "$CROPS" \
    --adapter "$V2" \
    --out-csv /home/bes/Desktop/Tin/csv/eval_pred_v2.csv

echo "===================================================================="
echo "[2/3] TRAIN — warm-start v2 -> v3 on targeted error set ($TRAIN)"
echo "  low LR (5e-5) + 2 epochs: nudge toward errors, avoid forgetting"
echo "===================================================================="
conda run -n $ENV python finetune_qwen_lora.py \
    --csv "$TRAIN" --crops-dir "$CROPS" \
    --model-id "$MODEL" \
    --resume-adapter "$V2" \
    --output-dir "$V3" \
    --epochs 2 --lr 5e-5 --batch-size 1 --grad-accum 8 \
    --valid-pct 0.05 --eval-every 300 --save-every 600

echo "===================================================================="
echo "[3/3] AFTER — eval v3 on the SAME held-out"
echo "  (eval 'final': trainer's internal 'best' is picked on a contaminated valid"
echo "   split — v2 already saw those rows — so it's not a reliable selector here)"
echo "===================================================================="
conda run -n $ENV python eval_ocr_bubble.py \
    --csv "$EVAL" --crops-dir "$CROPS" \
    --adapter "$V3/final" \
    --out-csv /home/bes/Desktop/Tin/csv/eval_pred_v3.csv

echo
echo "DONE. Compare 'all' CER between the [1/3] and [3/3] tables."
echo "  v2 preds: csv/eval_pred_v2.csv   |   v3 preds: csv/eval_pred_v3.csv"
echo "  Also check 'diacritic' row (where the win should show) and 'correct' row"
echo "  (must NOT regress — that would mean over-correction / forgetting)."
