#!/usr/bin/env bash
# Round 2: target v2's ACTUAL errors (not prefill's). Fix the round-1 `correct` regression
# with lower LR (2e-5), 1 epoch, and all-correct-rows kept while error rows are x2-duplicated.
set -euo pipefail

ENV=comic_ocr
HERE=/home/bes/Desktop/Tin/Capstone_project/voice_cloning/xtts_v2
CROPS=/home/bes/Desktop/Tin/labeling_task_v4/bubble_crops_vn
CSVD=/home/bes/Desktop/Tin/csv
TRAIN=$CSVD/ocr_ft_train.csv
TRAIN_R2=$CSVD/ocr_ft_train_r2.csv
EVAL=$CSVD/ocr_ft_eval.csv
CKPT=/mnt/nfs-data/tin_dataset/checkpoints
V2=$CKPT/qwen25vl_7b_vncomic_uppercase_lora_v2/best
V3B=$CKPT/qwen25vl_7b_vncomic_uppercase_lora_v3b
MODEL=Qwen/Qwen2.5-VL-7B-Instruct
cd "$HERE"

echo "==== [1/3] v2 predictions on full train (find v2's REAL errors) ===="
# Resume-friendly: skip if predictions already exist
if [ ! -s "$CSVD/train_v2_pred.csv" ]; then
  conda run -n $ENV python eval_ocr_bubble.py \
      --csv "$TRAIN" --crops-dir "$CROPS" --adapter "$V2" \
      --out-csv "$CSVD/train_v2_pred.csv"
else
  echo "  (train_v2_pred.csv exists — reuse)"
fi

echo "==== [2a] relabel from v2-pred, build round-2 targeted train ===="
conda run -n $ENV python build_r2_train.py \
    --pred "$CSVD/train_v2_pred.csv" --full "$TRAIN" --out "$TRAIN_R2" --err-dup 2

echo "==== [2b] TRAIN warm-start v2 -> v3b (lr 2e-5, 1 epoch) ===="
conda run -n $ENV python finetune_qwen_lora.py \
    --csv "$TRAIN_R2" --crops-dir "$CROPS" \
    --model-id "$MODEL" --resume-adapter "$V2" --output-dir "$V3B" \
    --epochs 1 --lr 2e-5 --batch-size 1 --grad-accum 8 \
    --valid-pct 0.05 --eval-every 99999 --save-every 99999

echo "==== [3/3] eval v3b on the SAME held-out (vs v2 baseline 5.90%) ===="
conda run -n $ENV python eval_ocr_bubble.py \
    --csv "$EVAL" --crops-dir "$CROPS" --adapter "$V3B/final" \
    --out-csv "$CSVD/eval_pred_v3b.csv"

echo
echo "DONE. Compare v3b table below vs round-1 baseline:"
echo "  v2  : all 5.90 | correct 3.16 | diacritic 5.52 | structural 9.17"
echo "  GOAL: diacritic down, correct NOT up (regression fixed)."
