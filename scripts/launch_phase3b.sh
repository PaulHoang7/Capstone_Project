#!/bin/bash
# Launch Phase 3b training — warmstart from Phase 3 G_519000 with higher c_sim.
#
# Hypothesis: Phase 3 plateau'd at cos_sim ~0.35 because c_sim=0.5 was too small
# relative to c_mel=45. Phase 3b uses c_sim=3.0 to push model toward speaker
# identity at some cost to mel quality.
#
# Usage:
#   cd /home/bes/Desktop/Tin
#   bash Capstone_project/scripts/launch_phase3b.sh           # foreground
#   bash Capstone_project/scripts/launch_phase3b.sh --bg      # background via nohup

set -euo pipefail

ROOT="/home/bes/Desktop/Tin"
CONFIG="Capstone_project/configs/vits2_vieneu_clone_phase3b.json"
MODEL_NAME="vieneu_clone_phase3b"
LOG_DIR_NFS="/mnt/nfs-data/tin_dataset/vits2_logs/${MODEL_NAME}"

cd "$ROOT"

# ── Pre-flight checks ──────────────────────────────────────────────────────────
echo "== Phase 3b launch checks =="
echo "Config: $CONFIG"

WARMSTART=$(/home/bes/miniconda3/envs/comic_ocr/bin/python -c "
import json
with open('$CONFIG') as f: c = json.load(f)
print(c['train']['warmstart_ckpt'])
")
echo "Warm-start checkpoint: $WARMSTART"

if [ ! -f "$WARMSTART" ]; then
    echo "ERROR: warmstart checkpoint not found: $WARMSTART"
    exit 1
fi
echo "  Size: $(du -h "$WARMSTART" | cut -f1)"

# ── GPU check ──────────────────────────────────────────────────────────────────
echo ""
echo "== GPU status =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits | awk -F, '{
        printf "GPU: util=%s%%, memory=%s/%s MiB\n", $1, $2, $3
        if ($2 > 2000) {
            printf "WARNING: GPU has %s MiB used — another process may be running\n", $2
        }
    }'

# ── Create log dir ─────────────────────────────────────────────────────────────
mkdir -p "./logs/${MODEL_NAME}"
if [ "$(ls -A "./logs/${MODEL_NAME}" 2>/dev/null)" ]; then
    echo ""
    echo "WARNING: ./logs/${MODEL_NAME}/ is not empty:"
    ls -la "./logs/${MODEL_NAME}/" | head -5
    echo "  (Existing checkpoints will cause resume instead of warmstart)"
    read -p "  Continue anyway? [y/N] " -n 1 -r
    echo ""
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# ── Config summary ─────────────────────────────────────────────────────────────
echo ""
echo "== Config summary =="
/home/bes/miniconda3/envs/comic_ocr/bin/python -c "
import json
with open('$CONFIG') as f: c = json.load(f)
t = c['train']
print(f\"  epochs:         {t['epochs']}\")
print(f\"  learning_rate:  {t['learning_rate']}\")
print(f\"  batch_size:     {t['batch_size']}\")
print(f\"  c_mel:          {t['c_mel']}\")
print(f\"  c_kl:           {t['c_kl']}\")
print(f\"  c_sim:          {t['c_sim']}  ← KEY CHANGE (was 0.5)\")
print(f\"  warmstart_ckpt: {t['warmstart_ckpt']}\")
"

echo ""
read -p "Start training? [y/N] " -n 1 -r
echo ""
[[ ! $REPLY =~ ^[Yy]$ ]] && exit 0

# ── Launch ─────────────────────────────────────────────────────────────────────
CMD="/home/bes/miniconda3/envs/comic_ocr/bin/python \
    Capstone_project/scripts/train_clone_phase3.py \
    -c $CONFIG \
    -m $MODEL_NAME"

if [[ "${1:-}" == "--bg" ]]; then
    LOG_FILE="./logs/${MODEL_NAME}/train_stdout.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "Starting background training..."
    echo "  Log: $LOG_FILE"
    nohup $CMD > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "  PID: $PID"
    echo ""
    echo "Monitor progress:"
    echo "  tail -f $LOG_FILE | grep -oP 'step \\d+ \\| mel=[\\d.]+ kl=[\\d.]+ sim=[\\d.]+ gen=[\\d.]+'"
    echo ""
    echo "Stop training:"
    echo "  kill $PID"
else
    exec $CMD
fi
