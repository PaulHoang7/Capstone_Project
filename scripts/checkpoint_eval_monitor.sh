#!/bin/bash
# =============================================================================
# Variant B Checkpoint Evaluation Monitor
# Watches training and triggers evaluation at specified step milestones.
# Usage: bash Capstone_project/scripts/checkpoint_eval_monitor.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/home/bes/Desktop/TTS"
CKPT_DIR="${PROJECT_ROOT}/vits2_pytorch/logs/vieneu_variant_b"
CONFIG="${PROJECT_ROOT}/Capstone_project/configs/vits2_vieneu_variant_b.json"
TEST_SET="${PROJECT_ROOT}/Capstone_project/evaluation/tone_test_set.json"
EVAL_SCRIPT="${PROJECT_ROOT}/Capstone_project/evaluation/evaluate_tts.py"
EXPERIMENT_BASE="/mnt/nfs-data/tin_dataset/experiments"

# Checkpoints to evaluate
EVAL_STEPS=(100000 120000 150000)

# Track which have been evaluated
DONE_FILE="${CKPT_DIR}/eval_done_steps.txt"
touch "${DONE_FILE}"

POLL_INTERVAL=120  # seconds between checks

echo "========================================"
echo " Variant B Checkpoint Eval Monitor"
echo "========================================"
echo " Watching:  ${CKPT_DIR}"
echo " Eval at:   ${EVAL_STEPS[*]}"
echo " Poll:      every ${POLL_INTERVAL}s"
echo "========================================"

while true; do
    ALL_DONE=true

    for STEP in "${EVAL_STEPS[@]}"; do
        # Skip if already evaluated
        if grep -q "^${STEP}$" "${DONE_FILE}" 2>/dev/null; then
            continue
        fi

        ALL_DONE=false
        CKPT_FILE="${CKPT_DIR}/G_${STEP}.pth"

        if [ -f "${CKPT_FILE}" ]; then
            echo ""
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checkpoint found: G_${STEP}.pth"
            echo "Starting evaluation for step ${STEP}..."

            OUTPUT_DIR="${EXPERIMENT_BASE}/variant_B_step${STEP}"
            mkdir -p "${OUTPUT_DIR}"

            # Run evaluation
            cd "${PROJECT_ROOT}"
            python "${EVAL_SCRIPT}" \
                --config "${CONFIG}" \
                --checkpoint "${CKPT_FILE}" \
                --test-set "${TEST_SET}" \
                --output-dir "${OUTPUT_DIR}" \
                --variant-name "B_tone_emb_step${STEP}" \
                --use-tone \
                --whisper-model large-v3 \
                2>&1 | tee "${OUTPUT_DIR}/eval.log"

            EVAL_EXIT=$?
            if [ ${EVAL_EXIT} -eq 0 ]; then
                echo "${STEP}" >> "${DONE_FILE}"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation complete for step ${STEP}"
                echo "  Results: ${OUTPUT_DIR}/results.json"

                # Print quick summary
                if [ -f "${OUTPUT_DIR}/results.json" ]; then
                    echo ""
                    echo "--- Quick Summary (step ${STEP}) ---"
                    python3 -c "
import json
with open('${OUTPUT_DIR}/results.json') as f:
    r = json.load(f)
tc = r.get('tone_confusion', {})
if tc:
    print(f'  Tone Accuracy: {tc.get(\"overall_accuracy\", \"N/A\")}')
f0 = r.get('f0_rmse', {})
if f0:
    print(f'  F0 RMSE:       {f0.get(\"f0_rmse\", \"N/A\")} Hz')
mcd = r.get('mcd', {})
if mcd:
    print(f'  MCD:           {mcd.get(\"mcd\", \"N/A\")} dB')
ss = r.get('speaker_similarity', {})
if ss:
    print(f'  Speaker Sim:   {ss.get(\"speaker_similarity\", \"N/A\")}')
"
                    echo "-----------------------------------"
                fi
            else
                echo "[WARN] Evaluation failed for step ${STEP} (exit code: ${EVAL_EXIT})"
                echo "  Check log: ${OUTPUT_DIR}/eval.log"
            fi
        fi
    done

    # Exit if all milestones evaluated
    if [ "${ALL_DONE}" = true ]; then
        echo ""
        echo "========================================"
        echo " All checkpoint evaluations complete!"
        echo "========================================"
        echo ""
        echo " Comparison:"
        for STEP in "${EVAL_STEPS[@]}"; do
            DIR="${EXPERIMENT_BASE}/variant_B_step${STEP}"
            if [ -f "${DIR}/results.json" ]; then
                python3 -c "
import json
with open('${DIR}/results.json') as f:
    r = json.load(f)
tc = r.get('tone_confusion', {})
acc = tc.get('overall_accuracy', 'N/A') if tc else 'N/A'
f0 = r.get('f0_rmse', {}).get('f0_rmse', 'N/A')
mcd = r.get('mcd', {}).get('mcd', 'N/A')
print(f'  Step {STEP:>7d}: Tone={acc}  F0={f0}  MCD={mcd}')
"
            fi
        done
        echo ""
        echo " Choose best checkpoint and proceed to Variant C."
        break
    fi

    sleep "${POLL_INTERVAL}"
done
