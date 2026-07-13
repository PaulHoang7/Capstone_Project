#!/bin/bash
# Run YOLO training in background — detached from terminal.
# Log saved to /mnt/nfs-data/tin_dataset/logs/yolo_train.log
#
# Usage:
#   bash Capstone_project/scripts/train_background.sh              # full training
#   bash Capstone_project/scripts/train_background.sh --subset     # subset test
#   bash Capstone_project/scripts/train_background.sh --resume     # resume
#
# Monitor:
#   tail -f /mnt/nfs-data/tin_dataset/logs/yolo_train.log
#   cat /mnt/nfs-data/tin_dataset/logs/yolo_train.pid
#
# Stop:
#   kill $(cat /mnt/nfs-data/tin_dataset/logs/yolo_train.pid)

LOG_DIR="/mnt/nfs-data/tin_dataset/logs"
LOG_FILE="$LOG_DIR/yolo_train.log"
PID_FILE="$LOG_DIR/yolo_train.pid"

mkdir -p "$LOG_DIR"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Training is already running (PID=$OLD_PID)"
        echo "  Log:  tail -f $LOG_FILE"
        echo "  Stop: kill $OLD_PID"
        exit 1
    fi
fi

# Launch detached
nohup python -u Capstone_project/scripts/train_yolo_manga109.py "$@" \
    >> "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$PID_FILE"

echo "Training started in background."
echo "  PID:  $PID"
echo "  Log:  $LOG_FILE"
echo "  Monitor: tail -f $LOG_FILE"
echo "  Stop:    kill $PID"
