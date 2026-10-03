#!/bin/bash
# Keeps models/gain_field_detection/yolo11n_v2/args.yaml pinned at
# epochs=20/patience=5 until the queue actually starts that step, in case
# something rewrites it back to 100/100 before then (observed once,
# 2026-09-23, cause unconfirmed). Stops itself once the queue log shows
# gain_field_detection has started or the queue is done.
ARGS=models/gain_field_detection/yolo11n_v2/args.yaml
QLOG=logs/train_queue_20260922.log
cd "$(dirname "$0")/.." || exit 1

while true; do
  if grep -q "START gain_field_detection\|QUEUE DONE" "$QLOG" 2>/dev/null; then
    exit 0
  fi
  sed -i 's/^epochs: 100$/epochs: 20/; s/^patience: 100$/patience: 5/' "$ARGS"
  sleep 30
done
