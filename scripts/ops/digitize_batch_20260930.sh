#!/bin/sh
cd /c/researches/research-ecg-digitization/repo/ecg-digitization-synthetic_realistic
for i in 1 2 3 4 5; do
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.digitize.run --config configs/digitize.yml --layout mac400 --out-dir ../../inferences/digitize_batch/baseline --list ../../logs/digitize_batch_inputs_20260930.txt && break
  echo "retry $i"
done
echo BATCH_DONE
