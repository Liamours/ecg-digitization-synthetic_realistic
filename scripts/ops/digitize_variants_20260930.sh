#!/bin/sh
cd /c/researches/research-ecg-digitization/repo/ecg-digitization-synthetic_realistic
for L in mac400_wide_unet mac400_wide_unet_overlap; do
  for i in 1 2 3; do
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.digitize.run --config configs/digitize.yml --layout $L --out-dir ../../inferences/digitize_batch/$L --list ../../logs/digitize_batch_inputs_20260930.txt && break
    echo "retry $i"
  done
  echo "DONE $L"
done
echo VARIANTS_DONE
