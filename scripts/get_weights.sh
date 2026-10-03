#!/bin/bash
# Download the pretrained trace network of Open-ECG-Digitizer (90 MB, CC BY-SA 4.0) to where configs/digitize.yml expects it.
# Usage: scripts/get_weights.sh
set -e
cd "$(dirname "$0")/.."
dir=../../models/open_ecg_digitizer-pretrained
file=$dir/unet_weights_07072025.pt
mkdir -p "$dir"
if [ -s "$file" ]; then echo "already there: $file"; exit 0; fi
curl -L --fail -o "$file" https://github.com/Ahus-AIM/Open-ECG-Digitizer/raw/main/weights/unet_weights_07072025.pt
echo "saved $file ($(wc -c < "$file") bytes, expected 90464067)"
