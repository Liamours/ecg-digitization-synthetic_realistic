#!/bin/bash
# Build the GPU environment the trace stage runs in (the repo's own .venv has CPU-only torch): ../../.venvs/digitize_gpu
# with CUDA 12.8 builds of torch and torchvision and every other dependency, so the whole pipeline (OCR and trace network) runs in it.
# Usage: scripts/setup_gpu_env.sh
set -e
cd "$(dirname "$0")/.."
env=../../.venvs/digitize_gpu
uv venv "$env" --python 3.12
py=$env/Scripts/python.exe
[ -f "$py" ] || py=$env/bin/python
uv pip install --python "$py" torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$py" docling matplotlib numpy opencv-python-headless pdf2image pillow pyyaml scikit-image scipy tqdm
"$py" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
