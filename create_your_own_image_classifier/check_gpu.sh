#!/bin/bash

echo "=== WSL Version Check ==="
cat /proc/version

echo -e "\n=== Checking for NVIDIA GPU driver in WSL ==="
ls -la /usr/lib/wsl/lib/ | grep -i nvidia

echo -e "\n=== Testing nvidia-smi ==="
/usr/lib/wsl/lib/nvidia-smi 2>&1 || echo "nvidia-smi not found or failed"

echo -e "\n=== PyTorch CUDA Check ==="
$HOME/miniconda3/envs/aipnd/bin/python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
