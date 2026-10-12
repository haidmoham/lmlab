#!/usr/bin/env bash

set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-13.3}"
export CUDA_HOME="$cuda_home"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

require_command() {
  local command_name="$1"

  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "missing required command: $command_name" >&2
    return 1
  fi
}

for command_name in nvidia-smi nvcc compute-sanitizer ncu nsys; do
  require_command "$command_name"
done

python_path="${GPU_PYTHON:-$workspace_root/.venv/bin/python}"
if [[ ! -x "$python_path" ]]; then
  echo "missing project environment: run uv sync in $workspace_root" >&2
  echo "or set GPU_PYTHON to a compatible CUDA Python environment" >&2
  exit 1
fi

echo "gpu"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader

echo
echo "toolchain"
nvcc --version | tail -n 1
compute-sanitizer --version | head -n 1
ncu --version | sed -n '/Version/p'
nsys --version | head -n 1

echo
echo "python"
"$python_path" - <<'PY'
import sys

import torch
import triton

print(sys.version.split()[0])
print(f"torch={torch.__version__}")
print(f"triton={triton.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
print(f"cuda_runtime={torch.version.cuda}")
print(f"device={torch.cuda.get_device_name(0)}")
print(f"capability={torch.cuda.get_device_capability(0)}")
PY

echo
echo "cuda c++ execution"
mkdir -p "$workspace_root/.build"
nvcc \
  -arch="${CUDA_ARCH:-sm_86}" \
  -std=c++17 \
  "$workspace_root/src/gpu_primitives/smoke_cuda.cu" \
  -o "$workspace_root/.build/smoke_cuda"
"$workspace_root/.build/smoke_cuda"
compute-sanitizer \
  --tool memcheck \
  --error-exitcode 1 \
  "$workspace_root/.build/smoke_cuda"

echo
echo "triton execution"
"$python_path" "$workspace_root/src/gpu_primitives/smoke_triton.py"
