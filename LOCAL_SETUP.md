# local GPU setup

this module adds a small CUDA C++ and Triton smoke path to lmlab. it verifies the local
toolchain before later GPU primitive study; it does not implement the curriculum exercises.

## verified machine target

- Windows 11 with WSL2 Ubuntu
- NVIDIA GeForce RTX 3080, 10 GB VRAM, compute capability 8.6
- Windows NVIDIA driver 610.47
- CUDA toolkit 13.3 inside WSL
- Python 3.12 managed by uv

the RTX 3080 supports the early CUDA and Triton work. it cannot execute Hopper- or
Blackwell-specific mechanisms such as TMA, FlashAttention-3 kernels, DeepGEMM, or Blackwell
tensor-memory instructions. Distributed inference work also needs remote or additional
hardware.

## verify the module

create lmlab's existing environment and run the verifier:

```bash
uv sync
./scripts/verify_cuda_setup.sh
```

the verifier reports the GPU, compiler and profiler versions, checks PyTorch CUDA access,
compiles and runs the CUDA smoke kernel under Compute Sanitizer, and JIT-compiles the Triton
smoke kernel. `CUDA_HOME` defaults to `/usr/local/cuda-13.3`, `CUDA_ARCH` defaults to `sm_86`,
and `GPU_PYTHON` can point to a separate compatible environment when a later project pins a
different Python or framework stack.

`notebooks/cuda_setup.ipynb` exposes the same smoke path as separate Python cells. it compiles
the editable `src/gpu_primitives/smoke_cuda.cu` file with `nvcc`; no C++ notebook kernel is
required. the notebook is intentionally committed without execution outputs so local evidence
stays tied to the machine that produced it.

in an earlier check on this machine, Nsight Systems tracing worked and Nsight Compute could
launch the smoke program, while hardware metrics remained locked. Enabling GPU performance
counters in the Windows NVIDIA App under System > Advanced > Developer should make those
metrics available; this verifier does not retest profiler capture or counter permissions.

do not install Linux NVIDIA display drivers inside WSL; the Windows driver supplies the WSL
GPU interface.
