#include <cuda_runtime.h>

#include <cstdio>

__global__ void add_one(int* values, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] += 1;
    }
}

int main() {
    constexpr int value_count = 256;
    int* values = nullptr;

    cudaError_t status = cudaMallocManaged(&values, value_count * sizeof(int));
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(status));
        return 1;
    }

    for (int index = 0; index < value_count; ++index) {
        values[index] = index;
    }

    add_one<<<1, value_count>>>(values, value_count);
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::fprintf(stderr, "kernel execution failed: %s\n", cudaGetErrorString(status));
        cudaFree(values);
        return 1;
    }

    for (int index = 0; index < value_count; ++index) {
        if (values[index] != index + 1) {
            std::fprintf(stderr, "unexpected result at %d: %d\n", index, values[index]);
            cudaFree(values);
            return 1;
        }
    }

    cudaFree(values);
    std::puts("cuda c++ smoke passed on 256 values");
    return 0;
}
