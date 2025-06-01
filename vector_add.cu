#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const int* A, const int* B, int* C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 16;
    size_t size = N * sizeof(int);

    // Host memory (CPU)
    int h_A[N], h_B[N], h_C[N];

    // 初始化输入数据
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Device memory (GPU)
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 拷贝数据到 GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 每个 block 8 个线程，共 (N+7)/8 个 block
    int threadsPerBlock = 8;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 启动 GPU kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    // 结果拷回 CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 打印结果
    std::cout << "A + B = C:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << "\n";
    }

    // 释放 GPU 内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
