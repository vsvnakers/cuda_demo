#include <iostream>
#include <cuda_runtime.h>

__global__ void blockReduceSum(int* input, int* partial, int N) {
    __shared__ int shared[1024];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    shared[tid] = (idx < N) ? input[idx] : 0;
    __syncthreads();

    // 对半规约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = shared[0]; // 每个 block 写出一个和
    }
}

int main() {
    const int N = 100000; // 要加的数数量
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;

    int* h_input = new int[N];
    for (int i = 0; i < N; ++i) h_input[i] = 1; // 所有元素为 1，方便验证

    int *d_input, *d_partial;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_partial, BLOCKS * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    blockReduceSum<<<BLOCKS, THREADS>>>(d_input, d_partial, N);

    int* h_partial = new int[BLOCKS];
    cudaMemcpy(h_partial, d_partial, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

    // 在 CPU 上求和（将每个 block 的和加起来）
    long long total = 0;
    for (int i = 0; i < BLOCKS; ++i) {
        total += h_partial[i];
    }

    std::cout << "Total sum = " << total << std::endl;

    delete[] h_input;
    delete[] h_partial;
    cudaFree(d_input);
    cudaFree(d_partial);

    return 0;
}
