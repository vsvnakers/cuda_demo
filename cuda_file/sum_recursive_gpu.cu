#include <iostream>
#include <cuda_runtime.h>

__global__ void reduceKernel(int* input, int* output, int N) {
    __shared__ int shared[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    shared[tid] = (idx < N) ? input[idx] : 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = shared[0];
}

int main() {
    const int N = 100000;
    const int THREADS = 256;
    const int BLOCKS_1 = (N + THREADS - 1) / THREADS;

    int* h_input = new int[N];
    for (int i = 0; i < N; ++i) h_input[i] = 1;

    int *d_input, *d_intermediate, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_intermediate, BLOCKS_1 * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // 第 1 轮 reduction
    reduceKernel<<<BLOCKS_1, THREADS>>>(d_input, d_intermediate, N);

    // 第 2 轮 reduction：对 intermediate 做规约
    int BLOCKS_2 = 1; // 足够小了
    reduceKernel<<<BLOCKS_2, THREADS>>>(d_intermediate, d_output, BLOCKS_1);

    int h_result;
    cudaMemcpy(&h_result, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Total sum = " << h_result << std::endl;

    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_intermediate);
    cudaFree(d_output);

    return 0;
}
// 纯 GPU - Reduction
// 第 1 轮把 10 万个数缩成 ~400 个 block → 得到 400 个“局部和”
// 第 2 轮用 1 个 block（256 线程）就可以轻松处理这 400 个中间和
