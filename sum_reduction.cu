#include <iostream>
#include <cuda_runtime.h>

__global__ void sumReduction(int* input, int* output, int N) {
    __shared__ int shared[256]; // 每个 block 有一块共享内存

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 每个线程将一个输入值写入共享内存
    shared[tid] = (idx < N) ? input[idx] : 0;

    __syncthreads(); // 所有线程必须等这一步做完！

    // reduction：线程对半折叠求和
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        printf("I am thread %d\n", tid);
        __syncthreads(); // 每步都要同步
    }

    // 把每个 block 的和写到输出数组中
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

int main() {
    const int N = 8;
    int h_in[N] = {1,2,3,4,5,6,7,8};
    int h_out;

    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, N * sizeof(int));
    cudaMalloc((void**)&d_out, sizeof(int));

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    sumReduction<<<1, 8>>>(d_in, d_out, N); // 启动 1 个 block，8 个线程
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sum = " << h_out << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
