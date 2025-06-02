#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

__global__ void reduce_float(const float* input, float* output, int N) {
    __shared__ float sum[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sum[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sum[tid] += sum[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sum[0];
}

__global__ void reduce_double(const double* input, double* output, int N) {
    __shared__ double sum[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sum[tid] = (idx < N) ? input[idx] : 0.0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sum[tid] += sum[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sum[0];
}

int main() {
    const int N = 100000; // 一共要加这么多小数
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;

    // 初始化 host 数据（float 和 double 两份）
    float* h_float = new float[N];
    double* h_double = new double[N];
    for (int i = 0; i < N; ++i) {
        h_float[i] = 1e-4f;
        h_double[i] = 1e-4;
    }

    // 分配 device 内存
    float *d_finput, *d_fpartial;
    double *d_dinput, *d_dpartial;
    cudaMalloc(&d_finput, N * sizeof(float));
    cudaMalloc(&d_fpartial, BLOCKS * sizeof(float));
    cudaMalloc(&d_dinput, N * sizeof(double));
    cudaMalloc(&d_dpartial, BLOCKS * sizeof(double));

    cudaMemcpy(d_finput, h_float, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dinput, h_double, N * sizeof(double), cudaMemcpyHostToDevice);

    // 执行两次规约
    reduce_float<<<BLOCKS, THREADS>>>(d_finput, d_fpartial, N);
    reduce_double<<<BLOCKS, THREADS>>>(d_dinput, d_dpartial, N);

    // 把中间结果拷回 host
    float* h_fpartial = new float[BLOCKS];
    double* h_dpartial = new double[BLOCKS];
    cudaMemcpy(h_fpartial, d_fpartial, BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dpartial, d_dpartial, BLOCKS * sizeof(double), cudaMemcpyDeviceToHost);

    // 在 CPU 上做最终合并
    float f_sum = 0.0f;
    double d_sum = 0.0;
    for (int i = 0; i < BLOCKS; ++i) {
        f_sum += h_fpartial[i];
        d_sum += h_dpartial[i];
    }

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Float  result: " << f_sum << std::endl;
    std::cout << "Double result: " << d_sum << std::endl;
    std::cout << "Expected      : " << N * 1e-4 << std::endl;

    // 清理内存
    delete[] h_float;
    delete[] h_double;
    delete[] h_fpartial;
    delete[] h_dpartial;
    cudaFree(d_finput);
    cudaFree(d_fpartial);
    cudaFree(d_dinput);
    cudaFree(d_dpartial);

    return 0;
}
