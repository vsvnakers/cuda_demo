#include <stdio.h>

__global__ void printThreadIds() {
    printf("I'm thread %d of block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    // Launch the kernel with 1 block and 10 threads
    printThreadIds<<<1, 10>>>();
    
    // Wait for the GPU to finish
    cudaDeviceSynchronize();
    
    return 0;
}
