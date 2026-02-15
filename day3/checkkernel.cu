#include "error.cuh"
#include <stdlib.h>
#include <stdio.h>

__global__ void hello_from_gpu()
{
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    printf("hello from thread %d\n", tid);
}

int main()
{
    int blockSize=1028, gridSize=2;
    hello_from_gpu<<<gridSize, blockSize>>>();
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    return 0;
}