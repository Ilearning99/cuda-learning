#include <stdio.h>

__global__ void hello_from_gpu()
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    printf("Hello World from block id %d and thread id %d\n", bid, tid);
}

int main()
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}