#include<stdio.h>
#include<stdlib.h>

__global__ void hello()
{
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    printf("Hello Thread! x=%d, y=%d, Block x=%d, Block y=%d, Thread x=%d, Thread y=%d\n",x,y,blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y);
}

int main()
{
    dim3 blockSize(4,2);
    dim3 gridSize(1,2);
    hello<<<gridSize, blockSize>>>();
    return 0;
}