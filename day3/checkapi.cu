#include "error.cuh"
#include <stdlib.h>

int main()
{
    const int M=sizeof(double)*10;
    double *h=(double*)malloc(M);
    for (int i=0;i<10;i++)
    {
        h[i]=1.0;
    }
    double *d;
    CHECK(cudaMalloc((void**)&d,M));
    CHECK(cudaMemcpy(d,h,M,cudaMemcpyDeviceToHost));
    free(h);
    cudaFree(d);
    return 0;
}