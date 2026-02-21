#include <stdio.h>

typedef float real;

__global__ void addGpu(const real *x, const real *y, real *z, const int N)
{
    int id=blockDim.x*blockIdx.x+threadIdx.x;
    if(id<N)
    {
        z[id]=x[id]*x[id]+y[id]*y[id];
    }
}

int main()
{
    int N=100000000;
    real *x, *y, *z;
    real *d_x, *d_y, *d_z;
    int M=N*sizeof(real);
    x=(real *)malloc(M);
    y=(real *)malloc(M);
    z=(real *)malloc(M);
    for(int i=0;i<N;i++)
    {
        x[i]=3.0;
        y[i]=4.0;
    }
    
    int bs=256;
    int gs=(N+bs-1)/bs;

    cudaMalloc((void**)&d_x,M);
    cudaMalloc((void**)&d_y,M);
    cudaMalloc((void**)&d_z,M);
    cudaMemcpy(d_x,x,M,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,y,M,cudaMemcpyHostToDevice);
    addGpu<<<gs,bs>>>(d_x,d_y,d_z,N);
    cudaMemcpy(z,d_z,M,cudaMemcpyDeviceToHost);
    printf("z[0]=%g\n",z[0]);
    free(x);
    free(y);
    free(z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}