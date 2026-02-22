#include <stdlib.h>
#include <stdio.h>
#include <time.h>

typedef float real;

__global__ void addGpu(const real *x, const real *y, real *z, const int N)
{
    int id=blockDim.x*blockIdx.x+threadIdx.x;
    if(id<N)
    {
        z[id]=x[id]*x[id]+y[id]*y[id];
    }
}

void addCpu(const real *x, const real *y, real *z, const int N)
{
    for(int i=0;i<N;i++)
    {
        z[i]=x[i]*x[i]+y[i]*y[i];
    }
}

int main()
{
    int N=100000000;
    real *x,*y,*z;
    int M=N*sizeof(real);
    x=(real*)malloc(M);
    y=(real*)malloc(M);
    z=(real*)malloc(M);
    for(int i=0;i<N;i++)
    {
        x[i]=3.0;
        y[i]=4.0;
    }
    float t1=clock();
    addCpu(x,y,z,N);
    float t2=clock();
    printf("z[0]=%g\n",z[0]);
    printf("cpu time=%gms\n", t2-t1);

    real *dx, *dy, *dz;
    cudaMalloc((void**)&dx,M);
    cudaMalloc((void**)&dy,M);
    cudaMalloc((void**)&dz,M);
    cudaMemcpy(dx,x,M,cudaMemcpyHostToDevice);
    cudaMemcpy(dy,y,M,cudaMemcpyHostToDevice);
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    int blockSize=256;
    int gridSize=(N+blockSize-1)/blockSize;
    addGpu<<<gridSize,blockSize>>>(dx,dy,dz,N);
    cudaEventRecord(end);
    cudaMemcpy(z,dz,M,cudaMemcpyDeviceToHost);
    printf("z[0]=%g\n",z[0]);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time,start,end);
    printf("gpu time=%gms\n",elapsed_time*1000/CLOCKS_PER_SEC);
    free(x);
    free(y);
    free(z);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    return 0;
}