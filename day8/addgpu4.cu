#include "stdlib.h"
#include "stdio.h"

typedef float real;

__global__ void addGPU(const real *x, const real *y, real *z, int H, int W)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    if(ix<H&&iy<W)
    {
        int id=ix*W+iy;
        z[id]=x[id]*x[id]+y[id]*y[id];
    }
}

int main()
{
    real *x,*y,*z;
    int H=1024;
    int W=1024;
    int N=H*W;
    int M=N*sizeof(real);
    x=(real*)malloc(M);
    y=(real*)malloc(M);
    z=(real*)malloc(M);
    for(int i=0;i<H;i++)
    {
        for(int j=0;j<W;j++)
        {
            x[i*W+j]=i;
            y[i*W+j]=j;
        }
    }

    real *dx,*dy,*dz;
    cudaMalloc((void**)&dx,M);
    cudaMalloc((void**)&dy,M);
    cudaMalloc((void**)&dz,M);
    cudaMemcpy(dx,x,M,cudaMemcpyHostToDevice);
    cudaMemcpy(dy,y,M,cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((H+15)/16,(W+15)/16);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    addGPU<<<gridDim, blockDim>>>(dx,dy,dz,H,W);
    cudaEventRecord(end);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time,start,end);
    printf("gpu time=%gms\n",elapsed_time);

    cudaMemcpy(z,dz,M,cudaMemcpyDeviceToHost);

    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            printf("i=%d,j=%d,x[i,j]=%g,y[i,j]=%g,z[i,j]=%g\n",i,j,x[i*W+j],y[i*W+j],z[i*W+j]);
        }
    }
    free(x);
    free(y);
    free(z);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    return 0;
}