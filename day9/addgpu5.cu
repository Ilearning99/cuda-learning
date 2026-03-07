
#include <stdio.h>
#include <stdlib.h>

typedef float real;

__global__ void add(const real *x, const real *y, real *z, const int nx, const int ny)
{
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    int idy=threadIdx.y+blockIdx.y*blockDim.y;
    if(idx<nx&&idy<ny)
    {
        int id=idy*nx+idx;
        z[id]=x[id]*y[id];
    }
}

int main()
{
    real *x, *y, *z;
    int nx=4096; // col
    int ny=4096; // row
    int n=nx*ny;
    int m=n*sizeof(real);
    x=(real*)malloc(m);
    y=(real*)malloc(m);
    z=(real*)malloc(m);

    real *dx, *dy, *dz;
    cudaMalloc((void**)&dx,m);
    cudaMalloc((void**)&dy,m);
    cudaMalloc((void**)&dz,m);

    int idx,idy;
    for(idy=0;idy<ny;idy++)
    {
        for(idx=0;idx<nx;idx++)
        {
            x[idy*nx+idx]=idx+1;
            y[idy*nx+idx]=idy+1;
        }
    }

    cudaMemcpy(dx,x,m,cudaMemcpyHostToDevice);
    cudaMemcpy(dy,y,m,cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((nx+15)/16,(ny+15)/16);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    add<<<gridDim, blockDim>>>(dx,dy,dz,ny,nx);
    cudaEventRecord(end);
    cudaMemcpy(z,dz,m,cudaMemcpyDeviceToHost);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, end);
    int i,j;
    for(i=0;i<3;i++)
    {
        for(j=0;j<3;j++)
        {
            printf("i=%d,j=%d,x[i,j]=%g, y[i,j]=%g, z[i,j]=%g\n",i,j,x[i*ny+j],y[i*ny+j],z[i*ny+j]);
        }
    }

    printf("elapsed time:%g\n", elapsed);
    free(x);
    free(y);
    free(z);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    return 0;
}