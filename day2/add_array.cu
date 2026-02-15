#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
const int N=100000005;
const double a=3.14;
const double b=5.38;
const double EPSILON=1.0e-12;
void addCpu(const double *x, const double *y, double *z, const int N)
{
    for (int i=0; i<N; i++)
    {
        z[i]=x[i]+y[i];
    }
}

__global__ void addGpu(const double *x, const double *y, double *z, const int N)
{
    const int id=blockDim.x*blockIdx.x+threadIdx.x;
    if(id<N)
    {
        z[id]=x[id]+y[id];
    }
}

void check(const double *z, double target, const int N)
{
    bool isCorrect=true;
    for(int i=0;i<N;i++)
    {
        if(fabs(z[i]-target)>EPSILON)
        {
            isCorrect=false;
        }
    }
    printf("%lf\n", z[0]);
    printf("%s\n", isCorrect?"No errors": "Has errors");
}

int main()
{
    const int M=sizeof(double)*N;
    double *h_x=(double*)malloc(M);
    double *h_y=(double*)malloc(M);
    double *h_z=(double*)malloc(M);
    for (int i=0;i<N;i++)
    {
        h_x[i]=a;
        h_y[i]=b;
    }
    auto start=std::chrono::high_resolution_clock::now();
    addCpu(h_x,h_y,h_z,N);
    auto end=std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed=end-start;
    std::cout<<"Elapsed time:"<<elapsed.count()<<"seconds"<<std::endl;
    check(h_z,8.52,N);
    double *d_x,*d_y,*d_z;
    cudaMalloc((void**)&d_x,M);
    cudaMalloc((void**)&d_y,M);
    cudaMalloc((void**)&d_z,M);
    auto preStart=std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_x,h_x,M,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_y,M,cudaMemcpyHostToDevice);
    int blockSize=256;
    int gridSize=(N+blockSize-1)/blockSize;
    auto start2=std::chrono::high_resolution_clock::now();
    addGpu<<<gridSize,blockSize>>>(d_x,d_y,d_z,N);
    double *h_r=(double*)malloc(M);
    cudaMemcpy(h_r,d_z,M,cudaMemcpyDeviceToHost);
    auto end2=std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2=end2-start2;
    std::cout<<"Elapsed time:"<<elapsed2.count()<<"seconds"<<std::endl;
    std::chrono::duration<double> elapsed3=end2-preStart;
    std::cout<<"Whole elapsed time:"<<elapsed3.count()<<"seconds"<<std::endl;
    check(h_r,8.52,N);
    free(h_x);
    free(h_y);
    free(h_z);
    free(h_r);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}