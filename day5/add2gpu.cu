#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;

__global__ void addGpu(const real *x, const real *y, real *z, const int N)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < N)
    {
        z[id] = x[id] + y[id];
    }
}

int main()
{
    const int N=100000000;
    const int M=sizeof(real)*N;
    real *x=(real*)malloc(M);
    real *y=(real*)malloc(M);
    real *z=(real*)malloc(M);

    for (int n=0;n<N;n++)
    {
        x[n]=1.0;
        y[n]=2.0;
    }

    real *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, M);
    cudaMalloc((void**)&d_y, M);
    cudaMalloc((void**)&d_z, M);
    cudaMemcpy(d_x, x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, M, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat=0;repeat<=NUM_REPEATS;repeat++)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaEventQuery(start);

        addGpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaMemcpy(z, d_z, M, cudaMemcpyDeviceToHost);
    printf("z[0] = %g\n", z[0]);

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Average time = %g ms. Error = %g ms.\n", t_ave, t_err);
    free(x);
    free(y);
    free(z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}
    