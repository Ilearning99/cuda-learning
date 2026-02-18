#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;

void addCpu(const real *x, const real *y, real *z, const int N)
{
    for (int i=0;i<N;i++)
    {
        z[i]=x[i]+y[i];
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

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat=0;repeat<=NUM_REPEATS;repeat++)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaEventQuery(start);

        addCpu(x, y, z, N);

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

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Average time = %g ms. Error = %g ms.\n", t_ave, t_err);

    return 0;
}
    