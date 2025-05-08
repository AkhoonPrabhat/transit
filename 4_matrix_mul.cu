% % writefile matrixmul.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define N 512
#define TOLERANCE 1e-6

        __global__ void
        matrixMul(float *A, float *B, float *C, int n)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n)
    {
        float sum = 0.0;
        for (int k = 0; k < n; k++)
        {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main()
{
    float *hA, *hB, *hC, *hGPU, *dA, *dB, *dC;
    size_t size = N * N * sizeof(float);
    cudaEvent_t start, stop;
    float cpuTime, gpuTime;

    hA = (float *)malloc(size);
    hB = (float *)malloc(size);
    hC = (float *)malloc(size);
    hGPU = (float *)malloc(size);

    for (int i = 0; i < N * N; i++)
    {
        hA[i] = (float)rand() / RAND_MAX;
        hB[i] = (float)rand() / RAND_MAX;
    }

    clock_t cpustart = clock();
    float sum = 0.0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                sum += hA[i * N + k] * hB[k * N + j];
            }
            hC[i * N + j] = sum;
            sum = 0.0;
        }
    }
    clock_t cpuend = clock();
    cpuTime = (float)(cpuend - cpustart) / CLOCKS_PER_SEC;
    printf("CPU time: %f ", cpuTime);

    cudaMalloc((void **)&dA, size);
    cudaMalloc((void **)&dB, size);
    cudaMalloc((void **)&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    dim3 threadsperBlock(16, 16);
    dim3 blocksPerGrid((N + threadsperBlock.x - 1) / threadsperBlock.x, (N + threadsperBlock.y - 1) / threadsperBlock.y);

    matrixMul<<<blocksPerGrid, threadsperBlock>>>(dA, dB, dC, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpuTime, start, stop);
    gpuTime /= 1000;
    printf("GPU time: %f ", gpuTime);

    cudaMemcpy(hGPU, dC, size, cudaMemcpyDeviceToHost);

    int isValid = 1;
    for (int i = 0; i < N * N; i++)
    {
        if (fabs(hC[i] - hGPU[i]) > TOLERANCE * fabs(hC[i]))
        {
            isValid = 0;
            break;
        }
    }

    printf("Verification: %s\n", isValid ? "TRUE" : "FALSE");

    free(hA);
    free(hB);
    free(hC);
    free(hGPU);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}