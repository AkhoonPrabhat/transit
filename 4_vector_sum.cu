% % writefile vecAdd.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define N 1000000
#define TOLERANCE 1e-6

        __global__ void
        vectorAdd(float *A, float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    float *hA, *hB, *hC, *hGPU, *dA, *dB, *dC;
    cudaEvent_t start, stop;
    size_t size = N * sizeof(float);
    float gpuTime, cpuTime;

    hA = (float *)malloc(size);
    hB = (float *)malloc(size);
    hC = (float *)malloc(size);
    hGPU = (float *)malloc(size);

    for (int i = 0; i < N; i++)
    {
        hA[i] = rand() / (float)RAND_MAX;
        hB[i] = rand() / (float)RAND_MAX;
    }

    clock_t cpuStart = clock();

    for (int i = 0; i < N; i++)
    {
        hC[i] = hA[i] + hB[i];
    }
    clock_t cpuEnd = clock();
    cpuTime = (float)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;
    printf("CPU Time: %f\n", cpuTime);

    cudaMalloc((void **)&dA, size);
    cudaMalloc((void **)&dB, size);
    cudaMalloc((void **)&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpuTime, start, stop);
    gpuTime /= 1000;

    cudaMemcpy(hGPU, dC, size, cudaMemcpyDeviceToHost);

    printf("GPU time: %f \n", gpuTime);

    bool isValid = true;

    for (int i = 0; i < N; i++)
    {
        if (fabs(hC[i] - hGPU[i]) > TOLERANCE)
        {
            isValid = false;
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
}
