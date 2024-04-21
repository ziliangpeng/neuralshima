#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// CUDA kernel for matrix multiplication (GEMM)
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows, int numAColumns, int numBColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBColumns) {
        float sum = 0.0f;
        for (int i = 0; i < numAColumns; ++i) {
            sum += A[row * numAColumns + i] * B[i * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}

// Host function to run GEMM
void gemm(float *h_A, float *h_B, float *h_C, int numARows, int numAColumns, int numBColumns) {
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    {
        // Allocation of mem turns out to be most time consuming. it took 0.4s out of total 0.5s.
        // nvprof result:
        /*
            ==19534== Profiling application: ./a.out
            ==19534== Profiling result:
                        Type  Time(%)      Time     Calls       Avg       Min       Max  Name
            GPU activities:   51.04%  26.783us         1  26.783us  26.783us  26.783us  matrixMultiply(float*, float*, float*, int, int, int)
                              35.79%  18.784us         2  9.3920us  9.2800us  9.5040us  [CUDA memcpy HtoD]
                              13.17%  6.9120us         1  6.9120us  6.9120us  6.9120us  [CUDA memcpy DtoH]
                 API calls:   92.18%  303.92ms         3  101.31ms  2.4430us  303.91ms  cudaMalloc
                               6.67%  21.988ms         1  21.988ms  21.988ms  21.988ms  cudaLaunchKernel
                               1.02%  3.3520ms       114  29.403us     226ns  1.8684ms  cuDeviceGetAttribute
                               0.07%  225.58us         3  75.192us  51.422us  101.55us  cudaMemcpy
                               0.05%  176.81us         3  58.935us  4.5960us  156.23us  cudaFree
                               0.00%  15.945us         1  15.945us  15.945us  15.945us  cuDeviceGetName
                               0.00%  6.8980us         1  6.8980us  6.8980us  6.8980us  cuDeviceGetPCIBusId
                               0.00%  3.3960us         3  1.1320us     408ns  2.5040us  cuDeviceGetCount
                               0.00%  1.4620us         2     731ns     314ns  1.1480us  cuDeviceGet
                               0.00%     859ns         1     859ns     859ns     859ns  cuDeviceTotalMem
                               0.00%     502ns         1     502ns     502ns     502ns  cuDeviceGetUuid
                               0.00%     470ns         1     470ns     470ns     470ns  cuModuleGetLoadingMode
        */
        // Based on nvprof result, pycuda uses cudaMemAlloc instead of cudaMalloc.
        auto start = std::chrono::high_resolution_clock::now();
        cudaMalloc((void **)&d_A, sizeof(float) * numARows * numAColumns);
        cudaMalloc((void **)&d_B, sizeof(float) * numAColumns * numBColumns);
        cudaMalloc((void **)&d_C, sizeof(float) * numARows * numBColumns);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Just malloc: " << (end - start).count() / 1e9 << " s" << std::endl;
    }

    // Copy host memory to device
    {
        auto start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_A, h_A, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, sizeof(float) * numAColumns * numBColumns, cudaMemcpyHostToDevice);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Just memcpy: " << (end - start).count() / 1e9 << " s" << std::endl;
    }

    // Set grid and block sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((numBColumns + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (numARows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    {
        auto start = std::chrono::high_resolution_clock::now();
        matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numARows, numAColumns, numBColumns);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Just matmul: " << (end - start).count() / 1e9 << " s" << std::endl;
    }

    // Copy result back to host
    {
        auto start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(h_C, d_C, sizeof(float) * numARows * numBColumns, cudaMemcpyDeviceToHost);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Just memcpy back: " << (end - start).count() / 1e9 << " s" << std::endl;
    }

    // Free device memory
    {
        auto start = std::chrono::high_resolution_clock::now();
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Just free: " << (end - start).count() / 1e9 << " s" << std::endl;
    }
}

int main() {
    int N = 1024;
    // Example usage:
    int numARows = N;
    int numAColumns = N;
    int numBColumns = N;

    // Allocate memory for the matrices on host
    float *h_A = new float[numARows * numAColumns];
    float *h_B = new float[numAColumns * numBColumns];
    float *h_C = new float[numARows * numBColumns];

    // Initialize matrices A and B with some values
    // ...

    // Perform GEMM
    #include <chrono>

    auto start = std::chrono::high_resolution_clock::now();
    gemm(h_A, h_B, h_C, numARows, numAColumns, numBColumns);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Total gemm(include allocation): " << (end - start).count() / 1e9 << " s" << std::endl;

    // Use the result matrix h_C
    // ...

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}