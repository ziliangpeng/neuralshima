// CUDA kernel for matrix multiplication (GEMM)
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows, int numAColumns, int numBColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBColumns)
    {
        float sum = 0.0f;
        for (int i = 0; i < numAColumns; ++i)
        {
            sum += A[row * numAColumns + i] * B[i * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}