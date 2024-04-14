from pycuda.compiler import SourceModule

# Define the CUDA kernel as a string.
# Here, B is read column by column so it's not good for coalescing.
# There are tricks to improve, like:
#    - shared memory and tiling (process a small tile each time)
#    - transpose B
gemm_kernel = """
__global__ void gemm(float *A, float *B, float *C, int N, int cnt = 1) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    // TODO: implement a multi-mul
    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""
# Compile the kernel code.
mod = SourceModule(gemm_kernel)
gemm_fn = mod.get_function("gemm")
