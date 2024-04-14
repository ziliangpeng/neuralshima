import time
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule


class Timed:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.elapsed = self.end - self.start
        print(f"{self.name} elapsed time: {self.elapsed} seconds")


# Define the CUDA kernel as a string.
kernel_code = """
__global__ void gemm(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""

# Compile the kernel code.
mod = SourceModule(kernel_code)
gemm = mod.get_function("gemm")

# Define the size of the matrices.
N = 1024 * 2  # Size of the matrix (N x N).

# Create random matrices A and B.
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)

# Create an empty matrix for the result.
C_cuda = np.empty_like(A)

# Define block and grid sizes.
block_size = (8, 8, 1)  # Threads per block (8x8).
grid_size = (int(np.ceil(N / block_size[0])), int(np.ceil(N / block_size[1])))

# Launch the kernel.
with Timed("GPU"):
    gemm(
        drv.In(A),
        drv.In(B),
        drv.Out(C_cuda),
        np.int32(N),
        block=block_size,
        grid=grid_size,
    )

with Timed("CPU"):
    C_cpu = np.matmul(A, B)

print("All close:", np.allclose(C_cuda, C_cpu, rtol=1e-3, atol=1e-5))
