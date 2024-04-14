import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

from timer import Timed


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

for scale in range(1, 8):
    # Define the size of the matrices.
    N = 1024 * scale  # Size of the matrix (N x N).

    # Create random matrices A and B.
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    # Create an empty matrix for the result.
    C_cuda = np.empty_like(A)

    # Define block and grid sizes.
    block_size = (8, 8, 1)  # Threads per block (8x8).
    grid_size = (int(np.ceil(N / block_size[0])), int(np.ceil(N / block_size[1])))

    # Launch the kernel.
    with Timed(f"GPU {N}"):
        gemm(
            drv.In(A),
            drv.In(B),
            drv.Out(C_cuda),
            np.int32(N),
            block=block_size,
            grid=grid_size,
        )

    with Timed(f"CPU {N}"):
        C_cpu = np.matmul(A, B)

    if not np.allclose(C_cuda, C_cpu, rtol=1e-3, atol=1e-3):
        print("Results do not match!")

for threads in [8, 16, 32]:
    # Define the size of the matrices.
    N = 1024 * 4  # Size of the matrix (N x N).

    # Create random matrices A and B.
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    # Create an empty matrix for the result.
    C_cuda = np.empty_like(A)

    # Define block and grid sizes.
    block_size = (threads, threads, 1)  # Threads per block.
    grid_size = (int(np.ceil(N / block_size[0])), int(np.ceil(N / block_size[1])))

    # Launch the kernel.
    with Timed(f"GPU {N}, threads {threads}"):
        gemm(
            drv.In(A),
            drv.In(B),
            drv.Out(C_cuda),
            np.int32(N),
            block=block_size,
            grid=grid_size,
        )