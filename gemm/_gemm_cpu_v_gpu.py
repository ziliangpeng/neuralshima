import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

from timer import Timed
from gemm_kernel import gemm_fn


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
        gemm_fn(
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

    print("=" * 80)
