import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from timer import Timed
from gemm_kernel import gemm_fn

with Timed("The entire process"):
    threads = 16
    # Define the size of the matrices.
    N = 1024  # Size of the matrix (N x N).

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
        gemm_fn(
            drv.In(A),
            drv.In(B),
            drv.Out(C_cuda),
            np.int32(N),
            block=block_size,
            grid=grid_size,
        )
