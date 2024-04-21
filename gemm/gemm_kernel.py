from pycuda.compiler import SourceModule

# Define the CUDA kernel as a string.
CU_GEMM_KERNEL_FILENAME = "gemm_kernel.cu"
with open(CU_GEMM_KERNEL_FILENAME, "r") as f:
    gemm_kernel = f.read()

# Compile the kernel code.
mod = SourceModule(gemm_kernel)
gemm_fn = mod.get_function("matrixMultiply")
