import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np

from sm_kernel import kernel_code

# Compile the kernel code
mod = SourceModule(kernel_code)
softmax_kernel = mod.get_function("softmax_kernel")

# Create input data
n = 102400  # Size of the input array
input_data = np.random.randn(n).astype(np.float32)
output_data = np.zeros_like(input_data)

# Allocate memory on the device
input_gpu = drv.mem_alloc(input_data.nbytes)
output_gpu = drv.mem_alloc(output_data.nbytes)

# Copy data to the device
drv.memcpy_htod(input_gpu, input_data)

# Launch the kernel
block_size = 256  # Number of threads per block
num_blocks = (n + block_size - 1) // block_size
softmax_kernel(input_gpu, output_gpu, np.int32(n), block=(block_size, 1, 1), grid=(num_blocks, 1, 1), shared=block_size * 4)

# Copy the result back to host
drv.memcpy_dtoh(output_data, output_gpu)

# Print the result
print(output_data)