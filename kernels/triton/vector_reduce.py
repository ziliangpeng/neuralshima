import torch
import time

import triton
import triton.language as tl
from triton.runtime.autotuner import autotune


@triton.jit
def reduce_kernel(x_ptr,  # *Pointer* to first input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.sum(x)
    tl.atomic_add(output_ptr, output)
    # tl.store(output_ptr, output)


# Define a list of configurations to try
@autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_elements'],
)
@triton.jit
def reduce_kernel_autotune(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    return reduce_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE)

def reduce(x: torch.Tensor):
    # We need to preallocate the output.
    output = torch.zeros(1, device='cuda')
    assert x.is_cuda and output.is_cuda
    n_elements = x.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # Using the autotuned kernel instead of the original one
    reduce_kernel_autotune[grid](x, output, n_elements)
    return output

torch.manual_seed(0)
size = 30000000
x = torch.rand(size, device='cuda')

# Time PyTorch implementation
torch.cuda.synchronize()  # Ensure GPU is synchronized before timing
start = time.perf_counter()
output_torch = torch.sum(x)
torch.cuda.synchronize()  # Ensure GPU is done before stopping timer
torch_time = (time.perf_counter() - start) * 1000  # Convert to milliseconds

# Time Triton implementation
output_triton = reduce(x)
triton_times = []
for i in range(10):  # Run 10 times
    x = torch.rand(size, device='cuda')
    print(x)
    torch.cuda.synchronize()
    start = time.perf_counter()
    output_triton = reduce(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000  # Convert to milliseconds
    triton_times.append(elapsed)
    print(f'Run {i+1}: {elapsed:.4f} ms')

    torch.cuda.synchronize()  # Ensure GPU is synchronized before timing
    start = time.perf_counter()
    output_torch = torch.sum(x)
    torch.cuda.synchronize()  # Ensure GPU is done before stopping timer
    torch_time = (time.perf_counter() - start) * 1000  # Convert to milliseconds
    print(f'Run {i+1}: {torch_time:.4f} ms')

triton_time = sum(triton_times) / len(triton_times)  # Average time
print(f'\nIndividual Triton runs (ms): {[f"{t:.4f}" for t in triton_times]}')
print(f'Average time: {triton_time:.4f} ms')
print(f'Torch time: {torch_time}')
print(f'Triton time: {triton_time}')
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')