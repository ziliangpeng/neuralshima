import triton
import triton.language as tl
import torch

@triton.jit
def kernel_matadd(
    a_ptr, b_ptr, c_ptr,
    M, N):
    pid = tl.program_id(axis=0)
    # print(pid)
    offs_m = (pid * M + tl.arange(0, M)) % M
    offs_n = (pid * N + tl.arange(0, N)) % N
    a_ptrs = a_ptr + offs_m[:, None] * tl.num_elements(a_ptr) // M + offs_n[None, :] * tl.element_size(a_ptr)
    b_ptrs = b_ptr + offs_m[:, None] * tl.num_elements(b_ptr) // M + offs_n[None, :] * tl.element_size(b_ptr)
    c_ptrs = c_ptr + offs_m[:, None] * tl.num_elements(c_ptr) // M + offs_n[None, :] * tl.element_size(c_ptr)
    tl.store(c_ptrs, tl.load(a_ptrs) + tl.load(b_ptrs))


def matmul(a, b):
    M, N = a.shape
    assert a.shape == b.shape, "Incompatible dimensions"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    grid = lambda META: (1, )
    kernel_matadd[grid](a, b, c, M, N)
    return c

a = torch.randn((2, 4), device='cuda', dtype=torch.float16)
b = torch.randn((2, 4), device='cuda', dtype=torch.float16)
c = matmul(a, b)
print(c)