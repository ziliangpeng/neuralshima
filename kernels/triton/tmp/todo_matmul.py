import triton
import triton.language as tl
import torch

@triton.jit
def kernel_matmul(
    a_ptr, b_ptr, c_ptr,
    M, N, K):
    pid = tl.program_id(axis=0)
    # print(pid)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    grid = lambda META: (1, )
    kernel_matmul[grid](a, b, c, M, N, K)
    return c

a = torch.randn((2, 8), device='cuda', dtype=torch.float16)
b = torch.randn((8, 4), device='cuda', dtype=torch.float16)
c = matmul(a, b)
print(c)