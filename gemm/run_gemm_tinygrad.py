from tinygrad import Tensor;

N = 1024

a, b = Tensor.rand(N, N), Tensor.rand(N, N)

c = a @ b

print(c.numpy().mean())