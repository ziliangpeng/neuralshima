from tinygrad import Tensor;

N = 1024

a, b = Tensor.rand(N, N), Tensor.rand(N, N)

# c = a @ b # typical matmul.
c = (a.reshape(N, 1, N) * b.T.reshape(1, N, N)).sum(axis=2); # interesting matmul.
print(c.numpy().mean())