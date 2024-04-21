import torch
import torch.nn.functional as F

# Create a tensor
N = 102400
input_tensor = torch.tensor(torch.rand(128, N).cuda(), dtype=torch.float32)

# Apply softmax along the last dimension
output_tensor = F.softmax(input_tensor, dim=1)
print(output_tensor)
