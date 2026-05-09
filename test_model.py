import torch
from src.model import MalwareMLP

model = MalwareMLP(input_dim=100)

x = torch.randn(32, 100)

output = model(x)

print("Output shape:", output.shape)
print(output[:5])
