import torch
from torch.utils.data import TensorDataset, DataLoader

from src.model import MalwareMLP
from src.purification import collect_activations, get_linear_layer_names

device = torch.device("cpu")
model = MalwareMLP(input_dim=100)

# Create a dummy dataloader
x = torch.randn(32, 100)
y = torch.zeros(32)
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=16)

target_names = get_linear_layer_names(model)
activations = collect_activations(model, loader, target_names, device)

print("\nCollected Activations:\n")

for layer_name, act in activations.items():
    print(layer_name, act.shape)