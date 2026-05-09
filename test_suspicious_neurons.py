import torch

from torch.utils.data import TensorDataset, DataLoader

from src.model import MalwareMLP
from src.purification import (
    collect_activations,
    compute_neuron_scores,
    detect_suspicious_neurons,
    get_linear_layer_names
)

device = torch.device("cpu")
model = MalwareMLP(input_dim=100)

x = torch.randn(128, 100)
y = torch.zeros(128)
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=32)

target_names = get_linear_layer_names(model)
activations = collect_activations(model, loader, target_names, device)

scores = compute_neuron_scores(activations)

suspicious = detect_suspicious_neurons(
    neuron_scores=scores,
    threshold=0.8
)

print("\nNeuron Scores:\n")

for layer, score in scores.items():
    print(layer, score.shape)

print("\nSuspicious Neurons:\n")

for layer, neurons in suspicious.items():
    print(layer, neurons)