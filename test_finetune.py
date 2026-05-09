import torch

from src.dataset import build_loaders
from src.model import MalwareMLP
from src.purification import (
    collect_activations,
    compute_neuron_scores,
    detect_suspicious_neurons,
    prune_suspicious_neurons,
    fine_tune_model,
    get_linear_layer_names
)

device = torch.device("cpu")

train_loader, test_loader, scaler, input_dim = build_loaders(
    csv_path="data/malware.csv",
    batch_size=32
)

model = MalwareMLP(input_dim=input_dim)

# Step 1: Collect activations
target_names = get_linear_layer_names(model)
activations = collect_activations(model, train_loader, target_names, device)

# Step 2: Detect suspicious neurons
scores = compute_neuron_scores(activations)

suspicious = detect_suspicious_neurons(
    neuron_scores=scores,
    threshold=0.8
)

# Step 3: Prune suspicious neurons
purified_model = prune_suspicious_neurons(
    model=model,
    suspicious_neurons=suspicious
)

# Step 4: Fine-tune purified model
purified_model = fine_tune_model(
    model=purified_model,
    train_loader=train_loader,
    val_loader=test_loader,
    epochs=3,
    learning_rate=0.0005
)

print("\nFine-tuning completed successfully.")