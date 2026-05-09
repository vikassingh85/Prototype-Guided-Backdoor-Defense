from src.dataset import build_loaders

train_loader, test_loader, scaler, input_dim = build_loaders(
    csv_path="data/malware.csv",
    batch_size=32
)

print("Input dimension:", input_dim)

for x, y in train_loader:
    print("Feature batch shape:", x.shape)
    print("Label batch shape:", y.shape)
    break