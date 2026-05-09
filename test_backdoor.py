import pandas as pd
import torch
from src.backdoor import poison_dataset, BackdoorConfig

df = pd.read_csv("data/malware.csv")

X = torch.from_numpy(df.drop(columns="label").to_numpy())
y = torch.from_numpy(df["label"].to_numpy())

config = BackdoorConfig(
    poison_ratio=0.1, 
    trigger_value=999, 
    trigger_size=10, 
    target_label=1
)

X_poisoned, y_poisoned, poison_mask = poison_dataset(X, y, config)

print("Original shape:", X.shape)
print("Poisoned shape:", X_poisoned.shape)

print("\nLast 10 features of first poisoned sample:\n")
poisoned_idx = torch.where(poison_mask)[0][0]
print(X_poisoned[poisoned_idx, -10:])