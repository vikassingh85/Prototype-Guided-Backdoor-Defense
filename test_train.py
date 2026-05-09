from src.train import train, TrainConfig

config = TrainConfig(
    csv_path="data/malware.csv",
    batch_size=32,
    max_epochs=5,
    lr=0.001,
    checkpoint_dir="outputs/models"
)

model = train(config)
print("Training complete!")