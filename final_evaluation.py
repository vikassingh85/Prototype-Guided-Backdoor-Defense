import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from torch import nn, Tensor
from torch.utils.data import DataLoader

from src.dataset import build_loaders
from src.model import MalwareMLP
from src.backdoor import BackdoorConfig, compute_asr
from src.purification import PurificationConfig, purify


def load_model(checkpoint_path: str, input_dim: int, device: torch.device) -> nn.Module:

    model = MalwareMLP(input_dim=input_dim)

    if os.path.exists(checkpoint_path):

        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device
            )
        )

    else:

        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using initialized weights."
        )

    model.to(device)

    model.eval()

    return model


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:

    model.eval()

    all_preds = []

    all_targets = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                device,
                non_blocking=True
            )

            logits = model(X_batch).squeeze(-1)

            probs = torch.sigmoid(logits)

            preds = (
                probs >= 0.5
            ).long().cpu()

            all_preds.extend(
                preds.numpy()
            )

            all_targets.extend(
                y_batch.numpy()
            )

    return {

        "Clean Accuracy":
            accuracy_score(
                all_targets,
                all_preds
            ),

        "Precision":
            precision_score(
                all_targets,
                all_preds,
                zero_division=0
            ),

        "Recall":
            recall_score(
                all_targets,
                all_preds,
                zero_division=0
            ),

        "F1-score":
            f1_score(
                all_targets,
                all_preds,
                zero_division=0
            )
    }


def plot_comparisons(
    poisoned_metrics: Dict[str, float],
    purified_metrics: Dict[str, float],
    output_dir: str
) -> None:

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    metrics = list(
        poisoned_metrics.keys()
    )

    poisoned_vals = [
        poisoned_metrics[m]
        for m in metrics
    ]

    purified_vals = [
        purified_metrics[m]
        for m in metrics
    ]

    x = np.arange(
        len(metrics)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width/2,
        poisoned_vals,
        width,
        label='Poisoned Model',
        color='salmon'
    )

    ax.bar(
        x + width/2,
        purified_vals,
        width,
        label='Purified Model',
        color='skyblue'
    )

    ax.set_ylabel('Scores')

    ax.set_title(
        'Model Performance Comparison: Poisoned vs Purified'
    )

    ax.set_xticks(x)

    ax.set_xticklabels(metrics)

    ax.legend()

    ax.set_ylim(0, 1.1)

    for i, v in enumerate(poisoned_vals):

        ax.text(
            i - width/2,
            v + 0.02,
            f"{v:.3f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    for i, v in enumerate(purified_vals):

        ax.text(
            i + width/2,
            v + 0.02,
            f"{v:.3f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            'metrics_comparison.png'
        ),
        dpi=300
    )

    plt.close()


def save_confusion_matrix(
    model,
    loader,
    device,
    title,
    filename
):

    model.eval()

    all_preds = []

    all_targets = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                device,
                non_blocking=True
            )

            logits = model(X_batch).squeeze(-1)

            probs = torch.sigmoid(logits)

            preds = (
                probs >= 0.5
            ).long().cpu()

            all_preds.extend(
                preds.numpy()
            )

            all_targets.extend(
                y_batch.numpy()
            )

    cm = confusion_matrix(
        all_targets,
        all_preds
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malware"]
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d"
    )

    ax.set_title(title)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()


def print_comparison_table(
    poisoned_metrics: Dict[str, float],
    purified_metrics: Dict[str, float]
) -> None:

    print("\n" + "="*65)

    print(
        f"{'Metric':<25} | {'Poisoned Model':<15} | {'Purified Model':<15}"
    )

    print("-" * 65)

    for metric in poisoned_metrics.keys():

        pm = poisoned_metrics[metric]

        pu = purified_metrics[metric]

        print(
            f"{metric:<25} | {pm:<15.4f} | {pu:<15.4f}"
        )

    print("="*65 + "\n")


def main() -> None:

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    data_path = "data/malware.csv"

    poisoned_ckpt = "models/poisoned_model.pt"

    graphs_dir = "outputs/graphs/"

    if not os.path.exists(data_path):

        print(
            f"Dataset not found at {data_path}. Please generate or provide the dataset."
        )

        return

    train_loader, test_loader, scaler, input_dim = build_loaders(
        csv_path=data_path,
        batch_size=256
    )

    X_test = test_loader.dataset.features

    y_test = test_loader.dataset.labels

    print("Loading poisoned model...")

    poisoned_model = load_model(
        poisoned_ckpt,
        input_dim,
        device
    )

    print("Evaluating poisoned model...")

    poisoned_metrics = evaluate_metrics(
        poisoned_model,
        test_loader,
        device
    )

    backdoor_config = BackdoorConfig()

    poisoned_asr = compute_asr(
        model=poisoned_model,
        X=X_test,
        y=y_test,
        config=backdoor_config,
        device=device
    )

    poisoned_metrics[
        "Attack Success Rate (ASR)"
    ] = poisoned_asr

    print("\nPurifying model...")

    purification_config = PurificationConfig(
        finetune_epochs=5
    )

    purified_model = purify(
        model=poisoned_model,
        clean_loader=train_loader,
        device=device,
        config=purification_config,
        val_loader=test_loader
    )

    print("\nEvaluating purified model...")

    purified_metrics = evaluate_metrics(
        purified_model,
        test_loader,
        device
    )

    purified_asr = compute_asr(
        model=purified_model,
        X=X_test,
        y=y_test,
        config=backdoor_config,
        device=device
    )

    purified_metrics[
        "Attack Success Rate (ASR)"
    ] = purified_asr

    print_comparison_table(
        poisoned_metrics,
        purified_metrics
    )

    print(
        f"Generating and saving comparison graphs to {graphs_dir}..."
    )

    plot_comparisons(
        poisoned_metrics,
        purified_metrics,
        graphs_dir
    )

    save_confusion_matrix(
        poisoned_model,
        test_loader,
        device,
        "Poisoned Model",
        os.path.join(
            graphs_dir,
            "confusion_poisoned.png"
        )
    )

    save_confusion_matrix(
        purified_model,
        test_loader,
        device,
        "Purified Model",
        os.path.join(
            graphs_dir,
            "confusion_purified.png"
        )
    )

    print("Evaluation complete.")


if __name__ == "__main__":
    main()