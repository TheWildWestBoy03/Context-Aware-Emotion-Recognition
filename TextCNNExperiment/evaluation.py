"""Evaluation utilities for the TextCNN MELD experiment: metrics, reports and plots."""
import csv
import os

import matplotlib
matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Must match the label2id order used for training
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL2ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}

HISTORY_KEYS = ["loss", "accuracy", "macro_f1", "weighted_f1"]


def compute_metrics(targets, predictions):
    return {
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(targets, predictions, average="weighted", zero_division=0),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run the model over a loader; return loss + metrics and the raw targets/predictions."""
    model.eval()

    total_loss = 0.0
    total_utterances = 0
    all_predictions = []
    all_targets = []

    for X, y, mask in loader:
        X = X.to(device)
        y = y.to(device)
        mask = mask.to(device)

        logits = model(X, mask)

        loss = criterion(logits[mask], y[mask])
        num_utterances = mask.sum().item()

        total_loss += loss.item() * num_utterances
        total_utterances += num_utterances

        predictions = logits.argmax(dim=-1)
        all_predictions.extend(predictions[mask].cpu().tolist())
        all_targets.extend(y[mask].cpu().tolist())

    return {
        "loss": total_loss / total_utterances,
        **compute_metrics(all_targets, all_predictions),
        "targets": all_targets,
        "predictions": all_predictions,
    }


def format_metrics(name, metrics):
    return (
        f"{name:<5} | Loss: {metrics['loss']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f} | "
        f"Weighted F1: {metrics['weighted_f1']:.4f}"
    )


class History:
    """Per-epoch train/val metrics."""

    def __init__(self):
        self.records = {split: {key: [] for key in HISTORY_KEYS} for split in ("train", "val")}

    def add(self, split, metrics):
        for key in HISTORY_KEYS:
            self.records[split][key].append(metrics[key])

    @property
    def num_epochs(self):
        return len(self.records["train"]["loss"])

    def best_epoch(self, metric="macro_f1"):
        """1-based epoch with the best validation value of `metric`."""
        return int(np.argmax(self.records["val"][metric])) + 1

    def to_csv(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch"] + [f"{split}_{key}" for split in ("train", "val") for key in HISTORY_KEYS]
            )
            for epoch in range(self.num_epochs):
                writer.writerow(
                    [epoch + 1]
                    + [self.records[split][key][epoch] for split in ("train", "val") for key in HISTORY_KEYS]
                )


def plot_training_curves(history, out_path):
    """Loss, accuracy and macro F1 per epoch, train vs val, with the best epoch marked."""
    epochs = range(1, history.num_epochs + 1)
    best = history.best_epoch("macro_f1")

    panels = [("loss", "Loss"), ("accuracy", "Accuracy"), ("macro_f1", "Macro F1")]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 4.5))

    for ax, (key, title) in zip(axes, panels):
        ax.plot(epochs, history.records["train"][key], marker="o", markersize=3, label="Train")
        ax.plot(epochs, history.records["val"][key], marker="o", markersize=3, label="Validation (dev)")
        ax.axvline(best, color="gray", linestyle="--", linewidth=1, label=f"Best epoch ({best})")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("TextCNN on MELD - training curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(targets, predictions, out_path, title="Confusion matrix"):
    """Two panels over the 7 emotions: raw counts and row-normalized (per-class recall)."""
    labels = list(range(len(EMOTION_LABELS)))
    cm = confusion_matrix(targets, predictions, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    panels = [(cm, "Counts", "d"), (cm_norm, "Normalized by true class", ".2f")]

    for ax, (matrix, subtitle, fmt) in zip(axes, panels):
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(labels, EMOTION_LABELS, rotation=45, ha="right")
        ax.set_yticks(labels, EMOTION_LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(subtitle)

        threshold = matrix.max() / 2
        for i in labels:
            for j in labels:
                ax.text(
                    j, i, format(matrix[i, j], fmt),
                    ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > threshold else "black",
                )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def report(name, metrics):
    """Print summary metrics and the per-class classification report."""
    print("\n" + format_metrics(name, metrics))
    print(f"\n{name} Classification Report:")
    print(classification_report(
        metrics["targets"],
        metrics["predictions"],
        labels=list(range(len(EMOTION_LABELS))),
        target_names=EMOTION_LABELS,
        zero_division=0,
    ))


def save_evaluation(history, test_metrics, out_dir="plots"):
    """Write history CSV, training curves and the test confusion matrix to `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "history": os.path.join(out_dir, "history.csv"),
        "curves": os.path.join(out_dir, "training_curves.png"),
        "confusion": os.path.join(out_dir, "confusion_matrix_test.png"),
    }
    history.to_csv(paths["history"])
    plot_training_curves(history, paths["curves"])
    plot_confusion_matrix(
        test_metrics["targets"], test_metrics["predictions"], paths["confusion"],
        title="TextCNN on MELD - test confusion matrix",
    )
    for path in paths.values():
        print("Saved", path)
    return paths
