"""Metrics, history, plots and saved outputs for DistilBERT runs."""
import csv
import json
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
    precision_recall_fscore_support,
)

from data import EMOTION_LABELS

HISTORY_KEYS = ["loss", "accuracy", "macro_f1", "weighted_f1"]


def compute_metrics(targets, predictions):
    return {
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(targets, predictions, average="weighted", zero_division=0),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Loss + metrics over a loader, plus the raw targets/predictions."""
    model.eval()
    total_loss, total = 0.0, 0
    targets, predictions = [], []
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        logits = model(input_ids, attention_mask)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        total += labels.size(0)
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        targets.extend(labels.cpu().tolist())
    return {
        "loss": total_loss / total,
        **compute_metrics(targets, predictions),
        "targets": targets,
        "predictions": predictions,
    }


def format_metrics(name, metrics):
    return (
        f"{name:<5} | Loss: {metrics['loss']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f} | "
        f"Weighted F1: {metrics['weighted_f1']:.4f}"
    )


class History:
    """Train/dev metrics at each evaluation point (x = fractional epoch)."""

    def __init__(self):
        self.epochs = []
        self.records = {split: {key: [] for key in HISTORY_KEYS} for split in ("train", "val")}

    def add(self, epoch, train_metrics, val_metrics):
        self.epochs.append(epoch)
        for split, metrics in (("train", train_metrics), ("val", val_metrics)):
            for key in HISTORY_KEYS:
                self.records[split][key].append(metrics[key])

    def best_index(self, metric="macro_f1"):
        return int(np.argmax(self.records["val"][metric]))

    def to_csv(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch"] + [f"{split}_{key}" for split in ("train", "val") for key in HISTORY_KEYS]
            )
            for i, epoch in enumerate(self.epochs):
                writer.writerow(
                    [round(epoch, 3)]
                    + [self.records[split][key][i] for split in ("train", "val") for key in HISTORY_KEYS]
                )


def plot_training_curves(history, out_path, title):
    best_epoch = history.epochs[history.best_index()]
    panels = [("loss", "Loss"), ("accuracy", "Accuracy"), ("macro_f1", "Macro F1")]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 4.5))
    for ax, (key, name) in zip(axes, panels):
        ax.plot(history.epochs, history.records["train"][key], marker="o", markersize=3, label="Train")
        ax.plot(history.epochs, history.records["val"][key], marker="o", markersize=3, label="Validation (dev)")
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
                   label=f"Best checkpoint (epoch {best_epoch:.2f})")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(targets, predictions, out_path, title):
    """Raw counts and row-normalized (per-class recall) over the 7 emotions."""
    labels = list(range(len(EMOTION_LABELS)))
    cm = confusion_matrix(targets, predictions, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, (matrix, subtitle, fmt) in zip(
        axes, [(cm, "Counts", "d"), (cm_norm, "Normalized by true class", ".2f")]
    ):
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
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=9,
                        color="white" if matrix[i, j] > threshold else "black")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def per_class_scores(targets, predictions):
    """{label: {precision, recall, f1, support}} on the test set."""
    labels = list(range(len(EMOTION_LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    return {
        label: {"precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i])}
        for i, label in enumerate(EMOTION_LABELS)
    }


def plot_per_class_bars(per_class, overall, out_path, title):
    """Grouped bars: precision / recall / F1 for each emotion, plus an 'overall' group
    with accuracy, macro F1 and weighted F1."""
    fig, (ax_cls, ax_all) = plt.subplots(
        1, 2, figsize=(17, 5.5), gridspec_kw={"width_ratios": [len(EMOTION_LABELS), 1.6]}
    )

    x = np.arange(len(EMOTION_LABELS))
    width = 0.27
    for offset, (key, name) in zip((-width, 0, width), (("precision", "Precision"),
                                                          ("recall", "Recall (per-class accuracy)"),
                                                          ("f1", "F1"))):
        values = [per_class[label][key] for label in EMOTION_LABELS]
        bars = ax_cls.bar(x + offset, values, width, label=name)
        ax_cls.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax_cls.set_xticks(x, [f"{label}\n(n={per_class[label]['support']})" for label in EMOTION_LABELS])
    ax_cls.set_ylim(0, 1.05)
    ax_cls.set_ylabel("Score")
    ax_cls.set_title("Per emotion")
    ax_cls.grid(axis="y", alpha=0.3)
    ax_cls.legend(loc="upper left")

    names = ["Accuracy", "Macro F1", "Weighted F1"]
    values = [overall["accuracy"], overall["macro_f1"], overall["weighted_f1"]]
    bars = ax_all.bar(names, values, color=["#4c72b0", "#dd8452", "#55a868"])
    ax_all.bar_label(bars, fmt="%.3f", fontsize=8, padding=1)
    ax_all.set_ylim(0, 1.05)
    ax_all.set_title("Overall")
    ax_all.tick_params(axis="x", rotation=20)
    ax_all.grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def classification_report_text(metrics):
    return classification_report(
        metrics["targets"], metrics["predictions"],
        labels=list(range(len(EMOTION_LABELS))), target_names=EMOTION_LABELS,
        digits=4, zero_division=0,
    )


def save_run_outputs(out_dir, run_name, config, history, test_metrics, best_val_metrics):
    """Write metrics.json, history.csv, classification report and both plots for one run."""
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "run": run_name,
        "config": config,
        "best_epoch": history.epochs[history.best_index()],
        "dev": {k: best_val_metrics[k] for k in HISTORY_KEYS},
        "test": {k: test_metrics[k] for k in HISTORY_KEYS},
        "test_per_class_f1": dict(zip(EMOTION_LABELS, f1_score(
            test_metrics["targets"], test_metrics["predictions"],
            labels=list(range(len(EMOTION_LABELS))), average=None, zero_division=0,
        ).round(4).tolist())),
        "test_per_class": per_class_scores(test_metrics["targets"], test_metrics["predictions"]),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "classification_report_test.txt"), "w") as f:
        f.write(format_metrics("Test", test_metrics) + "\n\n" + classification_report_text(test_metrics))
    history.to_csv(os.path.join(out_dir, "history.csv"))
    plot_training_curves(history, os.path.join(out_dir, "training_curves.png"),
                         f"DistilBERT on MELD - {run_name} - training curves")
    plot_confusion_matrix(test_metrics["targets"], test_metrics["predictions"],
                          os.path.join(out_dir, "confusion_matrix_test.png"),
                          f"DistilBERT on MELD - {run_name} - test confusion matrix")
    plot_per_class_bars(summary["test_per_class"], summary["test"],
                        os.path.join(out_dir, "per_class_metrics_test.png"),
                        f"DistilBERT on MELD - {run_name} - test metrics per emotion")
    print("Saved outputs to", out_dir)
    return summary
