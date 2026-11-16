import copy
import os
import random
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless SLURM nodes
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from proiect.biltsm.dataset import (
    EMOTION_LABELS,
    PAD_IDX,
    MELDSequenceDataset,
    build_vocabulary,
    collate_utterances,
    compute_smoothed_class_weights,
    ensure_meld_csvs,
    load_glove_embeddings,
    load_meld_split,
)
from proiect.biltsm.model import BiLSTMClassifier


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    metrics = {
        "loss": total_loss / max(total_samples, 1),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }
    return metrics["loss"], metrics, y_true, y_pred


def save_result_plots(
    history: Dict[str, List[float]],
    best_epoch: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    out_dir: str = "plots",
) -> None:
    """Generate and save training curves, per-class bar chart, and confusion matrix."""
    os.makedirs(out_dir, exist_ok=True)
    epochs = history["epoch"]

    # ---------------------------------------------------------
    # 1. Training & Validation Loss / F1 Curves
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, history["train_loss"], marker="o", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], marker="s", label="Validation Loss", color="#ff7f0e")
    ax1.axvline(best_epoch, color="gray", linestyle="--", label=f"Best Epoch ({best_epoch})")
    ax1.set_title("BiLSTM Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_xticks(epochs)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["val_macro_f1"], marker="o", label="Val Macro-F1", color="#2ca02c")
    ax2.plot(
        epochs, history["val_weighted_f1"], marker="s", label="Val Weighted-F1", color="#9467bd"
    )
    ax2.axvline(best_epoch, color="gray", linestyle="--", label=f"Best Epoch ({best_epoch})")
    ax2.set_title("Validation F1 Progression")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xticks(epochs)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------
    # 2. Per-Class Precision, Recall, and F1-Score Bar Chart
    # ---------------------------------------------------------
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), zero_division=0
    )
    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 5.5))
    rects1 = ax.bar(x - width, precision, width, label="Precision", color="#4c72b0")
    rects2 = ax.bar(x, recall, width, label="Recall", color="#55a868")
    rects3 = ax.bar(x + width, f1, width, label="F1-Score", color="#c44e52")

    for rects in (rects1, rects2, rects3):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    x_tick_labels = [f"{lbl.capitalize()}\n(n={sup})" for lbl, sup in zip(labels, support)]
    ax.set_title("BiLSTM Test Performance per Emotion Class (MELD)")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_class_metrics.png"), dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------
    # 3. Confusion Matrix Heatmap (Counts + Row Percentages)
    # ---------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-Normalized Proportion (Recall)")

    cap_labels = [lbl.capitalize() for lbl in labels]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(cap_labels, rotation=35, ha="right")
    ax.set_yticklabels(cap_labels)
    ax.set_xlabel("Predicted Emotion")
    ax.set_ylabel("True Emotion")
    ax.set_title("BiLSTM Confusion Matrix on MELD Test Set")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
            )

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300)
    plt.close(fig)

    print(f"\nSaved evaluation plots to '{out_dir}/':")
    print(f"  - {os.path.join(out_dir, 'training_curves.png')}")
    print(f"  - {os.path.join(out_dir, 'per_class_metrics.png')}")
    print(f"  - {os.path.join(out_dir, 'confusion_matrix.png')}")


def main() -> None:
    seed = 42
    batch_size = 64
    embed_dim = 300
    hidden_dim = 128
    num_layers = 2
    dropout = 0.3
    max_len = 80
    lr = 1e-3
    weight_decay = 1e-4
    max_epochs = 25
    patience = 5
    use_class_weights = True

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 1. Load official MELD splits
    csv_paths = ensure_meld_csvs("data")
    train_tokens, train_labels = load_meld_split(csv_paths["train"], split_name="train")
    dev_tokens, dev_labels = load_meld_split(csv_paths["dev"], split_name="dev")
    test_tokens, test_labels = load_meld_split(csv_paths["test"], split_name="test")

    vocab = build_vocabulary(train_tokens, min_freq=1)
    print(
        f"Loaded MELD | Train: {len(train_labels)}, "
        f"Dev: {len(dev_labels)}, Test: {len(test_labels)} | "
        f"Vocab size: {len(vocab)}"
    )

    # 2. Load pretrained GloVe embeddings
    glove_matrix = load_glove_embeddings(
        vocab=vocab,
        embed_dim=embed_dim,
        glove_txt_path=None,  # Optional path to local glove.6B.300d.txt
        seed=seed,
    )

    # 3. Create Datasets and DataLoaders
    train_ds = MELDSequenceDataset(train_tokens, train_labels, vocab, max_len=max_len)
    dev_ds = MELDSequenceDataset(dev_tokens, dev_labels, vocab, max_len=max_len)
    test_ds = MELDSequenceDataset(test_tokens, test_labels, vocab, max_len=max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_utterances
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_utterances
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_utterances
    )

    # 4. Initialize BiLSTM, CrossEntropyLoss, and AdamW
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=len(EMOTION_LABELS),
        dropout=dropout,
        pad_idx=PAD_IDX,
        pretrained_embeddings=glove_matrix,
        freeze_embeddings=False,
    ).to(device)

    class_weights = None
    if use_class_weights:
        class_weights = compute_smoothed_class_weights(
            train_labels, num_classes=len(EMOTION_LABELS), power=0.5
        ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # 5. Train with validation Macro-F1 selection & early stopping
    best_val_macro_f1 = -1.0
    best_epoch = 1
    best_model_state = None
    epochs_without_improvement = 0
    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
    }

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_metrics, _, _ = evaluate(model, dev_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_weighted_f1"].append(val_metrics["weighted_f1"])

        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"Val Weighted-F1: {val_metrics['weighted_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(best_model_state, "best_bilstm_meld.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best Val Macro-F1: {best_val_macro_f1:.4f})."
                )
                break

    # 6. Evaluate best checkpoint on the held-out MELD test split
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    _, test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    print("\n=== BiLSTM Test Set Results (MELD) ===")
    print(f"Macro-F1:    {test_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {test_metrics['weighted_f1']:.4f}\n")

    print("Per-Class Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=EMOTION_LABELS,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion Matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(y_true, y_pred)
    header = " " * 10 + " ".join(f"{lbl[:7]:>8}" for lbl in EMOTION_LABELS)
    print(header)
    for label, row in zip(EMOTION_LABELS, cm):
        row_str = " ".join(f"{val:8d}" for val in row)
        print(f"{label:<10}{row_str}")

    # 7. Save result graphics
    save_result_plots(
        history=history,
        best_epoch=best_epoch,
        y_true=y_true,
        y_pred=y_pred,
        labels=EMOTION_LABELS,
        out_dir="plots",
    )


if __name__ == "__main__":
    main()