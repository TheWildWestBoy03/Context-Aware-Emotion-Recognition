"""TextCNN on MELD with pretrained GloVe 100d embeddings.

Separate experiment: reuses the model, data pipeline and training loop from textcnn.py
without changing them, and writes its own checkpoint and plots so the baseline
results (best_textcnn.pth, plots/) are not overwritten.

    python textcnn_glove.py              # GloVe frozen
    FREEZE_EMBEDDINGS=0 python textcnn_glove.py   # GloVe fine-tuned
    FREEZE_EMBEDDINGS=0 CLASS_WEIGHTING=sqrt python textcnn_glove.py   # + softer class weights
    FREEZE_EMBEDDINGS=0 CLASS_WEIGHTING=sqrt EMBEDDING_LR=1e-4 WEIGHT_DECAY=1e-2 python textcnn_glove.py
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_preprocessing import Preprocessor, ensure_meld_csvs
from dataset import MELDDataset
from embeddings import build_embedding_matrix
from evaluation import LABEL2ID, History, evaluate, format_metrics, report, save_evaluation
from textcnn import TextCNN, compute_class_weights, set_seed, train_one_epoch

FREEZE_EMBEDDINGS = os.environ.get("FREEZE_EMBEDDINGS", "1") == "1"
# "inverse": total / (K * count_c) as in textcnn.py; "sqrt": square root of that (softer)
CLASS_WEIGHTING = os.environ.get("CLASS_WEIGHTING", "inverse")
if CLASS_WEIGHTING not in ("inverse", "sqrt"):
    raise ValueError(f"CLASS_WEIGHTING must be 'inverse' or 'sqrt', got {CLASS_WEIGHTING!r}")

LEARNING_RATE = 1e-3
# Separate learning rate for the GloVe vectors when fine-tuning (default: same as the rest)
EMBEDDING_LR = float(os.environ.get("EMBEDDING_LR", LEARNING_RATE))
# Decoupled (AdamW) weight decay on the non-embedding layers; 0 = plain Adam behaviour
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.0))

RUN_NAME = "glove_frozen" if FREEZE_EMBEDDINGS else "glove_finetuned"
if CLASS_WEIGHTING == "sqrt":
    RUN_NAME += "_sqrtw"
if EMBEDDING_LR != LEARNING_RATE and not FREEZE_EMBEDDINGS:
    RUN_NAME += f"_elr{EMBEDDING_LR:g}"
if WEIGHT_DECAY > 0:
    RUN_NAME += f"_wd{WEIGHT_DECAY:g}"
# Random seed; runs with a non-default seed get their own checkpoint/plots
SEED = int(os.environ.get("SEED", 42))
if SEED != 42:
    RUN_NAME += f"_seed{SEED}"
CHECKPOINT = f"best_textcnn_{RUN_NAME}.pth"
PLOTS_DIR = f"plots_{RUN_NAME}"


class GloVeTextCNN(TextCNN):
    """TextCNN whose embedding layer is initialised from a pretrained matrix."""

    def __init__(self, embedding_matrix, freeze_embeddings=True, **kwargs):
        super().__init__(
            vocab_size=embedding_matrix.size(0),
            embedding_dim=embedding_matrix.size(1),
            **kwargs,
        )
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze_embeddings, padding_idx=0
        )


def make_loader(df, vocab, shuffle):
    return DataLoader(
        MELDDataset(df, vocab, LABEL2ID),
        batch_size=32,
        shuffle=shuffle,
        collate_fn=MELDDataset.collate_fn,
    )


def main():
    set_seed(SEED)
    print(f"Run: {RUN_NAME} (checkpoint {CHECKPOINT}, plots in {PLOTS_DIR}/)")

    # Official MELD splits, vocabulary from train only (same as textcnn.py)
    meld_paths = ensure_meld_csvs()
    preprocessor = Preprocessor()
    train_df = preprocessor.start_pipeline(meld_paths["train"])
    val_df = preprocessor.start_pipeline(meld_paths["dev"])
    test_df = preprocessor.start_pipeline(meld_paths["test"])
    preprocessor.build_vocab(train_df)

    train_loader = make_loader(train_df, preprocessor.vocab, shuffle=True)
    val_loader = make_loader(val_df, preprocessor.vocab, shuffle=False)
    test_loader = make_loader(test_df, preprocessor.vocab, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    embedding_matrix = build_embedding_matrix(preprocessor.vocab, embedding_dim=100)
    network = GloVeTextCNN(
        embedding_matrix,
        freeze_embeddings=FREEZE_EMBEDDINGS,
        num_classes=7,
        num_filters=100,
        dialogue_filters=128,
        dropout=0.5,
    ).to(device)

    # Same loss, optimizer and early stopping as textcnn.py
    class_weights = compute_class_weights(train_loader.dataset, len(LABEL2ID))
    if CLASS_WEIGHTING == "sqrt":
        class_weights = class_weights.sqrt()
    print("Class weights:", {
        label: round(class_weights[idx].item(), 3) for label, idx in LABEL2ID.items()
    })
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # Embeddings get their own LR and no weight decay (decay would pull GloVe vectors to 0).
    # With WEIGHT_DECAY=0 and EMBEDDING_LR=LEARNING_RATE this is identical to plain Adam.
    other_params = [
        p for name, p in network.named_parameters()
        if p.requires_grad and not name.startswith("embedding.")
    ]
    param_groups = [{"params": other_params, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}]
    if not FREEZE_EMBEDDINGS:
        param_groups.append({
            "params": list(network.embedding.parameters()),
            "lr": EMBEDDING_LR,
            "weight_decay": 0.0,
        })
    optimizer = torch.optim.AdamW(param_groups)
    print(f"Optimizer: AdamW lr={LEARNING_RATE:g}, embedding lr="
          f"{'frozen' if FREEZE_EMBEDDINGS else f'{EMBEDDING_LR:g}'}, weight decay={WEIGHT_DECAY:g}")

    num_epochs = 30
    patience = 3
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history = History()

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(network, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(network, val_loader, criterion, device)
        history.add("train", train_metrics)
        history.add("val", val_metrics)

        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        print(format_metrics("Train", train_metrics))
        print(format_metrics("Val", val_metrics))

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(network.state_dict(), CHECKPOINT)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1} "
                      f"(no dev macro F1 improvement for {patience} epochs)")
                break

    print(f"\nBest validation Macro F1: {best_val_f1:.4f} (epoch {history.best_epoch()})")

    network.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    test_metrics = evaluate(network, test_loader, criterion, device)
    report("Test", test_metrics)
    save_evaluation(history, test_metrics, out_dir=PLOTS_DIR)


if __name__ == "__main__":
    main()
