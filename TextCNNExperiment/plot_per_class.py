"""Per-emotion bar charts for the saved TextCNN experiments, without retraining.

Loads each saved best checkpoint, predicts the official MELD test split and writes
results/<experiment>/[seed_<s>/]per_class_metrics_test.png:
    - precision / recall (= per-class accuracy) / F1 for each of the 7 emotions
    - an 'overall' group with accuracy, macro F1 and weighted F1

    python plot_per_class.py
    python plot_per_class.py --checkpoint-01 /path/to/best_textcnn.pth
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from data_preprocessing import Preprocessor, ensure_meld_csvs
from dataset import MELDDataset
from evaluation import EMOTION_LABELS, LABEL2ID, evaluate, format_metrics
from textcnn import TextCNN

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

SQRT = "04_glove_finetuned_sqrt_weights"
SQRT_ELR_WD = "05_glove_finetuned_sqrt_weights_emb_lr1e-4_wd0.01"
# results sub-folder -> checkpoint written by textcnn.py / textcnn_glove.py
EXPERIMENTS = {
    "01_class_weights_early_stopping": "best_textcnn.pth",
    "02_glove_frozen": "best_textcnn_glove_frozen.pth",
    "03_glove_finetuned": "best_textcnn_glove_finetuned.pth",
    f"{SQRT}/seed_42": "best_textcnn_glove_finetuned_sqrtw.pth",
    f"{SQRT}/seed_1": "best_textcnn_glove_finetuned_sqrtw_seed1.pth",
    f"{SQRT}/seed_2": "best_textcnn_glove_finetuned_sqrtw_seed2.pth",
    f"{SQRT_ELR_WD}/seed_42": "best_textcnn_glove_finetuned_sqrtw_elr0.0001_wd0.01.pth",
    f"{SQRT_ELR_WD}/seed_1": "best_textcnn_glove_finetuned_sqrtw_elr0.0001_wd0.01_seed1.pth",
    f"{SQRT_ELR_WD}/seed_2": "best_textcnn_glove_finetuned_sqrtw_elr0.0001_wd0.01_seed2.pth",
}


def per_class_scores(targets, predictions):
    labels = list(range(len(EMOTION_LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    return {
        label: {"precision": precision[i], "recall": recall[i], "f1": f1[i], "support": int(support[i])}
        for i, label in enumerate(EMOTION_LABELS)
    }


def plot_per_class_bars(per_class, overall, out_path, title):
    """Same layout as DistilBERTExperiment/evaluation.py, so the two models are comparable."""
    fig, (ax_cls, ax_all) = plt.subplots(
        1, 2, figsize=(17, 5.5), gridspec_kw={"width_ratios": [len(EMOTION_LABELS), 1.6]}
    )
    x = np.arange(len(EMOTION_LABELS))
    width = 0.27
    for offset, (key, name) in zip((-width, 0, width), (("precision", "Precision"),
                                                          ("recall", "Recall (per-class accuracy)"),
                                                          ("f1", "F1"))):
        bars = ax_cls.bar(x + offset, [per_class[label][key] for label in EMOTION_LABELS], width, label=name)
        ax_cls.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax_cls.set_xticks(x, [f"{label}\n(n={per_class[label]['support']})" for label in EMOTION_LABELS])
    ax_cls.set_ylim(0, 1.05)
    ax_cls.set_ylabel("Score")
    ax_cls.set_title("Per emotion")
    ax_cls.grid(axis="y", alpha=0.3)
    ax_cls.legend(loc="upper left")

    names = ["Accuracy", "Macro F1", "Weighted F1"]
    bars = ax_all.bar(names, [overall["accuracy"], overall["macro_f1"], overall["weighted_f1"]],
                      color=["#4c72b0", "#dd8452", "#55a868"])
    ax_all.bar_label(bars, fmt="%.3f", fontsize=8, padding=1)
    ax_all.set_ylim(0, 1.05)
    ax_all.set_title("Overall")
    ax_all.tick_params(axis="x", rotation=20)
    ax_all.grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_test_loader():
    """Same preprocessing and train-only vocabulary as textcnn.py / textcnn_glove.py."""
    paths = ensure_meld_csvs()
    preprocessor = Preprocessor()
    train_df = preprocessor.start_pipeline(paths["train"])
    test_df = preprocessor.start_pipeline(paths["test"])
    preprocessor.build_vocab(train_df)
    loader = DataLoader(MELDDataset(test_df, preprocessor.vocab, LABEL2ID), batch_size=32,
                        shuffle=False, collate_fn=MELDDataset.collate_fn)
    return loader, len(preprocessor.vocab)


def load_model(checkpoint, vocab_size):
    state = torch.load(checkpoint, map_location="cpu")
    rows, dim = state["embedding.weight"].shape
    if rows != vocab_size:
        raise ValueError(f"{checkpoint}: embedding has {rows} rows but the vocabulary has {vocab_size}; "
                         "this checkpoint was trained with a different vocabulary/split")
    model = TextCNN(vocab_size=vocab_size, embedding_dim=dim, num_classes=len(EMOTION_LABELS),
                    num_filters=100, dialogue_filters=128, dropout=0.5)
    model.load_state_dict(state)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-01", default=None,
                        help="checkpoint for 01_class_weights_early_stopping (default: best_textcnn.pth)")
    args = parser.parse_args()

    loader, vocab_size = build_test_loader()
    criterion = nn.CrossEntropyLoss()
    for experiment, checkpoint in EXPERIMENTS.items():
        if experiment.startswith("01_") and args.checkpoint_01:
            checkpoint = args.checkpoint_01
        else:
            checkpoint = os.path.join(HERE, checkpoint)
        if not os.path.exists(checkpoint):
            print(f"Skipping {experiment}: {checkpoint} not found")
            continue

        model = load_model(checkpoint, vocab_size)
        metrics = evaluate(model, loader, criterion, torch.device("cpu"))
        out_dir = os.path.join(RESULTS_DIR, experiment)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "per_class_metrics_test.png")
        plot_per_class_bars(per_class_scores(metrics["targets"], metrics["predictions"]), metrics, out_path,
                            f"TextCNN on MELD - {experiment.replace('/', ' / ')} - test metrics per emotion")
        print(f"{experiment:<62} {format_metrics('Test', metrics)}")


if __name__ == "__main__":
    main()
