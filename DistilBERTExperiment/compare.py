"""Compare finished DistilBERT runs: context size x speaker identification.

Reads results/ctx<k>_<speakers|no_speakers>/metrics.json and writes to results/comparison/:
    summary.csv               - one row per run (dev + test metrics, per-class test F1)
    metrics_vs_context.png    - test accuracy / macro F1 / weighted F1 vs context size, with vs without speakers
    per_class_f1_test.png     - test F1 per emotion for every run
"""
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import EMOTION_LABELS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUN_PATTERN = re.compile(r"^ctx(\d+)_(speakers|no_speakers)$")  # default seed only


def load_runs(results_dir=RESULTS_DIR):
    runs = []
    for path in glob.glob(os.path.join(results_dir, "*", "metrics.json")):
        match = RUN_PATTERN.match(os.path.basename(os.path.dirname(path)))
        if not match:
            continue
        with open(path) as f:
            summary = json.load(f)
        summary["context_size"] = int(match.group(1))
        summary["speakers"] = match.group(2) == "speakers"
        runs.append(summary)
    return sorted(runs, key=lambda r: (r["speakers"], r["context_size"]))


def write_summary(runs, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["run", "context_size", "speakers", "best_epoch"]
            + [f"{split}_{m}" for split in ("dev", "test") for m in ("accuracy", "macro_f1", "weighted_f1", "loss")]
            + [f"test_f1_{label}" for label in EMOTION_LABELS]
        )
        for r in runs:
            writer.writerow(
                [r["run"], r["context_size"], r["speakers"], r["best_epoch"]]
                + [round(r[split][m], 4) for split in ("dev", "test") for m in ("accuracy", "macro_f1", "weighted_f1", "loss")]
                + [r["test_per_class_f1"][label] for label in EMOTION_LABELS]
            )


def plot_metrics_vs_context(runs, path):
    panels = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1")]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 4.5))
    for ax, (key, title) in zip(axes, panels):
        for speakers, style in ((False, "o-"), (True, "s--")):
            subset = [r for r in runs if r["speakers"] == speakers]
            if not subset:
                continue
            xs = [r["context_size"] for r in subset]
            ys = [r["test"][key] for r in subset]
            ax.plot(xs, ys, style, label="with speakers" if speakers else "without speakers")
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=8)
        ax.set_xticks(sorted({r["context_size"] for r in runs}))
        ax.set_xlabel("Previous utterances in context")
        ax.set_title(f"Test {title}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("DistilBERT on MELD - effect of context size and speaker identification")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_per_class_f1(runs, path):
    matrix = np.array([[r["test_per_class_f1"][label] for label in EMOTION_LABELS] for r in runs])
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(runs) + 2))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(0.01, matrix.max()))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(EMOTION_LABELS)), EMOTION_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(runs)), [r["run"] for r in runs])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_title("Test F1 per emotion")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit(f"No finished runs found in {RESULTS_DIR}")
    out_dir = os.path.join(RESULTS_DIR, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    write_summary(runs, os.path.join(out_dir, "summary.csv"))
    plot_metrics_vs_context(runs, os.path.join(out_dir, "metrics_vs_context.png"))
    plot_per_class_f1(runs, os.path.join(out_dir, "per_class_f1_test.png"))

    print(f"{'run':<22} {'acc':>6} {'macroF1':>8} {'wF1':>6}")
    for r in runs:
        t = r["test"]
        print(f"{r['run']:<22} {t['accuracy']:6.4f} {t['macro_f1']:8.4f} {t['weighted_f1']:6.4f}")
    print("Saved comparison to", out_dir)


if __name__ == "__main__":
    main()
