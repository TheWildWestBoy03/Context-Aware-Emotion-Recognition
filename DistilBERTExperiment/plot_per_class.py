"""Per-emotion bar charts (precision / recall / F1 + overall accuracy, macro F1, weighted F1)
for runs that already finished, without retraining.

Reads results/<run>/metrics.json (overall scores) and per-class scores from
metrics.json["test_per_class"] when present, otherwise from classification_report_test.txt.
Writes results/<run>/per_class_metrics_test.png.

    python plot_per_class.py            # all runs in results/
    python plot_per_class.py ctx3_speakers
"""
import glob
import json
import os
import re
import sys

from data import EMOTION_LABELS
from evaluation import plot_per_class_bars

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
REPORT_ROW = re.compile(r"^\s*(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$")


def per_class_from_report(path):
    scores = {}
    with open(path) as f:
        for line in f:
            match = REPORT_ROW.match(line)
            if match and match.group(1) in EMOTION_LABELS:
                label, precision, recall, f1, support = match.groups()
                scores[label] = {"precision": float(precision), "recall": float(recall),
                                 "f1": float(f1), "support": int(support)}
    missing = set(EMOTION_LABELS) - set(scores)
    if missing:
        raise ValueError(f"{path}: no scores for {sorted(missing)}")
    return scores


def plot_run(run_dir):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        summary = json.load(f)
    per_class = summary.get("test_per_class") or per_class_from_report(
        os.path.join(run_dir, "classification_report_test.txt")
    )
    out_path = os.path.join(run_dir, "per_class_metrics_test.png")
    plot_per_class_bars(per_class, summary["test"], out_path,
                        f"DistilBERT on MELD - {summary['run']} - test metrics per emotion")
    print("Saved", out_path)


def main():
    names = sys.argv[1:]
    run_dirs = ([os.path.join(RESULTS_DIR, n) for n in names] if names else
                sorted(os.path.dirname(p) for p in glob.glob(os.path.join(RESULTS_DIR, "*", "metrics.json"))))
    if not run_dirs:
        raise SystemExit(f"No finished runs found in {RESULTS_DIR}")
    for run_dir in run_dirs:
        plot_run(run_dir)


if __name__ == "__main__":
    main()
