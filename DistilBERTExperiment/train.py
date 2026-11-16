"""Fine-tune DistilBERT for context-aware emotion recognition on MELD (one configuration).

Examples:
    python train.py --context-size 0                    # target utterance only
    python train.py --context-size 3 --speakers         # 3 previous utterances, with speaker names
    python train.py --context-size 50                   # -> error: too many previous utterances

Outputs go to results/ctx<k>_<speakers|no_speakers>/ (an existing run is never overwritten
unless --overwrite is given).
"""
import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import LABEL2ID, MELDContextDataset, load_meld, max_context_size, validate_context_size
from evaluation import compute_metrics, evaluate, format_metrics, History, save_run_outputs, classification_report_text
from model import DEFAULT_MODEL_NAME, DistilBertEmotionClassifier

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context-size", type=int, required=True,
                        help="number of previous utterances from the same dialogue given as context")
    parser.add_argument("--speakers", action=argparse.BooleanOptionalAction, default=False,
                        help="prefix every utterance with its speaker name (default: off)")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=256, help="max tokens per input (<= 512)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--class-weighting", choices=["none", "sqrt", "inverse"], default="sqrt",
                        help="class weights in the loss (sqrt worked best for TextCNN)")
    parser.add_argument("--evals-per-epoch", type=int, default=2, help="dev evaluations per epoch")
    parser.add_argument("--patience", type=int, default=3,
                        help="stop after this many dev evaluations without macro F1 improvement")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-dialogues", type=int, default=None,
                        help="debug: use only the first N dialogues of every split")
    parser.add_argument("--out-dir", default=None, help="default: results/<run name>")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing run")
    parser.add_argument("--save-model", action="store_true", help="also save the best weights (~260 MB)")
    return parser.parse_args(argv)


def run_name(args):
    name = f"ctx{args.context_size}_{'speakers' if args.speakers else 'no_speakers'}"
    if args.seed != 42:
        name += f"_seed{args.seed}"
    return name


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights(labels, num_classes, mode):
    counts = np.bincount(labels, minlength=num_classes)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    if mode == "sqrt":
        weights = np.sqrt(weights)
    elif mode == "none":
        weights = np.ones(num_classes)
    return torch.tensor(weights, dtype=torch.float)


def build_optimizer(model, lr, weight_decay):
    """AdamW without weight decay on biases and LayerNorm weights (standard BERT fine-tuning)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if name.endswith("bias") or "LayerNorm" in name or "layer_norm" in name else decay).append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def main(argv=None):
    args = parse_args(argv)
    name = run_name(args)
    out_dir = args.out_dir or os.path.join(RESULTS_DIR, name)
    if os.path.exists(os.path.join(out_dir, "metrics.json")) and not args.overwrite:
        raise FileExistsError(f"{out_dir} already has results; pass --overwrite to replace them.")
    if not 0 < args.max_length <= 512:
        raise ValueError(f"--max-length must be in 1..512 (DistilBERT limit), got {args.max_length}")

    set_seed(args.seed)
    splits = load_meld()
    validate_context_size(args.context_size, splits["train"])
    if args.limit_dialogues:
        splits = {s: df[df.dialogue_id < args.limit_dialogues] for s, df in splits.items()}

    device = get_device()
    print(f"Run {name} | device {device} | context size {args.context_size} "
          f"(max allowed {max_context_size(splits['train'])}) | speakers {args.speakers}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    datasets = {
        split: MELDContextDataset(df, tokenizer, args.context_size, args.speakers, args.max_length)
        for split, df in splits.items()
    }
    for split, dataset in datasets.items():
        dataset.describe(split)

    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size if split == "train" else 64,
                          shuffle=(split == "train"), collate_fn=dataset.collate,
                          generator=generator if split == "train" else None)
        for split, dataset in datasets.items()
    }

    model = DistilBertEmotionClassifier(len(LABEL2ID), args.model_name, args.dropout).to(device)
    weights = class_weights(datasets["train"].labels(), len(LABEL2ID), args.class_weighting)
    print("Class weights:", {label: round(weights[i].item(), 3) for label, i in LABEL2ID.items()})
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * args.epochs
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)

    history = History()
    best_f1, best_state, best_val = -1.0, None, None
    evals_without_improvement = 0
    interval_loss, interval_count, interval_targets, interval_preds = 0.0, 0, [], []
    step, stop = 0, False
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(loaders["train"]):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            interval_loss += loss.item() * labels.size(0)
            interval_count += labels.size(0)
            interval_preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())
            interval_targets.extend(labels.cpu().tolist())

            # Evaluate evals_per_epoch times per epoch, at evenly spaced batches; the last one
            # is always the end of the epoch (no extra 1-batch evaluation at the epoch boundary).
            batches_done = batch_idx + 1
            remaining = steps_per_epoch - batches_done
            last_batch = remaining == 0
            if (batches_done % eval_every == 0 and remaining >= eval_every) or last_batch:
                train_metrics = {"loss": interval_loss / interval_count,
                                 **compute_metrics(interval_targets, interval_preds)}
                val_metrics = evaluate(model, loaders["dev"], criterion, device)
                model.train()
                epoch_position = epoch + (batch_idx + 1) / steps_per_epoch
                history.add(epoch_position, train_metrics, val_metrics)
                interval_loss, interval_count, interval_targets, interval_preds = 0.0, 0, [], []

                print(f"\nEpoch {epoch_position:.2f} | step {step}/{total_steps} | "
                      f"{(time.time() - start_time) / 60:.1f} min")
                print(format_metrics("Train", train_metrics))
                print(format_metrics("Val", val_metrics))

                if val_metrics["macro_f1"] > best_f1:
                    best_f1, best_val = val_metrics["macro_f1"], val_metrics
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1
                    if evals_without_improvement >= args.patience:
                        print(f"\nEarly stopping: no dev macro F1 improvement in {args.patience} evaluations")
                        stop = True
                        break
        if stop:
            break

    print(f"\nBest dev macro F1 {best_f1:.4f} at epoch {history.epochs[history.best_index()]:.2f}")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], criterion, device)
    print("\n" + format_metrics("Test", test_metrics))
    print(classification_report_text(test_metrics))

    config = {k: v for k, v in vars(args).items() if k not in ("out_dir", "overwrite")}
    config["device"] = str(device)
    config["minutes"] = round((time.time() - start_time) / 60, 1)
    summary = save_run_outputs(out_dir, name, config, history, test_metrics, best_val)
    if args.save_model:
        torch.save(best_state, os.path.join(out_dir, "best_model.pt"))
    return summary


if __name__ == "__main__":
    main()
