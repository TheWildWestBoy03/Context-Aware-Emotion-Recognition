"""
DeBERTa-v3 pe MELD – recunoașterea emoțiilor la nivel de replică, în contextul conversației.

Format input:
[CLS] replici anterioare [SEP] vorbitor: replica țintă [SEP] replici următoare [SEP]

Fișiere necesare în data_dir:
    train_sent_emo.csv
    dev_sent_emo.csv
    test_sent_emo.csv

Instalare (Colab):
    !pip install -q transformers sentencepiece protobuf scikit-learn pandas matplotlib

Implicit, antrenarea folosește FP32 pentru a evita eroarea:
    ValueError: Attempting to unscale FP16 gradients.

Dacă dorești ulterior mixed precision, setează fp16=True. Modelul este convertit
explicit la FP32 înainte de crearea optimizerului, iar GradScaler este activat
numai în acel caz.
"""

import json
import math
import os
import random
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURARE
# ============================================================

CONFIG = {
    "data_dir": ".",                       # folderul cu cele 3 CSV-uri
    "model": "microsoft/deberta-v3-base",  # base, small sau xsmall
    "out_dir": "deberta_output",

    "k_past": 4,          # numărul replicilor anterioare
    "k_future": 2,        # numărul replicilor următoare; 0 = fără viitor
    "max_len": 192,       # lungimea maximă în tokeni

    "epochs": 5,
    "batch": 16,
    "eval_batch": 64,
    "accum": 1,           # gradient accumulation
    "lr": 2e-5,
    "wd": 0.01,
    "warmup_ratio": 0.10,
    "patience": 2,

    "fp16": False,        # recomandat False pentru prima rulare
    "seed": 42,
    "num_workers": 2,
}


SPLITS = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}


# ============================================================
# DATE ȘI CONTEXT
# ============================================================

def load_meld(data_dir):
    """Încarcă split-urile MELD și construiește etichete consistente."""
    data = {}

    for split, filename in SPLITS.items():
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Nu găsesc fișierul: {path}")

        df = pd.read_csv(path)
        required = {"Dialogue_ID", "Utterance_ID", "Utterance", "Speaker", "Emotion"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{filename}: lipsesc coloanele {sorted(missing)}")

        df = df.sort_values(
            ["Dialogue_ID", "Utterance_ID"]
        ).reset_index(drop=True)

        text = (
            df["Utterance"].fillna("").astype(str)
            .str.replace("\x92", "'", regex=False)
            .str.replace(r"[\x80-\x9f]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        data[split] = pd.DataFrame({
            "dialogue": df["Dialogue_ID"],
            "utt": df["Utterance_ID"],
            "speaker": df["Speaker"].fillna("").astype(str).str.strip(),
            "text": text,
            "emotion": df["Emotion"].astype(str).str.strip(),
        })

    # Etichetele sunt definite exclusiv din train.
    classes = sorted(data["train"]["emotion"].unique())
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    for split, df in data.items():
        unknown = sorted(set(df["emotion"]) - set(class_to_id))
        if unknown:
            raise ValueError(
                f"Split-ul {split} conține clase absente din train: {unknown}"
            )
        df["label"] = df["emotion"].map(class_to_id).astype(int)

    return data, classes


def dialogue_slices(df):
    """Returnează intervalele fiecărui dialog, presupunând date sortate."""
    dialogue_ids = df["dialogue"].to_numpy()
    cuts = np.flatnonzero(
        np.r_[True, dialogue_ids[1:] != dialogue_ids[:-1], True]
    )
    return list(zip(cuts[:-1], cuts[1:]))


def context_windows(df, k_past, k_future):
    """Construiește contextul fără a trece granița dintre dialoguri."""
    past, target, future = [], [], []

    def format_turn(i):
        return f"{df.speaker.iat[i]}: {df.text.iat[i]}"

    for start, end in dialogue_slices(df):
        for i in range(start, end):
            past.append([
                format_turn(j)
                for j in range(max(start, i - k_past), i)
            ])
            target.append(format_turn(i))
            future.append([
                format_turn(j)
                for j in range(i + 1, min(end, i + 1 + k_future))
            ])

    return past, target, future


def encode_example(tokenizer, past, target, future, max_len):
    """
    Codifică [CLS] trecut [SEP] țintă [SEP] viitor [SEP].
    Replica țintă primește prioritate și nu este eliminată de context.
    """
    target_ids = tokenizer.encode(
        target, add_special_tokens=False
    )[:min(96, max_len // 2)]

    left_ids = (
        tokenizer.encode(" ".join(past), add_special_tokens=False)
        if past else []
    )
    right_ids = (
        tokenizer.encode(" ".join(future), add_special_tokens=False)
        if future else []
    )

    # 4 tokeni speciali: CLS + 3 SEP.
    remaining = max(0, max_len - 4 - len(target_ids))

    if right_ids:
        left_budget = int(remaining * 0.65)
        right_budget = remaining - left_budget
    else:
        left_budget = remaining
        right_budget = 0

    if left_budget > 0:
        left_ids = left_ids[-left_budget:]
    else:
        left_ids = []

    if right_budget > 0:
        right_ids = right_ids[:right_budget]
    else:
        right_ids = []

    return (
        [tokenizer.cls_token_id]
        + left_ids
        + [tokenizer.sep_token_id]
        + target_ids
        + [tokenizer.sep_token_id]
        + right_ids
        + [tokenizer.sep_token_id]
    )


def class_weights(labels, num_classes, power=0.5):
    """Ponderi moderate pentru clase dezechilibrate."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Există cel puțin o clasă fără exemple în train.")

    weights = (len(labels) / (num_classes * counts)) ** power
    weights = weights / ((weights * counts).sum() / len(labels))
    return weights.astype(np.float32)


# ============================================================
# EVALUARE ȘI RAPORTARE
# ============================================================

def evaluate(model, loader, device, use_amp):
    import torch

    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            amp_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True,
                )
                if use_amp else nullcontext()
            )

            with amp_context:
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

            probabilities = torch.softmax(logits.float(), dim=-1)
            all_probs.append(probabilities.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def save_report(out_dir, y_true, probabilities, classes, history,
                test_df, config, title=""):
    os.makedirs(out_dir, exist_ok=True)

    num_classes = len(classes)
    predictions = probabilities.argmax(axis=1)
    order = np.argsort(-probabilities, axis=1)

    def per_class(metric):
        values = metric(
            y_true,
            predictions,
            average=None,
            labels=range(num_classes),
            zero_division=0,
        )
        return {name: float(value) for name, value in zip(classes, values)}

    results = {
        "test_accuracy": float(accuracy_score(y_true, predictions)),
        "test_precision_weighted": float(
            precision_score(y_true, predictions, average="weighted",
                            zero_division=0)
        ),
        "test_precision_macro": float(
            precision_score(y_true, predictions, average="macro",
                            zero_division=0)
        ),
        "test_recall_weighted": float(
            recall_score(y_true, predictions, average="weighted",
                         zero_division=0)
        ),
        "test_weighted_f1": float(
            f1_score(y_true, predictions, average="weighted")
        ),
        "test_macro_f1": float(
            f1_score(y_true, predictions, average="macro")
        ),
        "test_top2_accuracy": float(
            (order[:, :2] == y_true[:, None]).any(axis=1).mean()
        ),
        "test_top3_accuracy": float(
            (order[:, :3] == y_true[:, None]).any(axis=1).mean()
        ),
        "per_class_precision": per_class(precision_score),
        "per_class_recall": per_class(recall_score),
        "per_class_f1": per_class(f1_score),
        "config": config,
    }

    print(f"\n========== REZULTATE TEST {title} ==========")
    print(f"Accuracy:       {results['test_accuracy']:.4f}")
    print(f"Precision w:    {results['test_precision_weighted']:.4f}")
    print(f"Precision macro:{results['test_precision_macro']:.4f}")
    print(f"Weighted F1:    {results['test_weighted_f1']:.4f}")
    print(f"Macro F1:       {results['test_macro_f1']:.4f}")
    print(f"Top-2 accuracy: {results['test_top2_accuracy']:.4f}")
    print(f"Top-3 accuracy: {results['test_top3_accuracy']:.4f}")
    print("\n", classification_report(
        y_true,
        predictions,
        labels=range(num_classes),
        target_names=classes,
        zero_division=0,
    ))

    with open(os.path.join(out_dir, "rezultate.json"), "w",
              encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False, default=str)

    prediction_df = pd.DataFrame(
        probabilities,
        columns=[f"p_{name}" for name in classes],
    )
    prediction_df.insert(
        0, "top3",
        [",".join(classes[j] for j in row[:3]) for row in order],
    )
    prediction_df.insert(
        0, "pred", [classes[i] for i in predictions]
    )
    prediction_df.insert(
        0, "true", [classes[i] for i in y_true]
    )
    prediction_df.insert(0, "utterance", test_df["text"].to_numpy())
    prediction_df.insert(0, "speaker", test_df["speaker"].to_numpy())
    prediction_df.insert(0, "dialogue", test_df["dialogue"].to_numpy())
    prediction_df.to_csv(
        os.path.join(out_dir, "predictii_test.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Matricea de confuzie normalizată pe rând.
    matrix = confusion_matrix(
        y_true, predictions, labels=range(num_classes)
    )
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(classes)

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i,
                f"{normalized[i, j]:.2f}\n({matrix[i, j]})",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set_xlabel("Prezicere")
    ax.set_ylabel("Etichetă reală")
    ax.set_title(f"Matrice de confuzie – test {title}")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "matrice_confuzie.png"),
        dpi=150,
    )
    plt.close(fig)

    # Curbele de antrenare.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train loss")
    axes[0].plot(history["val_loss"], label="dev loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epocă")
    axes[0].legend()

    axes[1].plot(history["val_acc"], label="dev accuracy")
    axes[1].plot(history["val_prec"], label="dev precision")
    axes[1].plot(history["val_wf1"], label="dev weighted F1")
    axes[1].set_title("Metrici pe dev")
    axes[1].set_xlabel("Epocă")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "curbe_antrenare.png"),
        dpi=150,
    )
    plt.close(fig)

    return results


# ============================================================
# ANTRENARE
# ============================================================

def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    args = SimpleNamespace(**CONFIG)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_amp = (
        device.type == "cuda"
        and bool(args.fp16)
    )

    print(f"Device: {device} | model: {args.model}")
    print(f"Mixed precision FP16: {use_amp}")

    os.makedirs(args.out_dir, exist_ok=True)

    data, classes = load_meld(args.data_dir)
    num_classes = len(classes)
    print("Clase:", classes)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as exc:
        print(
            "Tokenizer rapid indisponibil; folosesc tokenizer lent:",
            type(exc).__name__,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_fast=False,
        )

    class MELDDataset(Dataset):
        def __init__(self, dataframe):
            past, target, future = context_windows(
                dataframe,
                args.k_past,
                args.k_future,
            )
            self.items = [
                (
                    encode_example(
                        tokenizer, p, t, f, args.max_len
                    ),
                    int(label),
                )
                for p, t, f, label in zip(
                    past, target, future, dataframe["label"]
                )
            ]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    def collate_fn(batch):
        max_length = max(len(ids) for ids, _ in batch)
        batch_size = len(batch)

        input_ids = torch.full(
            (batch_size, max_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
        )

        for i, (ids, _) in enumerate(batch):
            length = len(ids)
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = 1

        labels = torch.tensor(
            [label for _, label in batch],
            dtype=torch.long,
        )
        return input_ids, attention_mask, labels

    train_set = MELDDataset(data["train"])
    dev_set = MELDDataset(data["dev"])
    test_set = MELDDataset(data["test"])

    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_fn,
    }

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        **loader_options,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.eval_batch,
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch,
        shuffle=False,
        **loader_options,
    )

    print(
        f"Train {len(train_set)} | Dev {len(dev_set)} | "
        f"Test {len(test_set)}"
    )
    print(
        "Exemplu input:",
        tokenizer.decode(train_set[5][0])[:300],
        "...",
    )

    # IMPORTANT: modelul rămâne în FP32; nu apela half() sau
    # to(dtype=torch.float16). Acest lucru previne gradientele FP16
    # pe parametri și eroarea GradScaler.unscale_.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_classes,
    )
    model = model.to(device=device, dtype=torch.float32)

    # Verificare defensivă înainte de crearea optimizerului.
    non_fp32 = [
        name for name, param in model.named_parameters()
        if param.requires_grad and param.dtype != torch.float32
    ]
    if non_fp32:
        raise RuntimeError(
            "Parametri care nu sunt FP32: "
            + ", ".join(non_fp32[:10])
        )

    weights = class_weights(
        data["train"]["label"].to_numpy(),
        num_classes,
    )
    class_weight_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor)

    no_decay = ("bias", "LayerNorm.weight")
    decay_params = []
    no_decay_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(key in name for key in no_decay):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.wd},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )

    updates_per_epoch = math.ceil(len(train_loader) / args.accum)
    total_steps = args.epochs * updates_per_epoch
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # API-ul modern torch.amp. FP32 => scaler dezactivat.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_prec": [],
        "val_wf1": [],
    }

    best_dev_f1 = -1.0
    epochs_without_improvement = 0
    checkpoint_path = os.path.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen_examples = 0

        optimizer.zero_grad(set_to_none=True)

        for step, (input_ids, attention_mask, labels) in enumerate(
            train_loader,
            start=1,
        ):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(
                device, non_blocking=True
            )
            labels = labels.to(device, non_blocking=True)

            amp_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True,
                )
                if use_amp else nullcontext()
            )

            with amp_context:
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

            # Loss-ul se calculează în FP32 pentru stabilitate.
            loss = loss_fn(logits.float(), labels)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Loss NaN/Inf. Încearcă fp16=False, "
                    "un learning rate mai mic sau verifică datele."
                )

            scaled_loss = loss / args.accum
            scaler.scale(scaled_loss).backward()

            is_update_step = (
                step % args.accum == 0
                or step == len(train_loader)
            )

            if is_update_step:
                if use_amp:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            seen_examples += batch_size

        train_loss = running_loss / max(seen_examples, 1)

        dev_probs, dev_labels = evaluate(
            model, dev_loader, device, use_amp
        )
        dev_predictions = dev_probs.argmax(axis=1)

        dev_f1 = f1_score(
            dev_labels,
            dev_predictions,
            average="weighted",
        )
        dev_precision = precision_score(
            dev_labels,
            dev_predictions,
            average="weighted",
            zero_division=0,
        )
        dev_accuracy = accuracy_score(
            dev_labels,
            dev_predictions,
        )
        dev_loss = float(
            -np.log(
                dev_probs[
                    np.arange(len(dev_labels)),
                    dev_labels,
                ] + 1e-12
            ).mean()
        )

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(dev_loss))
        history["val_acc"].append(float(dev_accuracy))
        history["val_prec"].append(float(dev_precision))
        history["val_wf1"].append(float(dev_f1))

        print(
            f"Epoca {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"dev loss {dev_loss:.4f} | "
            f"dev accuracy {dev_accuracy:.4f} | "
            f"dev precision {dev_precision:.4f} | "
            f"dev weighted-F1 {dev_f1:.4f}"
        )

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "best_dev_weighted_f1": best_dev_f1,
                    "config": vars(args),
                },
                checkpoint_path,
            )
            print(f"Checkpoint salvat: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping.")
                break

    # Încarcă cel mai bun checkpoint, nu ultimul epoch.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_probs, test_labels = evaluate(
        model, test_loader, device, use_amp
    )

    # Confirmă alinierea ordinii exemplelor cu split-ul original.
    expected_labels = data["test"]["label"].to_numpy()
    if not np.array_equal(test_labels, expected_labels):
        raise AssertionError(
            "Etichetele testului nu sunt aliniate cu CSV-ul original."
        )

    final_config = {
        **vars(args),
        "best_dev_weighted_f1": float(best_dev_f1),
        "classes": classes,
    }

    save_report(
        args.out_dir,
        test_labels,
        test_probs,
        classes,
        history,
        data["test"],
        final_config,
        title=f"({args.model})",
    )

    print(f"\nToate rezultatele au fost salvate în: {args.out_dir}")


if __name__ == "__main__":
    main()
