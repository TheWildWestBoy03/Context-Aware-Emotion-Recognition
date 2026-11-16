"""MELD data pipeline for context-aware DistilBERT emotion classification.

Each example is one target utterance plus up to `context_size` previous utterances
from the same dialogue, optionally prefixed with the speaker name:

    [CLS] ctx_-k [SEP] ... [SEP] ctx_-1 [SEP] target [SEP]
    e.g. with speakers: [CLS] monica : what ? [SEP] ross : i'm fine . [SEP]

The target is always the last segment and is never truncated; when the input is
longer than `max_length`, the oldest context utterances are dropped first.
"""
import os
import urllib.request

import pandas as pd
import torch
from torch.utils.data import Dataset

# Same label order as TextCNNExperiment, so results are directly comparable
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL2ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}

MELD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "MELD.Raw")
MELD_RAW_URLS = {
    "train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
    "dev": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
    "test": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
}
EXPECTED_SPLIT_SIZES = {"train": 9989, "dev": 1109, "test": 2610}

# Leftover Windows-1252 punctuation in the MELD CSVs (e.g. \x92 instead of ')
CP1252_FIX = str.maketrans({
    "\x91": "'", "\x92": "'", "‘": "'", "’": "'",
    "\x93": '"', "\x94": '"', "“": '"', "”": '"',
    "\x85": "...", "\x96": "-", "\x97": "-", "\xa0": " ",
})


def ensure_meld_csvs(data_dir=MELD_DIR):
    """Return paths to the train/dev/test MELD CSVs, downloading any that are missing."""
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for split, url in MELD_RAW_URLS.items():
        path = os.path.join(data_dir, f"{split}_sent_emo.csv")
        if not os.path.exists(path):
            print(f"Downloading MELD {split} split to {path}...")
            urllib.request.urlretrieve(url, path)
        paths[split] = path
    return paths


def clean_text(text):
    """Fix encoding artefacts and whitespace; casing/punctuation are left to the tokenizer."""
    return " ".join(str(text).translate(CP1252_FIX).split())


def load_split(path):
    """Load one MELD split as a DataFrame sorted by dialogue and utterance order."""
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns={
        "Dialogue_ID": "dialogue_id",
        "Utterance_ID": "utterance_id",
        "Speaker": "speaker",
        "Utterance": "utterance",
        "Emotion": "emotion",
    })[["dialogue_id", "utterance_id", "speaker", "utterance", "emotion"]]
    df["utterance"] = df["utterance"].map(clean_text)
    df["speaker"] = df["speaker"].map(clean_text)
    df["label"] = df["emotion"].map(LABEL2ID)
    if df["label"].isna().any():
        raise ValueError(f"Unknown emotion labels in {path}: {set(df.loc[df.label.isna(), 'emotion'])}")
    return df.sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)


def load_meld(data_dir=MELD_DIR):
    splits = {split: load_split(path) for split, path in ensure_meld_csvs(data_dir).items()}
    for split, df in splits.items():
        if len(df) != EXPECTED_SPLIT_SIZES[split]:
            print(f"Warning: {split} has {len(df)} utterances, expected {EXPECTED_SPLIT_SIZES[split]}")
    return splits


def max_context_size(train_df):
    """Largest usable context: no training utterance has more previous utterances than this."""
    return int(train_df.groupby("dialogue_id").size().max()) - 1


def validate_context_size(context_size, train_df):
    limit = max_context_size(train_df)
    if not isinstance(context_size, int) or isinstance(context_size, bool):
        raise TypeError(f"context_size must be an int, got {type(context_size).__name__}")
    if context_size < 0:
        raise ValueError(f"context_size must be >= 0 (number of previous utterances), got {context_size}")
    if context_size > limit:
        raise ValueError(
            f"context_size={context_size} is too large: the longest training dialogue has "
            f"{limit + 1} utterances, so at most {limit} previous utterances can ever be used. "
            f"Choose a value between 0 and {limit}."
        )


def format_utterance(speaker, utterance, use_speakers):
    return f"{speaker}: {utterance}" if use_speakers else utterance


def build_examples(df, context_size, use_speakers):
    """One example per utterance: (list of context strings oldest->newest, target string, label)."""
    examples = []
    for _, dialogue in df.groupby("dialogue_id", sort=False):
        texts = [
            format_utterance(row.speaker, row.utterance, use_speakers)
            for row in dialogue.itertuples()
        ]
        for position, row in enumerate(dialogue.itertuples()):
            start = max(0, position - context_size)
            examples.append({
                "context": texts[start:position],
                "target": texts[position],
                "label": int(row.label),
                "dialogue_id": int(row.dialogue_id),
                "utterance_id": int(row.utterance_id),
            })
    return examples


class MELDContextDataset(Dataset):
    """Tokenised context + target examples for one split."""

    def __init__(self, df, tokenizer, context_size, use_speakers, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []
        self.truncated_examples = 0
        self.dropped_context_utterances = 0

        cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
        for example in build_examples(df, context_size, use_speakers):
            target_ids = tokenizer.encode(example["target"], add_special_tokens=False)
            # [CLS] target [SEP] must always fit
            target_ids = target_ids[: max_length - 2]
            input_ids = target_ids + [sep_id]

            # Prepend context from the most recent utterance backwards until the budget is used
            kept = 0
            for context_text in reversed(example["context"]):
                context_ids = tokenizer.encode(context_text, add_special_tokens=False) + [sep_id]
                if 1 + len(context_ids) + len(input_ids) > max_length:
                    break
                input_ids = context_ids + input_ids
                kept += 1
            dropped = len(example["context"]) - kept
            if dropped:
                self.truncated_examples += 1
                self.dropped_context_utterances += dropped

            self.items.append({
                "input_ids": [cls_id] + input_ids,
                "label": example["label"],
                "num_context": kept,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def labels(self):
        return [item["label"] for item in self.items]

    def collate(self, batch):
        """Dynamic padding to the longest sequence in the batch."""
        max_len = max(len(item["input_ids"]) for item in batch)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, item in enumerate(batch):
            ids = item["input_ids"]
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return input_ids, attention_mask, labels

    def describe(self, name):
        lengths = [len(item["input_ids"]) for item in self.items]
        mean_ctx = sum(item["num_context"] for item in self.items) / len(self.items)
        print(
            f"{name}: {len(self.items)} examples | tokens mean {sum(lengths) / len(lengths):.1f}, "
            f"max {max(lengths)} | context utterances used mean {mean_ctx:.2f} | "
            f"truncated examples {self.truncated_examples} "
            f"({self.dropped_context_utterances} context utterances dropped)"
        )
