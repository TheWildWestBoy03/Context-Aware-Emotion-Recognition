import os
import re
import urllib.request
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# Official MELD emotion label indexing
LABEL2IDX: Dict[str, int] = {
    "neutral": 0,
    "surprise": 1,
    "fear": 2,
    "sadness": 3,
    "joy": 4,
    "disgust": 5,
    "anger": 6,
}
IDX2LABEL: Dict[int, str] = {idx: label for label, idx in LABEL2IDX.items()}
EMOTION_LABELS: List[str] = [IDX2LABEL[i] for i in range(len(LABEL2IDX))]

EXPECTED_SPLIT_SIZES: Dict[str, int] = {
    "train": 9989,
    "dev": 1109,
    "test": 2610,
}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

MELD_RAW_URLS: Dict[str, str] = {
    "train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
    "dev": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
    "test": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
}


def ensure_meld_csvs(data_dir: str = "data") -> Dict[str, str]:
    """Ensure train, dev, and test MELD CSV files exist locally."""
    os.makedirs(data_dir, exist_ok=True)
    paths = {
        "train": os.path.join(data_dir, "train_sent_emo.csv"),
        "dev": os.path.join(data_dir, "dev_sent_emo.csv"),
        "test": os.path.join(data_dir, "test_sent_emo.csv"),
    }
    for split, path in paths.items():
        if not os.path.exists(path):
            print(f"Downloading MELD {split} split to {path}...")
            urllib.request.urlretrieve(MELD_RAW_URLS[split], path)
    return paths


def clean_and_tokenize(text: str) -> List[str]:
    """Normalize Windows-1252/Unicode punctuation and tokenize an utterance."""
    if not isinstance(text, str):
        return []
    text = (
        text.replace("\x92", "'")
        .replace("\x91", "'")
        .replace("\x93", '"')
        .replace("\x94", '"')
        .replace("\x85", " ... ")
        .replace("\x96", " - ")
        .replace("\x97", " - ")
        .replace("\xa0", " ")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("...", " ")
        .lower()
        .strip()
    )
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[!?.,;:\-]+", text)


def load_meld_split(
    csv_path: str, split_name: Optional[str] = None
) -> Tuple[List[List[str]], List[int]]:
    """Load utterances and integer emotion labels from an official MELD CSV."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin1")

    required_cols = {"Utterance", "Emotion"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required MELD columns in {csv_path}: {missing}")

    tokenized_utterances: List[List[str]] = []
    labels: List[int] = []

    for _, row in df.iterrows():
        emotion = str(row["Emotion"]).strip().lower()
        if emotion not in LABEL2IDX:
            raise ValueError(f"Unexpected emotion label '{emotion}' in {csv_path}")
        tokens = clean_and_tokenize(str(row["Utterance"]))
        tokenized_utterances.append(tokens)
        labels.append(LABEL2IDX[emotion])

    if split_name in EXPECTED_SPLIT_SIZES:
        expected = EXPECTED_SPLIT_SIZES[split_name]
        if len(labels) != expected:
            print(
                f"Warning: {split_name} split has {len(labels)} utterances "
                f"(expected {expected})."
            )

    return tokenized_utterances, labels


def build_vocabulary(
    tokenized_texts: List[List[str]], min_freq: int = 1
) -> Dict[str, int]:
    """Build word-to-index vocabulary strictly from the training split."""
    counter = Counter(token for tokens in tokenized_texts for token in tokens)
    vocab: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, freq in counter.most_common():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def compute_smoothed_class_weights(
    labels: List[int], num_classes: int = 7, power: float = 0.5
) -> torch.Tensor:
    """
    Compute normalized inverse-frequency class weights from training labels:
        w_c = (N / (K * count_c)) ** power
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = float(len(labels))
    raw_weights = (total / (num_classes * np.maximum(counts, 1.0))) ** power
    normalized_weights = raw_weights / raw_weights.mean()
    return torch.tensor(normalized_weights, dtype=torch.float32)


def load_glove_embeddings(
    vocab: Dict[str, int],
    embed_dim: int = 300,
    glove_txt_path: Optional[str] = None,
    seed: int = 42,
) -> torch.Tensor:
    """
    Build an embedding matrix aligned with `vocab` using pretrained GloVe vectors.
    Checks `glove_txt_path` first, then `gensim.downloader` ('glove-wiki-gigaword-300'),
    falling back to random normal initialization if offline.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=0.0, scale=0.1, size=(len(vocab), embed_dim)).astype(
        np.float32
    )
    matrix[PAD_IDX] = np.zeros(embed_dim, dtype=np.float32)
    matched = 0

    if glove_txt_path and os.path.exists(glove_txt_path):
        print(f"Loading GloVe embeddings from {glove_txt_path}...")
        with open(glove_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                if word in vocab and len(parts) - 1 == embed_dim:
                    matrix[vocab[word]] = np.asarray(parts[1:], dtype=np.float32)
                    matched += 1
    else:
        try:
            import gensim.downloader as api

            model_name = f"glove-wiki-gigaword-{embed_dim}"
            print(f"Loading pretrained embeddings via gensim ({model_name})...")
            wv = api.load(model_name)
            for word, idx in vocab.items():
                if word in wv:
                    matrix[idx] = wv[word]
                    matched += 1
        except Exception as exc:
            print(
                f"Warning: Could not load GloVe embeddings ({exc}). "
                "Using random initialization."
            )

    print(f"Matched {matched}/{len(vocab)} vocabulary tokens in GloVe.")
    return torch.tensor(matrix, dtype=torch.float32)


class MELDSequenceDataset(Dataset):
    """PyTorch Dataset converting tokenized MELD utterances into token ID sequences."""

    def __init__(
        self,
        tokenized_texts: List[List[str]],
        labels: List[int],
        vocab: Dict[str, int],
        max_len: int = 80,
    ) -> None:
        self.sequences: List[torch.Tensor] = []
        self.labels: List[int] = labels

        for tokens in tokenized_texts:
            truncated = tokens[:max_len]
            if not truncated:
                ids = [UNK_IDX]
            else:
                ids = [vocab.get(tok, UNK_IDX) for tok in truncated]
            self.sequences.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.sequences[idx], self.labels[idx]


def collate_utterances(
    batch: List[Tuple[torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamically pad variable-length sequences to the longest utterance in the batch."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return padded_seqs, lengths, label_tensor
