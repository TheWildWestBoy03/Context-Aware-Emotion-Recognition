"""Pretrained GloVe embeddings aligned with the pipeline vocabulary."""
import os
import re

import numpy as np
import torch

# GloVe 6B (Wikipedia + Gigaword, lowercase), 100d vectors
GLOVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "embeddings", "glove.6B.100d.txt"
)

# GloVe splits contractions the way the Stanford tokenizer does ("that's" -> "that" "'s",
# "don't" -> "do" "n't"); our tokenizer keeps them as one token.
CONTRACTION = re.compile(r"^(.+?)(n't|'[a-z]+)$")


def _load_glove(path, wanted):
    """Read only the GloVe vectors for tokens in `wanted`."""
    vectors = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, rest = line.split(" ", 1)
            if word in wanted:
                vectors[word] = np.asarray(rest.split(), dtype=np.float32)
    return vectors


def _contraction_parts(token):
    match = CONTRACTION.match(token)
    return list(match.groups()) if match else []


def build_embedding_matrix(vocab, path=GLOVE_PATH, embedding_dim=100, seed=42):
    """Return a (len(vocab), embedding_dim) tensor initialised from GloVe.

    - <PAD> stays zero.
    - Tokens found in GloVe get their vector.
    - Contractions missing from GloVe ("that's") get the mean of their parts ("that", "'s").
    - Everything else (incl. <UNK>) gets a small random vector with GloVe's scale.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GloVe file not found at {path}. Download glove.6B.zip from "
            "https://nlp.stanford.edu/projects/glove/ and extract glove.6B.100d.txt there."
        )

    wanted = set(vocab)
    for token in vocab:
        wanted.update(_contraction_parts(token))
    glove = _load_glove(path, wanted)

    std = float(np.std(np.stack(list(glove.values())))) if glove else 0.1
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, std, size=(len(vocab), embedding_dim)).astype(np.float32)

    exact, from_parts = 0, 0
    for token, idx in vocab.items():
        if token == "<PAD>":
            matrix[idx] = 0.0
        elif token in glove:
            matrix[idx] = glove[token]
            exact += 1
        else:
            parts = [glove[p] for p in _contraction_parts(token) if p in glove]
            if parts:
                matrix[idx] = np.mean(parts, axis=0)
                from_parts += 1

    real_tokens = len(vocab) - 2  # minus <PAD>, <UNK>
    print(
        f"GloVe coverage: {exact}/{real_tokens} exact, {from_parts} from contraction parts, "
        f"{real_tokens - exact - from_parts} random ({(exact + from_parts) / real_tokens:.1%} covered)"
    )
    return torch.from_numpy(matrix)
