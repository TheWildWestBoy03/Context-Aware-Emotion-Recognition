from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for isolated utterance emotion classification on MELD."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.3,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.embed_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Final forward/backward hidden states (2 * hidden_dim)
        # + masked mean-pooled sequence outputs (2 * hidden_dim)
        repr_dim = hidden_dim * 4
        self.norm = nn.LayerNorm(repr_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, input_ids: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.embed_dropout(self.embedding(input_ids))

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, (h_n, _) = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_ids.size(1),
        )

        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)

        mask = (input_ids != self.pad_idx).unsqueeze(-1).float()
        summed = (output * mask).sum(dim=1)
        valid_counts = mask.sum(dim=1).clamp(min=1.0)
        mean_pooled = summed / valid_counts

        utterance_repr = self.norm(torch.cat([last_hidden, mean_pooled], dim=1))
        return self.classifier(utterance_repr)
