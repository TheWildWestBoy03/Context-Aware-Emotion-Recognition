"""DistilBERT encoder with a classification head on the [CLS] token."""
import torch.nn as nn
from transformers import AutoModel

DEFAULT_MODEL_NAME = "distilbert-base-uncased"


class DistilBertEmotionClassifier(nn.Module):
    """[CLS] hidden state -> Linear -> ReLU -> Dropout -> Linear (7 emotions).

    Same head as transformers' DistilBertForSequenceClassification, written out explicitly.
    """

    def __init__(self, num_classes, model_name=DEFAULT_MODEL_NAME, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.dim
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = hidden[:, 0]
        return self.classifier(self.dropout(self.activation(self.pre_classifier(cls))))
