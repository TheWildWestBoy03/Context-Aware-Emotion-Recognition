from data_preprocessing import Preprocessor
from dataset import MELDDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import nltk
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        num_classes=7,
        num_filters=100,
        dialogue_filters=128,
        kernel_sizes=(2, 3, 4),
        padding_idx=0,
        dropout=0.5
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        # Word-level convolutions
        self.word_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k
            )
            for k in kernel_sizes
        ])

        utterance_dim = num_filters * len(kernel_sizes)

        self.dialogue_conv = nn.Conv1d(
            in_channels=utterance_dim,
            out_channels=dialogue_filters,
            kernel_size=3,
            padding=1
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dialogue_filters, num_classes)

    def forward(self, x, utterance_mask):
        # x: (B, U, L)
        # utterance_mask: (B, U)

        B, U, L = x.shape

        # --------------------------------------------------
        # 1. Embedding
        # --------------------------------------------------

        # (B, U, L) -> (B, U, L, D)
        embedded = self.embedding(x)

        # Treat each utterance as an independent sequence
        # (B, U, L, D) -> (B*U, D, L)
        embedded = embedded.reshape(B * U, L, -1)
        embedded = embedded.transpose(1, 2)

        # Number of real tokens in each utterance
        lengths = x.ne(0).sum(dim=2).reshape(B * U)

        # Ensure sequence length is at least the largest kernel
        max_kernel = max(
            conv.kernel_size[0]
            for conv in self.word_convs
        )

        if embedded.size(2) < max_kernel:
            embedded = F.pad(
                embedded,
                (0, max_kernel - embedded.size(2))
            )

        # --------------------------------------------------
        # 2. Utterance-level CNN
        # --------------------------------------------------

        utterance_features = []

        for conv in self.word_convs:

            features = F.relu(conv(embedded))

            kernel_size = conv.kernel_size[0]

            # Number of valid convolution windows
            valid_windows = (
                lengths - kernel_size + 1
            ).clamp(min=1)

            positions = torch.arange(
                features.size(2),
                device=x.device
            ).unsqueeze(0)

            valid_mask = positions < valid_windows.unsqueeze(1)

            # Exclude padded token windows from max pooling
            features = features.masked_fill(
                ~valid_mask.unsqueeze(1),
                float("-inf")
            )

            # Max pooling over the token dimension
            pooled = features.max(dim=2).values

            # Empty utterances receive zero representations
            pooled = pooled.masked_fill(
                (lengths == 0).unsqueeze(1),
                0.0
            )

            utterance_features.append(pooled)

        # Concatenate features from all kernel sizes
        # (B*U, num_filters * len(kernel_sizes))
        utterance_features = torch.cat(
            utterance_features,
            dim=1
        )

        # Restore dialogue structure
        # (B*U, F) -> (B, U, F)
        utterance_features = utterance_features.reshape(
            B, U, -1
        )

        # Zero padded utterance representations
        utterance_features = utterance_features.masked_fill(
            ~utterance_mask.unsqueeze(-1),
            0.0
        )

        # --------------------------------------------------
        # 3. Dialogue-level CNN
        # --------------------------------------------------

        # Conv1d expects (B, channels, sequence_length)
        dialogue_features = utterance_features.transpose(1, 2)

        dialogue_features = F.relu(
            self.dialogue_conv(dialogue_features)
        )

        # Restore (B, U, dialogue_filters)
        dialogue_features = dialogue_features.transpose(1, 2)

        # Exclude padded dialogue positions
        dialogue_features = dialogue_features.masked_fill(
            ~utterance_mask.unsqueeze(-1),
            0.0
        )

        # --------------------------------------------------
        # 4. Classification
        # --------------------------------------------------

        dialogue_features = self.dropout(dialogue_features)

        logits = self.classifier(dialogue_features)

        # (B, U, num_classes)
        return logits

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_utterances = 0
    all_predictions = []
    all_targets = []

    for X, y, mask in loader:
        X = X.to(device)
        y = y.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        logits = model(X, mask)

        # Only calculate loss for real utterances
        loss = criterion(logits[mask], y[mask])

        loss.backward()
        optimizer.step()

        num_utterances = mask.sum().item()
        
        total_loss += loss.item() * num_utterances
        total_utterances += num_utterances

        predictions = logits.argmax(dim=-1)

        all_predictions.extend(predictions[mask].cpu().tolist())
        all_targets.extend(y[mask].cpu().tolist())

    avg_loss = total_loss / total_utterances
    accuracy = accuracy_score(all_targets, all_predictions)
    macro_f1 = f1_score(
        all_targets, all_predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        all_targets, all_predictions, average="weighted", zero_division=0
    )

    return avg_loss, accuracy, macro_f1, weighted_f1

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_utterances = 0
    all_predictions = []
    all_targets = []

    for X, y, mask in loader:
        X = X.to(device)
        y = y.to(device)
        mask = mask.to(device)

        logits = model(X, mask)

        loss = criterion(logits[mask], y[mask])
        num_utterances = mask.sum().item()

        total_loss += loss.item() * num_utterances
        total_utterances += num_utterances

        predictions = logits.argmax(dim=-1)

        all_predictions.extend(predictions[mask].cpu().tolist())
        all_targets.extend(y[mask].cpu().tolist())

    avg_loss = total_loss / total_utterances
    accuracy = accuracy_score(all_targets, all_predictions)
    macro_f1 = f1_score(
        all_targets, all_predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        all_targets, all_predictions, average="weighted", zero_division=0
    )

    return avg_loss, accuracy, macro_f1, weighted_f1
    
def download_packages():
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def main():
    # read the dataframe into RAM memory for now, given its low dimensions

    preprocessor = Preprocessor()
    df = preprocessor.start_pipeline()

    # Split complete dialogues
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining emotion distribution:")
    print(train_df["dialogue"].apply(
        lambda dialogue: [
            item["emotion"] for item in dialogue
        ]
    ).explode().value_counts())

    print("\nValidation emotion distribution:")
    print(val_df["dialogue"].apply(
        lambda dialogue: [
            item["emotion"] for item in dialogue
        ]
    ).explode().value_counts())
    # Build vocabulary ONLY from training dialogues
    preprocessor.build_vocab(train_df)

    # Fixed MELD emotion mapping
    label2id = {
        "anger": 0,
        "disgust": 1,
        "fear": 2,
        "joy": 3,
        "neutral": 4,
        "sadness": 5,
        "surprise": 6
    }

    train_dataset = MELDDataset(
        train_df,
        preprocessor.vocab,
        label2id
    )

    val_dataset = MELDDataset(
        val_df,
        preprocessor.vocab,
        label2id
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=MELDDataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=MELDDataset.collate_fn
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    network = TextCNN(
        vocab_size=len(preprocessor.vocab),
        embedding_dim=100,
        num_classes=7,
        num_filters=100,
        dialogue_filters=128,
        dropout=0.5
    ).to(device)


    # Inspect one batch
    X, y, mask = next(iter(train_loader))

    print("Input:", X.shape)
    print("Targets:", y.shape)
    print("Mask:", mask.shape)

    print("Vocabulary size:", len(preprocessor.vocab))

    X = X.to(device)
    mask = mask.to(device)

    logits = network(X, mask)

    print("Logits:", logits.shape)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=0.001
    )

    # Training configuration
    num_epochs = 20
    best_val_f1 = -1.0

    for epoch in range(num_epochs):

        train_loss, train_acc, train_f1, train_weighted_f1 = train_one_epoch(
            network,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, val_acc, val_f1, val_weighted_f1 = evaluate(
            network,
            val_loader,
            criterion,
            device
        )

        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        print(
            f"Train | Loss: {train_loss:.4f} | "
            f"Accuracy: {train_acc:.4f} | "
            f"Macro F1: {train_f1:.4f}"
        )

        print(
            f"Val   | Loss: {val_loss:.4f} | "
            f"Accuracy: {val_acc:.4f} | "
            f"Macro F1: {val_f1:.4f}"
        )

        # Save the model with the best validation macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            torch.save(
                network.state_dict(),
                "best_textcnn.pth"
            )

    print(f"\nBest validation Macro F1: {best_val_f1:.4f}")

    # Load the best saved model
    network.load_state_dict(
        torch.load("best_textcnn.pth", map_location=device)
    )
    network.eval()

    # Collect validation predictions
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for X, y, mask in val_loader:
            X = X.to(device)
            y = y.to(device)
            mask = mask.to(device)

            logits = network(X, mask)
            predictions = logits.argmax(dim=-1)

            all_predictions.extend(predictions[mask].cpu().tolist())
            all_targets.extend(y[mask].cpu().tolist())

    # Print classification report
    print("\nValidation Classification Report:")
    print(classification_report(
        all_targets,
        all_predictions,
        labels=list(range(7)),
        target_names=[
            "anger",
            "disgust",
            "fear",
            "joy",
            "neutral",
            "sadness",
            "surprise"
        ],
        zero_division=0
    ))
    

if __name__ == "__main__":
    main()