import torch
from torch.utils.data import Dataset


class MELDDataset(Dataset):

    def __init__(self, dataframe, vocab, label2id):

        self.dialogues = []

        for dialogue in dataframe["dialogue"]:

            encoded_dialogue = []
            labels = []

            for item in dialogue:

                input_ids = [
                    vocab.get(token, vocab["<UNK>"])
                    for token in item["tokens"]
                ]

                encoded_dialogue.append(input_ids)
                labels.append(label2id[item["emotion"]])

            self.dialogues.append({
                "input_ids": encoded_dialogue,
                "labels": labels
            })

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        max_utterances = max(
            len(item["input_ids"])
            for item in batch
        )

        max_tokens = max(
            1,
            max(
                len(utterance)
                for item in batch
                for utterance in item["input_ids"]
            )
        )

        X = torch.zeros(
            (batch_size, max_utterances, max_tokens),
            dtype=torch.long
        )

        y = torch.zeros(
            (batch_size, max_utterances),
            dtype=torch.long
        )

        mask = torch.zeros(
            (batch_size, max_utterances),
            dtype=torch.bool
        )

        for i, item in enumerate(batch):

            for j, utterance in enumerate(item["input_ids"]):

                X[i, j, :len(utterance)] = torch.tensor(
                    utterance,
                    dtype=torch.long
                )

                y[i, j] = item["labels"][j]

                mask[i, j] = True

        return X, y, mask

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        return self.dialogues[idx]