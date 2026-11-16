import os
import urllib.request
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize, RegexpTokenizer
import nltk
from collections import Counter

pd.set_option('display.max_colwidth', None)
PATH = "./meld_full_13708_utterances.xlsx"

# Directory holding the official MELD CSVs (train/dev/test_sent_emo.csv)
MELD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "MELD.Raw")

MELD_RAW_URLS = {
    "train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
    "dev": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
    "test": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
}

# MELD CSV column names -> names used by this pipeline
MELD_COLUMNS = {
    "Dialogue_ID": "dialogue_id",
    "Utterance_ID": "utterance_id",
    "Speaker": "speaker",
    "Utterance": "utterance",
    "Emotion": "emotion",
}

# Leftover Windows-1252 punctuation in the MELD CSVs (e.g. \x92 instead of ')
CP1252_FIX = str.maketrans({
    "\x91": "'", "\x92": "'", "‘": "'", "’": "'",
    "\x93": '"', "\x94": '"', "“": '"', "”": '"',
    "\x85": " ", "\x96": " ", "\x97": " ", "\xa0": " ",
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

class Preprocessor:
    def __init__(self):
        self.dataframe = pd.DataFrame({})
        self.target = []
        self.raw_text = []
        self.cleaned_text = []
        self.tokens = []
        self.tokenizer = RegexpTokenizer(r"[a-z]+(?:'[a-z]+)*")

        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1
        }

    def build_vocab(self, dataframe, min_freq=1):
        counter = Counter()

        for dialogue in dataframe["dialogue"]:
            for item in dialogue:
                counter.update(item["tokens"])

        sorted_tokens = sorted(
            token
            for token, frequency in counter.items()
            if frequency >= min_freq
        )

        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        for token in sorted_tokens:
            self.vocab[token] = len(self.vocab)


    def build_label_mapping(self, dataframe):
        labels = sorted({
            item["emotion"]
            for dialogue in dataframe["dialogue"]
            for item in dialogue
        })

        self.label2id = {
            label: idx
            for idx, label in enumerate(labels)
        }

    def clean(self):
        self.dataframe["dialogue"] = self.dataframe["dialogue"].apply(
            lambda dialogue: [
                {
                    **item,
                    "utterance": item["utterance"].translate(CP1252_FIX).lower()
                }
                for item in dialogue
            ]
        )

        self.dataframe["dialogue"] = self.dataframe["dialogue"].apply(
            lambda dialogue: [
                {
                    **item,
                    "utterance": re.sub(r"[^a-zA-Z']+", " ", item["utterance"])
                }
                for item in dialogue
            ]
        )
        
    def preprocess(self):
        self.dataframe["dialogue"] = self.dataframe["dialogue"].apply(
            lambda dialogue: [
                {
                    **item,
                    "tokens": self.tokenizer.tokenize(item["utterance"])
                }
                for item in dialogue
            ]
        )

    @staticmethod
    def _load_raw(path):
        if path.endswith(".csv"):
            df = pd.read_csv(path, encoding="utf-8")
            return df.rename(columns=MELD_COLUMNS)[list(MELD_COLUMNS.values())]
        return pd.read_excel(path, sheet_name="meld_full_13708_utterances")

    def read_data(self, path=PATH):
        self.dataframe = self._load_raw(path)
        self.dataframe.sort_values(
            ["dialogue_id", "utterance_id"],
            inplace=True
        )
        self.dataframe.drop(["utterance_id", "speaker"], axis=1, inplace=True)

        self.dataframe = (
            self.dataframe
            .groupby('dialogue_id', sort=False)
            .agg({
                'utterance': lambda x: list(x),
                'emotion': lambda x: list(x),
            })
            .reset_index()
        )

        self.dataframe['dialogue'] = self.dataframe.apply(
            lambda row: [
                {'utterance': utterance, 'emotion': emotion, 'tokens': []}
                for utterance, emotion in zip(row['utterance'], row['emotion'])
            ],
            axis=1
        )

        self.dataframe = self.dataframe[['dialogue_id', 'dialogue']]

    def encode_dialogues(self):
            self.dataframe["dialogue"] = self.dataframe["dialogue"].apply(
                lambda dialogue: [
                    {
                        **item,
                        "input_ids": [
                            self.vocab.get(token, self.vocab["<UNK>"])
                            for token in item["tokens"]
                        ]
                    }
                    for item in dialogue
                ]
            )

    def start_pipeline(self, path=PATH):
        self.read_data(path)
        print(self.dataframe.head())
        self.clean()
        self.preprocess()

        return self.dataframe
