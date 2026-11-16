import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize, RegexpTokenizer
import nltk
from collections import Counter

pd.set_option('display.max_colwidth', None)
PATH = "./meld_full_13708_utterances.xlsx"

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
                    "utterance": item["utterance"].lower()
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

    def read_data(self):
        self.dataframe = pd.read_excel(PATH, sheet_name="meld_full_13708_utterances")
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

    def start_pipeline(self):
        self.read_data()
        print(self.dataframe.head())
        self.clean()
        self.preprocess()

        return self.dataframe
