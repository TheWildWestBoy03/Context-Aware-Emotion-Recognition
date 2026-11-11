import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize, RegexpTokenizer
import nltk

PATH = "./meld_full_13708_utterances.xlsx"

class Preprocessor:
    def __init__(self):
        self.dataframe = pd.DataFrame({})
        self.target = []
        self.raw_text = []
        self.cleaned_text = []
        self.tokens = []
        self.tokenizer = RegexpTokenizer(r"[a-z]+(?:'[a-z]+)*")

    def embed_text():
        pass

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

    def start_pipeline(self):
        self.read_data()
        self.clean()
        self.preprocess()

        print(self.dataframe['dialogue'][0])


def main():
    # read the dataframe into RAM memory for now, given its low dimensions

    preprocessor = Preprocessor()
    # nltk.download('punkt_tab')
    # nltk.download('punkt')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')

    preprocessor.start_pipeline()
    pass

if __name__ == "__main__":
    main()