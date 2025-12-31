import pandas as pd
from torch.utils.data import Dataset
import torch

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Class in order to make a dataloader
class TranslationDataset(Dataset):
    def __init__(self,
        df : pd.DataFrame,
        void_fr : int,
        void_en : int
        ):

        self.df = df.reset_index(drop=True)

        # Maximum length of the dataset
        self.max_len_fr = df['tokens fr'].map(len).max()
        self.max_len_en = df['tokens en'].map(len).max() + 1
        # Adding one to make predictions over the end token

        # Padding indices
        self.void_fr = void_fr
        self.void_en = void_en

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        padsX = self.max_len_fr - len(self.df.loc[idx, 'tokens fr'])
        padsY = self.max_len_en - len(self.df.loc[idx, 'tokens en'])
        return (torch.tensor(self.df.loc[idx, 'tokens fr'] + [self.void_fr] * padsX),
                torch.tensor(self.df.loc[idx, 'tokens en'] + [self.void_en] * padsY))


def load_dataset(dataset_file : str) -> pd.DataFrame:
    df = pd.read_csv(dataset_file, sep='\t', names=('en', 'fr'))

    df['sample fr'] = df.apply(lambda row: f"{row['fr']} eeeee", axis=1)
    df['sample en'] = df.apply(lambda row: f"sssss {row['en']} eeeee", axis=1)

    vec_fr, vec_en = make_vectorizers(df)
    analyzer_fr, analyzer_en = make_analyzers(vec_fr, vec_en)

    df['tokens fr'] = df['sample fr'].map(analyzer_fr)
    df['tokens fr'] = df['tokens fr'].map(lambda x: list(map(vec_fr.vocabulary_.get, x)))

    df['tokens en'] = df['sample en'].map(analyzer_en)
    df['tokens en'] = df['tokens en'].map(lambda x: list(map(vec_en.vocabulary_.get, x)))

    return df



def make_vectorizers(df : pd.DataFrame):
    # Making vocabularies
    vec_en = CountVectorizer(token_pattern=r'\b\w+\b')
    vec_en.fit(df['sample en'])

    vec_fr = CountVectorizer(token_pattern=r'\b\w+\b')
    vec_fr.fit(df['sample fr'])

    # Adding the void token to the output language
    vec_en.vocabulary_['vvvvv'] = len(vec_en.vocabulary_)
    vec_fr.vocabulary_['vvvvv'] = len(vec_fr.vocabulary_)

    return vec_fr, vec_en

def make_analyzers(vec_fr : CountVectorizer, vec_en : CountVectorizer):
    # Build analyzers
    analyzer_en = vec_en.build_analyzer()
    analyzer_fr = vec_fr.build_analyzer()
    return analyzer_fr, analyzer_en

def invert_vocabularies(vec_fr : CountVectorizer, vec_en : CountVectorizer):
    # Useful to convert tokens into strings
    invert_vocabulary_fr = {y: x for x, y in vec_fr.vocabulary_.items()}
    invert_vocabulary_en = {y: x for x, y in vec_en.vocabulary_.items()}
    return invert_vocabulary_fr, invert_vocabulary_en


def get_max_lenghts(df : pd.DataFrame):
    max_len_fr = df['tokens fr'].map(len).max()
    max_len_en = df['tokens en'].map(len).max()
    return max_len_fr, max_len_en

    
