from torch.utils.data import Dataset
import spacy
import pandas as pd
import torch

class SADataset(Dataset):
    def __init__(self, data_file, max_words):
        self.embedder = spacy.load('en_core_web_md')
        self.max_words = max_words
        self.df = pd.read_csv(data_file)
        self.df = self.df.sample(4096).reset_index()
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df["text"][idx]
        label = self.df["target"][idx]
        embedding = torch.tensor([tok.vector for tok in self.embedder(text)])
        if self.max_words > embedding.shape[0]:
            end_seq = torch.full((self.max_words - embedding.shape[0],embedding.shape[1]),0)
            embedding = torch.cat((embedding, end_seq))

        if label == 4:
            label = 1
        return embedding, label
    
        