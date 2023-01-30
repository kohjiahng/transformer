from torch.utils.data import Dataset
import spacy
import pandas as pd

class SADataset(Dataset):
    def __init__(self, data_file):
        self.embedder = spacy.load('en_core_web_md')
        self.df = pd.read_csv(data_file)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df["text"][idx]
        label = self.df["target"][idx]
        embedding = self.embedder(text).vector
        return embedding, label
    
        