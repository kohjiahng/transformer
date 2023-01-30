from dataset import SADataset
from torch.utils.data import DataLoader

dataset = SADataset('./sentiment_analysis/data/data.csv')
dataloader = DataLoader(dataset, batch_size = 64, shuffle=True)

