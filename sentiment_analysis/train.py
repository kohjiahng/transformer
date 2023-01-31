from model import SentimentModel
from dataset import SADataset
import torch
from torch.utils.data import DataLoader
from torch import nn
import gc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
gc.collect()

dataset = SADataset('./sentiment_analysis/data/data.csv', max_words=512)
dataloader = DataLoader(dataset, batch_size = 32, shuffle=True)

model = SentimentModel(
    n_encoders=6,
    n_heads=8,
    embed_dim=300,
    key_dim=300,
    value_dim=100,
    output_dim=2
)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

print(model.encoder[0].ff1.weight.device)
print(model.encoder[0].mha.mix_layer.weight.device)
print(model.encoder[0].mha.heads[0].query.weight.device)
print("Starting Training...")
for epoch in range(10):
    total_loss = 0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 1 == 0:
            print(f'Loss in epoch {epoch} and batch {i}: {total_loss/((i+1)*dataloader.batch_size)}')
        
        del inputs, labels, outputs, loss
    print(total_loss/len(dataset))