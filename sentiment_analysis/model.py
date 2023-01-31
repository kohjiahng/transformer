import torch
from torch import nn
from transformer import EncoderBlock
class SentimentModel(nn.Module):
    def __init__(self, n_encoders, n_heads, embed_dim, key_dim, value_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(*[
            EncoderBlock(
                n_heads,
                embed_dim,
                key_dim,
                value_dim,
                ff_dim=256
            )
            for _ in range(n_encoders)    
        ])
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(output_dim)
    def forward(self, X):
        encoding = self.encoder(X)
        logits = self.linear(self.flatten(encoding))

        return logits 

