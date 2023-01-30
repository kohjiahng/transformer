import torch
from torch import nn
from transformer.utils import MultiHeadAttentionModule
class SentimentModel(nn.Module):
    def __init__(self, n_encoders, n_heads, embed_dim, key_dim, value_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(*[
            MultiHeadAttentionModule(
                n_heads,
                embed_dim,
                key_dim,
                value_dim,
                output_dim=embed_dim
            )
            for _ in range(n_encoders)    
        ])
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(output_dim)
        self.softmax = nn.Softmax(-1)
    def forward(self, X):
        encoding = self.encoder(X)
        logits = self.linear(self.flatten(encoding))
        probs = self.softmax(logits)

        return probs

