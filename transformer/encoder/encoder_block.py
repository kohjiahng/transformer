import torch
from torch import nn
from transformer.utils import MultiHeadAttentionModule, SelfAttentionModule, add_sin_positional_encodings

class EncoderBlock(nn.Module):
    def __init__(self, nheads, embed_dim, key_dim, value_dim, ff_dim):
        super().__init__()
        self.mha = MultiHeadAttentionModule(nheads, embed_dim, key_dim, value_dim, output_dim=embed_dim)
        self.layer_norm = nn.LayerNorm((embed_dim,))
        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, X):
        attention = self.mha(X)

        attention = self.layer_norm(attention + X) # Skip connection

        ff1 = self.ff1(attention)
        ff1 = self.relu(ff1)
        ff2 = self.ff2(ff1)

        out = self.layer_norm(attention + ff2) # Skip connection
        return out







    
