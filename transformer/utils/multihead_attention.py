import torch
from torch import nn
from transformer.utils.self_attention import SelfAttentionModule

class MultiHeadAttentionModule(nn.Module):
    '''
    NxMxembed_dim -> NxMxoutput_dim
    N: batch size
    M: Number of words
    '''
    def __init__(self, n_heads, embed_dim, key_dim, value_dim, output_dim):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim

        self.heads = [SelfAttentionModule(embed_dim,key_dim,value_dim) for _ in range(n_heads)]
        
        self.mix_layer = nn.Linear(n_heads*value_dim, output_dim) # To mix the embeddings from each head

    def forward(self, X):
        if X.ndim != 3:
            raise Exception(f"Wrong number of dimensions passed into MultiheadAttentionModule.forward: Expected 3, got {X.ndim}")

        if X.shape[-1] != self.embed_dim:
            raise Exception(f"Wrong embed dimension passed into MultiheadAttentionModule.forward: Expected {self.embed_dim}, got {X.shape[-1]}")

        attention_outputs = [head(X) for head in self.heads]

        embeddings = torch.cat(attention_outputs, dim=-1) # Concatenate the embeddings
        # embeddings is a NxMx(nheads*value_dim) tensor

        mixed_embeddings = self.mix_layer(embeddings)

        return mixed_embeddings

        
