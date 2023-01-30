import torch
from torch import nn
class SelfAttentionModule(nn.Module):
    '''
    NxMxd -> NxMxvalue_dim
    N: batch size
    M: Number of words
    d: Embedding dimension
    '''
    def __init__(self, embed_dim, key_dim, value_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        self.query = nn.Linear(self.embed_dim, self.key_dim)
        self.key = nn.Linear(self.embed_dim, self.key_dim)
        self.value = nn.Linear(self.embed_dim, self.value_dim)

    def forward(self, X):
        if X.shape[-1] != self.embed_dim:
            raise Exception(f"Wrong embed dimension passed into AttentionModule.forward: Expected {self.embed_dim}, received {X.shape[-1]}")

        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        A = Q @ K.transpose(1,2) # NxMxM

        result = A @ V

        return result

