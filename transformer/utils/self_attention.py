import torch
from torch import nn
from math import sqrt
class SelfAttentionModule(nn.Module):
    '''
    NxMxembed_dim -> NxMxvalue_dim
    N: batch size
    M: Number of words
    '''
    def __init__(self, embed_dim, key_dim, value_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        self.query = nn.Linear(self.embed_dim, self.key_dim)
        self.key = nn.Linear(self.embed_dim, self.key_dim)
        self.value = nn.Linear(self.embed_dim, self.value_dim)

        self.softmax = nn.Softmax(dim = 2)
    def forward(self, X):
        if X.ndim != 3:
            raise Exception(f"Wrong number of dimensions passed into SelfAttentionModule.forward: Expected 3, got {X.ndim}")

        if X.shape[-1] != self.embed_dim:
            raise Exception(f"Wrong embed dimension passed into SelfAttentionModule.forward: Expected {self.embed_dim}, got {X.shape[-1]}")

        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        A = Q @ K.transpose(1,2) / sqrt(self.embed_dim) # NxMxM
        A = self.softmax(A)

        result = A @ V

        return result

