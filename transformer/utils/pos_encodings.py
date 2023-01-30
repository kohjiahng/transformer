import torch
import math

def add_sin_positional_encodings(X):
    '''
    X is a NxMxd tensor

    Encodings are done using the formula (taken from paper): 

    PE(pos, 2i) = sin(pos*10000^(-2i/d)))
    PE(pos, 2i+1) = cos(pos*10000^(-2i/d)))

    where d is the embed dimension

    This function returns a tensor Y of shape NxMxd such that

    Y_ijk = X_ijk + PE(j,k)
    '''

    d = X.shape[-1]

    wavelengths = torch.pow(10000, -torch.arange(0,d,step=2) / d).unsqueeze(0) # [1,d/2]

    pos = torch.arange(0,X.shape[1],dtype=torch.float32).unsqueeze(-1) # [X.shape[1], 1]
    
    inputs = pos @ wavelengths

    even = torch.sin(inputs)
    odd = torch.cos(inputs)


    PE = torch.zeros([X.shape[1], X.shape[2]])

    PE[:, ::2] = even
    PE[:, 1::2] = odd
   
    return X + PE



