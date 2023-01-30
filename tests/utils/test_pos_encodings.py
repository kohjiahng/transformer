import pytest
import torch
from transformer.utils.pos_encodings import add_sin_positional_encodings 
import math
def test_encoding():
    inp = torch.rand((3, 10, 10), dtype=torch.float32)
    out = add_sin_positional_encodings(inp)

    assert out.shape == torch.Size([3, 10, 10])
    
    d = inp.shape[2]
    for pos in range(inp.shape[1]):
        for i in range(inp.shape[2]):
            if i % 2 == 0:
                pe = math.sin(pos*(10000**(-i/d)))
            else:
                pe = math.cos(pos*(10000**(-(i-1)/d)))
            for n in range(inp.shape[0]):
                assert out[n][pos][i].item() == pytest.approx(inp[n][pos][i].item() + pe)
