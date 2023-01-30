import pytest
import torch
from transformer.attention import SelfAttentionModule

@pytest.fixture
def module():
    return SelfAttentionModule(10,10,5)
def test_dim(module):
    inp = torch.rand((3, 10, 10))
    out = module(inp)

    assert out.shape == torch.Size([3, 10, 5])

def test_dim_exception(module):
    inp = torch.rand((3,10,4))
    with pytest.raises(Exception):
        out = module(inp)