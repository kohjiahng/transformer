import pytest
import torch
from transformer.utils import MultiHeadAttentionModule


@pytest.fixture
def module():
    return MultiHeadAttentionModule(
        n_heads=3,
        embed_dim=10,
        key_dim=12,
        value_dim=6,
        output_dim=5
    )


def test_dim(module):
    inp = torch.rand((3, 9, 10))
    out = module(inp)

    assert out.shape == torch.Size([3, 9, 5])

def test_ndim_exception(module):
    inp = torch.rand((10,10))
    with pytest.raises(Exception) as ecp:
        out = module(inp)
    assert str(ecp.value) == "Wrong number of dimensions passed into MultiheadAttentionModule.forward: Expected 3, got 2"

def test_dim_exception(module):
    inp = torch.rand((3, 10, 4))
    with pytest.raises(Exception) as ecp:
        out = module(inp)
    assert str(ecp.value) == "Wrong embed dimension passed into MultiheadAttentionModule.forward: Expected 10, got 4"
