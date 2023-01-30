import pytest
import torch
from transformer.utils.multihead_attention import MultiHeadAttentionModule


@pytest.fixture
def module():
    return MultiHeadAttentionModule(
        nheads=3,
        embed_dim=10,
        key_dim=12,
        value_dim=6,
        output_dim=5
    )


def test_dim(module):
    inp = torch.rand((3, 10, 10))
    out = module(inp)

    assert out.shape == torch.Size([3, 10, 5])


def test_dim_exception(module):
    inp = torch.rand((3, 10, 4))
    with pytest.raises(Exception):
        out = module(inp)
