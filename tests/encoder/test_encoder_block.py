import pytest
import torch
from transformer.encoder.encoder_block import EncoderBlock

@pytest.fixture
def encoderblock():
    return EncoderBlock(
        nheads=6,
        embed_dim=10,
        key_dim=5,
        value_dim=6,
        ff_dim=20
    )

def test_encoder_block(encoderblock):
    inp = torch.rand((3,5,10))
    out = encoderblock(inp)
    assert out.shape == torch.Size([3,5,10])

