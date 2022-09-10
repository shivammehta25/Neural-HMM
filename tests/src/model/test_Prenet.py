import pytest
import torch

from src.model.Prenet import Prenet


@pytest.mark.parametrize(
    "in_dim, n_layers, prenet_dim, prenet_dropout, dropout_flag",
    [(80, 1, 256, 0.0, False), (160, 2, 512, 0.5, True)],
)
def test_Prenet(dummy_data, in_dim, n_layers, prenet_dim, prenet_dropout, dropout_flag):
    model = Prenet(in_dim, n_layers, prenet_dim, prenet_dropout)
    (
        text_padded,
        input_lengths,
        mel_padded,
        gate_padded,
        output_lengths,
    ) = dummy_data

    prenet_output = model(torch.randn(1, in_dim), dropout_flag)

    assert prenet_output.shape == (1, prenet_dim)
