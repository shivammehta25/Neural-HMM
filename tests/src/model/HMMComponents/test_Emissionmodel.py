import pytest
import torch

from src.model.HMMComponents.EmissionModel import EmissionModel


@pytest.mark.parametrize("n_mel_channels", [80, 4])
def test_emissionmodel(test_batch_size, dummy_data, n_mel_channels):
    _, input_lengths, _, _, _ = dummy_data
    model = EmissionModel()
    x_t = torch.randn(test_batch_size, n_mel_channels)
    means = torch.zeros(test_batch_size, input_lengths.max(), n_mel_channels)
    stds = torch.ones(test_batch_size, input_lengths.max(), n_mel_channels)

    out = model(x_t, means, stds, input_lengths)
    assert out.shape == (test_batch_size, input_lengths.max())
