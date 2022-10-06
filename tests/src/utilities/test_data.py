import pytest
import torch

from src.utilities.data import Normalise, TextMelCollate


def test_collate_function(dummy_data_uncollated, hparams):
    text_padded, input_lengths, mel_padded, _, output_lengths = TextMelCollate(hparams.n_frames_per_step)(
        dummy_data_uncollated
    )
    assert text_padded.shape[1] == torch.max(input_lengths).item()
    assert mel_padded.shape[2] == torch.max(output_lengths).item()


@pytest.mark.parametrize("mean, std", [(2, 0.5), (0, 1.0), (None, None), (torch.randn(80), torch.rand(80))])
def test_Normalisation(dummy_data, hparams, mean, std):
    if mean is None and std is None:
        normaliser = hparams.normaliser
    else:
        normaliser = Normalise(mean, std)

    _, _, mel_padded, _, _ = dummy_data
    mel_padded.transpose_(1, 2)

    normalised = normaliser(mel_padded)
    inverted = normaliser.inverse_normalise(normalised)

    assert torch.allclose(mel_padded, inverted, atol=1e-5)
