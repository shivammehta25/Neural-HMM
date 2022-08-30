import pytest
import torch

from src.hparams import create_hparams
from src.utilities.data import TextMelCollate
from tests.test_utilities import get_a_text_mel_pair


@pytest.fixture
def hparams():
    hparams = create_hparams()
    hparams.checkpoint_path = None
    return hparams


@pytest.fixture
def test_batch_size():
    return 3


@pytest.fixture
def dummy_data_uncollated(test_batch_size):
    return [get_a_text_mel_pair() for _ in range(test_batch_size)]


@pytest.fixture
def dummy_data(dummy_data_uncollated, hparams):
    (text_padded, input_lengths, mel_padded, gate_padded, output_lengths,) = TextMelCollate(
        hparams.n_frames_per_step
    )(dummy_data_uncollated)
    return text_padded, input_lengths, mel_padded, gate_padded, output_lengths


@pytest.fixture
def dummy_embedded_data(dummy_data, hparams):
    text_padded, input_lengths, mel_padded, gate_padded, output_lengths = dummy_data
    embedded_input = torch.nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim)(text_padded)
    return (embedded_input, input_lengths, mel_padded, gate_padded, output_lengths)
