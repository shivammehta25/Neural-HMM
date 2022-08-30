import pytest
import torch

from src.model.Encoder import Encoder


@pytest.mark.parametrize("state_per_phone", [1, 2, 3])
def test_Encoder_forward(
    hparams,
    dummy_data,
    state_per_phone,
):
    hparams.state_per_phone = state_per_phone
    encoder = Encoder(hparams)
    text_padded, input_lengths, _, _, _ = dummy_data
    emb = torch.nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim)(text_padded)
    encoder_outputs, text_lengths_post_enc = encoder(emb.transpose(1, 2), input_lengths)
    assert encoder_outputs.shape[1] == (emb.shape[1] * state_per_phone)
    assert (text_lengths_post_enc == (input_lengths * state_per_phone)).all()
