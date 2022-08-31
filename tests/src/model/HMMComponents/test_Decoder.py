import pytest
import torch

from src.model.HMMComponents.Decoder import Decoder, ParameterModel
from src.utilities.functions import inverse_sigmod, inverse_softplus


@pytest.mark.parametrize(
    "parameternetwork, input_size, output_size, step_size, \
    init_transition_probability, init_mean, init_std",
    [([256, 256], 162, 81, 40, 0.125, 2, 5), ([125, 125], 20, 9, 4, 0.256, 0, 1)],
)
def test_parameter_model_flat_start(
    test_batch_size,
    parameternetwork,
    input_size,
    output_size,
    step_size,
    init_transition_probability,
    init_mean,
    init_std,
):
    model = ParameterModel(
        parameternetwork,
        input_size,
        output_size,
        step_size,
        init_transition_probability,
        init_mean,
        init_std,
    )
    test_data = torch.randn(test_batch_size, input_size)
    output = model(test_data)
    assert (output[:, 0:step_size] == init_mean).all()
    assert (output[:, step_size : 2 * step_size] == inverse_softplus(init_std)).all()
    assert (output[:, 2 * step_size :] == inverse_sigmod(init_transition_probability)).all()


@pytest.mark.parametrize(
    "post_prenet_rnn_dim, encoder_embedding_dim, n_mel_channels, \
    init_transition_probability, init_mean, init_std",
    [(1024, 512, 80, 0.125, 0, 1), (256, 128, 4, 0.99, 2, 3)],
)
def test_decoder_flat_start(
    hparams,
    test_batch_size,
    post_prenet_rnn_dim,
    encoder_embedding_dim,
    n_mel_channels,
    init_transition_probability,
    init_mean,
    init_std,
):
    hparams.post_prenet_rnn_dim = post_prenet_rnn_dim
    hparams.encoder_embedding_dim = encoder_embedding_dim
    hparams.n_mel_channels = n_mel_channels
    hparams.init_transition_probability = init_transition_probability
    hparams.init_mean = init_mean
    hparams.init_std = init_std

    test_data = torch.randn(test_batch_size, hparams.post_prenet_rnn_dim)
    test_state = torch.randn(test_batch_size, 10, hparams.encoder_embedding_dim)

    model = Decoder(hparams)
    mean, std, transition_vector = model(test_data, test_state)
    assert torch.isclose(mean, torch.tensor(hparams.init_mean * 1.0)).all()
    assert torch.isclose(std, torch.tensor(hparams.init_std * 1.0)).all()
    assert torch.isclose(
        transition_vector.sigmoid(),
        torch.tensor(hparams.init_transition_probability * 1.0),
    ).all()


@pytest.mark.parametrize("std, variance_floor", [(1, 0.5), (0.0001, 0.001), (0.1, 1)])
def test_variance_floor(hparams, test_batch_size, std, variance_floor):
    hparams.variance_floor = variance_floor
    model = Decoder(hparams)
    std = torch.empty(test_batch_size, hparams.n_mel_channels).fill_(std)
    out_std = model.floor_variance(std)
    assert (out_std >= variance_floor).all()
