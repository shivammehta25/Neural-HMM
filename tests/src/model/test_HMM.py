import pytest
import torch
from src.model.HMM import HMM


def test_hmm_forward(hparams, dummy_embedded_data, test_batch_size):
    model = HMM(hparams)
    (
        embedded_input,
        input_lengths,
        mel_padded,
        _,
        output_lengths,
    ) = dummy_embedded_data
    loss = model.forward(embedded_input, input_lengths, mel_padded, output_lengths)
    assert (not torch.isnan(loss).any()) and (not torch.isinf(loss).any())
    assert model.N == embedded_input.shape[1]
    assert model.log_alpha_scaled.shape == (
        test_batch_size,
        mel_padded.shape[2],
        model.N,
    )
    assert model.transition_vector.shape == (
        test_batch_size,
        mel_padded.shape[2],
        model.N,
    )
    assert (not torch.isnan(model.transition_vector).any()) and (
        not torch.isinf(model.transition_vector).any()
    )


def test_mask_lengths(hparams, dummy_embedded_data, test_batch_size):
    model = HMM(hparams)
    (
        embedded_input,
        input_lengths,
        mel_padded,
        _,
        output_lengths,
    ) = dummy_embedded_data
    model.forward(embedded_input, input_lengths, mel_padded, output_lengths)
    log_c = torch.randn(test_batch_size, mel_padded.shape[2])

    log_c = model.mask_lengths(mel_padded.transpose(1, 2), output_lengths, log_c)

    for i in range(test_batch_size):
        assert log_c[i, output_lengths[i] :].sum() == 0.0, "Not masked properly"

    # TODO: remove the input_lengths[i] from the assertion
    #       but then we have avoid masking with -inf
    for i in range(test_batch_size):
        assert (
            model.log_alpha_scaled[i, output_lengths[i] :, : input_lengths[i]].sum()
            == 0.0
        ), "Not masked properly"


def test_init_lstm_states(hparams, dummy_embedded_data, test_batch_size):
    model = HMM(hparams)
    (
        embedded_input,
        input_lengths,
        mel_padded,
        _,
        output_lengths,
    ) = dummy_embedded_data
    h_post_prenet, c_post_prenet = model.init_lstm_states(
        test_batch_size, hparams.post_prenet_rnn_dim, mel_padded
    )
    assert h_post_prenet.shape == (test_batch_size, hparams.post_prenet_rnn_dim)
    assert c_post_prenet.shape == (test_batch_size, hparams.post_prenet_rnn_dim)


@pytest.mark.parametrize(
    "t, data_dropout_flag, prenet_dropout_flag",
    [(0, True, True), (1, True, False), (2, False, True), (3, False, False)],
)
def test_process_ar_timestep(
    t,
    data_dropout_flag,
    prenet_dropout_flag,
    hparams,
    dummy_embedded_data,
    test_batch_size,
):
    model = HMM(hparams)
    (
        embedded_input,
        input_lengths,
        mel_padded,
        _,
        output_lengths,
    ) = dummy_embedded_data
    mel_padded = mel_padded.transpose(1, 2)
    h_post_prenet, c_post_prenet = model.init_lstm_states(
        test_batch_size, hparams.post_prenet_rnn_dim, mel_padded
    )

    h_post_prenet, c_post_prenet = model.process_ar_timestep(
        t,
        mel_padded,
        h_post_prenet,
        c_post_prenet,
        data_dropout_flag,
        prenet_dropout_flag,
    )

    assert h_post_prenet.shape == (test_batch_size, hparams.post_prenet_rnn_dim)
    assert c_post_prenet.shape == (test_batch_size, hparams.post_prenet_rnn_dim)


def test_add_go_token(hparams, dummy_embedded_data):
    model = HMM(hparams)
    (
        embedded_input,
        input_lengths,
        mel_padded,
        _,
        output_lengths,
    ) = dummy_embedded_data
    mel_padded = mel_padded.transpose(1, 2)
    ar_mel = model.add_go_token(mel_padded)
    assert ar_mel.shape == mel_padded.shape
    assert (ar_mel[:, 1:] == mel_padded[:, :-1]).all(), "Go token not appended properly"


def test_forward_algorithm_variables(hparams, dummy_embedded_data, test_batch_size):
    model = HMM(hparams)
    (
        embedded_input,
        input_lengths,
        mel_padded,
        _,
        output_lengths,
    ) = dummy_embedded_data
    mel_padded = mel_padded.transpose(1, 2)

    model.N = embedded_input.shape[1]
    log_c = model.initialize_forward_algorithm_variables(mel_padded)
    assert log_c.shape == (test_batch_size, mel_padded.shape[1])
    assert model.log_alpha_scaled.shape == (
        test_batch_size,
        mel_padded.shape[1],
        embedded_input.shape[1],
    )

