import torch

from src.model.NeuralHMM import NeuralHMM


def test_parse_batch(hparams, dummy_data):
    neural_hmm = NeuralHMM(hparams)
    parsed_batch = neural_hmm.parse_batch(dummy_data)
    text_padded, input_lengths, mel_padded, max_len, mel_lengths = parsed_batch[0]
    mel_padded, _ = parsed_batch[1]
    assert text_padded.shape[1] == max_len
    assert mel_padded.shape[2] == torch.max(mel_lengths).item()


def test_forward(hparams, dummy_data, test_batch_size):
    neural_hmm = NeuralHMM(hparams)
    log_probs = neural_hmm.forward(dummy_data)
    assert log_probs.shape == (test_batch_size,)


def test_inference(hparams, dummy_data_uncollated):
    neural_hmm = NeuralHMM(hparams)
    text = dummy_data_uncollated[0][0].unsqueeze(0)
    mel_output, states_travelled = neural_hmm.inference(text)
    assert len(mel_output[0]) == hparams.n_mel_channels


# TODO: remove inference and merge with sampling later
def test_sample(hparams, dummy_data_uncollated):
    neural_hmm = NeuralHMM(hparams)
    text = dummy_data_uncollated[0][0]
    (
        mel_output,
        states_travelled,
        input_parameters,
        output_parameters,
    ) = neural_hmm.sample(text, torch.tensor(len(text)))
    assert len(mel_output[0]) == hparams.n_mel_channels
    assert input_parameters[0][0].shape[-1] == hparams.n_mel_channels
    assert output_parameters[0][0].shape[-1] == hparams.n_mel_channels
    assert output_parameters[0][1].shape[-1] == hparams.n_mel_channels
