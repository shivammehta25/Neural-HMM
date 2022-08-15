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


def test_inference(hparams, dummy_data):
    neural_hmm = NeuralHMM(hparams)
    text = dummy_data[0:1]
    mel_output, states_travelled = neural_hmm.inference(text)
    assert mel_output.shape[-1] == hparams.n_mel_channels
