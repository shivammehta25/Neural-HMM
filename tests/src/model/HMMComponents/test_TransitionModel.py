import torch

from src.model.HMMComponents.TransitionModel import TransitionModel


def test_transitionmodel(test_batch_size, dummy_data):
    _, input_lengths, _, _, _ = dummy_data
    model = TransitionModel()
    prev_t_log_scaled_alpha = torch.randn(test_batch_size, input_lengths.max())
    transition_vector = torch.randn(input_lengths.max())
    out = model(prev_t_log_scaled_alpha, transition_vector, input_lengths)
    assert out.shape == (test_batch_size, input_lengths.max())
