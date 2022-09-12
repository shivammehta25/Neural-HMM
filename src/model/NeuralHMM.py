from math import sqrt

import torch
from torch import nn

from src.model.Encoder import Encoder
from src.model.HMM import HMM
from src.utilities.functions import sequence_mask
from src.utilities.monotonic_align import maximum_path


class NeuralHMM(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim)
        if hparams.warm_start or hparams.checkpoint_path:
            # If warm start or resuming training do not re-initialize embeddings
            std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
            val = sqrt(3.0) * std  # uniform bounds for std
            self.embedding.weight.data.uniform_(-val, val)

        self.encoder = Encoder(hparams)
        self.hmm = HMM(hparams)
        self.logger = hparams.logger

    def parse_batch(self, batch):
        """
        Takes batch as an input and returns all the tensor to GPU
        Args:
            batch:

        Returns:

        """
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = mel_padded.float()
        gate_padded = gate_padded.float()
        output_lengths = output_lengths.long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded),
        )

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, mel_lengths = inputs
        text_lengths, mel_lengths = text_lengths.data, mel_lengths.data
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs, text_lengths = self.encoder(embedded_inputs, text_lengths)

        log_probs = self.hmm(encoder_outputs, text_lengths, mels, mel_lengths)

        text_mask = sequence_mask(text_lengths).float().unsqueeze(1)
        mel_mask = sequence_mask(mel_lengths).float().unsqueeze(1)
        attn_mask = (torch.unsqueeze(text_mask, -1) * torch.unsqueeze(mel_mask, 2)).squeeze(1)
        attn = maximum_path(self.hmm.log_alpha_scaled.transpose(1, 2).contiguous(), attn_mask).detach()

        filtered_states = torch.matmul(attn.squeeze(1).transpose(1, 2), encoder_outputs)

        viterbi_prob = self.hmm.viterbi_algorithm(filtered_states, text_lengths, mels, mel_lengths)

        filtered_states, viterbi_prob == 0, 0

        return log_probs

    @torch.inference_mode()
    def inference(self, text_inputs):
        r"""
        Sampling audio based on single text input
        Args:
            text_inputs (int tensor) : shape: (1, x) where x is length of phoneme input
        Returns:
            mel_outputs (list): list of len of the output of mel spectrogram each
                    containing n_mel_channels channels
                shape: (len, n_mel_channels)
            states_travelled (list): list of phoneme travelled at each time step t
                shape: (len)
        """

        text_lengths = text_inputs.new_tensor(text_inputs.shape[1]).unsqueeze(0)
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs, text_lengths = self.encoder(embedded_inputs, text_lengths)

        (
            mel_output,
            states_travelled,
            input_parameters,
            output_parameters,
        ) = self.hmm.sample(encoder_outputs)

        return mel_output, states_travelled

    @torch.inference_mode()
    def sample(self, text_inputs, text_lengths):
        r"""
        Sampling mel spectrogram based on text inputs
        Args:
            text_inputs (int tensor) : shape ([x]) where x is the phoneme input
            text_lengths (int tensor):  single value scalar with length of input (x)

        Returns:
            mel_outputs (list): list of len of the output of mel spectrogram
                    each containing n_mel_channels channels
                shape: (len, n_mel_channels)
            states_travelled (list): list of phoneme travelled at each time step t
                shape: (len)
        """
        text_inputs, text_lengths = text_inputs.unsqueeze(0), text_lengths.unsqueeze(0)
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs, text_lengths = self.encoder(embedded_inputs, text_lengths)

        (
            mel_output,
            states_travelled,
            input_parameters,
            output_parameters,
        ) = self.hmm.sample(encoder_outputs)

        return mel_output, states_travelled, input_parameters, output_parameters
