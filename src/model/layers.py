"""layers.py.

Layer modules used in the model design
"""
import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn

from src.utilities.audio import dynamic_range_compression, dynamic_range_decompression
from src.utilities.stft import STFT


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LinearReluInitNorm(nn.Module):
    r"""
    Contains a Linear Layer with Relu activation and a dropout
    Args:
        inp (tensor): size of input to the linear layer
        out (tensor): size of output from the linear layer
        init (bool): model initialisation with xavier initialisation
            default: False
        w_init_gain (str): gain based on the activation function used
            default: relu
    """

    def __init__(self, inp, out, init=True, w_init_gain="relu", bias=True):
        super().__init__()

        self.w_init_gain = w_init_gain
        self.linear = nn.Sequential(nn.Linear(inp, out, bias=bias), nn.ReLU())

        if init:
            self.linear.apply(self._weights_init)

    def _weights_init(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight.data, gain=torch.nn.init.calculate_gain(self.w_init_gain))

    def forward(self, x):
        return self.linear(x)


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class TacotronSTFT(torch.nn.Module):
    """Short Time Fourier Transformation."""

    def __init__(
        self,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        sampling_rate=22050,
        mel_fmin=0.0,
        mel_fmax=8000.0,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels  # 80
        self.sampling_rate = sampling_rate  # 22050
        self.stft_fn = STFT(filter_length, hop_length, win_length)  # default values
        # """This produces a linear transformation matrix to project FFT bins onto Mel-frequency bins."""
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax
        )  # all default values

        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output
