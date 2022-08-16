from torch import nn
from torch.nn import functional as F

from src.model.layers import LinearReluInitNorm, ConvNorm


class Prenet(nn.Module):
    r"""
    MLP prenet module
    """

    def __init__(self, in_dim, n_layers, prenet_dim, prenet_dropout):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + [prenet_dim for _ in range(n_layers)]
        self.prenet_dropout = prenet_dropout
        self.layers = nn.ModuleList(
            [
                LinearReluInitNorm(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes[:-1], in_sizes[1:])
            ]
        )

    def forward(self, x, dropout_flag):
        for linear in self.layers:
            x = F.dropout(linear(x), p=self.prenet_dropout, training=dropout_flag)
        return x


class ConvPrenet(nn.Module):
    r"""
    Convolution prenet module
    """

    def __init__(self, in_dim, n_convolutions, kernel_size, prenet_dim, prenet_dropout):
        r"""
        Args:
            in_dim (int): input dimension
            n_convolutions (int): number of convolutions in prenet
            kernel_size (int): size of kernel
            prenet_dim (int): dimension of output
            prenet_dropout (float): float value of range (1-0)
        """
        super(ConvPrenet, self).__init__()
        in_sizes = [in_dim] + [prenet_dim for _ in range(n_convolutions)]
        self.prenet_dropout = prenet_dropout
        self.layers = nn.ModuleList(
            [
                ConvNorm(in_size, out_size, kernel_size=kernel_size)
                for (in_size, out_size) in zip(in_sizes[:-1], in_sizes[1:])
            ]
        )

    def forward(self, x):
        r"""
        Args:
            x (float tensor): (batch, max_len, n_frames_per_step * n_mel_channels)

        Returns:
            x (float tensor): (batch, max_len, prenet_dim)
        """
        x = x.permute(0, 2, 1)

        for conv in self.layers:
            x = F.dropout(F.relu(conv(x)), p=self.prenet_dropout, training=True)

        x = x.permute(0, 2, 1)
        return x


class LSTMPostPrenet(nn.Module):
    r"""
    Takes sequence of Convolution inputs and returns output from LSTM network
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        batch_first=True,
        bidirectional=False,
    ):
        super(LSTMPostPrenet, self).__init__()

        self.batch_first = batch_first
        self.layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(self, inputs, inputs_size):
        r"""
        Takes output from convolutional prenet and then feeds it into the LSTM and
        returns its outputs
        Args:
            inputs (torch.float): shape (batch, mel_max_len, prenet_dim)
            inputs_size (torch.int): shape (batch)

        Returns:
            outputs (torch.float):
        """

        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            inputs,
            inputs_size.cpu(),
            batch_first=self.batch_first,
            enforce_sorted=False,
        )
        packed_outputs, (hidden_state, cell_state) = self.layer(packed_inputs)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        return outputs.transpose(1, 0)
