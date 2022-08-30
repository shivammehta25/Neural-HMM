from torch import nn
from torch.nn import functional as F

from src.model.layers import ConvNorm


class Encoder(nn.Module):
    """Encoder module:

    - Three 1-d convolution banks
    - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super().__init__()

        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.state_per_phone = hparams.state_per_phone

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    hparams.encoder_embedding_dim,
                    hparams.encoder_embedding_dim,
                    kernel_size=hparams.encoder_kernel_size,
                    stride=1,
                    padding=int((hparams.encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(hparams.encoder_embedding_dim),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            hparams.encoder_embedding_dim,
            int(hparams.encoder_embedding_dim / 2) * hparams.state_per_phone,
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, input_lengths):
        r"""
        Takes embeddings as inputs and returns encoder representation of them
        Args:
            x (torch.float) : input shape (32, 512, 139)
            input_lengths (torch.int) : (32)

        Returns:
            outputs (torch.float):
                shape (batch, text_len * phone_per_state, encoder_embedding_dim)
            input_lengths (torch.float): shape (batch)
        """

        batch_size = x.shape[0]
        t_len = x.shape[2]

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths_np = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths_np, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)  # We do not use the hidden or cell states

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        outputs = outputs.reshape(batch_size, t_len * self.state_per_phone, self.encoder_embedding_dim)
        input_lengths = input_lengths * self.state_per_phone

        return outputs, input_lengths  # (32, 139, 519)
