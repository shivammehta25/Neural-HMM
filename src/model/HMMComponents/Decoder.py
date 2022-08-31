"""Parameter Model This model takes state as an input and generates its parameters i.e mean, standard deviation and
the probability of transition to the next state."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.layers import LinearReluInitNorm
from src.utilities.functions import inverse_sigmod, inverse_softplus


class ParameterModel(nn.Module):
    r"""
    Generate means of the current state based on current state value
    and previous observations
    """

    def __init__(
        self,
        parameternetwork,
        input_size,
        output_size,
        step_size,
        init_transition_probability,
        init_mean,
        init_std,
    ):
        r"""

        Args:
            parameternetwork (list[int]): the architecture of the parameter model
            input_size (int): size of input for the first layer
            output_size (int): size of output i.e size of the feature dim
            step_size (int): feature dim to set the flat start bias
            init_transition_probability (float): flat start transition probability
            init_mean (float): flat start mean
            init_std (float): flat start std between 0 and 1
        """
        super().__init__()

        self.output_size = output_size

        self.layers = nn.ModuleList(
            [LinearReluInitNorm(inp, out) for inp, out in zip([input_size] + parameternetwork[:-1], parameternetwork)]
        )
        last_layer = nn.Linear(parameternetwork[-1], output_size)
        last_layer.weight.data.zero_()
        last_layer.bias.data[0:step_size] = init_mean
        last_layer.bias.data[step_size : 2 * step_size] = inverse_softplus(init_std)
        last_layer.bias.data[2 * step_size :] = inverse_sigmod(init_transition_probability)
        self.layers.append(last_layer)

    def forward(self, x):
        r"""
        Inputs 2nd order auto regression values and returns the means for that
            observations
        Args:
            x (torch.FloatTensor): shape (batch, maxlength, prenet_dim)
            states (torch.FloatTensor):  shape (hidden_states)
            dropout_flag (bool): Flag to keep on dropout during evaluation
        Returns:
            x: model output
                shape: (batch, maxlength, hidden_states, self.output_size)
        """

        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)

        x = self.layers[-1](x)

        return x


class Decoder(nn.Module):
    r"""
    This network takes current state and previous observed values as input
    and returns its parameters, mean, standard deviation and probability
    of transition to the next state
    """

    def __init__(self, hparams):
        r"""

        Args:
            hparams (argparse.Namespace): hyperparameters
        """
        super().__init__()
        self.hparams = hparams

        input_size = hparams.post_prenet_rnn_dim + hparams.encoder_embedding_dim
        output_size = 2 * hparams.n_mel_channels + 1

        self.validate_parameters()

        self.decoder_network = ParameterModel(
            hparams.parameternetwork,
            input_size,
            output_size,
            hparams.n_mel_channels,
            hparams.init_transition_probability,
            hparams.init_mean,
            hparams.init_std,
        )

    def validate_parameters(self):
        """Validate the hyperparameters.

        Raises:
            ValueError: when the parameters network is not defined
            ValueError: transition probability is not between 0 and 1
        """
        if len(self.hparams.parameternetwork) < 1:
            raise ValueError(
                "Parameter Network must have atleast one layer check the config file \
                for parameter network"
            )
        if not (0 < self.hparams.init_transition_probability < 1):
            raise ValueError(
                "Invalid value for initial transitioning probability should be within \
                    (0, 1) but was given {}".format(
                    self.hparams.init_transition_probability
                )
            )

    def forward(self, ar_mel_inputs, states):
        r"""
        Inputs observation and returns the means, stds and transition probability
        for the current state

        Args:
            ar_mel_inputs (torch.FloatTensor): shape (batch, prenet_dim)
            states (torch.FloatTensor):  (hidden_states, hidden_state_dim)

        Returns:
            means: means for the emission observation for each feature
                shape: (batch, hidden_states, feature_size)
            stds: standard deviations for the emission observation for each feature
                shape: (batch, hidden_states, feature_size)
            transition_vectors: transition vector for the current hidden state
                shape: (batch, hidden_states)
        """
        batch_size, prenet_dim = ar_mel_inputs.shape[0], ar_mel_inputs.shape[1]
        N = states.shape[1]

        ar_mel_inputs = ar_mel_inputs.unsqueeze(1).expand(batch_size, N, prenet_dim)
        ar_mel_inputs = torch.cat((ar_mel_inputs, states), dim=2)
        ar_mel_inputs = self.decoder_network(ar_mel_inputs)

        mean, std, transition_vector = (
            ar_mel_inputs[:, :, 0 : self.hparams.n_mel_channels],
            ar_mel_inputs[:, :, self.hparams.n_mel_channels : 2 * self.hparams.n_mel_channels],
            ar_mel_inputs[:, :, 2 * self.hparams.n_mel_channels :].squeeze(2),
        )
        std = F.softplus(std)
        std = self.floor_variance(std)

        return mean, std, transition_vector

    def floor_variance(self, std):
        r"""
        It clamps the standard deviation to not to go below some level
        This removes the problem the model over learns for a state and the gaussian
        is converted to point mass resulting in improper probability for data points

        Args:
            std (float Tensor): tensor containing the standard deviation to be
        """
        original_tensor = std.clone().detach()
        std = torch.clamp(std, min=self.hparams.variance_floor)
        if torch.any(original_tensor != std):
            print("Variance Floored")
        return std
