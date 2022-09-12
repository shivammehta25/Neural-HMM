"""Transition Model.

This network is responsible for the models response of whether to switch to next state or keep emitting at the current
state.
"""
import torch
import torch.nn as nn

from src.utilities.functions import log_clamped, logsumexp


class TransitionModel(nn.Module):
    r"""
    Transition Model of the HMM, it represents the probability of transitioning
        form current state to all other states
    """

    def __init__(self):
        super().__init__()

    def set_staying_and_transitioning_probability(self, staying, transitioning):
        r"""
        Make reference of the staying and transitioning probabilities as instance
        parameters of class
        """
        self.staying_probability = staying
        self.transition_probability = transitioning

    def forward(self, log_alpha_scaled, transition_vector, state_lengths):
        r"""
        It is the product of the past state with transitional probabilities
        and since it is in log scale, the product will be converted to logsumexp
        Args:
            log_alpha_scaled (torch.Tensor): Multiply previous timestep's alphas by
                        transition matrix (in log domain)
                shape: (batch size, N)
            transition_vector (torch.tensor): transition vector for each state
                shape: (N)
            state_lengths (int tensor): Lengths of states in a batch
                shape: (batch)

        Returns:
            out (torch.FloatTensor): log probability of transitioning to each state
        """
        T_max = log_alpha_scaled.shape[1]

        transition_probability = torch.sigmoid(transition_vector)
        staying_probability = torch.sigmoid(-transition_vector)

        self.set_staying_and_transitioning_probability(staying_probability, transition_probability)

        log_staying_probability = log_clamped(staying_probability)
        log_transition_probability = log_clamped(transition_probability)

        staying = log_alpha_scaled + log_staying_probability
        leaving = log_alpha_scaled + log_transition_probability
        leaving = leaving.roll(1, dims=1)
        leaving[:, 0] = -float("inf")

        mask_tensor = log_alpha_scaled.new_zeros(T_max)
        not_state_lengths_mask = ~(
            torch.arange(T_max, out=mask_tensor).expand(len(state_lengths), T_max) < (state_lengths).unsqueeze(1)
        )

        out = logsumexp(torch.stack((staying, leaving), dim=2), dim=2)

        out = out.masked_fill(not_state_lengths_mask, -float("inf"))

        return out

    def maxmul(self, log_chi, transition_vector, state_lengths):
        r"""
        Multiplies and returns the maximum of values from the transition matrix
        Args:
            log_chi (torch.FloatTensor): viterbi probability variable of previous time step
            transition_vector (torch.FloatTensor): transition vector for each state
            state_lengths (torch.LongTensor): lengths of states in a batch
        """
        T_max = log_chi.shape[1]

        transition_probability = torch.sigmoid(transition_vector)
        staying_probability = torch.sigmoid(-transition_vector)

        log_staying_probability = log_clamped(staying_probability)
        log_transition_probability = log_clamped(transition_probability)

        staying = log_chi + log_staying_probability
        leaving = log_chi + log_transition_probability
        leaving = leaving.roll(1, dims=1)
        leaving[:, 0] = -float("inf")

        # Mask for max ending
        mask_tensor = log_chi.new_zeros(T_max)
        not_state_lengths_mask = ~(
            torch.arange(T_max, out=mask_tensor).expand(len(state_lengths), T_max) < (state_lengths).unsqueeze(1)
        ).unsqueeze(2)

        transition_matrix = torch.stack((staying, leaving), dim=2).masked_fill(not_state_lengths_mask, -float("inf"))

        # TODO: we can mask here as well check
        max_val, max_arg = torch.max(transition_matrix, dim=2)

        return max_val, max_arg
