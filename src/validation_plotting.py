import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_only

from src.utilities.plotting import (
    plot_alpha_scaled_to_numpy,
    plot_go_tokens_to_numpy,
    plot_hidden_states_to_numpy,
    plot_mel_spectrogram_to_numpy,
    plot_transition_matrix,
    plot_transition_probabilities_to_numpy,
)


@rank_zero_only
def log_validation(
    logger,
    model,
    mel_output,
    mel_output_normalised,
    state_travelled,
    mel_targets,
    input_parameters,
    output_parameters,
    iteration,
):
    """
    Args:
        logger (SummaryWriter): logger from pytorch lightning
        model: model to plot alpha scaled
        mel_output: mel generated
        mel_output_normalised: normalised version of mel output
        state_travelled: phones/states travelled
        mel_targets: target mel
        input_parameters: input parameters to decoder model while sampling
        output_parameters: output parameters from the decoder model while sampling
        iteration: iteration number

    Returns:
        None
    """
    # plot distribution of parameters
    for tag, value in model.named_parameters():
        tag = tag.replace(".", "/")
        logger.add_histogram(tag, value.data.cpu().numpy(), iteration)

    logger.add_image(
        "alignment/log_alpha_scaled",
        plot_alpha_scaled_to_numpy(model.hmm.log_alpha_scaled[0, :, :].T, plot_logrithmic=True),
        iteration,
        dataformats="HWC",
    )

    logger.add_image(
        "transition_probabilities",
        plot_transition_matrix(torch.sigmoid(model.hmm.transition_vector[0, :, :])),
        iteration,
        dataformats="HWC",
    )

    logger.add_image(
        "alignment/alpha_scaled",
        plot_alpha_scaled_to_numpy(torch.exp(model.hmm.log_alpha_scaled[0, :, :]).T),
        iteration,
        dataformats="HWC",
    )

    logger.add_image(
        "mel_target",
        plot_mel_spectrogram_to_numpy(mel_targets),
        iteration,
        dataformats="HWC",
    )

    logger.add_image(
        "synthesised/mel_synthesised",
        plot_mel_spectrogram_to_numpy(np.array(mel_output).T),
        iteration,
        dataformats="HWC",
    )

    logger.add_image(
        "synthesised/mel_synthesised_normalised",
        plot_mel_spectrogram_to_numpy(mel_output_normalised.T),
        iteration,
        dataformats="HWC",
    )

    logger.add_image(
        "synthesised/hidden_state_travelled",
        plot_hidden_states_to_numpy(state_travelled),
        iteration,
        dataformats="HWC",
    )

    logger.add_image(
        "parameters/go_tokens",
        plot_go_tokens_to_numpy(model.hmm.go_tokens.clone().detach()),
        iteration,
        dataformats="HWC",
    )

    states = [p[1] for p in input_parameters]
    transition_probability_synthesising = [p[2].cpu().numpy() for p in output_parameters]

    for i in range((len(transition_probability_synthesising) // 200) + 1):
        start = i * 200
        end = (i + 1) * 200
        logger.add_image(
            f"synthesised_transition_probabilities/{i}",
            plot_transition_probabilities_to_numpy(states[start:end], transition_probability_synthesising[start:end]),
            iteration,
            dataformats="HWC",
        )

    # Plotting means of most probable state
    max_state_numbers = torch.max(model.hmm.log_alpha_scaled[0, :, :], dim=1)[1]
    means = torch.stack(model.hmm.means, dim=1).squeeze(0)

    max_len = means.shape[0]
    n_mel_channels = means.shape[2]

    max_state_numbers = max_state_numbers.unsqueeze(1).unsqueeze(1).expand(max_len, 1, n_mel_channels)
    means = torch.gather(means, 1, max_state_numbers).squeeze(1)

    logger.add_image(
        "mel_from_the_means_predicted_by_most_probable_state",
        plot_mel_spectrogram_to_numpy(means.T),
        iteration,
        dataformats="HWC",
    )
