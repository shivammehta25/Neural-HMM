r"""
The file creates a pickle file where the values needed for loading of dataset is stored and the model can load it
when needed.

Parameters from hparam.py will be used
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.hparams import create_hparams
from src.utilities.data import TextMelCollate, TextMelLoader


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def parse_batch(batch):
    r"""
    Takes batch as an input and returns all the tensor to GPU
    Args:
        batch: batch of data
    """
    text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    max_len = torch.max(input_lengths.data).item()
    mel_padded = to_gpu(mel_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    output_lengths = to_gpu(output_lengths).long()

    return (
        (text_padded, input_lengths, mel_padded, max_len, output_lengths),
        (mel_padded, gate_padded),
    )


def get_data_parameters_for_flat_start(train_loader, hparams):
    r"""

    Args:
        dataloader (torch.util.data.DataLoader) : datalaoder containing text, mel
        hparams (hparam.py): hyperparemters

    Returns:
        mean (single value float tensor): Mean of the dataset
        std (single value float tensor): Standard deviation of the dataset
        total_sum (single value float tensor): Sum of all values in the observations
        total_observations_all_timesteps (single value float tensor): Sum of all length of observations
    """

    # N related information, useful for transition prob init
    total_state_len = torch.zeros(1).cuda().type(torch.double)
    total_mel_len = torch.zeros(1).cuda().type(torch.double)

    # Useful for data mean and data std
    total_mel_sum = torch.zeros(1).cuda().type(torch.double)
    total_mel_sq_sum = torch.zeros(1).cuda().type(torch.double)

    # For go token
    sum_first_observation = torch.zeros(hparams.n_mel_channels).cuda().type(torch.double)

    print("For exact calculation we would do it with two loops")
    print("We first get the mean:")
    start = time.perf_counter()

    for i, batch in enumerate(tqdm(train_loader)):
        (text_inputs, text_lengths, mels, max_len, mel_lengths), (
            _,
            gate_padded,
        ) = parse_batch(batch)

        total_state_len += torch.sum(text_lengths)
        total_mel_len += torch.sum(mel_lengths)

        total_mel_sum += torch.sum(mels)
        total_mel_sq_sum += torch.sum(torch.pow(mels, 2))

        sum_first_observation += torch.sum(mels[:, :, 0], dim=0)

    data_mean = total_mel_sum / (total_mel_len * hparams.n_mel_channels)
    data_std = torch.sqrt((total_mel_sq_sum / (total_mel_len * hparams.n_mel_channels)) - torch.pow(data_mean, 2))

    N_mean = total_state_len / len(train_loader.dataset)

    average_mel_len = total_mel_len / len(train_loader.dataset)
    average_duration_each_state = average_mel_len / N_mean
    init_transition_prob = 1 / average_duration_each_state

    go_token_init_value = sum_first_observation / len(train_loader.dataset)
    go_token_init_value.sub_(data_mean).div_(data_std)

    print("Total Processing Time:", time.perf_counter() - start)

    print("Getting standard deviation")
    sum_of_squared_error_data = torch.zeros(1).cuda().type(torch.double)
    start = time.perf_counter()

    for i, batch in enumerate(tqdm(train_loader)):
        (text_inputs, text_lengths, mels, max_len, mel_lengths), (
            _,
            gate_padded,
        ) = parse_batch(batch)
        x_minus_mean_square = (mels - data_mean).pow(2)
        T_max_batch = torch.max(mel_lengths)

        mask_tensor = mels.new_zeros(T_max_batch)
        mask = (
            (
                torch.arange(float(T_max_batch), out=mask_tensor).expand(len(mel_lengths), T_max_batch)
                < (mel_lengths).unsqueeze(1)
            )
            .unsqueeze(1)
            .expand(len(mel_lengths), hparams.n_mel_channels, T_max_batch)
        )

        x_minus_mean_square *= mask

        sum_of_squared_error_data += torch.sum(x_minus_mean_square)

    std = torch.sqrt(sum_of_squared_error_data / (total_mel_len * hparams.n_mel_channels))

    print("Total Processing Time:", time.perf_counter() - start)

    data_mean = data_mean.type(torch.float).cpu()
    data_std = std.type(torch.float).cpu()
    go_token_init_value = go_token_init_value.type(torch.float).cpu()
    init_transition_prob = init_transition_prob.type(torch.float).cpu()

    return data_mean, data_std, go_token_init_value, init_transition_prob


def main(args):
    hparams = create_hparams(generate_parameters=True)

    hparams.batch_size = args.batch_size

    trainset = TextMelLoader(hparams.training_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    train_loader = DataLoader(
        trainset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    (
        data_mean,
        data_std,
        go_token_init_value,
        init_transition_prob,
    ) = get_data_parameters_for_flat_start(train_loader, hparams)

    print(
        {
            "data_mean": data_mean.item(),
            "data_std": data_std.item(),
            "init_transition_prob": init_transition_prob.item(),
            "go_token_init_value": go_token_init_value,
        }
    )

    torch.save(
        {
            "data_mean": data_mean,
            "data_std": data_std,
            "init_transition_prob": init_transition_prob,
            "go_token_init_value": go_token_init_value,
        },
        args.output_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="data_parameters.pt",
        required=False,
        help="checkpoint path",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=256,
        required=False,
        help="batch size to fetch data properties",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        required=False,
        help="force overwrite the file",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_file) and not args.force:
        print("File already exists. Use -f to force overwrite")
        sys.exit(1)

    main(args)
