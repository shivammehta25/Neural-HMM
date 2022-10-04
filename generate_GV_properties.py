"""
Generate Global Variance (GV)

"""
import argparse
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.hparams import create_hparams
from src.training_module import TrainingModule
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


def main(args):
    hparams = create_hparams()
    hparams.num_workers = 0
    hparams.batch_size = 6
    val_loader = get_val_dataloader(hparams)

    model = TrainingModule.load_from_checkpoint(args.checkpoint_path)
    model = model.cuda() if torch.cuda.is_available() else model
    # Turn of normalisation
    model.model.hmm.normaliser = None
    mel_outputs = []
    mel_outputs_len = []
    for i, batch in enumerate(tqdm(val_loader)):
        x, _ = parse_batch(batch)
        text_inputs, text_lengths, mels, max_len, mel_lengths = x

        for i in tqdm(range(len(text_inputs))):
            mel_output, *_ = model.sample(text_inputs[i][: text_lengths[i]], text_lengths[i])
            mel_outputs.append(torch.stack([torch.tensor(t) for t in mel_output]))
            mel_outputs_len.append(len(mel_output))

            break
        break

    mel_outputs = pad_sequence(mel_outputs, batch_first=True)

    # T_max = mel_outputs.shape[1]
    # mask = torch.arange(T_max).expand(len(mel_outputs_len), T_max) < (torch.tensor(mel_outputs_len)).unsqueeze(1)

    mean_gv = torch.mean(mel_outputs, dim=[0, 1])
    std_gv = torch.std(mel_outputs, dim=[0, 1])
    return mean_gv, std_gv


def get_val_dataloader(hparams):
    val_set = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    validation_loader = DataLoader(
        val_set, batch_size=hparams.batch_size, num_workers=hparams.num_workers, collate_fn=collate_fn
    )
    return validation_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default="checkpoints/TestRun/checkpoint_50000.ckpt",
        required=False,
        help="checkpoint path",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        required=False,
        help="force overwrite the file",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="gv_parameters.pt",
        required=False,
        help="checkpoint path",
    )
    args = parser.parse_args()

    if args.checkpoint_path and not os.path.exists(args.checkpoint_path):
        raise FileExistsError("Check point not present recheck the name")

    if os.path.exists(args.output_file) and not args.force:
        raise FileExistsError("File already exists. Use -f to force overwrite")

    mean_gv, std_gv = main(args)

    torch.save({"mean_gv": mean_gv, "std_gv": std_gv}, "gv_parameters.pt")
