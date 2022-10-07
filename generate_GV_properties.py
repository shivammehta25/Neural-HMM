"""
Generate Global Variance (GV)

"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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

    model = load_model(args.checkpoint_path)

    mean_gv, std_gv = generate_gv(val_loader, model)
    return mean_gv, std_gv


def generate_gv(val_loader, model):
    """Takes in validation dataloader and model and returns mean and std of global variance

    Args:
        val_loader (dataset.DataLoader): Validation dataloader
        model (nn.Module): Model

    Returns:
        _type_: _description_
    """
    mel_means = 0
    mel_means_sq = 0
    count = 0

    print("Synthesise and compute mean!")
    mel_outputs = []
    for j, batch in enumerate(tqdm(val_loader, leave=False)):
        x, _ = parse_batch(batch)
        text_inputs, text_lengths, mels, max_len, mel_lengths = x

        for i in tqdm(range(len(text_inputs)), leave=False):
            mel_output, *_ = model.sample(text_inputs[i][: text_lengths[i]], text_lengths[i])
            mel_output = torch.tensor(mel_output, device=mels.device, dtype=mels.dtype)
            mel_outputs.append(mel_output)
            mel_means += mel_output.sum(dim=0)
            count += mel_output.shape[0]

    mean_gv = mel_means / count

    sq_count = 0
    print("Now get standard deviation!")
    for mel_output in tqdm(mel_outputs, leave=False):
        mel_means_sq += (mel_output.sub(mean_gv)).pow(2).sum(dim=0)
        sq_count += mel_output.shape[0]

    assert count == sq_count, "heh?"

    std_gv = torch.sqrt(mel_means_sq.div(sq_count))
    return mean_gv, std_gv


def load_model(checkpoint):
    model = TrainingModule.load_from_checkpoint(checkpoint)
    model = model.cuda() if torch.cuda.is_available() else model
    # Turn of normalisation
    model.model.hmm.normaliser = None
    return model


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

    # if os.path.exists(args.output_file) and not args.force:
    #     raise FileExistsError("File already exists. Use -f to force overwrite")

    mean_gv, std_gv = main(args)
    output = {"mean_gv": mean_gv, "std_gv": std_gv}
    print(output)
    torch.save(output, args.output_file)
    print(f"Saved successfully as {args.output_file}")
