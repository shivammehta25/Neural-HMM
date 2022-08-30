r"""
data_module.py

Contains PyTorch-Lightning's datamodule and dataloaders
"""
import nltk
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.utilities.data import TextMelCollate, TextMelLoader


class LightningLoader(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams.update(vars(hparams))
        self.collate_fn = TextMelCollate(self.hparams.n_frames_per_step)
        self.num_workers = hparams.num_workers

    def prepare_data(self):
        try:
            nltk.data.find("tokenizers/punkt")
            print("NLTK Tokenizer already present will not download it!")
        except LookupError:
            nltk.download("punkt")

    def setup(self, stage=None):
        self.trainset = TextMelLoader(self.hparams.training_files, self.hparams, [self.hparams.normaliser])
        self.valset = TextMelLoader(self.hparams.validation_files, self.hparams, [self.hparams.normaliser])

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
