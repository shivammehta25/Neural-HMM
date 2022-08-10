r"""
training_model.py 

This file contains PyTorch Lightning's main module where code of the main model is implemented
"""
import os
from argparse import Namespace
from typing import Any, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.NeuralHMM import NeuralHMM
from src.validation_plotting import log_validation
from src.hparams import create_hparams


class TrainingModule(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        if type(hparams) != Namespace:
            hparams = Namespace(**hparams)

        self.save_hyperparameters(hparams)
        hparams.logger = self.logger

        self.model = NeuralHMM(hparams)

    def forward(self, x):
        r"""
        Forward pass of the model

        Args:
            x (Any): input to the forward function

        Returns:
            output (Any): output of the forward function
        """
        log_probs = self.model(x)
        return log_probs

    def configure_optimizers(self):
        r"""
        Configure optimizer

        Returns:
            (torch.optim.Optimizer) 
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)

    def training_step(self, train_batch, batch_idx):
        r"""
        Main training loop of your model

        Args:
            train_batch (List): batch of input data
            batch_idx (Int): index of the current batch

        Returns:
            loss (torch.FloatTensor): loss of the forward run of your model
        """
        x, y = self.model.parse_batch(train_batch)
        log_probs = self(x)
        loss = -log_probs.mean()
        self.log("loss/train", loss.item(), prog_bar=True,
                 on_step=True, sync_dist=True, logger=True)
        self.log("Global_Step", int(self.global_step),
                 prog_bar=True, on_step=True, sync_dist=True, logger=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        r"""
        Validation step

        Args:
            val_batch (Any): output depends what you are returning from the train loop
            batch_idx (): batch index
        """
        x, y = self.model.parse_batch(val_batch)
        log_probs = self(x)
        loss = -log_probs.mean()
        self.log('loss/val', loss.item(), prog_bar=True,
                 sync_dist=True, logger=True)
        return loss

    def on_before_zero_grad(self, optimizer):
        r"""
        Takes actions before zeroing the gradients.
        We use it to plot the output of the model at the save_model_checkpoint iteration from hparams.

        Args:
            optimizer ([type]): [description]
        """

        if self.trainer.is_global_zero and (self.global_step % self.hparams.save_model_checkpoint == 0):
            text_inputs, text_lengths, mels, max_len, mel_lengths = self.get_an_element_of_validation_dataset()
            mel_output, state_travelled, input_parameters, output_parameters = self.model.sample(text_inputs[0],
                                                                                                 text_lengths[0])
            mel_output_normalised = self.model.hmm.normaliser(
                mels.new_tensor(mel_output))

            with torch.no_grad():
                _ = self.model((text_inputs, text_lengths,
                                mels, max_len, mel_lengths))

            log_validation(self.logger.experiment, self.model, mel_output, mel_output_normalised, state_travelled, mels[0],
                           input_parameters, output_parameters, self.global_step)

            self.trainer.save_checkpoint(os.path.join(
                self.hparams.checkpoint_dir, self.hparams.run_name, f"checkpoint_{self.global_step}.ckpt"))

    def get_an_element_of_validation_dataset(self):
        r"""
        Gets an element of the validation dataset.

        Returns:
            text_inputs (torch.FloatTensor): The text inputs.
            text_lengths (torch.LongTensor): The text lengths.
            mels (torch.FloatTensor): The mels spectrogram.
            max_len (int): The maximum length of the mels spectrogram.
            mel_lengths (torch.LongTensor): The lengths of the mel spectrogram.
        """
        x, y = self.model.parse_batch(next(iter(self.val_dataloader())))
        (text_inputs, text_lengths, mels, max_len, mel_lengths) = x
        text_inputs = text_inputs[0].unsqueeze(0).to(self.device)
        text_lengths = text_lengths[0].unsqueeze(0).to(self.device)
        mels = mels[0].unsqueeze(0).to(self.device)
        max_len = torch.max(text_lengths).data
        mel_lengths = mel_lengths[0].unsqueeze(0).to(self.device)
        # Sometimes in a batch the element which has the maximum mel len
        # is not the same as the element which has the maximum text len.
        # This prevent the model to break down when plotting validation.
        mels = mels[:, :, :mel_lengths.item()]

        return text_inputs, text_lengths, mels, max_len, mel_lengths

    def inference(self, text_inputs):
        return self.model.inference(text_inputs)

    def sample(self, text_inputs, text_lengths):
        return self.model.sample(text_inputs, text_lengths)

    def log_grad_norm(self, grad_norm_dict):
        r"""
        Lightning method to log the grad norm of the model.
        change prog_bar to True to track on progress bar

        Args:
            grad_norm_dict: Dictionary containing current grad norm metrics

        """
        self.log_dict(grad_norm_dict, on_step=True,
                      on_epoch=True, prog_bar=False, logger=True)
