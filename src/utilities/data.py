r"""
data.py

Utilities for processing of Data
"""
import random

import numpy as np
import torch
from nltk import word_tokenize
from scipy.io.wavfile import read
from torch.utils.data.dataset import Dataset

from src.model.layers import TacotronSTFT
from src.utilities.text import phonetise_text, text_to_sequence


def load_wav_to_torch(full_path):
    r"""
    Uses scipy to convert the wav file into torch tensor
    Args:
        full_path: "Wave location"

    Returns:
        torch.FloatTensor of wav data and sampling rate
    """
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


class TextMelCollate:
    r"""
    Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        r"""
        Collate's training batch from normalized text and mel-spectrogram

        Args:
            batch (List): [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max(x[1].size(1) for x in batch)

        # include mel padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, : mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        # torch.empty is a substite for gate_padded, will be removed later when more
        # test ensures there is no regression
        return text_padded, input_lengths, mel_padded, torch.empty([1]), output_lengths


class TextMelLoader(Dataset):
    r"""
    Taken from Nvidia-Tacotron-2 implementation

    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams, transform=None):
        r"""
        Args:
            audiopaths_and_text:
            hparams:
            transform (list): list of transformation
        """
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.transform = transform
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.phonetise = hparams.phonetise
        self.cmu_phonetiser = hparams.cmu_phonetiser
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = TacotronSTFT(
            hparams.filter_length,
            hparams.hop_length,
            hparams.win_length,
            hparams.n_mel_channels,
            hparams.sampling_rate,
            hparams.mel_fmin,
            hparams.mel_fmax,
        )
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        r"""
        Takes audiopath_text list input where list[0] is location for wav file
            and list[1] is the text
        Args:
            audiopath_and_text (list): list of size 2
        """
        # separate filename and text (string)
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        # This text is int tensor of the input representation
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        if self.transform:
            for t in self.transform:
                mel = t(mel)

        return (text, mel)

    def get_mel(self, filename):
        r"""
        Takes filename as input and returns its mel spectrogram
        Args:
            filename (string): Example: 'LJSpeech-1.1/wavs/LJ039-0212.wav'
        """
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError(f"{sampling_rate} SR doesn't match target {self.stft.sampling_rate} SR")
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, "Mel dimension mismatch: given {}, expected {}".format(
                melspec.size(0), self.stft.n_mel_channels
            )

        return melspec

    def get_text(self, text):
        if self.phonetise:
            text = phonetise_text(self.cmu_phonetiser, text, word_tokenize)

        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class Normalise:
    r"""
    Z-Score normalisation class / Standardisation class
    normalises the data with mean and std, when the data object is called

    Args:
        mean (int/tensor): Mean of the data
        std (int/tensor): Standard deviation
    """

    def __init__(self, mean, std):
        super().__init__()

        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        if not torch.is_tensor(std):
            std = torch.tensor(std)

        self.mean = mean
        self.std = std

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        r"""
        Takes an input and normalises it

        Args:
            x (Any): Input to the normaliser

        Returns:
            (torch.FloatTensor): Normalised value
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)

        x = x.sub(self.mean).div(self.std)
        return x

    def inverse_normalise(self, x):
        r"""
        Takes an input and de-normalises it

        Args:
            x (Any): Input to the normaliser

        Returns:
            (torch.FloatTensor): Normalised value
        """
        if not torch.is_tensor(x):
            x = torch.tensor([x])

        x = x.mul(self.std).add(self.mean)
        return x
