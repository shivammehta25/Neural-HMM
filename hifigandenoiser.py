"""Waveglow style denoiser can be used to remove the artifacts from the HiFiGAN generated audio."""
import torch

from src.utilities.stft import STFT


class Denoiser(torch.nn.Module):
    """Removes model bias from audio produced with waveglow"""

    def __init__(self, vocoder, filter_length=1024, n_overlap=4, win_length=1024, mode="zeros"):
        super().__init__()
        self.stft = STFT(
            filter_length=filter_length, hop_length=int(filter_length / n_overlap), win_length=win_length
        ).cuda()

        dtype, device = next(vocoder.parameters()).dtype, next(vocoder.parameters()).device
        if mode == "zeros":
            mel_input = torch.zeros((1, 80, 88), dtype=dtype, device=device)
        elif mode == "normal":
            mel_input = torch.randn((1, 80, 88), dtype=dtype, device=device)
        else:
            raise Exception(f"Mode {mode} if not supported")

        with torch.no_grad():
            bias_audio = vocoder(mel_input).float().squeeze(0)
            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer("bias_spec", bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.cuda().float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised
