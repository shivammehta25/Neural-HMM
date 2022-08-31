import os
import sys
import warnings

import matplotlib.pyplot as plt
import nltk
import numpy as np
import soundfile
import streamlit as st
import torch
from nltk import word_tokenize
from PIL import Image
from waveglow.denoiser import Denoiser

from src.hparams import create_hparams
from src.training_module import TrainingModule
from src.utilities.text import phonetise_text, text_to_sequence

# print(os.getcwd())


# if os.getcwd().split('/')[-1] == 'deployment':
#     os.chdir('../')

nltk.download("punkt")

sys.path.append("src/model")
sys.path.append("waveglow/")

# ===========================================#
#                Configs                    #
# ===========================================#

title = "Neural HMM"
image = Image.open("NeuralHMMTTS.png")
desc = "Generate audio with the Neural HMM, \
    more information available at  \
    https://shivammehta007.github.io/Neural-HMM/"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "neur-hmm.ckpt"
waveglow_path = "waveglow_256channels_universal_v5.pt"


# ===========================================#
#        Loads Model and Pipeline           #
# ===========================================#
# Load Waveglow Vocoder

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    waveglow = torch.load(waveglow_path)["model"]
    waveglow.to(device).eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)


hparams = create_hparams()


# Load Neural-HMM
def load_model(checkpoint_path):
    model = TrainingModule.load_from_checkpoint(checkpoint_path)
    _ = model.to(device).eval()
    return model


model = load_model(checkpoint_path)


# Phonetising
def prepare_text(text):
    text = phonetise_text(hparams.cmu_phonetiser, text, word_tokenize)
    sequence = np.array(text_to_sequence(text, ["english_cleaners"]))[None, :]
    sequence = torch.from_numpy(sequence).to(device).long()
    return sequence


# Plotting mel
def plot_spectrogram_to_numpy(spectrogram):

    fig.canvas.draw()
    plt.close()
    return fig


# ===========================================#
#              Streamlit Code               #
# ===========================================#


st.title(title)
st.write(desc)
st.image(image, caption="Neural HMM Architecture")

speaking_rate = st.slider("Speaker rate", min_value=0.1, max_value=0.9, value=0.4, step=0.1)

user_input = st.text_input("Text to generate")
if st.button("Generate Audio"):
    with torch.no_grad():
        model.model.hmm.hparams.duration_quantile_threshold = speaking_rate
        text = prepare_text(user_input)
        mel_output, _ = model.inference(text)
        mel_output = torch.tensor(mel_output).T.unsqueeze(0).cuda()
        audio = waveglow.infer(mel_output, sigma=0.666)
        audio_denoised = denoiser(audio, strength=0.01)[:, 0].cpu().numpy()

    # import pdb; pdb.set_trace()

    sample_rate = 22050

    spectrogram = mel_output.cpu().numpy()[0]
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.title("Synthesised Mel-Spectrogram")
    st.pyplot(fig)

    soundfile.write("temp.wav", audio_denoised.T, sample_rate)
    st.audio("temp.wav", format="audio/wav")
    os.remove("temp.wav")
