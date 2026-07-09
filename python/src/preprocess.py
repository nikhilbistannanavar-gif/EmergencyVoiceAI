"""
preprocess.py

This script converts audio files into Mel Spectrograms
for training the Emergency Voice AI model.
"""

import os
import librosa
import numpy as np


# Audio settings
SAMPLE_RATE = 16000
DURATION = 1.0          # seconds
N_MELS = 64


def load_audio(file_path):
    """
    Load an audio file.
    """

    audio, sample_rate = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    return audio


def audio_to_melspectrogram(audio):
    """
    Convert waveform to Mel Spectrogram.
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel)

    return mel_db


def process_file(file_path):

    audio = load_audio(file_path)

    mel = audio_to_melspectrogram(audio)

    return mel


def main():

    print("Emergency Voice AI")
    print("Audio preprocessing ready.")


if __name__ == "__main__":
    main()