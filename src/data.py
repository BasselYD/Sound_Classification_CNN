"""Data loading and preprocessing for ESC-50.

The ESC-50 folder has two important parts:
- `audio/` with WAV files
- `meta/esc50.csv` with labels and fold numbers

This file loads waveforms, converts them to a fixed length, normalizes
amplitude, and supports basic data augmentation.
"""

import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TARGET_SR = 16000
TARGET_SECONDS = 5
TARGET_LENGTH = TARGET_SR * TARGET_SECONDS


def load_audio(path, sr=TARGET_SR):
    path = Path(path)
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio.astype(np.float32)


def fix_length(audio, length=TARGET_LENGTH):
    if len(audio) < length:
        pad = length - len(audio)
        audio = np.pad(audio, (0, pad), mode="constant")
    elif len(audio) > length:
        audio = audio[:length]
    return audio


def normalize(audio):
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return audio / peak


def random_shift(audio, max_shift=0.1):
    if max_shift <= 0:
        return audio
    shift_amount = int(len(audio) * max_shift)
    shift = random.randint(-shift_amount, shift_amount)
    return np.roll(audio, shift)


def random_gain(audio, min_gain=0.7, max_gain=1.3):
    gain = random.uniform(min_gain, max_gain)
    return audio * gain


class ESC50Dataset(Dataset):
    def __init__(self, csv_file, audio_folder, folds=None, augment=False, transform=None):
        self.csv_file = Path(csv_file)
        self.audio_folder = Path(audio_folder)
        self.augment = augment
        self.transform = transform

        self.meta = pd.read_csv(self.csv_file)
        if "filename" not in self.meta.columns or "target" not in self.meta.columns:
            raise ValueError("ESC-50 CSV needs 'filename' and 'target' columns")

        if folds is not None:
            self.meta = self.meta[self.meta["fold"].isin(folds)].reset_index(drop=True)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        path = self.audio_folder / row["filename"]

        audio = load_audio(path)
        audio = fix_length(audio)
        audio = normalize(audio)

        if self.augment:
            audio = random_shift(audio)
            audio = random_gain(audio)
            audio = normalize(audio)

        audio = torch.from_numpy(audio).float().unsqueeze(0)
        if self.transform is not None:
            audio = self.transform(audio)

        sample = {
            "waveform": audio,
            "label": int(row["target"]),
            "filename": row["filename"],
            "category": row.get("category"),
            "esc10": bool(row.get("esc10", False)),
            "src_file": row.get("src_file"),
            "take": row.get("take"),
        }

        if "fold" in row:
            sample["fold"] = int(row["fold"])

        return sample


def build_fold_dataset(csv_file, audio_folder, fold, augment=False, transform=None):
    return ESC50Dataset(csv_file, audio_folder, folds=[fold], augment=augment, transform=transform)

