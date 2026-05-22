# Sound Classification with a 1D CNN

A simple ESC-50 audio classification project using raw waveform input and a 1D convolutional network built with PyTorch.

## What this project does

- loads ESC-50 audio data from `ESC-50-master/audio`
- reads labels from `ESC-50-master/meta/esc50.csv`
- applies audio preprocessing and augmentation
- trains a small 1D CNN on raw waveforms
- supports fold-based cross-validation for evaluation

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the ESC-50 dataset is in the project folder as:

- `ESC-50-master/audio/`
- `ESC-50-master/meta/esc50.csv`

If you already have the dataset in another place, update the file paths when you run the code.

## How to run

There is no command-line entry script, so run the training functions directly from Python.

Example:

```python
from src.train import run_fold

run_fold(
    csv_file="ESC-50-master/meta/esc50.csv",
    audio_folder="ESC-50-master/audio",
    val_fold=1,
    num_classes=50,
    batch_size=16,
    lr=1e-3,
    epochs=30,
)
```

To run full cross-validation:

```python
from src.train import run_cross_validation

run_cross_validation(
    csv_file="ESC-50-master/meta/esc50.csv",
    audio_folder="ESC-50-master/audio",
    num_classes=50,
    batch_size=16,
    lr=1e-3,
    epochs=30,
)
```

Alternatively, you can also use `ESC50_1D_CNN.ipynb` for interactive experiments and visualization.

## Notes

- The model uses CPU by default, but it will use CUDA if available.
- `src/data.py` handles audio loading, fixed-length padding, normalization, and augmentation.
- `src/model.py` defines the 1D CNN.
- `src/train.py` contains training, evaluation, and cross-validation routines.
