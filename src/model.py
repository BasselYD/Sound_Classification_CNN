"""Model definitions for the ESC-50 1D CNN.

This file contains a simple 1D CNN for raw audio input. The model is
kept relatively small so it can train on CPU without blowing memory.
"""

import torch
import torch.nn as nn


class SoundCNN1D(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(5),

            nn.Conv1d(128, 256, kernel_size=9, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(5),
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = x.mean(dim=-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def get_model(num_classes=50):
    """Create a model instance."""
    return SoundCNN1D(num_classes=num_classes)

