"""Training and evaluation routines for ESC-50 1D CNN."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

from src.data import ESC50Dataset
from src.model import get_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataloader(csv_file, audio_folder, folds, batch_size, augment=False):
    dataset = ESC50Dataset(csv_file, audio_folder, folds=folds, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=augment, num_workers=0)


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device):
    model.train()
    losses = []
    all_preds = []
    all_labels = []

    for batch in loader:
        inputs = batch["waveform"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = np.mean(losses) if losses else 0.0
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return avg_loss, acc


def evaluate(model, loader, loss_fn, device):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["waveform"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            losses.append(loss.item())
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = np.mean(losses) if losses else 0.0
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    cm = confusion_matrix(all_labels, all_preds) if all_labels else np.zeros((1, 1), dtype=int)
    return avg_loss, acc, cm


def run_fold(csv_file, audio_folder, val_fold, num_classes=50, batch_size=16, lr=1e-3, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_folds = [f for f in range(1, 6) if f != val_fold]

    train_loader = make_dataloader(csv_file, audio_folder, train_folds, batch_size, augment=True)
    valid_loader = make_dataloader(csv_file, audio_folder, [val_fold], batch_size, augment=False)

    model = get_model(num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-5)

    best_val_acc = 0.0
    best_model = None

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
        val_loss, val_acc, _ = evaluate(model, valid_loader, loss_fn, device)

        print("Fold", val_fold, "Epoch", epoch + 1)
        print("  train_loss", round(train_loss, 4), "train_acc", round(train_acc, 4))
        print("  val_loss", round(val_loss, 4), "val_acc", round(val_acc, 4))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

    return best_val_acc, best_model


def run_cross_validation(csv_file, audio_folder, num_classes=50, batch_size=16, lr=1e-3, epochs=30):
    results = []
    for fold in range(1, 6):
        print("\n=== Fold", fold, "===")
        best_acc, _ = run_fold(csv_file, audio_folder, fold, num_classes, batch_size, lr, epochs)
        results.append(best_acc)
        print("Fold", fold, "best acc", round(best_acc, 4))

    mean_acc = np.mean(results) if results else 0.0
    std_acc = np.std(results) if results else 0.0
    print("\nFinal cross-validation mean acc", round(mean_acc, 4))
    print("Final cross-validation std", round(std_acc, 4))
    return results

