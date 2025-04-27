"""
Author: Jesús Maldonado
Description: This script trains a Deep Learning model for EEG data classification.
"""

from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.EEGDataset import Data, EEGDataset, get_dataloader
from src.cnn1d import CNN1D
from src.stats import Stats
from scripts.config import Config
from scripts.logger_utils import setup_logger

logger = setup_logger(name=Path(__file__).name, level=10)

def train_epoch(model: nn.Module, loader: DataLoader, loss_fn: Callable, optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * y.size(0)
        _, predicted = torch.max(outputs.detach(), 1)
        train_total += y.size(0)
        train_correct += (predicted == y).sum().item()
    train_epoch_loss = train_loss / train_total
    train_epoch_acc = train_correct / train_total
    logger.info(f"Train | Loss: {train_epoch_loss:.4f} | Accuracy: {train_epoch_acc:.4f}")
    return train_epoch_loss, train_epoch_acc
    

def eval_epoch(model: nn.Module, loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]], loss_fn: nn.Module, device: torch.device) -> float:
    y_true = []
    y_pred = []

    model.eval()
    validation_loss = 0.0
    validation_correct = 0
    validation_total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            validation_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            validation_total += y.size(0)
            validation_correct += (predicted == y).sum().item()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return float(accuracy_score(y_true, y_pred))
    

def save_model(model: nn.Module, name: str) -> None:
    Config().MODELS_PATH.mkdir(parents=True, exist_ok=True)
    file = Config().MODELS_PATH / name
    torch.save(model.state_dict(), file)
    logger.info(f"Model saved to {file}")

def train_and_validation(data: pd.DataFrame, models: list[nn.Module], params: dict[str, Any], device: torch.device) -> None:
    logger.info("Starting training and validation models")
    ws: int   = params['window_size']
    fo: int   = params['fold']
    bs: int   = params['batch_size']
    ne: int   = params['n_epochs']
    lr: float = params['lr']
    sh: bool  = params['shuffle']
    nw: int   = params['num_workers']
    pm: bool  = params['pin_memory']

    train_dataset:      EEGDataset = EEGDataset(data, window_size=ws, split='train', fold=fo)
    validation_dataset: EEGDataset = EEGDataset(data, window_size=ws, split='val',   fold=fo)

    train_dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(train_dataset,      batch_size=bs, shuffle=sh,    num_workers=nw, pin_memory=pm)
    val_dataloader:   DataLoader[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(validation_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)
    
    loss_function: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    for model in models:
        logger.info(f"Training model: {model.__class__.__name__}")
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_validation_accuracy = 0.0
        for epoch in range(1, ne + 1):
            logger.info(f"Epoch {epoch}/{ne}")
            logger.info(f"Training")
            train_epoch(model, train_dataloader, loss_function, optimizer, device)
            logger.info(f"Validating")
            validation_accuracy = eval_epoch(model, val_dataloader, loss_function, device)
            logger.info(f"Validation Accuracy: {validation_accuracy:.4f}")
            
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                save_model(model, f"{model.__class__.__name__}_{epoch}.pth")
                

def test_model(data: pd.DataFrame, models: list[nn.Module], params: dict[str, Any], device: torch.device, loss_fn: nn.Module) -> None:
    ws: int   = params['window_size']
    sp: str   = params['split']
    fo: int   = params['fold']
    bs: int   = params['batch_size']
    ne: int   = params['n_epochs']
    lr: float = params['lr']
    sh: bool  = params['shuffle']
    nw: int   = params['num_workers']
    pm: bool  = params['pin_memory']

    test_dataset: EEGDataset = EEGDataset(data, window_size=ws, split='test')

    dataloader_test:  DataLoader[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(test_dataset, batch_size=bs, shuffle=sh if sp == 'train' else False, num_workers=nw, pin_memory=pm)

    for model in models:
        logger.info(f"Testing model: {model.__class__.__name__}")
        model.load_state_dict(torch.load(f"best_model_{model.__class__.__name__}.pth"))
        model.eval()

        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X, y in dataloader_test:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = loss_fn(outputs, y)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += y.size(0)
                test_correct += (predicted == y).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        test_epoch_loss = float(test_loss / len(dataloader_test))
        test_epoch_acc = float(test_correct / test_total)
        test_epoch_f1 = float(f1_score(all_labels, all_preds, average='macro'))
        test_epoch_cm: np.ndarray = confusion_matrix(all_labels, all_preds)
        logger.info(f"Test | Loss: {test_epoch_loss:.4f} | Accuracy: {test_epoch_acc:.4f} | F1 Score: {test_epoch_f1:.4f}")
        logger.info(f"Confusion Matrix:\n{test_epoch_cm}")

def main():
    in_channels:     int       = 20
    num_classes:     int       = 2
    hidden_channels: list[int] = [16, 32, 64]
    kernel_sizes:    list[int] = [5, 3, 2]
    pool_kernel:     int       = 2
    dropout:         float     = 0.5

    data: pd.DataFrame = Data().data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cnn1d: CNN1D = CNN1D(in_channels=in_channels, num_classes=num_classes, hidden_channels=hidden_channels, kernel_sizes=kernel_sizes, pool_kernel=pool_kernel, dropout=dropout).to(device)
    models: list[nn.Module] = [cnn1d]

    params = {
        'window_size': 100,     # Window size in milliseconds.
        'fold':        0,       # Fold index for validation (0, 1, 2).
        'batch_size':  16,      # Batch size: number of samples (museData rows) processed before the model is updated.
        'n_epochs':    5,       # Epochs: number of times the model will see the entire dataset during training.
        'lr':          1e-3,    # Learning rate for the optimizer. It controls how much to change the model in response to the estimated error each time the model weights are updated.
        'shuffle':     True,    # Whether to shuffle the data at every epoch.
        'num_workers': 4,       # Number of subprocesses to use for data loading. More workers = more speed loading data = more memory usage.
        'pin_memory':  False    # Whether to pin memory for faster data transfer to GPU. It is recommended to set this to True if you are using a GPU. If you are using a CPU, set it to False.
    }
    
    train_and_validation(data, models, params, device)

if __name__ == '__main__':
    main()
