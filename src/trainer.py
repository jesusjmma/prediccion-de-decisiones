"""
Author: Jesús Maldonado
Description: This module provides a Trainer class to manage the training and evaluation of multiple EEG models.
"""

from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from EEGData import EEGData, EEGTimeSeries
from EEGDataset import EEGDataset, get_dataloader
from model import Model
from scripts.config import Config
from scripts.logger_utils import setup_logger

logger = setup_logger("DEBUG")

class Trainer:
    def __init__(self) -> None:
        # self.models[i] = {'model', 'train_dataloader', 'validation_dataloader', 'loss_fn', 'optimizer', 'device'}
        self.models: dict[int, dict[str, Any]] = {}

    def _save_if_best(self, model_id: int, metrics: dict[str, Any]) -> None:
        """Save the model if it is learning well or if it has a better accuracy than the previous one.
        Args:
            model_id (int): ID of the model to save.
            validation_metrics (dict[str, Any]): Dictionary with validation metrics.
        """
        if 'previous_metrics' not in self.models[model_id] or (self.models[model_id]['previous_metrics']['accuracy'] > metrics['accuracy'] and self.models[model_id]['previous_metrics']['loss'] > metrics['loss']):
            self.models[model_id]['previous_metrics'] = metrics
            self.models[model_id]['best_acc_version'] = self.models[model_id]['model'].version
            self.models[model_id]['best_loss_version'] = self.models[model_id]['model'].version
            self.models[model_id]['model'].save_model()
        elif self.models[model_id]['previous_metrics']['accuracy'] > metrics['accuracy']:
            self.models[model_id]['previous_metrics'] = metrics
            self.models[model_id]['best_acc_version'] = self.models[model_id]['model'].version
            self.models[model_id]['model'].save_model()
        elif self.models[model_id]['previous_metrics']['loss'] > metrics['loss']:
            self.models[model_id]['previous_metrics'] = metrics
            self.models[model_id]['best_loss_version'] = self.models[model_id]['model'].version
            self.models[model_id]['model'].save_model()

    def add_model(self, model: int|Model, training_routine: dict[str, Any]) -> None:
        if isinstance(model, int):
            model = Model.get_model(model)

        routine = training_routine.copy()
        routine['optimizer'] = routine['optimizer'](model.module.parameters())

        new_model: dict[str, Any] = {'model': model}
        new_model.update(routine)

        self.models[model.id] = new_model

    def add_models(self, models: Sequence[int|Model], training_routine: dict[str, Any]) -> None:
        for m in models:
            model = Model.get_model(m) if isinstance(m, int) else m
            self.add_model(model, training_routine)

    def read_model(self, ) -> None:
        pass

    def create_models_from_file(self, models: List[Model]) -> None:
        pass

    def train_model_epoch(self, model_id: int) -> dict[str, Any]:
        model:     Model                 = self.models[model_id]['model']
        dl:        DataLoader            = self.models[model_id]['train_val_dls'][(model.window_size, int(model.exact_time_before_event.total_seconds()*1000), int(model.exact_time_after_event.total_seconds()*1000))][model.val_fold][0]
        loss_fn:   Callable              = self.models[model_id]['loss_fn']
        optimizer: optim.Optimizer       = self.models[model_id]['optimizer']
        device:    torch.device          = self.models[model_id]['device']
        
        model.metrics['train'].reset()
        model.train()
        X: torch.Tensor
        y: torch.Tensor
        for X, y in dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model.module(X).squeeze(1)
            loss = loss_fn(outputs, y.float())
            loss.backward()
            optimizer.step()
            model.metrics['train'].update_metrics(y, outputs, loss.item())
        
        return model.metrics['train'].to_dict

    def eval_model_epoch(self, model_id: int) -> dict[str, Any]:
        model:  Model        = self.models[model_id]['model']
        dl:     DataLoader   = self.models[model_id]['train_val_dls'][(model.window_size, int(model.exact_time_before_event.total_seconds()*1000), int(model.exact_time_after_event.total_seconds()*1000))][model.val_fold][1]
        loss_fn:   Callable  = self.models[model_id]['loss_fn']
        device: torch.device = self.models[model_id]['device']

        model.metrics['val'].reset()
        model.eval()
        X: torch.Tensor
        y: torch.Tensor
        with torch.no_grad():
            for X, y in dl:
                X, y = X.to(device), y.to(device)
                outputs = model.module(X).squeeze(1)
                loss = loss_fn(outputs, y.float())
                model.metrics['val'].update_metrics(y, outputs, loss.item())

        return model.metrics['val'].to_dict
    
    def individual_training(self, model_id: int, epochs: int = 50) -> None:
        """Train a model for a number of epochs.
        Args:
            model_id (int): ID of the model to train.
            epochs (int): Number of epochs to train the model. Defaults to 1.
        """
        for epoch in range(epochs):
            logger.info(f"Training model {model_id} for epoch {epoch + 1}/{epochs}")
            train_metrics = self.train_model_epoch(model_id)
            logger.info(f"[TRAIN] Epoch {epoch + 1}/{epochs} | Loss: {train_metrics['loss']:.4f} | Accuracy: {train_metrics['accuracy']:.4f}")
            validation_metrics = self.eval_model_epoch(model_id)
            logger.info(f"[VALID] Epoch {epoch + 1}/{epochs} | Loss: {validation_metrics['loss']:.4f} | Accuracy: {validation_metrics['accuracy']:.4f}")

            self._save_if_best(model_id, validation_metrics)

    def group_training(self, epochs_for_each_batch: int = 1, total_batches: int = 50) -> None:
        """Train all models for a number of epochs.
        Args:
            epochs (int): Number of epochs to train each model. Defaults to 1.
        """
        for i in range(1, total_batches+1):
            logger.info(f"Training batch {i}/{total_batches}")
            for model_id in self.models:
                logger.info(f"Training model {model_id} for {epochs_for_each_batch} epochs consecutively in this batch (nº {i}/{total_batches}).")
                self.individual_training(model_id, epochs_for_each_batch)
        
        logger.info("Training finished for all models.")
    
    def get_dataloaders(self,
            train_validation_datasets: dict[int, Tuple[EEGDataset, EEGDataset]], 
            test_dataset: EEGDataset,
            params: dict[str, Any]
            ) -> Tuple[dict[int, Tuple[DataLoader, DataLoader]], DataLoader]:
        """
        Creates dataloaders for training and validation datasets.
        Args:
            train_validation_datasets (dict[int, Tuple[EEGDataset, EEGDataset]]): Dictionary with fold number as key and a tuple of train and validation EEGDatasets as value.
            test_dataset (EEGDataset): EEGDataset for testing.
            params (dict[str, Any]): Dictionary with parameters for dataloaders.
        
        Returns:
            DataLoaders (Tuple[dict[int, Tuple[DataLoader, DataLoader]], DataLoader]):
                - **train_validation_dataloaders**: *dict[int, Tuple[DataLoader, DataLoader]]*: Dictionary with fold number as key and a tuple of train and validation DataLoaders as value.
                - **test_dataloader**: *DataLoader*: DataLoader for test dataset.
        """
        bs: int  = params['batch_size']
        sh: bool = params['shuffle']
        nw: int  = params['num_workers']
        pm: bool = params['pin_memory']

        train_validation_dataloaders: dict[int, Tuple[DataLoader, DataLoader]] = {}
        test_dataloader: DataLoader = get_dataloader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm, split='test', fold=None)

        for fold, datasets in train_validation_datasets.items():
            train_dataloader = get_dataloader(datasets[0], batch_size=bs, shuffle=sh, num_workers=nw, pin_memory=pm, split='train', fold=fold)
            validation_dataloader = get_dataloader(datasets[1], batch_size=bs, shuffle=sh, num_workers=nw, pin_memory=pm, split='validation', fold=fold)
            train_validation_dataloaders[fold] = (train_dataloader, validation_dataloader)
        
        return train_validation_dataloaders, test_dataloader

    def get_datasets(self, data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]],
                    folds: int
                    ) -> Tuple[dict[int, Tuple[EEGDataset, EEGDataset]], EEGDataset]:
        """Creates datasets for training and validation.
        Args:
            data (dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]): Dictionary with the data.
            folds (int): Number of folds for cross-validation.
        Returns:
            Datasets (Tuple[dict[int, Tuple[EEGDataset, EEGDataset]], EEGDataset]):
                - **train_validation_datasets**: *dict[int, Tuple[EEGDataset, EEGDataset]]*: Dictionary with fold number as key and a tuple of train and validation EEGDatasets as value.
                - **test_dataset**: *EEGDataset*: EEGDataset for testing.
        """
        train_validation_datasets: dict[int, Tuple[EEGDataset, EEGDataset]] = {}
        test_dataset:    EEGDataset = EEGDataset(data, split='test', validation_fold=None)

        for fold in range(folds):
            train_dataset: EEGDataset = EEGDataset(data, split='train', validation_fold=fold)
            validation_dataset: EEGDataset = EEGDataset(data, split='validation', validation_fold=fold)
            train_validation_datasets[fold] = (train_dataset, validation_dataset)
        
        return train_validation_datasets, test_dataset

    def load_data(self, file: Path | str | None = None) -> dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]:
        """
        Load the data from a file.
        Args:
            file (Path | str | None): Path to the file. If None, use the default path.
        Returns:
            data (dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]): Dictionary with the data.
        """
        if file is None:
            file = Config().EEGDATA_FILE
        elif isinstance(file, str):
            file = Path(file)

        logger.info(f"Loading data from {file}")
        data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]] = EEGData.load_data(file)

        logger.info(f"Data loaded from file {file.stem}")

        return data



def main() -> None:
    dataloader_params: dict[str, Any] = {
        "batch_size":  16,
        "shuffle":     True,
        "num_workers": 4,
        "pin_memory":  False
    }

    trainer = Trainer()
    models: list[int] = Model.create_from_csv("new_models_to_train_03")

    windows: set[tuple[int,int,int]] = set()
    for idx in models:
        windows.add((Model.get_model(idx).window_size,int(Model.get_model(idx).exact_time_before_event.total_seconds()*1000),int(Model.get_model(idx).exact_time_after_event.total_seconds()*1000)))

    eeg_data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]
    train_validation_datasets: dict[int, Tuple[EEGDataset, EEGDataset]]
    test_dataset: EEGDataset
    train_val_dataloaders: dict[tuple[int, int, int], dict[int, Tuple[DataLoader, DataLoader]]] = {}
    test_dataloader: dict[tuple[int, int, int], DataLoader] = {}

    for w in windows:
        eeg_data = trainer.load_data(Config().PROCESSED_FILES_PATH / "windows" / f"eegdata_{w[0]}_-{w[1]}_+{w[2]}_({w[0]//int(Config().SAMPLING_OFFSET)} samples).pkl")
        train_validation_datasets, test_dataset = trainer.get_datasets(eeg_data, folds=3)
        train_val_dataloaders[w], test_dataloader[w] = trainer.get_dataloaders(train_validation_datasets, test_dataset, dataloader_params)

    loss_function: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    optimizer: type[optim.Optimizer] = optim.Adam
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_routine: dict[str, Any] = {
        'train_val_dls': train_val_dataloaders,
        'test_dl':       test_dataloader,
        'loss_fn':       loss_function,
        'optimizer':     optimizer,
        'device':        device
    }

    trainer.add_models(models, training_routine)

    trainer.group_training(epochs_for_each_batch=1, total_batches=60)

    logger.info("Script finished successfully.")

if __name__ == '__main__':
    main()
