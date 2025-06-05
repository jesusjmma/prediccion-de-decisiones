"""
Author: Jesús Maldonado
Description: This script trains a Deep Learning model for EEG data classification.
"""

from pathlib import Path
import time
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

from EEGData import EEGData, EEGTimeSeries
from EEGDataset import EEGDataset, get_dataloader
from cnn1d import CNN1D
from lstm import LSTM
from gru import GRU
from transformer import TransformerEncoder
from model import Model
from scripts.config import Config
from scripts.logger_utils import setup_logger

logger = setup_logger("INFO")

def get_params(models_params: dict[type[nn.Module], dict[str, Any]],
               model_type: type[nn.Module]
               ) -> dict[str, Any]:
    """
    Get the parameters for the model.
    Args:
        models_params (dict[type[nn.Module], dict[str, Any]]): Dictionary with model parameters.
        model_type (type[nn.Module]): Model type.
    Returns:
        params (dict[str, Any]): Dictionary with model parameters.
    """
    if model_type not in models_params:
        logger.warning(f"Model type {model_type} not found in model_params.")
        return {}
    
    params = {}
    if model_type == CNN1D:
        params = {
            'in_channels': models_params[model_type]['in_channels'],
            'hidden_channels': models_params[model_type]['hidden_channels'],
            'kernel_sizes': models_params[model_type]['kernel_sizes'],
            'dropout_rate': models_params[model_type]['dropout_rate']
        }
    elif model_type == LSTM:
        params = {
            'in_channels': models_params[model_type]['in_channels'],
            'hidden_size': models_params[model_type]['hidden_size'],
            'num_layers': models_params[model_type]['num_layers'],
            'bidirectional': models_params[model_type]['bidirectional'],
            'dropout_rate': models_params[model_type]['dropout_rate']
        }
    elif model_type == GRU:
        params = {
            'in_channels': models_params[model_type]['in_channels'],
            'hidden_size': models_params[model_type]['hidden_size'],
            'num_layers': models_params[model_type]['num_layers'],
            'bidirectional': models_params[model_type]['bidirectional'],
            'dropout_rate': models_params[model_type]['dropout_rate']
        }
    elif model_type == TransformerEncoder:
        params = {
            'in_channels': models_params[model_type]['in_channels'],
            'd_model': models_params[model_type]['d_model'],
            'nhead': models_params[model_type]['nhead'],
            'num_layers': models_params[model_type]['num_layers'],
            'dim_feedforward': models_params[model_type]['dim_feedforward'],
            'dropout_rate': models_params[model_type]['dropout_rate']
        }
    
    return params

def load_models(test_dataloader: DataLoader,
                models: dict[type[nn.Module], dict[int, nn.Module]],
                device: torch.device,
                loss_fn: nn.Module
                ) -> dict[type[nn.Module], dict[int, dict[str, float | np.ndarray]]]:
    """Test the models on the test dataset.
    Args:
        test_dataloader (DataLoader): The dataloader for the test data.
        models (dict[type[nn.Module], dict[int, nn.Module]]): List of models to test.
        device (torch.device): The device to use for testing.
        loss_fn (nn.Module): The loss function.
    Returns:
        stats (dict[type[nn.Module], dict[int, dict[str, float | np.ndarray]]]): Dictionary with model type as key and a dictionary with fold number as key and a dictionary with model as key and a dictionary with statistics name as key and statistics value as value.
    """
    stats: dict[type[nn.Module], dict[int, dict[str, float | np.ndarray]]] = {}

    for model_type, model_dict in models.items():
        stats[model_type] = {}
        for fold, model in model_dict.items():
            logger.info(f"Testing model: {model.__class__.__name__}")
            model.eval()

            test_loss = 0.0
            test_correct = 0
            test_total = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for X, y in test_dataloader:
                    X, y = X.to(device), y.to(device)
                    outputs = model(X).squeeze(1)
                    loss = loss_fn(outputs, y.float())

                    test_loss += loss.item() * y.size(0) #TODO Revisar esta (y todas) las estadísticas
                    probs = torch.sigmoid(outputs)
                    predicted = (probs >= 0.5).long()
                    test_total += y.size(0)
                    test_correct += (predicted == y).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
            test_epoch_loss = float(test_loss / test_total) #TODO Revisar esta (y todas) las estadísticas
            test_epoch_acc = float(test_correct / test_total)
            test_epoch_f1 = float(f1_score(all_labels, all_preds, average='macro'))
            test_epoch_cm: np.ndarray = confusion_matrix(all_labels, all_preds)
            stats[model_type][fold] = {
                'loss': test_epoch_loss,
                'accuracy': test_epoch_acc,
                'f1_score': test_epoch_f1,
                'confusion_matrix': test_epoch_cm}
            logger.info(f"Test | Loss: {test_epoch_loss:.4f} | Accuracy: {test_epoch_acc:.4f} | F1 Score: {test_epoch_f1:.4f}")
            logger.info(f"Confusion Matrix:\n{test_epoch_cm}")

    return stats

def load_and_test_models(eeg_data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]],
                         saved_models: list[tuple[type[nn.Module], Path]],
                         models_params: dict[type[nn.Module], dict[str, Any]],
                         device: torch.device,
                         params: dict[str, Any],
                         loss_function: nn.Module = nn.BCEWithLogitsLoss()
                         ) -> None:
    
    logger.info("Starting training and testing process")
    folds: int = params['folds']
    start_time: float = time.time()
    train_validation_datasets: dict[int, Tuple[EEGDataset, EEGDataset]]
    test_dataset: EEGDataset
    train_validation_datasets, test_dataset = get_datasets(eeg_data, folds=folds)
    end_time: float = time.time()
    logger.info(f"Time to create datasets: {end_time - start_time:.3f} seconds")

    start_time: float = time.time()
    test_dataloader: DataLoader
    _, test_dataloader = get_dataloaders(train_validation_datasets, test_dataset, params)
    end_time: float = time.time()
    logger.info(f"Time to create dataloaders: {end_time - start_time:.3f} seconds")

    trained_models: list[dict[type[nn.Module], dict[int, nn.Module]]] = []

    print (f"Saved models: {saved_models}")
    for model_type, model_file in saved_models:
        if not model_file.exists():
            logger.warning(f"Model file {model_file} does not exist. Training from scratch.")
            continue
        
        model = load_model(model_type, {model_type: model_file}, models_params, device)
        if model is None:
            logger.warning(f"Model {model_type.__name__} not found.")
            continue
        model = model.to(device)

        trained_models.append({model_type: {0: model}})

    stats: dict[type[nn.Module], dict[int, dict[str, float | np.ndarray]]]

    for trained_model in trained_models:
        stats = load_models(test_dataloader, trained_model, device, loss_function)
        logger.info(stats)

def load_model(model_type: type[nn.Module],
               saved_models: dict[type[nn.Module], Path],
               models_params: dict[type[nn.Module], dict[str, Any]],
               device: torch.device
               ) -> nn.Module | None:
    if saved_models.get(model_type) is None:
        logger.warning(f"Model type {model_type} not found in saved models.")
        return None
    
    if models_params.get(model_type) is None:
        logger.warning(f"Model type {model_type} not found in model parameters.")
        return None

    model_file: Path = Config().MODELS_PATH / saved_models[model_type]
    if not model_file.exists():
        logger.warning(f"Model file {model_file} does not exist. Training from scratch.")
        return None
    
    logger.info(f"Loading model {model_type.__name__} from {model_file}")
    params = get_params(models_params, model_type)
    model = model_type(**params)
    model.load_state_dict(torch.load(model_file, map_location=device))
    logger.info(f"Model {model_type.__name__} loaded from {model_file}")

    return model                

def get_dataloaders(
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

def get_datasets(data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]],
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

def load_data(file: Path | str | None = None) -> dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]:
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

    # Comprobar que cada instancia de EEGTimeSeries tiene eeg de longitud exactamente 300 
    for key, value in data.items():
        if isinstance(value, dict):
            for _, sub_value in value.items():
                for instance in sub_value:
                    if len(instance.eeg) != 300:
                        logger.warning(f"Instance {instance.subject}.{instance.trial}.{instance.response} has eeg of length {len(instance.eeg)}")
        else:
            for instance in value:
                if len(instance.eeg) != 300:
                    logger.warning(f"Instance {instance.subject}.{instance.trial}.{instance.response} has eeg of length {len(instance.eeg)}")

    logger.info("Data loaded")

    return data

def main() -> None:
    params = {
        'folds':       3,       # Number of folds for cross-validation.
        'batch_size':  16,      # Batch size: number of samples (museData rows) processed before the model is updated.
        'n_epochs':    100,   # Epochs: number of times the model will see the entire dataset during training.
        'lr':          1e-3,    # Learning rate for the optimizer. It controls how much to change the model in response to the estimated error each time the model weights are updated.
        'shuffle':     True,    # Whether to shuffle the data at every epoch.
        'num_workers': 4,       # Number of subprocesses to use for data loading. More workers = more speed loading data = more memory usage.
        'pin_memory':  False    # Whether to pin memory for faster data transfer to GPU. It is recommended to set this to True if you are using a GPU. If you are using a CPU, set it to False.
    }
    models_params: dict[type[nn.Module], dict[str, Any]] = {
        #CNN1D: {
        #    'in_channels':     28,
        #    'hidden_channels': [16, 32],
        #    'kernel_sizes':    [5, 3],
        #    'dropout_rate':    0.5
        #},
        LSTM: {
            'in_channels': 28,
            'hidden_size': 64,
            'num_layers': 1,
            'bidirectional': False,
            'dropout_rate': 0.2
        }
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eeg_data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]] = load_data(Config().ROOT_PATH / "eegdata_2025_05_04_22_27_03.pkl")  # Load the data from a file
    loss_function: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    saved_models_test: list[tuple[type[nn.Module], str]] = [
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold0_Epoch86_Acc0.5263157894736842"),
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold0_Epoch27_Acc0.5247208931419458"),
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold0_Epoch7_Acc0.5151515151515151"),
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold0_Epoch4_Acc0.507177033492823  "),
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold0_Epoch1_Acc0.48484848484848486"),
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold1_Epoch68_Acc0.5709728867623605"),
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold1_Epoch66_Acc0.569377990430622"),
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold1_Epoch62_Acc0.5598086124401914"),
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold1_Epoch18_Acc0.543859649122807"),
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold1_Epoch2_Acc0.5406698564593302"),
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold1_Epoch1_Acc0.5247208931419458"),
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold2_Epoch36_Acc0.5357710651828299"),
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold2_Epoch36_Acc0.5357710651828299"),
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold2_Epoch26_Acc0.5325914149443561"),
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold2_Epoch20_Acc0.5278219395866455"),
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold2_Epoch4_Acc0.5262321144674086"),
        (LSTM, "LSTM_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold2_Epoch1_Acc0.5246422893481717"),
    ]

    saved_models_to_test: list[tuple[type[nn.Module], Path]] = []
    for model_type, model_file in saved_models_test:
        saved_models_to_test.append((model_type, Config().MODELS_PATH / Path(model_file+".pth")))
    
    load_and_test_models(eeg_data, saved_models_to_test, models_params, device, params, loss_function)
    return

















def print_stats(stats: dict[int, dict[str, Any]]) -> None:
    """
    Print the statistics of the models.
    Args:
        stats (dict[int, dict[str, Any]]): Dictionary with model id as key and a dictionary with statistics as value.
    """
    for model_id, model_stats in stats.items():
        logger.info(f"Model {model_id}:")
        for stat_name, stat_value in model_stats.items():
            logger.info(f"{stat_name}: {stat_value}")

def load_data2(test_data_file: str, params: dict[str, Any]) -> DataLoader:
    eeg_data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]] = load_data(Config().ROOT_PATH / test_data_file)

    train_validation_datasets, test_dataset = get_datasets(eeg_data, folds=int(Config().TOTAL_FOLDS))

    test_dataloader: DataLoader
    _, test_dataloader = get_dataloaders(train_validation_datasets, test_dataset, params)

    return test_dataloader

def load_models2(models_files: list[str] ) -> list[int]:
    for file in models_files:
        model = Model(path=file)
        logger.info(f"Model loaded: {model}")

    models: list[int] = Model.get_available_models()

    return models

def test_model2(m: Model,
                test_dataloader: DataLoader,
                loss_fn: nn.Module = nn.BCEWithLogitsLoss()
               ) -> dict[str, Any]:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Testing model: {m}")
    model = m.module
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze(1)
            loss = loss_fn(outputs, y.float())

            test_loss += loss.item() * y.size(0) #TODO Revisar esta (y todas) las estadísticas
            probs = torch.sigmoid(outputs)
            predicted = (probs >= 0.5).long()
            test_total += y.size(0)
            test_correct += (predicted == y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    test_epoch_loss = float(test_loss / test_total) #TODO Revisar esta (y todas) las estadísticas
    test_epoch_acc = float(test_correct / test_total)
    test_epoch_f1 = float(f1_score(all_labels, all_preds, average='macro'))
    test_epoch_cm: np.ndarray = confusion_matrix(all_labels, all_preds)
    stats = {
        'loss': test_epoch_loss,
        'accuracy': test_epoch_acc,
        'f1_score': test_epoch_f1,
        'confusion_matrix': test_epoch_cm
    }
    logger.info(f"Test | Loss: {test_epoch_loss:.4f} | Accuracy: {test_epoch_acc:.4f} | F1 Score: {test_epoch_f1:.4f}")
    logger.info(f"Confusion Matrix:\n{test_epoch_cm}")

    return stats

def test_models2(models: list[int],
                 test_dataloader: DataLoader,
                 loss_function: nn.Module = nn.BCEWithLogitsLoss()
                ) -> dict[int, dict[str, Any]]:
    
    stats: dict[int, dict[str, Any]] = {}

    for m in models:
        start_time: float = time.time()
        model = Model.get_model(m)
        stats[m] = {}
        stats[m] = test_model2(model, test_dataloader, loss_function)
        end_time: float = time.time()
        logger.info(f"Time to test model {model.id}: {end_time - start_time:.3f} seconds")

    return stats

def main2() -> None:
    params = {
        'batch_size':  16,      # Batch size: number of samples (museData rows) processed before the model is updated.
        'shuffle':     True,    # Whether to shuffle the data at every epoch.
        'num_workers': 4,       # Number of subprocesses to use for data loading. More workers = more speed loading data = more memory usage.
        'pin_memory':  False    # Whether to pin memory for faster data transfer to GPU. It is recommended to set this to True if you are using a GPU. If you are using a CPU, set it to False.
    }

    test_data_file = "eegdata_2025_05_04_22_27_03.pkl"
    models_files = [
        "LSTM_200011_{'in_channels': 28, 'hidden_size': 64, 'num_layers': 1, 'bidirectional': False, 'dropout_rate': 0.2}_ValFold1_Epoch68_Acc0.5709728867623605",
        "TransformerEncoder_400003_{'in_channels': 28, 'd_model': 128, 'nhead': 8, 'num_layers': 1, 'dim_feedforward': 256, 'dropout_rate': 0.2}_ValFold0_Epoch4_Acc0.5311004784688995"
    ]

    test_dataloader = load_data2(test_data_file, params)
    models = load_models2(models_files)
    models_stats = test_models2(models, test_dataloader)
    print_stats(models_stats)

    return

if __name__ == '__main__':
    main2()
