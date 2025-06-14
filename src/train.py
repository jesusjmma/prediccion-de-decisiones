"""
Author: Jesús Maldonado
Description: This script trains a Deep Learning model for EEG data classification.
"""

from pathlib import Path
import time
from typing import Any, Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from EEGData import EEGData, EEGTimeSeries
from EEGDataset import EEGDataset, get_dataloader
from cnn1d import CNN1D
from lstm import LSTM
from gru import GRU
from transformer import TransformerEncoder
from scripts.metrics import Metrics
from trainer import Trainer
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

def test_models(test_dataloader: DataLoader,
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

def save_model(model: nn.Module,
               file_base_name: Path | str
               ) -> None:
    """Save the model to a file.
    Args:
        model (nn.Module): The model to save.
        name (str): The name of the file to save the model to.
    """
    if isinstance(file_base_name, str):
        file_base_name = Path(file_base_name)
    
    Config().MODELS_PATH.mkdir(parents=True, exist_ok=True)
    file = Config().MODELS_PATH / file_base_name
    if file.exists():
        file = Config().MODELS_PATH / f"{file_base_name.stem}_{time.strftime('%Y_%m_%d_%H_%M_%S')}{file_base_name.suffix}"

    torch.save(model.state_dict(), file)
    logger.info(f"Model {model.__class__.__name__} saved to {file}")

def eval_epoch(model: nn.Module,
               loader: DataLoader,
               device: torch.device
               ) -> float:
    """Evaluate the model on the validation fold.
    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The dataloader for the validation data.
        device (torch.device): The device to use for evaluation.
    Returns:
        accuracy (float): The accuracy of the model on the validation fold.
    """
    start_time: float = time.time()
    logger.info("Evaluating model")
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze(1)
            probs = torch.sigmoid(outputs)
            predicted = (probs >= 0.5).long()
            y_true.extend(y.float().cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    end_time: float = time.time()
    logger.info(f"Time for evaluation epoch: {end_time - start_time:.3f} seconds")
    return float(accuracy_score(y_true, y_pred))

def train_epoch(model: nn.Module,
                loader: DataLoader,
                loss_fn: Callable,
                optimizer: optim.Optimizer,
                device: torch.device
                ) -> Tuple[float, float]:
    """Train the model for one epoch.
    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): The dataloader for the training data.
        loss_fn (Callable): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device): The device to use for training.
    Returns:
        (train_epoch_loss, train_epoch_acc) (Tuple[float, float]): The training loss and accuracy.
    """
    start_time: float = time.time()
    logger.info("Training model")
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X).squeeze(1)
        loss = loss_fn(outputs, y.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * y.size(0)
        probs = torch.sigmoid(outputs)
        predicted = (probs >= 0.5).long()
        y: torch.Tensor
        model.metrics["train"].update_metrics(y, outputs, loss.item())
        train_total += y.size(0)
        train_correct += (predicted == y).sum().item()
    train_epoch_loss = train_loss / train_total
    train_epoch_acc = train_correct / train_total
    logger.info(f"Train | Loss: {train_epoch_loss:.4f} | Accuracy: {train_epoch_acc:.4f}")
    end_time: float = time.time()
    logger.info(f"Time for training epoch: {end_time - start_time:.3f} seconds")
    return train_epoch_loss, train_epoch_acc

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

def train_and_validation(train_val_dataloaders: dict[int, Tuple[DataLoader, DataLoader]],
                         models: list[type[nn.Module]],
                         models_params: dict[type[nn.Module], dict[str, Any]],
                         saved_models: dict[type[nn.Module], Path],
                         params: dict[str, Any],
                         device: torch.device,
                         loss_function: nn.Module
                         ) -> dict[type[nn.Module], dict[int, nn.Module]]:
    """Train and validate the model.
    Args:
        train_val_dataloaders (dict[int, Tuple[DataLoader, DataLoader]]): Dictionary with fold number as key and a tuple of train and validation DataLoaders as value.
        models (list[nn.Module]): List of models to train.
        params (dict[str, Any]): Dictionary with parameters for training.
        device (torch.device): Device to use for training.
        loss_function (nn.Module): Loss function to use for training.
    Returns:
        trained_models (dict[type[nn.Module], dict[int, nn.Module]]): Dictionary with model type as key and a dictionary with fold number as key and the trained model as value.
    """
    logger.info("Starting training and validation models")

    ne: int   = params['n_epochs']
    lr: float = params['lr']

    trained_models: dict[type[nn.Module], dict[int, nn.Module]] = {}
    
    for model_type in models:
        start_model_time: float = time.time()
        trained_models[model_type] = {}
        logger.info(f"Starting processing for model: {model_type.__name__}")

        params = get_params(models_params, model_type)
    
        for fold, (train_dl, val_dl) in train_val_dataloaders.items():
            logger.info(f"Fold {fold}/{len(train_val_dataloaders)}")

            model_fold: nn.Module | None = load_model(model_type, saved_models, models_params, device)

            if model_fold is None:
                logger.info(f"Creating model {model_type.__name__}")
                model_fold = model_type(**params)
                logger.info(f"Model {model_fold.__class__.__name__} created for fold {fold}")

            model_fold = model_fold.to(device)

            optimizer = optim.Adam(model_fold.parameters(), lr=lr)

            trained_models[model_type][fold] = model_fold
            
            best_validation_accuracy_for_this_fold = 0.0
            for epoch in range(1, ne + 1):
                start_time: float = time.time()
                logger.info(f"Epoch {epoch}/{ne}")
                logger.info(f"Training")
                train_epoch(model_fold, train_dl, loss_function, optimizer, device)
                logger.info(f"Validating")
                val_accuracy = eval_epoch(model_fold, val_dl, device)
                logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
                
                if val_accuracy > best_validation_accuracy_for_this_fold:
                    best_validation_accuracy_for_this_fold = val_accuracy
                    save_model(model_fold, f"{model_fold.__class__.__name__}_{get_params(models_params, model_type)}_ValFold{fold}_Epoch{epoch}_Acc{val_accuracy}.pth")
                
                end_time: float = time.time()
                logger.info(f"Time for the full epoch ({epoch}): {end_time - start_time:.3f} seconds")
        
        end_model_time: float = time.time()
        logger.info(f"Total time for processing model {model_type.__name__}: {int((end_model_time - start_model_time) // 3600)}:{int(((end_model_time - start_model_time) % 3600) // 60)}:{int((end_model_time - start_model_time) % 60)}")
    return trained_models

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

    train_val_dl: dict[int, Tuple[DataLoader, DataLoader]] = {}
    test_dl: DataLoader = get_dataloader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm, split='test', fold=None)

    for fold, datasets in train_validation_datasets.items():
        train_dataloader = get_dataloader(datasets[0], batch_size=bs, shuffle=sh, num_workers=nw, pin_memory=pm, split='train', fold=fold)
        validation_dataloader = get_dataloader(datasets[1], batch_size=bs, shuffle=sh, num_workers=nw, pin_memory=pm, split='validation', fold=fold)
        train_val_dl[fold] = (train_dataloader, validation_dataloader)
    
    return train_val_dl, test_dl

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

def train_and_test(eeg_data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]],
                   models: list[type[nn.Module]],
                   models_params: dict[type[nn.Module], dict[str, Any]],
                   saved_models: dict[type[nn.Module], Path],
                   params: dict[str, Any],
                   device: torch.device,
                   loss_function: nn.Module = nn.BCEWithLogitsLoss()
                   ) -> None:
    """
    Train and test the models.
    Args:
        eeg_data (dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]): Dictionary with the data.
        models (list[nn.Module]): List of models to train.
        params (dict[str, Any]): Dictionary with parameters for training.
        device (torch.device): Device to use for training.
    """
    logger.info("Starting training and testing process")
    folds: int = params['folds']
    start_time: float = time.time()
    train_validation_datasets: dict[int, Tuple[EEGDataset, EEGDataset]]
    test_dataset: EEGDataset
    train_validation_datasets, test_dataset = get_datasets(eeg_data, folds=folds)
    end_time: float = time.time()
    logger.info(f"Time to create datasets: {end_time - start_time:.3f} seconds")

    start_time: float = time.time()
    train_val_dataloaders: dict[int, Tuple[DataLoader, DataLoader]]
    test_dataloader: DataLoader
    train_val_dataloaders, test_dataloader = get_dataloaders(train_validation_datasets, test_dataset, params)
    end_time: float = time.time()
    logger.info(f"Time to create dataloaders: {end_time - start_time:.3f} seconds")

    trained_models: dict[type[nn.Module], dict[int, nn.Module]] = train_and_validation(train_val_dataloaders, models, models_params, saved_models, params, device, loss_function)

    stats: dict[type[nn.Module], dict[int, dict[str, float | np.ndarray]]] = test_models(test_dataloader, trained_models, device, loss_function)

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

def save_data(data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]],
              file: Path | str | None = None
              ) -> None:
    """
    Save the data to a file.
    Args:
        data (dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]): Dictionary with the data.
    """
    if file is None:
        file = Config().EEGDATA_FILE
    elif isinstance(file, str):
        file = Path(file)

    logger.info(f"Saving data to {file}")
    final_file = EEGData.save_data(file, data)
    logger.info(f"Data saved to {final_file}")

def extract_data(eeg_data: EEGData) -> dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]:
    """
    Extract the data from the EEGData object.
    Args:
        eeg_data (EEGData): EEGData object.
    Returns:
        data (dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]): Dictionary with the data.
    """
    logger.info("Extracting data")
    data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]] = eeg_data.get_data()
    logger.info("Data extracted")

    return data

def create_data() -> EEGData:
    """
    Create the data for the model.
    Returns:
        eeg_data (EEGData): EEGData object.
    """
    logger.info("Creating data")
    eeg_data: EEGData = EEGData.initialize()
    logger.info("Data created")
    
    return eeg_data

def main() -> None:
    new_data: bool = False
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
        #LSTM: {
        #    'in_channels': 28,
        #    'hidden_size': 64,
        #    'num_layers': 1,
        #    'bidirectional': False,
        #    'dropout_rate': 0.2
        #},
        #GRU: {
        #    'in_channels': 28,
        #    'hidden_size': 64,
        #    'num_layers': 1,
        #    'bidirectional': False,
        #    'dropout_rate': 0.2
        #},
        TransformerEncoder: {
            'in_channels': 28,
            'd_model': 128,
            'nhead': 8,
            'num_layers': 1,
            'dim_feedforward': 256,
            'dropout_rate': 0.2
        }
    }

    models: list[type[nn.Module]] = [
        #CNN1D,
        #LSTM,
        #GRU,
        TransformerEncoder
        ]
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eeg_data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]
    if new_data:
        data: EEGData = create_data()  # Create the EEGData object
        eeg_data = extract_data(data)  # Extract the data from the EEGData object
        save_data(eeg_data)  # Save the data to a file
    else:
        eeg_data = load_data(Config().ROOT_PATH / "eegdata_2025_05_04_22_27_03.pkl")  # Load the data from a file
    loss_function: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    saved_models: dict[type[nn.Module], Path] = {}
    '''
    saved_models = {
        CNN1D: Config().MODELS_PATH / Path("CNN1D_ValFold0_Epoch132_Acc0.5582137161084529.pth")
    }
    #'''
    
    train_and_test(eeg_data, models, models_params, saved_models, params, device, loss_function)

if __name__ == '__main__':
    main()
