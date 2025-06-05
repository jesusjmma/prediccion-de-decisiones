"""
Author: Jesús Maldonado
Description: This class is a wrapper for the model architectures used in the EEG classification task.
"""

from pathlib import Path
from typing import Any
from random import randint
import json
import ast

import pandas as pd
import torch
import torch.nn as nn

from cnn1d import CNN1D
from lstm import LSTM
from gru import GRU
from transformer import TransformerEncoder
from scripts.metrics import Metrics
from scripts.config import Config
from scripts.logger_utils import setup_logger
from datetime import datetime

logger = setup_logger(level="DEBUG")

MODELS: dict[str, type[nn.Module]] = {
    "CNN1D": CNN1D,
    "LSTM": LSTM,
    "GRU": GRU,
    "TransformerEncoder": TransformerEncoder
}

PARAMS_KEYS: dict[str, type] = {
    "in_channels": int,
    "hidden_channels": list[int],
    "hidden_size": int,
    "kernel_sizes": list[int],
    "d_model": int,
    "nhead": int,
    "num_layers": int,
    "bidirectional": bool,
    "dim_feedforward": int,
    "dropout_rate": float
}

class Model:
    """
    Wrapper class for model architectures.
    This class manages the model's parameters, training history, and provides methods to save and load models.
    It also maintains a registry of all model instances to allow easy access by ID.
    Attributes:
        __instances (dict[int, "Model"]): A dictionary to store all model instances by their ID.
    Methods:
        __init__(model_type: type[nn.Module] | None = None, params: dict[str, Any] | None = None, path: str | None = None):
            Initializes the Model class with the given model type, parameters, or path to a saved model.
        get_available_models() -> list[int]:
            Returns a list of IDs of all available models.
        get_model(id: int) -> "Model":
            Retrieves a model instance by its ID.
        create_from_csv(file_base_name: str) -> list[int]:
            Loads models from a CSV file and returns their IDs.
    """
    __instances: dict[int, "Model"] = {}
    
    def __init__(self, model_type: type[nn.Module] | None = None, params: dict[str, Any] | None = None, path: str | None = None) -> None:
        """
        Initializes the Model class.
        Args:
            model_type (type[nn.Module] | None): The type of the model to be created. If None, a model will be loaded from the given path.
            params (dict[str, Any] | None): The parameters for the model. If None, the model will be loaded from the given path.
            path (str | None): The path to a saved model file. If None, a new model will be created with the given type and parameters.
        """
        if path is None and model_type is None and params is None:
            return
        
        if path is not None and (model_type is not None or params is not None):
            raise ValueError("If 'path' is provided, 'model_type' and 'params' should not be provided.")

        self.id:            int
        self.version:       int
        self.model_type:    type[nn.Module]
        self.module:        nn.Module
        self.exact_time_before_event: pd.Timedelta
        self.exact_time_after_event: pd.Timedelta
        self.params:        dict[str, Any]
        self.training:      dict[int, int]
        self.history:       dict[int, dict[int, dict[int, int]]]
        self.val_fold:      int
        self.metrics:       dict[str, Metrics]

        if path is not None:
            self._read_model(path)
            return
        
        assert model_type is not None, "Model type must be provided."
        assert params is not None and params != {}, "Model parameters must not be empty."
                
        self._change_id()
        self.version     = 0
        self.model_type  = model_type
        self.params      = params
        self.training    = {}
        self.history     = {self.id: {self.version: {}}}
        self.val_fold    = -1

        self.metrics = {
            "train": Metrics(),
            "val":   Metrics(),
            "test":  Metrics()
        }

        for i in range(Config().TOTAL_FOLDS):
            self.training[i] = 0

        self.module: nn.Module = model_type(**params)
        
        Model.__instances[self.id] = self

    def __repr__(self) -> str:
        return f"Model(id={self.id}, type={self.model_type.__name__}, window_size={self.window_size} (-{self.exact_time_before_event.total_seconds()*1000}, +{self.exact_time_after_event.total_seconds()*1000}), params={self.params}, val_fold={self.val_fold})"

    def __str__ (self) -> str:
        return f"Model ID: {self.id}, Type: {self.model_type.__name__}, Window Size: {self.window_size} (-{self.exact_time_before_event.total_seconds()*1000}, +{self.exact_time_after_event.total_seconds()*1000}), Params: {self.params}, Val Fold: {self.val_fold}"
    
    @classmethod
    def get_available_models(cls) -> list[int]:
        return list(cls.__instances.keys())
    
    @classmethod
    def get_model(cls, id: int) -> "Model":
        if id not in cls.__instances:
            raise ValueError(f"ID {id} not found in DLModel._instances")
        return cls.__instances[id]
    
    @classmethod
    def create_from_csv(cls, file_base_name: str) -> list[int]:
        """ Load models from a CSV file and return their IDs.
        Args:
            file_base_name (str): The base name of the file containing model configurations.
        Returns:
            list[int]: A list of model IDs loaded from the file.
        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        models: list[int] = []
        file = Config().MODELS_PATH / (file_base_name + ".csv")
        if not file.exists():
            logger.error(f"File {file} does not exist.")
            raise FileNotFoundError(f"File {file} does not exist.")

        df = pd.read_csv(file, index_col=0)
        if df.empty:
            logger.warning(f"CSV file {file} is empty.")
            return models
        
        df.columns = df.columns.get_level_values(0)

        def _parse_int_list(s: Any) -> Any:
            if isinstance(s, str) and s.startswith("[") and s.endswith("]"):
                try:
                    return [int(x) for x in ast.literal_eval(s)]
                except (ValueError, SyntaxError):
                    pass
            return s
        
        for col, type_ in PARAMS_KEYS.items():
            if col in df.columns:
                df[col] = df[col].apply(_parse_int_list)

        for idx, row in df.iterrows():
            row = row[row.notna()]
            exact_time_before_event: pd.Timedelta = pd.Timedelta(milliseconds=row["exact_time_before_event"])
            exact_time_after_event: pd.Timedelta = pd.Timedelta(milliseconds=row["exact_time_after_event"])
            val_fold: int       = int(row["val_fold"])
            model_type_str: str = str(row["model_type"])

            params: dict[str, Any] = {str(k): v for k, v in row.items() if k not in ["exact_time_before_event", "exact_time_after_event", "val_fold", "model_type"]}

            for key, value in params.items():
                if key not in PARAMS_KEYS:
                    continue
                expected_type = PARAMS_KEYS[key]
                if expected_type == int:
                    params[key] = int(value)
                elif expected_type == float:
                    params[key] = float(value)
                elif expected_type == bool:
                    params[key] = bool(value)
                elif expected_type == list[int]:
                    continue
                else:
                    raise ValueError(f"Unsupported parameter type for key {key}. Expected: {expected_type}. Got: {type(value)}. Value: {value}")

            assert model_type_str in MODELS, f"Model type {model_type_str} not found in MODELS dictionary."
            model_type: type[nn.Module] = MODELS[model_type_str]

            model = cls(model_type, params)
            model.initialize(exact_time_before_event, exact_time_after_event, val_fold)
            model._change_id(int(str(idx)), overwrite=True)

            models.append(int(str(idx)))
        
        models.sort()
        logger.info(f"Loaded {len(models)} models from {file}")
        return models
      
    @property
    def window_size(self) -> int:
        return int((self.exact_time_after_event + self.exact_time_before_event).total_seconds() * 1000.0)

    def _change_id(self, new_id: int|None = None, overwrite: bool = False) -> None:
        """Generate a new unique ID for the model."""
        if new_id is None:
            new_id = randint(0, 9999999999)
        while new_id in Model.__instances and not overwrite:
            logger.warning(f"ID {new_id} already exists. Generating a new ID.")
            new_id = randint(0, 9999999999)
        self.id = new_id
        Model.__instances[self.id] = self
    
    def _get_model_info(self) -> dict[str, Any]:
        """Get the model information in a json-serializable format."""
        return {
            "id": self.id,
            "model_type": self.model_type.__name__,
            "exact_time_before_event": int(self.exact_time_before_event.total_seconds() * 1000),
            "exact_time_after_event": int(self.exact_time_after_event.total_seconds() * 1000),
            "window_size": self.window_size,
            "params": self.params,
            "val_fold": self.val_fold,
            "training": self.training,
            "version": self.version,
            "history": self.history
        }
    
    def _set_model_info(self, model_info: dict[str, Any]) -> None:
        """Set the model information from a json-serializable format."""
        self.id = model_info["id"]
        self.model_type = MODELS[model_info["model_type"]]
        self.exact_time_before_event = pd.Timedelta(milliseconds=model_info["exact_time_before_event"])
        self.exact_time_after_event = pd.Timedelta(milliseconds=model_info["exact_time_after_event"])
        self.params = model_info["params"]
        self.val_fold = model_info["val_fold"]
        self.training = model_info["training"]
        self.version = model_info["version"]
        self.history = model_info["history"]

        if self.id in Model.__instances:
            logger.warning(f"Model with ID {self.id} already exists. Changing ID of the model to load to a new one.")
            self._change_id()
    
    def save_model(self) -> None:
        """Save the model to a file."""
        folder = Config().MODELS_PATH / datetime.today().strftime("%Y_%m_%d")
        folder.mkdir(parents=True, exist_ok=True)
        file_base_name = str(self.id).zfill(10) + "_" + str(self.version).zfill(4)
        model_extension = ".pth"
        info_extension = ".json"
        metrics_extension = "_metrics.json"

        model_file   = folder / (file_base_name + model_extension)
        info_file    = folder / (file_base_name + info_extension)
        metrics_file = folder / (file_base_name + metrics_extension)

        while model_file.exists() or info_file.exists() or metrics_file.exists():
            subversion   = 1
            model_file   = Config().MODELS_PATH / (file_base_name + "_" + str(subversion).zfill(2) + model_extension)
            info_file    = Config().MODELS_PATH / (file_base_name + "_" + str(subversion).zfill(2) + info_extension)
            metrics_file = Config().MODELS_PATH / (file_base_name + "_" + str(subversion).zfill(2) + metrics_extension)
            subversion  += 1

        torch.save(self.module.state_dict(), model_file)
        self.metrics['val'].to_json(metrics_file.resolve())

        model_info = self._get_model_info()
        model_info["model_file"] = str(model_file)
        model_info["metrics_file"] = str(metrics_file)

        with open(info_file, 'w') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=4)

        if self.id not in self.history:
            self.history[self.id] = {}
        self.history[self.id][self.version] = self.training.copy()
        logger.info(f"Model {self.model_type.__name__} ID \"{self.id}\" v{self.version} saved to {model_file}")
        self.version += 1

    def _read_model(self, model_path: str) -> None:
        """
        Load a model from a .pth file and a .txt file containing model information.
        Args:
            model_file (str): The full path of the model file.
        """
        model_file = Path(model_path)
        info_file = model_file.with_suffix('.txt')

        if not model_file.exists():
            logger.warning(f"Model file {model_file} does not exist.")
            raise FileNotFoundError
        if not info_file.exists():
            logger.warning(f"Info file {info_file} does not exist.")
            raise FileNotFoundError

        with open(info_file, 'r') as f:
            model_info = json.load(f)

        self._set_model_info(model_info)
        self.module = self._load_model(model_file, self.model_type, self.params)

    def _load_model(self, model_path: Path, model_type: type[nn.Module], params: dict[str,Any]) -> nn.Module:
        model = model_type(**params)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Model {model_type.__name__} loaded from {model_path}")

        return model
    
    def initialize(self, exact_time_before_event: pd.Timedelta, exact_time_after_event: pd.Timedelta, val_fold: int = -1) -> None:
        """
        Initialize the model with the given window size and validation fold.
        Args:
            window_size (int): The size of the input window.
            val_fold (int): The validation fold number.
        """
        self.exact_time_before_event = exact_time_before_event
        self.exact_time_after_event = exact_time_after_event
        self.val_fold = val_fold
        logger.debug(f"Model {self.id} initialized with window size {self.window_size} (-{self.exact_time_before_event.total_seconds()*1000}, +{self.exact_time_after_event.total_seconds()*1000}) and validation fold {self.val_fold}")

    def train(self) -> None:
        self.module.train()
        for fold in self.training:
            if fold == self.val_fold:
                continue
            self.training[fold] += 1

    def eval(self) -> None:
        self.module.eval()
        self.training[self.val_fold] += 1

def main():
    # Example usage
    new_models_to_train_file = "new_models_to_train"

    models = Model.create_from_csv(new_models_to_train_file)

    for model_id in models:
        model = Model.get_model(model_id)
        logger.info(f"Model ID: {model.id}, Type: {model.model_type.__name__}, Params: {model.params}, Window Size: {model.window_size} (-{model.exact_time_before_event.total_seconds()*1000}, +{model.exact_time_after_event.total_seconds()*1000}), Val Fold: {model.val_fold}")
    
if __name__ == "__main__":
    main()