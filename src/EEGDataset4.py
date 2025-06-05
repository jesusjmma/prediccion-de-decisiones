from pathlib import Path

import numpy as np
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import torch
from torch import Tensor

from src.EEGData import EEGData, EEGTimeSeries
from scripts.logger_utils import setup_logger

logger = setup_logger("DEBUG")

class EEGDataset(Dataset):
    def __init__(self, eeg_data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]], split: str, validation_fold: int|None=None, windows_size: int = 300) -> None:
        """
        Inicializa el dataset EEGDataset.
        Args:
            eeg_data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]
            split: 'train', 'validation' or 'test'.
            validation_fold: índice del fold de validación para la validación cruzada.
        """
        super().__init__()
        self.samples: List[Tensor] = []
        self.labels:  List[Tensor] = []

        letter_to_num   = {'c': 0, 'd': 1, 'l': 2, 'm': 3, 'n': 4, 'r': 5, 's': 6, 't': 7}
        letter_to_label = {'p': 0, 'q': 1}

        if split in ['train', 'validation']:
            sp = 'train'
        elif split == 'test':
            sp = 'test'
        else:
            raise ValueError(f"Invalid split: {split}")

        data_split: set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]] = eeg_data[sp]
        data:       set[EEGTimeSeries]

        if split == 'test':
            assert validation_fold is None and isinstance(data_split, set), "Fold should be None for test split, and data_split should be a set"
            #isinstance(data_split, dict) and fold is not None:
            data = data_split
        elif split == 'validation':
            assert validation_fold is not None and isinstance(data_split, dict), "Validation fold should be provided for validation split, and data_split should be a dict"
            data = data_split[validation_fold]
        elif split == 'train':
            assert validation_fold is not None and isinstance(data_split, dict), "Validation fold should be provided for train split, and data_split should be a dict"
            data = set()
            splits = list(data_split.keys())
            splits.remove(validation_fold)
            for fold in splits:
                data.update(data_split[fold])
        else:
            raise ValueError(f"Invalid split and fold combination: {split}, {validation_fold}")
        
        for sequence in data:
            df: DataFrame = sequence.eeg.drop(columns=['TimeStamp'])
            
            np_df: np.ndarray = df.to_numpy(dtype=np.float32)

            x_seq: torch.Tensor = torch.tensor(np_df.T, dtype=torch.float32)

            letter_id = letter_to_num[sequence.observed_letter]
            one_hot_encoding = torch.zeros(len(letter_to_num), dtype=torch.float32)
            one_hot_encoding[letter_id] = 1.0
            x_cat = one_hot_encoding.unsqueeze(1).repeat(1, x_seq.shape[1])
            sample = torch.cat((x_seq, x_cat), dim=0)

            key = letter_to_label[sequence.chosen_key]
            label = torch.tensor(key, dtype=torch.long)

            self.samples.append(sample)
            self.labels.append(label)
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Returns the EEG time series and its corresponding class.

        Args:
            idx (int): Index of the EEG time series.

        Returns:
            Tuple[Tensor, Tensor]: Tuple containing the EEG time series and its corresponding class.
        """
        return self.samples[idx], self.labels[idx]
    
def get_dataloader(
        eeg_data: EEGDataset | EEGData | dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]] | None,
        batch_size: int = 16,
        shuffle: bool = False,
        num_workers: int = 4,
        pin_memory: bool = False,
        split: str | None = None,
        fold: int | None = None
    ) -> DataLoader:
    """
    Crea un DataLoader para el EEGDataset.
    Args:
        eeg_data: EEGData o dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]
        batch_size: tamaño del batch.
        shuffle: si se deben mezclar los datos.
        num_workers: número de workers para el DataLoader.
        pin_memory: si se debe usar pin_memory.
        split: 'train' o 'test'.
        fold: índice de la validación cruzada.

    Returns:
        torch.utils.data.DataLoader[Tuple[Tuple[Tensor, Tensor], Tensor]]
    """
    if isinstance(eeg_data, EEGDataset):
        eeg_dataset: EEGDataset = eeg_data
    elif split is not None:
        if isinstance(eeg_data, dict):
            eeg_dataset: EEGDataset = EEGDataset(eeg_data, split=split, validation_fold=fold)
        elif isinstance(eeg_data, EEGData):
            eeg_dataset: EEGDataset = EEGDataset(eeg_data.get_data(), split=split, validation_fold=fold)
        elif eeg_data is None:
            eeg_dataset: EEGDataset = EEGDataset(EEGData.initialize().get_data(), split=split, validation_fold=fold)
        else:
            raise ValueError(f"Invalid eeg_data type: {type(eeg_data)}")
    else:
        raise ValueError(f"If split is None, eeg_data must be an EEGDataset instance")

    return DataLoader(
        eeg_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
        collate_fn=None,
        pin_memory=pin_memory,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory_device='',
        in_order=True
    )

def create_all_dataloaders():
    """
    Crea todos los DataLoaders necesarios para el entrenamiento, validación y prueba.
    Returns:
        dict[str, DataLoader]: Diccionario con los DataLoaders.
    """
    eeg_data: EEGData = EEGData.initialize()
    dataloaders: dict[str, DataLoader] = {
        'train': get_dataloader(eeg_data, split='train', fold=0, batch_size=16, shuffle=True),
        'validation': get_dataloader(eeg_data, split='validation', fold=0, batch_size=16, shuffle=False),
        'test': get_dataloader(eeg_data, split='test', fold=None, batch_size=16, shuffle=False)
    }
    return dataloaders

if __name__ == "__main__":
    create_all_dataloaders()