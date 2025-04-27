"""
Author: Jesús Maldonado
Description: This script loads the EEG dataset for training and testing Deep Learning models.
"""

# 1. Standard library imports
from dataclasses import dataclass, field
import gc
from pathlib import Path
from time import sleep
from typing import List, Tuple, Optional

# 2. Third-party library imports
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 3. Local application imports
from scripts.config import Config
from scripts.logger_utils import setup_logger

# Logger setup
logger = setup_logger(name=Path(__file__).name, level=10)

@dataclass(slots=True)
class Data:
    data: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)

    def __post_init__(self):
        # Load all the CSV files in PROCESSED_MUSEDATA_PATH
        data_path = Config().PROCESSED_MUSEDATA_PATH

        for muse_file in sorted(data_path.glob('museData*.csv'), key=lambda f: int(f.stem.replace('museData', ''))):
            sid = int(muse_file.stem.replace('museData', ''))
            logger.info(f"Loading Subject {sid} from  file {muse_file}")
            # Load processed muse data
            df: pd.DataFrame = pd.read_csv(muse_file, parse_dates=True)
            # Load the corresponding processed results (to get the true labels)
            results_file = Config().PROCESSED_RESULTS_PATH / f'results{sid}.csv'
            results: pd.DataFrame = pd.read_csv(results_file, parse_dates=True)

            # Merge to attach 'Tecla elegida' as label
            df = df.merge(
                results[['Trial', 'Respuesta', 'Tecla elegida']],
                on=['Trial', 'Respuesta'], how='left'
            )
            df['Subject'] = sid
            self.data = pd.concat([self.data, df], ignore_index=True)
        
        self.data.rename(columns={'Tecla elegida': 'label'}, inplace=True)

class EEGDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):   
    """PyTorch Dataset for loading EEG decision-prediction data.
    Each sample is a time-series window of shape (channels, time_steps) and a label (0 or 1).
    """

    def __init__(self, data: pd.DataFrame, window_size: int, split: str = 'train', fold: int = 0, subject_ids: Optional[List[int]] = None):
        """
        Args:
            window_size (int): Sliding window length in milliseconds (e.g., 100, 250, ...).
            split (str): Split type ('train', 'test', 'val').
            fold (int): fold index for cross-validation (0..TOTAL_FOLDS-1).
            subject_ids (Optional[List[int]]): Optional list of subject IDs to include. If None, all subjects are included.
        """
        super().__init__()
        logger.info(f"Initializing EEGDataset with window size {window_size} ms, split '{split}'"+(f", fold {fold}" if split != 'test' else ''))
        self.window_size = window_size
        self.split = split
        self.fold = fold
        self.subject_ids = subject_ids

        # Get feature column names from config
        self.feature_columns = Config().MUSEDATA_IMPORTANT_COLUMNS

        # Full DataFrame
        self.df: pd.DataFrame = data
        
        # Filter by subject IDs if provided
        if self.subject_ids and 'Subject' in self.df.columns:
            self.df = self.df[self.df['Subject'].isin(self.subject_ids)]

        # Prepare samples and labels
        self.samples: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self._prepare_samples()

    def _prepare_samples(self) -> None:
        df = self.df
        # Filter by split
        if self.split == 'train':
            df = df[(df['Set'] == True) & (df['Fold'] != self.fold)]
        elif self.split == 'val':
            df = df[(df['Set'] == True) & (df['Fold'] == self.fold)]
        elif self.split == 'test':
            df = df[df['Set'] == False]
        else:
            raise ValueError(f"Unknown split '{self.split}', use 'train', 'val', or 'test'.")

        obs_col = f'chunk_{self.window_size}_observation'
        win_col = f'chunk_{self.window_size}_window'

        # Group into windows
        grouped = df.groupby(['Subject', obs_col, win_col])
        for (subject, obs, win), group in grouped:
            # skip invalid segments
            if obs < 0 or win < 0 or pd.isna(obs) or pd.isna(win):
                continue

            # Get the label (0 or 1)
            label: bool = bool(group['label'].iloc[0])
            # Get the features for the current window
            features: np.ndarray = group[self.feature_columns].values
            # Convert to tensor
            sample_tensor: torch.Tensor = torch.tensor(features, dtype=torch.float64)
            # Append to samples and labels
            self.samples.append(sample_tensor)
            self.labels.append(torch.tensor(label, dtype=torch.bool))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx], self.labels[idx]
    
    def __getitems__(self, indices: List[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return [self.samples[i] for i in indices], [self.labels[i] for i in indices]


def get_dataloader(data: pd.DataFrame , window_size: int, split: str = 'train', fold: int = 0, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4) -> DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Utility function to create a DataLoader for the EEGDataset.

    Args:
        window_size: window length in milliseconds.
        split: 'train', 'val', or 'test'.
        fold: fold index for validation.
        batch_size: number of samples per batch.
        shuffle: whether to shuffle the data.
        num_workers: number of subprocesses for data loading.

    Returns:
        torch.utils.data.DataLoader
    """
    egg_dataset: EEGDataset = EEGDataset(data, window_size=window_size, split=split, fold=fold)
    return DataLoader(egg_dataset, batch_size=batch_size, shuffle=shuffle if split == 'train' else False, num_workers=num_workers, pin_memory=False)

def main():
    data: pd.DataFrame = Data().data
    ds: EEGDataset = EEGDataset(data, window_size=100, split='train', fold=0)
    #dl = DataLoader(ds, batch_size=16, shuffle = True, num_workers=4, pin_memory=False)
    del data
    gc.collect()
    
    logger.info(f"DataSet length: {len(ds)}")


if __name__ == '__main__':
    main()
