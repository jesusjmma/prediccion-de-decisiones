"""
Author: Jesús Maldonado
Description: Classes for preprocessing EEG data and exporting it in a format suitable for machine learning.
"""

from __future__ import annotations
from configparser import ConfigParser
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
import random
import re
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from config import Config
from logger_utils import setup_logger
logger = setup_logger(name=Path(__file__).name, level=10)
@dataclass
class Subject:
    """Holds EEG results and time-series data for one participant, with methods to split into training and test sets, assign folds, and save/load CSVs."""
    subject_id:                 np.uint8 = field(init=False)
    results:                    pd.DataFrame
    muse_data:                  pd.DataFrame
    process_raw:                bool                       = field(default=True, repr=False)
    training_assignment:        Optional[bool]             = None
    num_trials:                 int                        = 0
    num_observations:           int                        = 0
    num_steps:                  int                        = 0
    num_observations_per_trial: dict[np.uint8, int]             = field(default_factory=dict)
    num_steps_per_observation:  dict[tuple[np.uint32, np.uint32], int] = field(default_factory=dict)
    chunks_count_per_window:    dict[int, np.uint16]             = field(default_factory=dict)

    def __post_init__(self):
        self.subject_id = self.results['ID del participante'].iloc[0]
        self.training_assignment = self.muse_data['Set'].iloc[0] if not self.process_raw and 'Set' in self.muse_data.columns else None
        self.muse_data = self.muse_data.resample(f"{Config().SAMPLING_OFFSET}ms").mean().interpolate(method='linear').round(8) if self.process_raw else self.muse_data
        if self.process_raw:
            self._assign_series_to_musedata()
        self.num_trials = self.results['Trial'].max()
        self.num_observations_per_trial = self.results.groupby('Trial')['Respuesta'].nunique().to_dict()
        self.num_steps_per_observation = self.muse_data.groupby(['Trial', 'Respuesta']).size().to_dict()
        self.results['key'] = list(zip(self.results['Trial'], self.results['Respuesta']))
        self.results['Steps'] = self.results['key'].map(self.num_steps_per_observation).fillna(0).astype('uint32')
        self.num_observations = sum(self.num_observations_per_trial.values())
        self.num_steps = sum(self.num_steps_per_observation.values())
        
        self.muse_data['Set'] = np.full(shape=(self.muse_data.shape[0],), fill_value=self.training_assignment, dtype=bool)
        self._calculate_chunks_count_per_window() if self.process_raw else None
        
    def __str__(self) -> str:
        return (
            f"\n{'_' * 22}\n"
            f"| Subject: {str(self.subject_id).ljust(10)}|\n"
            f"| Set:     {("Training" if self.training_assignment else "Test").ljust(10)}|\n"
            f"| Trials:  {str(self.num_trials).ljust(10)}|\n"
            f"| Observations: {str(self.num_observations).ljust(5)}|\n"
            f"| Steps:   {str(self.num_steps).ljust(10)}|\n"
            f"|{'_' * 20}|" 
        )
    
    def _calculate_chunks_count_per_window(self) -> None:
        logger.debug(f"Calculating chunks for subject {self.subject_id}...")
        press_times: pd.Series[pd.Timestamp] = self.results['Tiempo de la pulsación']
        steps = pd.Series(list(self.num_steps_per_observation.values()), index=self.results.index, dtype=int)
        
        for w in Config().WINDOWS:
            chunk_observation = f'chunk_{w}_observation'
            chunk_window = f'chunk_{w}_window'
            num_windows: NDArray[np.uint16] = np.floor(steps*int(Config().SAMPLING_OFFSET) / int(w)).astype(np.uint16)
            first_steps: pd.Series[pd.Timestamp] = press_times - pd.to_timedelta(num_windows * w, unit='ms')
            
            start_times = first_steps
            end_times = press_times
            mask = start_times != end_times

            observations_intervals: pd.IntervalIndex = pd.IntervalIndex.from_arrays(pd.DatetimeIndex(first_steps[mask]), pd.DatetimeIndex(press_times[mask]), closed='right')

            idx: NDArray[np.intp] = observations_intervals.get_indexer(self.muse_data.index)

            self.muse_data[chunk_observation] = idx.astype(np.int16)

            aux = self.muse_data[chunk_observation]
            aux = aux[aux != -1]
            count = aux.value_counts()
            windows: pd.Series[float] = count[count >= 0] / float(w / Config().SAMPLING_OFFSET)
            timestamps: list[pd.Timestamp] = aux.drop_duplicates().index.tolist()
            window_unit = np.ones(int(w / Config().SAMPLING_OFFSET), dtype=int)
            
            chunk_id = 0
            for i in range(len(timestamps)):
                start: pd.Timestamp = timestamps[i]
                end = start + pd.Timedelta(milliseconds=w)
                for j in range(int(windows[i])):
                    mask = (self.muse_data.index >= start) & (self.muse_data.index < end)
                    self.muse_data.loc[mask, chunk_window] = window_unit * chunk_id
                    start = end
                    end += pd.Timedelta(milliseconds=w)
                    chunk_id += 1

    @staticmethod
    def _normalize_subjects(subjects) -> list['Subject']:
        if isinstance(subjects, dict):
            return list(subjects.values())
        elif isinstance(subjects, Subject):
            return [subjects]
        return subjects
    
    @staticmethod
    def split_training_test_data(subjects: dict[int, 'Subject'] | list['Subject'] | 'Subject') -> None:
        """Splits subjects into training and test sets, then divides the training set into folds.

        Args:
            subjects: One or more Subject instances to split into sets and folds.
        """
        subjects = Subject._normalize_subjects(subjects)

        total_subjects = len(subjects)

        training_subjects_size = int(total_subjects * Config().TRAINING_DATA_RATIO)
        test_subjects_size = total_subjects - training_subjects_size

        training_test_set = [True] * training_subjects_size + [False] * test_subjects_size
        random.shuffle(training_test_set)

        for subject, training_test in zip(subjects, training_test_set):
            subject.training_assignment = training_test
            subject.muse_data['Set'] = np.full(shape=(subject.muse_data.shape[0],), fill_value=training_test, dtype=bool)

        Subject.split_data_into_folds(subjects)
    
    @staticmethod
    def split_data_into_folds(subjects: dict[int, 'Subject'] | list['Subject'] | 'Subject') -> None:
        """Splits the training set into folds for cross-validation.
        It assigns -1 to the fold for test data and assigns a fold number (0 to TOTAL_FOLDS-1) for training data.
        
        Args:
            subjects: One or more Subject instances to split into folds.
        """
        subjects = Subject._normalize_subjects(subjects)
        
        total_observations = sum(subject.num_observations for subject in subjects if subject.training_assignment)
        observations_per_fold = int(total_observations / Config().TOTAL_FOLDS)

        fold_assignment: list[int] = []
        for i in range(Config().TOTAL_FOLDS-1):
            fold_assignment += [i] * observations_per_fold
        
        fold_assignment += [int(Config().TOTAL_FOLDS - 1)] * (total_observations - len(fold_assignment))

        # Debugging: Count the number of observations per fold
        counts = {}
        for fold in fold_assignment:
            if fold in counts:
                counts[fold] += 1
            else:
                counts[fold] = 1
        for fold in sorted(counts):
            logger.debug(f"Fold {fold}: {counts[fold]} observaciones")

        random.shuffle(fold_assignment)

        offset = 0
        for subject in subjects:
            logger.debug(f"Assigning folds to subject {subject.subject_id}...")
            subject_observations = subject.num_observations
            if subject.training_assignment:
                subject_observations_assignment = fold_assignment[offset:offset + subject_observations]
                subject_observations_assignment_to_step = np.repeat(subject_observations_assignment, subject.results['Steps'].to_numpy())
                subject.muse_data['Fold'] = np.array(subject_observations_assignment_to_step, dtype=np.int8)  # Training data
                offset += subject_observations
            else:
                subject.muse_data['Fold'] = np.full(shape=(subject.muse_data.shape[0],), fill_value=-1, dtype=np.int8)  # Test data

    def _assign_series_to_musedata(self) -> None:
        """Assigns the 'Trial' and 'Respuesta' columns to the muse_data DataFrame based on the results DataFrame.

        Args:
            None
        """
        # Initialize columns for 'Trial' and 'Respuesta' with NaN values
        self.muse_data['Trial'] = np.nan
        self.muse_data['Respuesta'] = np.nan

        # Convert to arrays for faster access
        muse_timestamps = self.muse_data.index.values.astype('datetime64[ns]')
        start_times = self.results['Tiempo de inicio'].values.astype('datetime64[ns]')
        end_times = self.results['Tiempo de la pulsación'].values.astype('datetime64[ns]')
        trials = self.results['Trial'].values.astype(np.uint8)
        observations = self.results['Respuesta'].values.astype(np.uint8)

        # Find start/end indices via binary search
        start_idx = np.searchsorted(muse_timestamps, start_times, side='left')
        end_idx = np.searchsorted(muse_timestamps, end_times, side='right')

        for i in range(len(self.results)):
            s_idx, e_idx = start_idx[i], end_idx[i]
            #e_idx = min(e_idx + 1, len(self.muse_data))
            # Assign values in the range '[s_idx, e_idx]' (closed interval)
            self.muse_data.loc[self.muse_data.index[s_idx:e_idx], 'Trial'] = trials[i]
            self.muse_data.loc[self.muse_data.index[s_idx:e_idx], 'Respuesta'] = observations[i]

        # Remove rows with NaN values in 'Trial' and 'Respuesta'
        self.muse_data.dropna(subset=['Trial', 'Respuesta'], inplace=True)

        # Convert columns to appropriate types (uint8)
        self.muse_data['Trial'] = self.muse_data['Trial'].astype(np.uint8)
        self.muse_data['Respuesta'] = self.muse_data['Respuesta'].astype(np.uint8)
        
        self.num_steps_per_observation = self.muse_data.groupby(['Respuesta', 'Trial']).size().to_dict()
        self.results['key'] = list(zip(self.results['Respuesta'], self.results['Trial']))
        self.results['Steps'] = self.results['key'].map(self.num_steps_per_observation).fillna(0).astype('uint32')
        self.results.drop(columns=['key'], inplace=True)

    def _get_observations_windows(self) -> pd.Series[pd.Timedelta]:
        """Returns the time difference between 'Tiempo de la pulsación' and 'Tiempo de inicio' for each observation.

        Args:
            None
        """
        press_time: pd.Series[pd.Timestamp] = self.results['Tiempo de la pulsación']
        start_time: pd.Series[pd.Timestamp] = self.results['Tiempo de inicio']
        observations_windows: pd.Series[pd.Timedelta] = press_time - start_time

        return observations_windows
    
    @staticmethod
    def get_observations_windows(subjects: dict[np.uint8, 'Subject'] | list['Subject'] | 'Subject') -> dict[np.uint8, pd.Series[pd.Timedelta]]:
        """Return the difference between "Tiempo de la pulsación" and "Tiempo de inicio" in milliseconds for each subject.
        
        Args:
            *subjects: Subject objects to get observation windows from.

        Returns:
            A dictionary with subject numbers as keys and their corresponding observation windows as values.
        """
        subjects = Subject._normalize_subjects(subjects)
        
        observations_windows: dict[np.uint8, pd.Series[pd.Timedelta]] = {}
        for subject in subjects:
            observations_windows[subject.subject_id] = subject._get_observations_windows()

        return observations_windows
    
    def _save_subject(self) -> None:
        """Saves this subject's results and muse data to CSV files named by subject ID.

        Args:
            None
        """
        logger.info(f"Saving subject {self.subject_id} data to CSV files...")
        results_filename = f"{Config().RESULTS_FILES_PREFIX}{self.subject_id}.csv"
        self.results.to_csv(results_filename, index=False)
        muse_data_filename = f"{Config().MUSEDATA_FILES_PREFIX}{self.subject_id}.csv"
        self.muse_data.to_csv(muse_data_filename, index=True)
    
    @staticmethod
    def save_subjects(subjects: dict[int, 'Subject'] | list['Subject'] | 'Subject') -> None:
        """Saves each subject's results and muse data to CSV files named by subject ID.

        Args:
            subjects: One or more Subject instances to save.
        """
        subjects = Subject._normalize_subjects(subjects)

        for subject in subjects:
            subject._save_subject()
    
    @staticmethod
    def _sanitize_data(results: pd.DataFrame) -> pd.Series[bool]:
        prev_press = results['Tiempo de la pulsación'].shift(1)
        mask: pd.Series[bool] = results['Tiempo de inicio'] <= prev_press
                
        results.loc[mask, 'Tiempo de inicio'] = prev_press[mask] + pd.Timedelta(milliseconds=1)

        return mask

    @staticmethod
    def load_data(process_raw = True) -> dict[int, 'Subject']:
        """Loads raw or processed EEG data from CSV files, creates and processes all Subject instances, and returns a dict of Subject instances.

        Args:
            process_raw (bool): If True, process raw data files. If False, load processed data files.
        """

        if process_raw:
            results_path = Config().LOCAL_PATH
            museData_path = Config().MUSE_PATH
            Config().PROCESSED_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            Config().PROCESSED_MUSEDATA_PATH.mkdir(parents=True, exist_ok=True)
            training_assignment = None
            converter={"Tecla elegida": Subject._bool_converter}
        else:
            results_path = Config().PROCESSED_RESULTS_PATH
            museData_path = Config().PROCESSED_MUSEDATA_PATH
            converter=None

        # Look for files in the directories
        files_dict = Subject._look_for_files(results_path, museData_path)

        subjects: dict[int, Subject] = {}

        for id, results_file in sorted(files_dict[str(results_path.resolve())].items()):
            museData_file = files_dict[str(museData_path.resolve())].get(id)
            if museData_file is None:
                logger.warning(f"Warning: There's no museData file for subject {id} in {museData_path}.")
                continue
            
            logger.info(f"Processing subject {id}...")
            
            with open(museData_file, 'r') as f:
                header = f.readline().strip().split(',')
            
            columns = Config().MUSEDATA_COLUMNS
            dtype_map = {'Set': bool, 'Fold': np.uint8, 'Trial': np.uint8, 'Respuesta': np.uint8}
            selected_columns = [col for col in columns if col in header]
            dtypes = {col: dtype_map[col] for col in selected_columns if col in dtype_map}

            museData = pd.read_csv(museData_file, low_memory=False, date_format="%Y-%m-%d %H:%M:%S.%f", parse_dates=[0], index_col=0, dtype=dtypes, usecols=selected_columns)
            museData.replace([float('inf'), float('-inf')], np.nan, inplace=True)
            cols_to_check = ['Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10','Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10','Alpha_TP9','Alpha_AF7','Alpha_AF8','Alpha_TP10','Beta_TP9','Beta_AF7','Beta_AF8','Beta_TP10','Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10']
            museData.dropna(subset=cols_to_check, how='all', inplace=True)

            results = pd.read_csv(results_file, low_memory=False, date_format="%Y-%m-%d %H:%M:%S.%f", parse_dates=[3,4,5,7], converters=converter, dtype={'ID del participante': np.uint8, 'Trial': np.uint8, 'Respuesta': np.uint8, 'Letra observada': "category"})
            mask = Subject._sanitize_data(results)

            subjects[id] = Subject(results, museData, process_raw=process_raw)

        Subject.split_training_test_data(subjects) if process_raw else None

        return subjects
    
    @staticmethod
    def _bool_converter(value: str) -> bool:
        _map = {'p': True, 'q': False}
        try:
            return _map[value.lower()]
        except KeyError:
            logger.error(f"Error: Invalid value '{value}' for boolean conversion.")
            raise ValueError(f"Invalid value '{value}' for boolean conversion.")

    @staticmethod
    def _look_for_files(*directories):
        """Looks for files in the specified directories.

        Args:
            *directories: Directories to search for files.

        Returns:
            files_per_directory: Dictionary with directory names as keys and dictionaries of file numbers to filenames as values.
        """
        files_per_directory: dict[str, dict[int, str]] = {}

        for directory in directories:
            path_dir = Path(directory)
            if not path_dir.is_dir():
                logger.warning(f"Warning: '{directory}' is not a valid directory.")
                continue

            files = {}
            for file in path_dir.iterdir():
                if file.is_file():
                    match = re.search(r'(\d+)', file.name)
                    if match:
                        number = int(match.group(1))
                        files[number] = str(file.resolve())

            files_per_directory[str(path_dir.resolve())] = files

        return files_per_directory

def main(process_raw: bool):
    """Entry point: loads the config file, processes the data, and saves it to processed CSV files.
    
    Args:
        process_raw (bool): If True, process raw data files. If False, load processed data files.
    """
    subjects = Subject.load_data(process_raw)    
    for subject in subjects.values():
        logger.info(f"{subject}\n")
    
    if process_raw:
        logger.info("Saving processed data to CSV files...")
        Subject.save_subjects(subjects)

if __name__ == "__main__":
    main(process_raw = False)  # False if you want to load the data from the processed files, True if you want to process the raw data again
