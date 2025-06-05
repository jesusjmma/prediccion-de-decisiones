# Python 3.13
from __future__ import annotations
from dataclasses import dataclass, field
import os
from pathlib import Path
import pickle
import random
import re
import time
from typing import Literal, ClassVar

import pandas as pd

from scripts.config import Config
from scripts.logger_utils import setup_logger

logger = setup_logger("DEBUG")

##### GLOBAL VARIABLES ####################################################################
local_path: Path = Config().LOCAL_PATH
muse_path: Path = Config().MUSE_PATH
processed_files_path: Path = Config().PROCESSED_FILES_PATH
training_data_ratio: float = float(Config().TRAINING_DATA_RATIO)
letters: list[str] = Config().LETTERS
musedata_columns: list[str] = Config().MUSEDATA_COLUMNS
musedata_important_columns: list[str] = Config().MUSEDATA_IMPORTANT_COLUMNS
total_folds: int = int(Config().TOTAL_FOLDS)
sampling_offset: int = int(Config().SAMPLING_OFFSET)
time_before_event_min: pd.Timedelta = Config().TIME_BEFORE_EVENT_MIN
time_after_event_min: pd.Timedelta = Config().TIME_AFTER_EVENT_MIN
time_before_event_max: pd.Timedelta = Config().TIME_BEFORE_EVENT_MAX
time_after_event_max: pd.Timedelta = Config().TIME_AFTER_EVENT_MAX
exact_time_before_event: pd.Timedelta = Config().EXACT_TIME_BEFORE_EVENT
exact_time_after_event: pd.Timedelta  = Config().EXACT_TIME_AFTER_EVENT
exact_time: bool = Config().EXACT_TIME
##### GLOBAL FUNCTIONS ####################################################################
def window_size() -> int:
    return int((exact_time_after_event + exact_time_before_event).total_seconds() * 1000.0)
def expected_len() -> int:
    return window_size() // sampling_offset 
###########################################################################################

@dataclass(slots=True)
class EEGTimeSeries:
    """
    Clase para almacenar los datos de una serie temporal EEG junto a su metadato de evento.
    """
    # Metadatos del ensayo
    subject: int
    trial: int
    response: int
    chosen_key: Literal['p', 'q']   # Tecla elegida (P/Q)
    start_time: pd.Timestamp        # Tiempo de inicio del ensayo
    display_time: pd.Timestamp      # Tiempo de aparición de letras
    observation_time: pd.Timestamp  # Tiempo de aparición de la letra observada
    press_time: pd.Timestamp        # Tiempo de la pulsación
    observed_letter: str            # Letra observada

    # Datos EEG
    eeg: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        # Comprobar la longitud de la serie temporal
        if self.eeg.empty:
            logger.error(f"Error: The EEG time series for subject {self.subject}, trial {self.trial}, response {self.response} is empty.")
            return
        if len(self.eeg) == 0:
            logger.error(f"Error: The EEG time series for subject {self.subject}, trial {self.trial}, response {self.response} has length 0.")
            return
        # Comprobar si hay datos NaN
        if self.eeg.isna().any().any():
            logger.error(f"Error: NaN values found in EEG data for subject {self.subject}, trial {self.trial}, response {self.response}.")
            logger.debug(f"NaN values: {self.eeg.isna().sum()}")
            logger.debug(f"NaN values: {self.eeg.isna()}")
        # Comprobar si hay datos infinitos
        if self.eeg.isin([float('inf'), float('-inf')]).any().any():
            logger.error(f"Error: Infinite values found in EEG data for subject {self.subject}, trial {self.trial}, response {self.response}.")
            logger.debug(f"Inf values: {self.eeg.isin([float('inf'), float('-inf')]).sum()}")
            logger.debug(f"Inf values: {self.eeg.isin([float('inf'), float('-inf')])}")
        # Comprobar si hay datos duplicados
        if self.eeg.duplicated().any():
            logger.warning(f"Warning: Duplicated values found in EEG data for subject {self.subject}, trial {self.trial}, response {self.response}.")
            logger.debug(f"Duplicated values: {self.eeg.duplicated().sum()}")
            logger.debug(f"Duplicated values: {self.eeg[self.eeg.duplicated()]}")
        # Imprimir el número de datos
        real_len = len(self.eeg)
        if real_len != expected_len():
            logger.error(f"Error: The EEG time series for subject {self.subject}, trial {self.trial}, response {self.response}, has {real_len} data points, but it should have {expected_len()}.")
            time.sleep(1)
            return

    def __hash__(self):
        return hash((self.subject, self.trial, self.response))
    
    def __repr__(self) -> str:
        return f"EEGTimeSeries(subject={self.subject}, trial={self.trial}, response={self.response})"
    
    def __str__(self) -> str:
        return f"EEGTimeSeries(subject={self.subject}, trial={self.trial}, response={self.response})"

    def to_csv(self, path: Path) -> None:
        """
        Guarda la serie temporal EEG en un archivo CSV y el resto de datos en otro archivo CSV.

        :param path: Ruta al archivo CSV donde se guardará la serie temporal.
        """
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        self.eeg.to_csv(path, index=False)
        # Guardar los metadatos en un archivo CSV separado
        metadata = {
            'ID del participante': self.subject,
            'Trial': self.trial,
            'Respuesta': self.response,
            'Tecla elegida': self.chosen_key,
            'Tiempo de inicio': self.start_time,
            'Tiempo de aparición de letras': self.display_time,
            'Tiempo de aparición de la letra observada': self.observation_time,
            'Tiempo de la pulsación': self.press_time,
            'Letra observada': self.observed_letter
        }
        metadata_df = pd.DataFrame([metadata])
        metadata_path = path.with_suffix('.metadata.csv')
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"Saved EEG time series and metadata to {path} and {metadata_path}")

@dataclass(slots=True)
class EEGData:
    """
    Clase para almacenar un diccionario de series temporales EEG (EEGTimeSeries) en base al ID de participante, Trial y Respuesta.
    Permitirá buscar y acceder a las series temporales de manera eficiente, tanto a 1 exacta como a un rango de Trials de un mismo participante.
    """
    # Diccionario de series temporales EEG
    series:                          dict[tuple[int, int, int], EEGTimeSeries]      = field(default_factory=dict)
    subjects:                        list[int]                                      = field(default_factory=list) #TODO añadir si es training o test
    trials_for_subject:              dict[int, list[int]]                           = field(default_factory=dict)
    responses_for_subject_and_trial: dict[tuple[int, int], set[int]]                = field(default_factory=dict) #TODO añadir el fold de cada uno
    responses_count_for_subject:     dict[int, int]                                 = field(default_factory=dict)
    training_or_val_subjects:        list[int]                                      = field(default_factory=list)
    testing_subjects:                list[int]                                      = field(default_factory=list)
    fold:                            dict[int, list[int]]                           = field(default_factory=dict)
    series2:                         dict[int, dict[int, dict[int, EEGTimeSeries]]] = field(default_factory=dict)

    def __repr__(self) -> str:
        return "EEGData"
    
    def __str__(self) -> str:
        return "EEGData"

    def save(self, 
                  file: Path, 
                  data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]] | None = None
                  ) -> None:
        """
        Save the EEG time series dictionary to a pickle file.

        Args:
            file (Path): Path to the file where the data will be saved.
            data (dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]], optional): Diccionario de series temporales EEG. Si no se proporciona, se utilizará el diccionario actual.
        """        
        # Crear el directorio si no existe
        if not file.parent.exists():
            file.parent.mkdir(parents=True, exist_ok=True)
        
        # Crear el archivo si no existe
        if not file.exists():
            file.touch()

        if data is None:
            data = self.get_data()

        with open(file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def save_data(cls, 
                  file: Path, 
                  data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]] | EEGData
                  ) -> Path:
        """
        Save the EEG time series dictionary to a pickle file.
        Args:
            file (Path): Path to the file where the data will be saved.
            data (dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]], EEGData): Diccionario de series temporales EEG o instancia de EEGData.
        """        
        # Crear el directorio si no existe
        if not file.parent.exists():
            file.parent.mkdir(parents=True, exist_ok=True)
        
        if file.exists():
            file = file.with_name(f"{file.stem}_{time.strftime('%Y_%m_%d_%H_%M_%S')}{file.suffix}")
        
        if isinstance(data, EEGData):
            data = data.get_data()

        file.touch()

        with open(file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return file

    @classmethod
    def load_data(cls, file: Path) -> dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]:
        """
        Load the EEG time series dictionary from a pickle file.
        Args:
            file (Path): Path to the file from which the data will be loaded.
        Returns:
            data (dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]): Dictionary of EEG time series.
        """
        data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]] = {}
        
        with open(file, "rb") as f:
            data = pickle.load(f)

        return data
    
    def get_data(self) -> dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]:
        data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]] = {}
        data['test']  = set[EEGTimeSeries]()
        data['train'] = dict[int, set[EEGTimeSeries]]()
        for i in range(total_folds):
            data['train'][i] = set[EEGTimeSeries]()
        
        for s in self.testing_subjects:
            for t in self.trials_for_subject[s]:
                for r in self.responses_for_subject_and_trial[(s, t)]:
                    data['test'].add(self.series[(s, t, r)])
        
        for s in self.training_or_val_subjects:
            subject_folds = self.fold[s].copy()
            print(f"Subject {s} folds: {len(subject_folds)}")
            print(f"Subject {s} responses: {self.responses_count_for_subject[s]}")
            for t in self.trials_for_subject[s]:
                for r in self.responses_for_subject_and_trial[(s, t)]:
                    fold = subject_folds.pop(0)
                    data['train'][fold].add(self.series[(s, t, r)])
                    
        return data

    def _split_training_test_data_and_folds(self) -> None:
        total_subjects = len(self.subjects)

        training_subjects_size = int(total_subjects * training_data_ratio)
        test_subjects_size = total_subjects - training_subjects_size

        training_test_set = [True] * training_subjects_size + [False] * test_subjects_size
        random.shuffle(training_test_set)

        for subject, training_or_val in zip(self.subjects, training_test_set):
            if training_or_val:
                self.training_or_val_subjects.append(subject)
            else:
                self.testing_subjects.append(subject)
                self.fold[subject] = [-1]*self.responses_count_for_subject[subject]

        self._split_data_into_folds()

    def _split_data_into_folds(self) -> None:
        responses_count = 0
        for subject in self.training_or_val_subjects:
            responses_count += self.responses_count_for_subject[subject]
            
        responses_per_fold = int(responses_count / total_folds)

        fold_assignment: list[int] = []
        for i in range(total_folds-1):
            fold_assignment += [i] * responses_per_fold
        
        fold_assignment += [total_folds - 1] * (responses_count - len(fold_assignment))
        random.shuffle(fold_assignment)

        offset = 0
        for subject in self.training_or_val_subjects:
            logger.debug(f"Assigning folds to subject {subject}...")
            subject_responses = self.responses_count_for_subject[subject]
            subject_observations_assignment = fold_assignment[offset:offset + subject_responses]
            self.fold[subject] = subject_observations_assignment
            offset += subject_responses

    def from_paths(self, results_path: Path, musedata_path: Path) -> tuple[int, dict[int, int]]:
        """
        Añade múltiples series temporales EEG desde archivos CSV.

        :param results_path: Ruta al directorio de archivos CSV de resultados.
        :param musedata_path: Ruta al directorio de archivos CSV de musedata.
        """

        # Look for files in the directories

        files: dict[int, tuple[Path, Path]] = {}
        
        if not results_path.is_dir():
            logger.warning(f"Warning: '{results_path}' is not a valid directory.")
            return 0, {}
        
        if not musedata_path.is_dir():
            logger.warning(f"Warning: '{musedata_path}' is not a valid directory.")
            return 0, {}
        
        for results_file in results_path.iterdir():
            if results_file.is_file():
                match = re.search(r'(\d+)', results_file.name)
                if match:
                    subject = int(match.group(1))
                    musedata_file = musedata_path / f'museData{subject}.csv'
                    if not musedata_file.exists():
                        logger.warning(f"Warning: '{musedata_file}' does not exist.")
                        continue
                    files[subject] = Path(results_file.resolve()), musedata_path / f'museData{subject}.csv'
        
        if not files:
            logger.warning(f"Warning: No files found in '{results_path}' or '{musedata_path}'.")
            return 0, {}
        logger.debug(f"Found {len(files)} files in '{results_path}' and '{musedata_path}'.")

        files = dict(sorted(files.items(), key=lambda item: item[0]))
        
        dataset_size: tuple[int, dict[int, int]] = (0, {})

        for subject, (results_file, musedata_file) in files.items():
            logger.debug(f"Processing files: {results_file} and {musedata_file}")
            added_series_count = self._from_csv(results_file, musedata_file)

            if added_series_count == 0:
                logger.warning(f"Warning: No EEG time series were added for subject {subject}.")
                continue
            
            dataset_size = (dataset_size[0] + added_series_count, {**dataset_size[1], subject: dataset_size[1].get(subject, 0) + added_series_count})

        return dataset_size

    def _from_csv(self, results_path: Path, musedata_path: Path) -> int:
        """
        Añade múltiples series temporales EEG desde archivos CSV.

        :param results_path: Ruta al archivo CSV de resultados.
        :param musedata_path: Ruta al archivo CSV de musedata.
        """
        with open(musedata_path, 'r') as f:
            header = f.readline().strip().split(',')

        dtype_map = {'Set': pd.BooleanDtype(), 'Fold': pd.Int8Dtype(), 'Trial': pd.Int8Dtype(), 'Respuesta': pd.Int8Dtype()}
        selected_columns = [col for col in musedata_columns if col in header]
        dtypes = {col: dtype_map[col] for col in selected_columns if col in dtype_map}
        dtypes.update({col: pd.Float64Dtype() for col in musedata_important_columns if col not in dtype_map})

        results = pd.read_csv(results_path)

        musedata = pd.read_csv(musedata_path, low_memory=False, date_format="%Y-%m-%d %H:%M:%S.%f", parse_dates=[0], dtype=dtypes, usecols=selected_columns)
        musedata['TimeStamp'] = pd.to_datetime(musedata['TimeStamp'], unit='ms', errors='raise')
        musedata.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        
        return self._from_dataframes(results, musedata)
    
    def _from_dataframes(self, results: pd.DataFrame, musedata: pd.DataFrame) -> int:
        """
        Añade múltiples series temporales EEG desde DataFrames.

        :param results: DataFrame de resultados.
        :param musedata: DataFrame de musedata.
        """
        added_series_count = 0

        # Resample the musedata DataFrame to a fixed sampling rate
        musedata = self._resample(musedata)
        if musedata.empty:
            logger.warning("Warning: The musedata DataFrame is empty after resampling.")
            return 0

        for _, results_row in results.iterrows():
            if not self._add_series(results_row, musedata):
                continue
            added_series_count += 1

        if added_series_count == 0:
            logger.warning("No EEG time series were added.")
        else:
            logger.debug(f"Added {added_series_count} EEG time series.")
        
        return added_series_count
    
    def _resample(self, musedata: pd.DataFrame) -> pd.DataFrame:
        """
        Resample the musedata DataFrame to a fixed sampling rate.

        :param musedata: DataFrame of musedata.
        :return: Resampled DataFrame.
        """
        logger.debug("Resampling musedata DataFrame to fixed sampling rate.")

        musedata.set_index('TimeStamp', inplace=True)
        musedata.dropna(axis='columns', how='all', inplace=True)
        musedata = musedata.resample(f'{sampling_offset}ms').mean()
        musedata = musedata.interpolate(method='linear', limit_direction='both')
        musedata = musedata.round(8)
        musedata.reset_index(inplace=True)
        if musedata.isna().any().any():
            raise ValueError(f"Quedan {musedata.isna().sum()} valores NaN después del resampleo e interpolación.")


        return musedata

    def _add_series(self, results_row: pd.Series, musedata: pd.DataFrame) -> bool:
        """
        Crea una serie temporal EEG y la añade al diccionario de series.

        :param results_row: Fila de resultados que contiene los metadatos del ensayo.
        :param musedata: DataFrame de musedata que contiene los datos EEG.
        """
        # Extraer metadatos del ensayo
        subject:          int          = int(results_row['ID del participante'])
        trial:            int          = int(results_row['Trial'])
        response:         int          = int(results_row['Respuesta'])
        chosen_key:       str          = str(results_row['Tecla elegida'])
        start_time:       pd.Timestamp = pd.to_datetime(results_row['Tiempo de inicio'])
        display_time:     pd.Timestamp = pd.to_datetime(results_row['Tiempo de aparición de letras'])
        observation_time: pd.Timestamp = pd.to_datetime(results_row['Tiempo de aparición de la letra observada'])
        press_time:       pd.Timestamp = pd.to_datetime(results_row['Tiempo de la pulsación'])
        observed_letter:  str          = str(results_row['Letra observada'])

        if not isinstance(observation_time, pd.Timestamp):
            logger.warning(f"Invalid observation time for subject {subject}, trial {trial}, response {response}.")
            logger.debug(f"observation_time: {observation_time}, type: {type(observation_time)}")
            return False

        if not (start_time <= display_time <= observation_time <= press_time):
            logger.warning(f"Invalid time order for subject {subject}, trial {trial}, response {response}.")
            logger.debug(f"start_time: {start_time}, display_time: {display_time}, observation_time: {observation_time}, press_time: {press_time}")
            return False

        if chosen_key not in ('p', 'q'):
            logger.warning(f"Invalid chosen key '{chosen_key}' for subject {subject}, trial {trial}, response {response}.")
            return False
        
        if observed_letter not in letters:
            logger.warning(f"Invalid observed letter '{observed_letter}' for subject {subject}, trial {trial}, response {response}.")
            return False

        # Filtrar los datos EEG para el ensayo actual según el tiempo de inicio y el tiempo de finalización
        if exact_time:
            eeg: pd.DataFrame = musedata[(musedata['TimeStamp'] >= (observation_time - exact_time_before_event)) & (musedata['TimeStamp'] < (observation_time + exact_time_after_event))]
        else:
            max_time_before_event = max(observation_time - time_before_event_max, start_time)
            min_time_before_event = observation_time - time_before_event_min
            if min_time_before_event < max_time_before_event:
                logger.warning(f"Invalid time range for subject {subject}, trial {trial}, response {response}.")
                logger.debug(f"max_time_before_event: {max_time_before_event}, min_time_before_event: {min_time_before_event}, difference: {min_time_before_event - max_time_before_event}")
                return False

            max_time_after_event = min(observation_time + time_after_event_max, press_time)
            min_time_after_event = observation_time + time_after_event_min
        
            if min_time_after_event > max_time_after_event:
                logger.warning(f"Invalid time range for subject {subject}, trial {trial}, response {response}.")
                logger.debug(f"max_time_after_event: {max_time_after_event}, min_time_after_event: {min_time_after_event}, difference: {min_time_after_event - max_time_after_event}")
                return False

            eeg: pd.DataFrame = musedata[(musedata['TimeStamp'] >= max_time_before_event) & (musedata['TimeStamp'] <= max_time_after_event)]

        # Comprobar si hay datos EEG para el ensayo actual
        if eeg.empty:
            logger.warning(f"No EEG data for subject {subject}, trial {trial}, response {response}.")
            return False

        if eeg.empty:
            logger.warning(f"No EEG data after dropping NaN for subject {subject}, trial {trial}, response {response}.")
            return False

        # Crear una serie temporal EEG
        eeg_time_series = EEGTimeSeries(
            subject=subject,
            trial=trial,
            response=response,
            chosen_key=chosen_key,
            start_time=start_time,
            display_time=display_time,
            observation_time=observation_time,
            press_time=press_time,
            observed_letter=observed_letter,
            eeg=eeg
        )

        if subject not in self.series2:
            self.series2[subject] = {}
        if trial not in self.series2[subject]:
            self.series2[subject][trial] = {}


        # Añadir la serie temporal a la colección
        self.series[(subject, trial, response)] = eeg_time_series
        self.series2[subject][trial][response] = eeg_time_series
        if subject not in self.subjects:
            self.subjects.append(subject)
        if subject not in self.trials_for_subject:
            self.trials_for_subject[subject] = []
        if trial not in self.trials_for_subject[subject]:
            self.trials_for_subject[subject].append(trial)
        if (subject, trial) not in self.responses_for_subject_and_trial:
            self.responses_for_subject_and_trial[(subject, trial)] = set()
        self.responses_for_subject_and_trial[(subject, trial)].add(response)
        if subject not in self.responses_count_for_subject:
            self.responses_count_for_subject[subject] = 0
        self.responses_count_for_subject[subject] += 1

        return True
    
    @classmethod
    def initialize(cls) -> EEGData:
        logger.debug("Initializing EEGData.")
        dataset = EEGData()
        logger.debug("Loading EEGData from paths.")
        dataset_size = dataset.from_paths(local_path, muse_path)
        logger.info(f"Dataset size: {dataset_size[0]} series, {len(dataset.subjects)} subjects, {len(dataset.trials_for_subject)} trials, and {len(dataset.responses_for_subject_and_trial)} responses.")
        
        dataset._split_training_test_data_and_folds()
        dataset._normalize()

        return dataset
    
    def min_max(self) -> dict[str, tuple[float, float]]:
        """Get the max and the min of Delta_TP9,Delta_AF7,Delta_AF8,Delta_TP10,Theta_TP9,Theta_AF7,Theta_AF8,Theta_TP10,Alpha_TP9,Alpha_AF7,Alpha_AF8,Alpha_TP10,Beta_TP9,Beta_AF7,Beta_AF8,Beta_TP10,Gamma_TP9,Gamma_AF7,Gamma_AF8,Gamma_TP10
        Returns:
            dict[str, tuple[float, float]]: max and min of each channel
        """
        channels = [
            'Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10',
            'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10',
            'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10',
            'Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10',
            'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10'
        ]
        min_max: dict[str, tuple[float, float]] = {}
        for s in self.series.values():
            # First print all NA values in the dataframes
            if s.eeg.isna().any().any():
                logger.warning(f"Warning: NaN values found in EEG data for subject {s.subject}, trial {s.trial}, response {s.response}.")
                logger.debug(f"NaN values: {s.eeg.isna().sum()}")
                logger.debug(f"NaN values: {s.eeg.isna}")

            for channel in channels:
                if channel not in min_max:
                    min_max[channel] = (s.eeg[channel].min(skipna=True), s.eeg[channel].max(skipna=True))
                else:
                    min_max[channel] = (min(min_max[channel][0], s.eeg[channel].min(skipna=True)), max(min_max[channel][1], s.eeg[channel].max(skipna=True)))
        return min_max
    
    def _normalize(self) -> None:
        """
        Normaliza los datos EEG de cada serie temporal.
        """
        logger.debug("Normalizing EEG data.")
        min_max = self.min_max()
        for s in self.series.values():
            logger.debug(f"Normalizing series for subject {s.subject}, trial {s.trial}, response {s.response}.")
            for channel in min_max:
                s.eeg[channel] = (s.eeg[channel] - min_max[channel][0]) / (min_max[channel][1] - min_max[channel][0])
                s.eeg[channel].fillna(0, inplace=True)

def main():
    times_before_and_after_event = [
        (9900,100)
    ]
    folder: Path = Path(processed_files_path / 'windows')

    times_td = [(pd.Timedelta(milliseconds=before), pd.Timedelta(milliseconds=after)) for before, after in times_before_and_after_event]

    global exact_time_before_event, exact_time_after_event
    for time in times_td:
        exact_time_before_event = time[0]
        exact_time_after_event = time[1]
        
        logger.info(f"Processing with window_size={window_size()}, exact_time_before_event={exact_time_before_event.total_seconds()*1000}, exact_time_after_event={exact_time_after_event.total_seconds()*1000} and samples_expected={expected_len()}")
        eeg_data = EEGData.initialize()

        file = folder / f"eegdata_{window_size()}_-{int(exact_time_before_event.total_seconds() * 1000)}_+{int(exact_time_after_event.total_seconds() * 1000)}_({expected_len()} samples).pkl"
        eeg_data.save(file)
        logger.info(f"Data saved into {file}")



    '''
    saved_count = 1
    for _, series in eeg_data.series.items():
        logger.debug(f"Saving series {saved_count}/{dataset_size}...")
        series.to_csv(processed_files_path / "dataset" / f"subject_{series.subject}_trial_{series.trial}_response_{series.response}.csv")
        saved_count += 1
        logger.debug(f"Saved series for subject {series.subject}, trial {series.trial}, response {series.response} to CSV.")
    
    logger.info("All series saved to CSV files.")
    #'''

if __name__ == "__main__":
    logger = setup_logger("DEBUG") 
    main()
