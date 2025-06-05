"""
Author: Jesús Maldonado
Description: Configuration class for reading an INI file and storing configuration parameters in a singleton. This class is used to manage application configuration settings, including paths, sampling rates, and other parameters.
"""

from __future__ import annotations
from configparser import ConfigParser
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
import random
from typing import Literal

import numpy as np
import pandas as pd

from scripts.logger_utils import setup_logger
logger = setup_logger("DEBUG")

class SingletonMeta(type):
    """Metaclass that ensures a class has only one instance and provides a global point of access to it."""
    _instances: dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # First call: creates a new instance
            cls._instances[cls] = super().__call__(*args, **kwargs)
        # Subsequent calls: returns the existing instance without creating a new one
        return cls._instances[cls]

@dataclass(slots=True)
class Config(metaclass=SingletonMeta):
    """Lee el archivo de configuración e inicializa parámetros globales (singleton)."""
    __config:                   ConfigParser    = field(default_factory=ConfigParser, init=False)
    ROOT_BASE_PATH:             str             = ".."
    RELATIVE_PATH:              str             = "config.ini"

    # Campos que se calculan en __post_init__
    ROOT_PATH:                  Path            = field(init=False)
    CONFIG_INI_FILE:            Path            = field(init=False)
    SAMPLING_RATE:              np.uint16       = field(init=False)
    SAMPLING_OFFSET:            np.uint16       = field(init=False)
    TRAINING_DATA_RATIO:        np.float64      = field(init=False)
    TOTAL_FOLDS:                np.uint8        = field(init=False)
    TIME_BEFORE_EVENT_MAX:      pd.Timedelta    = field(init=False)
    TIME_BEFORE_EVENT_MIN:      pd.Timedelta    = field(init=False)
    TIME_AFTER_EVENT_MAX:       pd.Timedelta    = field(init=False)
    TIME_AFTER_EVENT_MIN:       pd.Timedelta    = field(init=False)
    EXACT_TIME_BEFORE_EVENT:    pd.Timedelta    = field(init=False)
    EXACT_TIME_AFTER_EVENT:     pd.Timedelta    = field(init=False)
    EXACT_TIME:                 bool            = field(init=False)
    SEED:                       int             = field(init=False)
    WINDOWS:                    list[np.uint16] = field(init=False)
    LOCAL_PATH:                 Path            = field(init=False)
    MUSE_PATH:                  Path            = field(init=False)
    PROCESSED_FILES_PATH:       Path            = field(init=False)
    PROCESSED_RESULTS_PATH:     Path            = field(init=False)
    PROCESSED_MUSEDATA_PATH:    Path            = field(init=False)
    SPLITS_FILE:                Path            = field(init=False)
    SUBJECTS_FILE:              Path            = field(init=False)
    RESULTS_FILES_PREFIX:       Path            = field(init=False)
    MUSEDATA_FILES_PREFIX:      Path            = field(init=False)
    MUSEDATA_COLUMNS:           list[str]       = field(init=False)
    MUSEDATA_IMPORTANT_COLUMNS: list[str]       = field(init=False)
    LETTERS:                    list[str]       = field(init=False)
    MODELS_PATH:                Path            = field(init=False)
    STATS_PATH:                 Path            = field(init=False)
    CONFUSION_MATRICES_PATH:    Path            = field(init=False)
    STATS_MODELS_FILE:          Path            = field(init=False)
    STATS_AGGREGATED_FILE:      Path            = field(init=False)
    CONFUSION_MATRICES_PREFIX:  Path            = field(init=False)
    CONFUSION_MATRICES_SUFFIX:  str             = field(init=False)
    EEGDATA_FILE:               Path            = field(init=False)

    def __post_init__(self):
        self.ROOT_PATH = Path(__file__).resolve().parent / self.ROOT_BASE_PATH
        self.CONFIG_INI_FILE = self.ROOT_PATH / self.RELATIVE_PATH

        if not self.CONFIG_INI_FILE.exists():
            logger.error(f"Configuration file '{self.CONFIG_INI_FILE}' not found.")
            raise FileNotFoundError(f"Configuration file '{self.CONFIG_INI_FILE}' not found.")
        self.__config.read(self.CONFIG_INI_FILE, encoding='utf-8')

        cfg_data  = self.__config['Data']
        cfg_rand  = self.__config['Random']
        cfg_paths = self.__config['Paths']

        self.SAMPLING_RATE       = np.uint16(cfg_data['sampling_rate'])
        self.TRAINING_DATA_RATIO = np.float64(cfg_data['training_data_ratio'])
        self.TOTAL_FOLDS         = np.uint8(cfg_data['folds_number'])
        self.WINDOWS             = [np.uint16(x) for x in cfg_data['window_sizes'].split(',')]
        self.MUSEDATA_COLUMNS    = cfg_data['museData_columns'].replace(" ", "").split(',')
        self.MUSEDATA_IMPORTANT_COLUMNS    = cfg_data['museData_important_columns'].replace(" ", "").split(',')
        self.SAMPLING_OFFSET     = np.uint16(1000.0 / self.SAMPLING_RATE)
        gcd_windows              = np.uint16(gcd(*self.WINDOWS))
        self.SAMPLING_OFFSET     = np.uint16(max(d for d in range(self.SAMPLING_OFFSET, 0, -1) if gcd_windows % d == 0))

        self.LETTERS = [letter for letter in cfg_data['letters'].replace(" ", "").split(',')]

        self.TIME_BEFORE_EVENT_MAX   = pd.Timedelta(int(cfg_data['time_before_event_max']), unit='ms')
        self.TIME_BEFORE_EVENT_MIN   = pd.Timedelta(int(cfg_data['time_before_event_min']), unit='ms')
        self.TIME_AFTER_EVENT_MAX    = pd.Timedelta(int(cfg_data['time_after_event_max']), unit='ms')
        self.TIME_AFTER_EVENT_MIN    = pd.Timedelta(int(cfg_data['time_after_event_min']), unit='ms')
        self.EXACT_TIME_BEFORE_EVENT = pd.Timedelta(int(cfg_data['exact_time_before_event']), unit='ms')
        self.EXACT_TIME_AFTER_EVENT  = pd.Timedelta(int(cfg_data['exact_time_after_event']), unit='ms')
        if (self.EXACT_TIME_BEFORE_EVENT > pd.Timedelta(0) or self.EXACT_TIME_AFTER_EVENT > pd.Timedelta(0)):
            self.EXACT_TIME = True
        else:
            self.EXACT_TIME = False

        self.SEED = int(cfg_rand['seed'])
        random.seed(self.SEED)

        self.LOCAL_PATH              = self.ROOT_PATH / cfg_paths['local_raw_data_path']
        self.MUSE_PATH               = self.ROOT_PATH / cfg_paths['muse_raw_data_path']
        self.PROCESSED_FILES_PATH    = self.ROOT_PATH / cfg_paths['processed_files_path']
        self.PROCESSED_RESULTS_PATH  = self.ROOT_PATH / cfg_paths['processed_results_path']
        self.PROCESSED_MUSEDATA_PATH = self.ROOT_PATH / cfg_paths['processed_musedata_path']
        self.SPLITS_FILE             = self.ROOT_PATH / cfg_paths['splits_file']
        self.SUBJECTS_FILE           = self.ROOT_PATH / cfg_paths['subjects_file']
        self.RESULTS_FILES_PREFIX    = self.ROOT_PATH / cfg_paths['results_files_prefix']
        self.MUSEDATA_FILES_PREFIX   = self.ROOT_PATH / cfg_paths['museData_files_prefix']
        self.MODELS_PATH             = self.ROOT_PATH / cfg_paths['models_path']

        self.STATS_PATH                = self.ROOT_PATH / cfg_paths['stats_path']
        self.CONFUSION_MATRICES_PATH   = self.ROOT_PATH / cfg_paths['confusion_matrices_path']
        self.STATS_MODELS_FILE         = self.ROOT_PATH / cfg_paths['stats_models_file']
        self.STATS_AGGREGATED_FILE     = self.ROOT_PATH / cfg_paths['stats_aggregated_file']
        self.CONFUSION_MATRICES_PREFIX = self.ROOT_PATH / cfg_paths['confusion_matrices_files_prefix']
        self.CONFUSION_MATRICES_SUFFIX = cfg_paths['confusion_matrices_files_extension']

        self.EEGDATA_FILE = self.ROOT_PATH / cfg_paths['eegdata_file']
