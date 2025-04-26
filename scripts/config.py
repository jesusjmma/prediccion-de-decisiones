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

import numpy as np

from logger_utils import setup_logger
logger = setup_logger(name=Path(__file__).name, level=10)

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
    ROOT_BASE_PATH:          str             = ".."
    RELATIVE_PATH:           str             = "config.ini"
    config:                  ConfigParser    = field(default_factory=ConfigParser, init=False)

    # Campos que se calculan en __post_init__
    ROOT_PATH:               Path            = field(init=False)
    CONFIG_INI_FILE:         Path            = field(init=False)
    SAMPLING_RATE:           np.uint16       = field(init=False)
    SAMPLING_OFFSET:         np.uint16       = field(init=False)
    TRAINING_DATA_RATIO:     np.float64      = field(init=False)
    TOTAL_FOLDS:             np.uint8        = field(init=False)
    SEED:                    int             = field(init=False)
    WINDOWS:                 list[np.uint16] = field(init=False)
    LOCAL_PATH:              Path            = field(init=False)
    MUSE_PATH:               Path            = field(init=False)
    PROCESSED_FILES_PATH:    Path            = field(init=False)
    PROCESSED_RESULTS_PATH:  Path            = field(init=False)
    PROCESSED_MUSEDATA_PATH: Path            = field(init=False)
    SPLITS_FILE:             Path            = field(init=False)
    SUBJECTS_FILE:           Path            = field(init=False)
    RESULTS_FILES_PREFIX:    Path            = field(init=False)
    MUSEDATA_FILES_PREFIX:   Path            = field(init=False)
    MUSEDATA_COLUMNS:        list[str]       = field(init=False)

    def __post_init__(self):
        self.ROOT_PATH = Path(__file__).resolve().parent / self.ROOT_BASE_PATH
        self.CONFIG_INI_FILE = self.ROOT_PATH / self.RELATIVE_PATH

        if not self.CONFIG_INI_FILE.exists():
            logger.error(f"Configuration file '{self.CONFIG_INI_FILE}' not found.")
            raise FileNotFoundError(f"Configuration file '{self.CONFIG_INI_FILE}' not found.")
        self.config.read(self.CONFIG_INI_FILE, encoding='utf-8')

        cfg_data  = self.config['Data']
        cfg_rand  = self.config['Random']
        cfg_paths = self.config['Paths']

        self.SAMPLING_RATE       = np.uint16(cfg_data['sampling_rate'])
        self.TRAINING_DATA_RATIO = np.float64(cfg_data['training_data_ratio'])
        self.TOTAL_FOLDS         = np.uint8(cfg_data['folds_number'])
        self.WINDOWS             = [np.uint16(x) for x in cfg_data['window_sizes'].split(',')]
        self.MUSEDATA_COLUMNS    = cfg_data['museData_columns'].replace(" ", "").split(',')
        self.SAMPLING_OFFSET     = np.uint16(1000.0 / self.SAMPLING_RATE)
        gcd_windows              = np.uint16(gcd(*self.WINDOWS))
        self.SAMPLING_OFFSET     = np.uint16(max(d for d in range(self.SAMPLING_OFFSET, 0, -1) if gcd_windows % d == 0))

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
