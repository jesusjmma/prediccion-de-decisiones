"""
Author: Jesús Maldonado
Description: This script creates processed data files from raw data files, including splitting data into training and testing sets, and saving the processed data in a specified directory.
"""
from pathlib import Path
import pandas as pd
from src.EEGData import EEGData, EEGTimeSeries
from config import Config
from logger_utils import setup_logger
logger = setup_logger("DEBUG")

def save_data(data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]],
              file: Path | None = None
              ) -> Path:
    """
    Save the data to a file.
    Args:
        data (dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]): Dictionary with the data.
    """
    if file is None:
        file = Config().EEGDATA_FILE

    logger.info(f"Saving data to {file}")
    path: Path = EEGData.save_data(file, data)
    logger.info(f"Data saved to {path}")
    return path

def create_data() -> dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]:
    """
    Create the data for the model.
    Returns:
        eeg_data (EEGData): EEGData object.
    """
    logger.info("Creating data")
    eeg_data: EEGData = EEGData.initialize()
    logger.info("Data created")
    logger.info("Extracting data")
    data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]] = eeg_data.get_data()
    logger.info("Data extracted")
    return data

def validate_data(data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]]) -> None:
    for key, value in data.items():
        if isinstance(value, set):
            logger.info(f"{key}: {len(value)} items")
            for item in value:
                logger.info(f"{item.subject}/{item.trial}/{item.response}: {item.eeg.shape} (shape)")
        elif isinstance(value, dict):
            logger.info(f"{key}: {len(value)} keys")
            for sub_key, sub_value in value.items():
                logger.info(f"{sub_key}: {len(sub_value)} items")
                for item in sub_value:
                    logger.info(f"{item.subject}/{item.trial}/{item.response}: {item.eeg.shape} (shape)")
        else:
            logger.warning(f"Unexpected type for key '{key}': {type(value)}")


if __name__ == "__main__":
    #file: Path = Path("aaa")
    data: dict[str, set[EEGTimeSeries] | dict[int, set[EEGTimeSeries]]] = create_data()
    #save_data(data, file)
    validate_data(data)
