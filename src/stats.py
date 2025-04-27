"""
Author: Jesús Maldonado
Description: Statistics and metrics for models evaluation.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.metrics import geometric_mean_score as gmean
from sklearn.metrics import (
    confusion_matrix as cm,
    accuracy_score as acc,
    precision_score as prec,
    recall_score as rec,
    f1_score as f1
)

from scripts.config import Config
from scripts.logger_utils import setup_logger

logger = setup_logger(name=Path(__file__).name, level=10)

@dataclass(slots=True)
class Metrics:
    model:            str
    fold:             int
    y_true:           np.ndarray
    y_pred:           np.ndarray
    training_time:    float
    final_loss:       float
    accuracy:         float             = -1.0
    precision:        float             = -1.0
    recall:           float             = -1.0
    specificity:      float             = -1.0
    f1_score:         float             = -1.0
    f1_score_macro:   float             = -1.0
    g_mean:           float             = -1.0
    confusion_matrix: NDArray[np.int64] = field(default_factory=lambda: np.zeros((2, 2), dtype=np.int64))

    def __post_init__(self):
        self.accuracy = float(acc(self.y_true, self.y_pred))
        self.precision = float(prec(self.y_true, self.y_pred, zero_division=0))
        self.recall = float(rec(self.y_true, self.y_pred, zero_division=0))
        self.confusion_matrix = cm(self.y_true, self.y_pred, labels=[0, 1])
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        self.specificity = float(tn / (tn + fp)) if (tn + fp) != 0 else -1.0
        self.f1_score = float(f1(self.y_true, self.y_pred, zero_division=0))
        self.f1_score_macro = float(f1(self.y_true, self.y_pred, average='macro', zero_division=0))
        self.g_mean = float(gmean(self.y_true, self.y_pred))

        logger.info(f"Metrics created: Model={self.model}, Fold={self.fold}, Accuracy={self.accuracy:.4f}")

        if self.accuracy < 0.5:
            logger.warning(f"Low accuracy ({self.accuracy:.4f}) for Model={self.model}, Fold={self.fold}")
        if self.specificity != -1.0 and self.specificity < 0.5:
            logger.warning(f"Low specificity ({self.specificity:.4f}) for Model={self.model}, Fold={self.fold}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Metrics object to a dictionary.
        
        Returns:
            dict: Dictionary representation of the Metrics object.
        """
        logger.debug(f"Converting Metrics to dict: Model={self.model}, Fold={self.fold}")
        data = asdict(self)
        data.pop('confusion_matrix', None)
        return data
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the Metrics object to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame representation of the Metrics object.
        """
        return pd.DataFrame([self.to_dict()])
    
    def log_summary(self) -> None:
        """Log the summary of the metrics for the model and fold."""
        logger.info(f"===== METRICS SUMMARY FOR [{self.model}] [fold {self.fold}] =====")
        logger.info(f"Training Time (s):    {self.training_time:.2f}")
        logger.info(f"Final Loss:           {self.final_loss:.6f}")
        logger.info(f"Accuracy:             {self.accuracy:.4f}")
        logger.info(f"Precision:            {self.precision:.4f}")
        logger.info(f"Recall (Sensitivity): {self.recall:.4f}")
        logger.info(f"Specificity:          {self.specificity:.4f}")
        logger.info(f"F1-Score:             {self.f1_score:.4f}")
        logger.info(f"F1-Score (Macro):     {self.f1_score_macro:.4f}")
        logger.info(f"G-Mean:               {self.g_mean:.4f}")
        logger.info(f"Confusion Matrix:\n{self.confusion_matrix}")
        logger.info("==============================================================")


class Stats:
    def __init__(self):
        Config().STATS_PATH.mkdir(parents=True, exist_ok=True)
        Config().CONFUSION_MATRICES_PATH.mkdir(parents=True, exist_ok=True)
        self.models_metrics:   list[Metrics] = []
        self.stats_models:     pd.DataFrame  = pd.DataFrame()
        self.stats_aggregated: pd.DataFrame  = pd.DataFrame()

    def add_metric(self, model_name: str, fold: int, y_true: np.ndarray, y_pred: np.ndarray, train_time: float, final_loss: float) -> None:
        model_metrics = Metrics(model_name, fold, y_true, y_pred, train_time, final_loss)
        self.models_metrics.append(model_metrics)
        model_metrics.log_summary()
        self.stats_models = pd.concat([self.stats_models, model_metrics.to_dataframe()], ignore_index=True)
        self.save_confusion_matrix(model_name, fold, model_metrics)

    def save_confusion_matrix(self, model_name: str, fold: int, matrix: np.ndarray | Metrics) -> None:
        file_name = f"{Config().CONFUSION_MATRICES_PREFIX}_{model_name}_fold{fold}{Config().CONFUSION_MATRICES_SUFFIX}"
        if isinstance(matrix, Metrics):
            matrix = matrix.confusion_matrix
        np.save(Config().CONFUSION_MATRICES_PATH / file_name, matrix)

    def save_stats_models(self) -> None:
        self.stats_models.to_csv(Config().STATS_MODELS_FILE, index=False)
        logger.info(f"Saved general statistics CSV at {Config().STATS_MODELS_FILE}")

    def load_stats_models(self) -> pd.DataFrame:
        if not Config().STATS_MODELS_FILE.exists():
            logger.error(f"No statistics file found at {Config().STATS_MODELS_FILE}")
            raise FileNotFoundError(f"No statistics file found at {Config().STATS_MODELS_FILE}")
        
        logger.info(f"Loading general statistics CSV from {Config().STATS_MODELS_FILE}")
        self.stats_models = pd.read_csv(Config().STATS_MODELS_FILE)

        if self.stats_models.empty:
            logger.warning(f"Statistics file is empty: {Config().STATS_MODELS_FILE}")
        else:
            logger.info(f"Loaded {len(self.stats_models)} records from {Config().STATS_MODELS_FILE}")

        # Create a Metrics object for each row
        for index, row in self.stats_models.iterrows():
            model_metrics = Metrics(
                model=row['model'],
                fold=row['fold'],
                y_true=np.array(row['y_true'].split(','), dtype=int),
                y_pred=np.array(row['y_pred'].split(','), dtype=int),
                training_time=row['training_time'],
                final_loss=row['final_loss']
            )
            self.models_metrics.append(model_metrics)
            model_metrics.log_summary()

        return self.stats_models

    def aggregate_stats(self) -> pd.DataFrame:
        if self.stats_models.empty:
            logger.info("No statistics loaded. Loading from CSV.")
            self.load_stats_models()
        
        logger.info("Aggregating statistics...")

        # Group by model and fold, and calculate the mean of the metrics
        aggregation = self.stats_models.drop(columns=['y_true', 'y_pred']).groupby('model').agg(['mean', 'std', 'min', 'max']).drop(columns='fold')
        aggregation.columns = ['_'.join(col).strip() for col in aggregation.columns.values]
        aggregation.reset_index(inplace=True)
        self.stats_aggregated = aggregation
        self.save_aggregated_stats()
        return aggregation

    def save_aggregated_stats(self) -> None:
        self.stats_aggregated.to_csv(Config().STATS_AGGREGATED_FILE, index=False)
        logger.info(f"Saved aggregated statistics CSV at {Config().STATS_AGGREGATED_FILE}")
    
    def load_aggregated_stats(self) -> pd.DataFrame:
        if not Config().STATS_AGGREGATED_FILE.exists():
            logger.error(f"No aggregated statistics file found at {Config().STATS_AGGREGATED_FILE}")
            raise FileNotFoundError(f"No aggregated statistics file found at {Config().STATS_AGGREGATED_FILE}")
        
        logger.info(f"Loading aggregated statistics from {Config().STATS_AGGREGATED_FILE}")
        self.stats_aggregated = pd.read_csv(Config().STATS_AGGREGATED_FILE)

        if self.stats_aggregated.empty:
            logger.warning(f"Aggregated statistics file is empty: {Config().STATS_AGGREGATED_FILE}")
        else:
            logger.info(f"Loaded {len(self.stats_aggregated)} records from {Config().STATS_AGGREGATED_FILE}")

        return self.stats_aggregated

    def print_summary(self) -> None:
        try:
            df = pd.read_csv(Config().STATS_AGGREGATED_FILE)
            logger.info(f"Loaded aggregated statistics from {Config().STATS_AGGREGATED_FILE}")
            logger.info("Aggregated Statistics:")
            logger.info(df)
        except FileNotFoundError:
            logger.error(f"Aggregated statistics file not found: {Config().STATS_AGGREGATED_FILE}. Run aggregate_stats() first.")

    def plot_confusion_matrix(self, model_name: str, fold: int, width: int = 6, height: int = 5) -> None:
        file_name = f"{Config().CONFUSION_MATRICES_PREFIX}_{model_name}_fold{fold}{Config().CONFUSION_MATRICES_SUFFIX}"
        matrix_file = Config().CONFUSION_MATRICES_PATH / file_name
        logger.debug(f"Loading confusion matrix from {matrix_file}")
        if not matrix_file.exists():
            logger.error(f"Confusion matrix file not found: {matrix_file}")
            raise FileNotFoundError(f"Confusion matrix file not found: {matrix_file}")

        try:
            matrix = np.load(matrix_file)
        except Exception as e:
            logger.error(f"Error loading confusion matrix [{model_name}] [fold {fold}]: {e}")
            raise

        plt.figure(figsize=(width, height))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix for {model_name} (fold {fold})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plot_file = f"{Config().CONFUSION_MATRICES_PREFIX}_{model_name}_fold{fold}.png"
        plt.savefig(Config().CONFUSION_MATRICES_PATH / f"{plot_file}", dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix plot at {Config().CONFUSION_MATRICES_PATH / plot_file}") 
        plt.show()
        plt.close()       

# Example usage
if __name__ == "__main__":
    stats = Stats()
    
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])

    # --- Ejemplo modelo CNN1D ---
    cnn1d_0 = np.array([1, 1, 0, 0, 1, 1, 1, 0])
    cnn1d_1 = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    cnn1d_2 = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    stats.add_metric("CNN1D", 0, y_true, cnn1d_0, train_time=120.5, final_loss=0.35)
    stats.add_metric("CNN1D", 1, y_true, cnn1d_1, train_time=118.2, final_loss=0.33)
    stats.add_metric("CNN1D", 2, y_true, cnn1d_2, train_time=119.0, final_loss=0.34)

    # --- Ejemplo modelo CNN2D ---
    cnn2d_0 = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    cnn2d_1 = np.array([0, 1, 1, 0, 0, 0, 1, 1])
    cnn2d_2 = np.array([0, 1, 0, 0, 1, 0, 1, 0])
    stats.add_metric("CNN2D", 0, y_true, cnn2d_0, train_time=140.7, final_loss=0.40)
    stats.add_metric("CNN2D", 1, y_true, cnn2d_1, train_time=138.9, final_loss=0.38)
    stats.add_metric("CNN2D", 2, y_true, cnn2d_2, train_time=139.5, final_loss=0.39)

    # --- Ejemplo modelo LSTM ---
    lstm_0 = np.array([0, 1, 1, 1, 1, 0, 0, 0])
    lstm_1 = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    lstm_2 = np.array([1, 1, 1, 0, 1, 0, 1, 0])
    stats.add_metric("LSTM", 0, y_true, lstm_0, train_time=200.2, final_loss=0.45)
    stats.add_metric("LSTM", 1, y_true, lstm_1, train_time=198.7, final_loss=0.43)
    stats.add_metric("LSTM", 2, y_true, lstm_2, train_time=199.5, final_loss=0.44)

    # --- Ejemplo modelo GRU ---
    gru_0 = np.array([0, 1, 0, 0, 1, 0, 1, 0])
    gru_1 = np.array([0, 1, 1, 0, 1, 1, 1, 0])
    gru_2 = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    stats.add_metric("GRU", 0, y_true, gru_0, train_time=180.1, final_loss=0.42)
    stats.add_metric("GRU", 1, y_true, gru_1, train_time=179.4, final_loss=0.41)
    stats.add_metric("GRU", 2, y_true, gru_2, train_time=179.9, final_loss=0.40)

    # --- Ejemplo modelo MLP (perceptrón multicapa) ---
    mlp_0 = np.array([1, 1, 1, 0, 1, 0, 1, 0])
    mlp_1 = np.array([0, 0, 1, 0, 1, 0, 1, 0])
    mlp_2 = np.array([0, 1, 1, 0, 1, 0, 0, 0])
    stats.add_metric("MLP", 0, y_true, mlp_0, train_time=95.3, final_loss=0.50)
    stats.add_metric("MLP", 1, y_true, mlp_1, train_time=94.7, final_loss=0.48)
    stats.add_metric("MLP", 2, y_true, mlp_2, train_time=95.0, final_loss=0.49)

    stats.save_stats_models()
    agg = stats.aggregate_stats()
    stats.print_summary()
    stats.plot_confusion_matrix("CNN1D", 0)
