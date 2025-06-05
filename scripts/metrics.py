"""
Author: Jesús Maldonado
Description: This module defines a Metrics class that calculates and stores various evaluation metrics for a binary classification model.
"""

import json
from pathlib import Path

import torch
from torch import Tensor
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from imblearn.metrics import geometric_mean_score as sk_gmean
from sklearn.metrics import (
    confusion_matrix as sk_cm,
    accuracy_score as sk_acc,
    precision_score as sk_prec,
    recall_score as sk_rec,
    f1_score as sk_f1
)

from scripts.config import Config
from scripts.logger_utils import setup_logger

logger = setup_logger(level="DEBUG")

class Metrics:

    def __init__(self):
        self._y_trues:  list[int] = []
        self._y_preds:  list[int] = []
        self._y_scores: list[int] = []
        self._TP:      int
        self._TN:      int
        self._FP:      int
        self._FN:      int
        self._loss:    dict[str, float|int]

        self.reset()

    @property
    def loss(self) -> float:
        """The final loss (cost) of the loss function measures how poorly the model predicts overall.
        The lower the better.
        It is used to compare models or tune hyperparameters.
        """
        return self._loss["loss_sum"] / self._loss["num_samples"] if self._loss["num_samples"] > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """ACCURACY = (TP + TN) / (TP + TN + FP + FN)
        The accuracy measures the proportion of correct predictions (both true positives and true negatives) out of all predictions made.
        The higher the better.
        It is used to evaluate the overall performance of the model, as it indicates how many predictions were correct.
        It is not suitable for imbalanced datasets, as it can be misleading.
        
        Returns:
            float: The accuracy of the model.
        """
        total = self._TP + self._TN + self._FP + self._FN
        acc = float(sk_acc(self._y_trues, self._y_preds))
        acc2 = (self._TP + self._TN) / total if total > 0 else 0.0
        logger.debug(f"Accuracy (sklearn): {acc}")
        logger.debug(f"Accuracy  (manual): {acc2}")
        return acc

    @property
    def precision(self) -> float:
        """PRECISION = TP / (TP + FP)
        The precision measures the proportion of true positive predictions out of all positive predictions made by the model.
        The higher the better.
        It is used to evaluate the model's ability to correctly identify positive instances, especially in cases where false positives are costly.
        
        Returns:
            float: The precision of the model.
        """
        prec = float(sk_prec(self._y_trues, self._y_preds))
        prec2 = self._TP / (self._TP + self._FP) if (self._TP + self._FP) > 0 else 0.0
        logger.debug(f"Precision (sklearn): {prec}")
        logger.debug(f"Precision  (manual): {prec2}")
        return prec
    
    @property
    def _precision_negative(self) -> float:
        """NEGATIVE PRECISION = TN / (TN + FN)
        The negative precision measures the proportion of true negative predictions out of all negative predictions made by the model.
        The higher the better.
        It is used to evaluate the model's ability to correctly identify negative instances, especially in cases where false negatives are costly.
        
        Returns:
            float: The negative precision of the model.
        """
        return self._TN / (self._TN + self._FN) if (self._TN + self._FN) > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """RECALL = TP / (TP + FN)
        The recall (also known as sensitivity or true positive rate) measures the proportion of true positive predictions out of all actual positive instances.
        The higher the better.
        It is used to evaluate the model's ability to correctly identify all positive instances, especially in cases where false negatives are costly.
        
        Returns:
            float: The recall of the model.
        """
        rec = float(sk_rec(self._y_trues, self._y_preds))
        rec2 = self._TP / (self._TP + self._FN) if (self._TP + self._FN) > 0 else 0.0
        logger.debug(f"Recall (sklearn): {rec}")
        logger.debug(f"Recall  (manual): {rec2}")
        return rec

    @property
    def specificity(self) -> float:
        """SPECIFICITY = TN / (TN + FP)
        The specificity (also known as true negative rate) measures the proportion of true negative predictions out of all actual negative instances.
        The higher the better.
        It is used to evaluate the model's ability to correctly identify negative instances, especially in cases where false positives are costly.
        
        Returns:
            float: The specificity of the model.
        """
        return self._TN / (self._TN + self._FP) if (self._TN + self._FP) > 0 else 0.0
    
    @property
    def g_mean(self) -> float:
        """G-MEAN = sqrt(recall * specificity)
        The G-mean is the geometric mean of recall and specificity (negative recall), providing a balance between the two metrics.
        The higher the better.
        It is used to evaluate the model's performance when there is an uneven class distribution, as it considers both false positives and false negatives.
        
        Returns:
            float: The G-mean of the model.
        """
        gmean = sk_gmean(self._y_trues, self._y_preds)
        gmean2 = np.sqrt(self.recall * self.specificity)
        logger.debug(f"G-Mean (sklearn): {gmean}")
        logger.debug(f"G-Mean  (manual): {gmean2}")
        return gmean
    
    @property
    def f1_score(self) -> float:
        """F1-SCORE = 2 * (precision * recall) / (precision + recall)
        The F1-score is the harmonic mean of precision and recall, providing a balance between the two metrics.
        The higher the better.
        It is used to evaluate the model's performance when there is an uneven class distribution, as it considers both false positives and false negatives.
        
        Returns:
            float: The F1-score of the model.
        """
        f1_score = float(sk_f1(self._y_trues, self._y_preds))
        f1_score2 = 2 * (self.precision * self.recall) / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0.0
        logger.debug(f"F1-Score (sklearn): {f1_score}")
        logger.debug(f"F1-Score  (manual): {f1_score2}")
        return f1_score
    
    @property
    def _f1_score_negative(self) -> float:
        """F1-SCORE NEGATIVE = 2 * (precision_negative * specificity) / (precision_negative + specificity)
        The negative F1-score is the harmonic mean of negative precision and negative recall, providing a balance between the two metrics.
        The higher the better.
        It is used to evaluate the model's performance when there is an uneven class distribution, as it considers both false positives and false negatives.
        
        Returns:
            float: The negative F1-score of the model.
        """
        return 2 * (self._precision_negative * self.specificity) / (self._precision_negative + self.specificity) if (self._precision_negative + self.specificity) > 0 else 0.0
    
    @property
    def f1_score_macro(self) -> float:
        """F1-SCORE MACRO = (F1-score + F1-score negative) / 2
        The macro F1-score is the average of the F1-score and the negative F1-score.
        The higher the better.
        It is used to evaluate the model's performance when there is an uneven class distribution, as it considers both false positives and false negatives.
        
        Returns:
            float: The macro F1-score of the model.
        """
        f1_score_macro = float(sk_f1(self._y_trues, self._y_preds, average='macro'))
        f1_score_macro2 = (self.f1_score + self._f1_score_negative) / 2.0
        logger.debug(f"F1-Score (sklearn): {f1_score_macro}")
        logger.debug(f"F1-Score  (manual): {f1_score_macro2}")
        return f1_score_macro
    
    @property
    def confusion_matrix(self) -> NDArray[np.int64]:
        """Returns the confusion matrix as a NumPy array.
        
        Returns:
            NDArray[np.int64]: The confusion matrix.
        """
        cm = sk_cm(self._y_trues, self._y_preds)
        cm2 = np.array([[self._TN, self._FP], [self._FN, self._TP]], dtype=np.int64)
        logger.debug(f"Confusion Matrix (sklearn):\n{cm}")
        logger.debug(f"Confusion Matrix  (manual):\n{cm2}")
        return cm

    @property
    def to_dict(self) -> dict[str, float|NDArray[np.int64]]:
        """Returns a dictionary with all metrics.
        
        Returns:
            dict[str, float]: A dictionary containing all metrics.
        """
        return {
            "loss":              self.loss,
            "accuracy":          self.accuracy,
            "precision":         self.precision,
            "recall":            self.recall,
            "specificity":       self.specificity,
            "g_mean":            self.g_mean,
            "f1_score":          self.f1_score,
            "f1_score_macro":    self.f1_score_macro,
            "confusion_matrix":  self.confusion_matrix.tolist()
        }

    def _update_loss(self, loss: float, num_samples: int) -> None:
        # It is correct if and only if loss is a mean loss per batch (reduction='mean' in loss function).
        self._loss["loss_sum"] += loss * num_samples
        self._loss["num_samples"] += num_samples

    def _update_predictions(self, y_true: Tensor, y_pred: Tensor) -> None:
        self._TP += int(((y_pred == y_true) & (y_pred == 1)).sum().item())
        self._TN += int(((y_pred == y_true) & (y_pred == 0)).sum().item())
        self._FP += int(((y_pred != y_true) & (y_pred == 1)).sum().item())
        self._FN += int(((y_pred != y_true) & (y_pred == 0)).sum().item())

    def reset(self) -> None:
        """Reset the metrics to their initial state."""
        self._TP = 0
        self._TN = 0
        self._FP = 0
        self._FN = 0
        self._loss = {
            "loss_sum": 0.0,
            "num_samples": 0,
        }
        self._y_trues.clear()
        self._y_preds.clear()
        self._y_scores.clear()

    def update_metrics(self, y_true: Tensor, outputs: Tensor, loss: float) -> None:
        """Update the metrics based on true and predicted values.
        Args:
            y_true (Tensor): The true labels.
            y_pred (Tensor): The predicted labels.
            loss (Tensor): The loss value for the current batch.
        """
        probs = torch.sigmoid(outputs)
        y_pred = (probs >= 0.5).long()

        self._update_predictions(y_true, y_pred)
        self._update_loss(loss, y_true.size(0))

        self._y_trues.extend(y_true.cpu().numpy().tolist())
        self._y_preds.extend(y_pred.cpu().numpy().tolist())
        self._y_scores.extend(probs.detach().cpu().numpy().flatten().tolist())

    def log_summary(self) -> None:
        """Log the summary of the metrics."""
        logger.info("=================== METRICS SUMMARY ===================")
        logger.info(f"                Loss: {self.loss:.6f}")
        logger.info(f"            Accuracy: {self.accuracy:.4f}")
        logger.info(f"           Precision: {self.precision:.4f}")
        logger.info(f"Recall (Sensitivity): {self.recall:.4f}")
        logger.info(f"         Specificity: {self.specificity:.4f}")
        logger.info(f"              G-Mean: {self.g_mean:.4f}")
        logger.info(f"            F1-Score: {self.f1_score:.4f}")
        logger.info(f"    F1-Score (Macro): {self.f1_score_macro:.4f}")
        logger.info(f"    Confusion Matrix: \n{self.confusion_matrix}")
        logger.info("=======================================================")

    def to_csv(self, file_base_name: str) -> None:
        """Save the metrics to a CSV file.
        Args:
            file_base_name (str): The name of the file where save the metrics.
        """
        if not Config().STATS_PATH.exists():
            Config().STATS_PATH.mkdir(parents=True, exist_ok=True)
        
        if not file_base_name.endswith('.csv'):
            file_base_name += '.csv'

        file = (Config().STATS_PATH / file_base_name).resolve()
        if file.exists():
            logger.warning(f"File {file} already exists. Overwriting it.")

        df = pd.DataFrame([self.to_dict])
        df.to_csv(file, index=False)
        logger.info(f"Metrics saved to {file}")

    def to_json(self, full_file_path: Path) -> None:
            """Save the metrics to a JSON file.
            Args:
                file_base_name (str): The name of the file where save the metrics.
            """
            if full_file_path.suffix != '.json':
                full_file_path = full_file_path.with_suffix('.json')

            if full_file_path.exists():
                logger.warning(f"File {full_file_path} already exists. Overwriting it.")

            with open(full_file_path, 'w') as f:
                json.dump(self.to_dict, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Metrics saved to {full_file_path}")