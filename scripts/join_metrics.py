"""
Author: Jesús Maldonado
Description: This script joins multiple metrics JSON files into a single CSV file.
"""

from pathlib import Path
import json
from typing import Iterator
import pandas as pd

from config import Config

def join_metrics_json_to_csv(json_files: list[Path], output_csv: Path) -> None:
    """Join multiple metrics JSON files into a single CSV file.
    Args:
        json_files (list): List of paths to JSON files containing metrics data.
        output_csv (Path): Path to the output CSV file where combined data will be saved.
    """
    joined = []
    
    for metric_file in json_files:
        data_file = Path(str(metric_file).replace('_metrics', ''))
        
        with open(metric_file, 'r') as mf, open(data_file, 'r') as df:
            print(mf)
            metrics = json.load(mf)
            print(df)
            data = json.load(df)
            # Flatten the confusion matrix into TP, TN, FP, FN
            tp = metrics['confusion_matrix'][1][1]
            tn = metrics['confusion_matrix'][0][0]
            fp = metrics['confusion_matrix'][0][1]
            fn = metrics['confusion_matrix'][1][0]
            
            # Create a flat dictionary for the row
            row = {
                'model_type': data['model_type'],
                'model_id': data['id'],
                'model': data['version'],
                'window_size': data['window_size'],
                'hidden_channels': data['params']['hidden_channels'],
                'kernel_sizes': data['params']['kernel_sizes'],
                'dropout_rate': data['params']['dropout_rate'],
                'epochs': data['training']['0'],
                'loss': metrics['loss'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'specificity': metrics['specificity'],
                'g_mean': metrics['g_mean'],
                'f1_score': metrics['f1_score'],
                'f1_score_macro': metrics['f1_score_macro'],
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn
            }
            joined.append(row)
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(joined)
    df.to_csv(output_csv, index=False)

def get_metrics_json_files(directory: Path) -> list[Path]:
    """Get all metrics JSON files in the specified directory.
    Args:
        directory (Path): Directory to search for JSON files.
    Returns:
        list: List of paths to JSON files.
    """
    return list(directory.glob('*_metrics.json'))

if __name__ == "__main__":
    metrics_dir = '2025_06_02'
    metrics_full_dir = Config().MODELS_PATH / metrics_dir
    output_csv = Config().MODELS_PATH / 'metrics' / (metrics_dir+f"_combined_metrics.csv")
    
    if not output_csv.parent.exists():
        output_csv.parent.mkdir(parents=True, exist_ok=True)
    json_files = get_metrics_json_files(metrics_full_dir)
    if json_files:
        join_metrics_json_to_csv(json_files, output_csv)
        print(f"Combined metrics saved to {output_csv}")
    else:
        print("No metrics JSON files found.")
